import feedparser
import requests
import json
import os
import time
import re
import sys
import random
from datetime import datetime
from time import mktime
from typing import List, Dict, Set, Tuple, Any, Optional
from dotenv import load_dotenv

from google import genai
from google.genai import types

# .envファイルから環境変数を読み込む（ローカルテスト用）
load_dotenv()

# ==========================================
# 1. 設定項目 (Configuration)
# ==========================================

# --- APIキーとWebhook URLの取得 ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

# キーがない場合はエラーを出して終了
if not GEMINI_API_KEY or not SLACK_WEBHOOK_URL:
    print("Error: GEMINI_API_KEY or SLACK_WEBHOOK_URL is missing.")
    # GitHub Actionsでエラーに気づけるよう終了コード1を返す
    sys.exit(1)

# --- 動作設定 ---
BATCH_SIZE = 100               # AIに一度に送信する論文数
HISTORY_FILE = "checked_history.txt"  # 既読論文リスト
MODEL_NAME = "gemini-2.5-flash"       # Geminiモデル

# アクセス集中回避設定
STARTUP_RANDOM_DELAY_MINUTES = 1
REQUEST_INTERVAL_MIN = 1
REQUEST_INTERVAL_MAX = 3

# --- スクレイピング対策ヘッダー (UserAgent設定) ---
# マニュアルにある通り、GitHub Secrets "MyUserAgent" を優先使用
# 設定がない場合は一般的なChromeのUserAgentを使用
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
USER_AGENT = os.environ.get("MyUserAgent", DEFAULT_USER_AGENT)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

# --- 外部設定ファイルの読み込み ---

# 1. プロンプトテンプレートの読み込み (マニュアル準拠: prompts.txt)
PROMPT_TEMPLATE = ""
try:
    with open("prompts.txt", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    print("Error: prompts.txt が見つかりません。作成してください。")
    sys.exit(1)

# 2. RSS URLリストの読み込み (feed_list.txt)
RSS_URLS = []
try:
    with open("feed_list.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        RSS_URLS = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    print("Error: feed_list.txt が見つかりません。作成してください。")
    sys.exit(1)

# Geminiクライアントの初期化
client = genai.Client(api_key=GEMINI_API_KEY)


# ==========================================
# 2. ヘルパー関数 (Utility Functions)
# ==========================================

def load_history() -> Set[str]:
    """過去にチェックした論文のURLリストをファイルから読み込む"""
    if not os.path.exists(HISTORY_FILE):
        return set()
    with open(HISTORY_FILE, "r") as f:
        return set(line.strip() for line in f)

def save_history(new_links: List[str]):
    """新しくチェックした論文のURLをファイルに追記保存する"""
    with open(HISTORY_FILE, "a") as f:
        for link in new_links:
            f.write(f"{link}\n")

def clean_text(text: str) -> str:
    """HTMLタグ除去・空白整理"""
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', '', text)
    clean = clean.replace('\n', ' ')
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

def clean_source_name(title: str) -> str:
    """雑誌名の整形"""
    remove_list = [
        "ScienceDirect Publication: ", "Table of Contents", "Wiley:", "JGR:", "Advance Access"
    ]
    cleaned = title
    for prefix in remove_list:
        cleaned = cleaned.replace(prefix, "")
    return cleaned.strip().rstrip(':').strip()

def parse_elsevier_date(description: str) -> Tuple[Optional[datetime.date], str, bool]:
    """Elsevier系の日付解析"""
    if not description:
        return None, "Unknown", False
    match = re.search(r"Publication date:\s*(?:Available online\s+)?([a-zA-Z0-9\s,]+)", description)
    if not match:
        match = re.search(r"Available online\s+([a-zA-Z0-9\s,]+)", description)
    if match:
        date_text = clean_text(match.group(1))
        try:
            d = datetime.strptime(date_text, "%d %B %Y").date()
            return d, date_text, False
        except ValueError:
            pass
        try:
            d = datetime.strptime(date_text, "%B %Y").date()
            return d, date_text, True 
        except ValueError:
            pass
    return None, "Unknown", False

def get_first_author(entry: Any) -> str:
    """第一著者抽出"""
    raw_author = ""
    if hasattr(entry, 'authors') and entry.authors:
        raw_author = entry.authors[0].get('name', '')
    elif hasattr(entry, 'author') and entry.author:
        raw_author = entry.author
    elif hasattr(entry, 'dc_creator') and entry.dc_creator:
        raw_author = entry.dc_creator
    
    if (not raw_author or "Unknown" in raw_author) and hasattr(entry, 'description'):
        match = re.search(r"Author\(s\):\s*([^<]+)", entry.description)
        if match:
            raw_author = match.group(1)
    
    if not raw_author:
        return "Unknown Author"

    clean_author_text = clean_text(raw_author).replace('*', '')
    parts = re.split(r',|\s+and\s+|&', clean_author_text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts: return "Unknown Author"
    
    first_author = parts[0]
    return f"{first_author} et al." if len(parts) > 1 else first_author

def get_entry_info(entry: Any) -> Tuple[Optional[datetime.date], str, bool]:
    """日付情報の正規化"""
    if hasattr(entry, 'description'):
        d, s, m = parse_elsevier_date(entry.description)
        if d: return d, s, m
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        d = datetime.fromtimestamp(mktime(entry.published_parsed)).date()
        return d, d.strftime('%Y-%m-%d'), False
    if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
        d = datetime.fromtimestamp(mktime(entry.updated_parsed)).date()
        return d, d.strftime('%Y-%m-%d'), False
    return None, "Unknown", False

def is_target_date(date_obj: Optional[datetime.date], is_month_only: bool) -> bool:
    """日付フィルタリング（直近3日または今月以降）"""
    if date_obj is None: return True
    today = datetime.now().date()
    
    if is_month_only:
        if date_obj.year > today.year: return True
        if date_obj.year == today.year and date_obj.month >= today.month: return True
        return False
    else:
        delta = today - date_obj
        return delta.days <= 3

# ==========================================
# 3. AI分析 & 通知 (AI Analysis & Notification)
# ==========================================

def analyze_papers_batch(candidates: List[Dict]) -> Dict:
    """
    Gemini APIを使用して論文リストを一括分析する
    """
    # 論文リストのテキストを作成
    papers_text = ""
    for c in candidates:
        papers_text += f"""
        [ID: {c['id']}]
        Journal: {c['source']}
        Title: {c['title']}
        Abstract: {clean_text(c['abstract'])}
        -----------------------------------
        """

    # prompts.txt の中の {papers_text} を実際の論文リストに置換する
    # .format() だとJSONの波括弧と競合するため、replaceを使用
    final_prompt = PROMPT_TEMPLATE.replace("{papers_text}", papers_text)

    try:
        # Gemini API呼び出し
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=final_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        text_to_parse = response.text.strip()
        if text_to_parse.startswith("```json"): text_to_parse = text_to_parse[7:]
        if text_to_parse.startswith("```"): text_to_parse = text_to_parse[:-3]
        return json.loads(text_to_parse, strict=False)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {"relevant_papers": []}

def send_to_slack(paper: Dict, analysis: Dict):
    """Slack通知"""
    clean_title_text = clean_text(paper['title'])
    display_title = clean_title_text[:140] + "..." if len(clean_title_text) > 140 else clean_title_text

    summary_data = analysis.get('summary', [])
    summary_text = "N/A"
    if isinstance(summary_data, list):
        summary_text = "\n".join([f"• {item}" for item in summary_data])
    elif isinstance(summary_data, str):
        cleaned = summary_data.replace("['", "").replace("']", "").replace("', '", "\n• ").replace('", "', "\n• ")
        summary_text = f"• {cleaned}" if not cleaned.startswith("•") else cleaned

    meta_info = (
        f"URL: {paper['link']} | "
        f"Title: {clean_title_text} | "
        f"Author: {paper['author']} | "
        f"Date: {paper['iso_date']} | "
        f"Journal: {paper['source']}"
    )

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": display_title, "emoji": True}
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"*{paper['source']}* | {paper['author']} | {paper['date']}"}]
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*URL:*\n{paper['link']}"}
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*選定理由:*\n{analysis.get('reason', 'N/A')}"}
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*要約:*\n{summary_text}"}
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": meta_info}]
        }
    ]

    try:
        requests.post(SLACK_WEBHOOK_URL, json={"blocks": blocks})
    except Exception as e:
        print(f"[Network Error] {e}")

# ==========================================
# 4. メイン実行 (Main Execution)
# ==========================================
def main():
    history = load_history()
    all_candidates = []
    processed_links = []
    seen_links_in_session = set()

    # マニュアルに従い、サーバー負荷分散のため待機（GitHub Actionsで有効）
    start_delay_seconds = random.randint(0, STARTUP_RANDOM_DELAY_MINUTES * 60)
    print(f"==================================================")
    print(f"Paper Checker Started.")
    print(f"Start Delay: {start_delay_seconds}s...")
    print(f"Using UserAgent: {USER_AGENT[:50]}...")
    print(f"==================================================")
    time.sleep(start_delay_seconds)

    print(f"既読ログ数: {len(history)}")
    print("論文を巡回中...")

    # RSSフィードの巡回
    for rss_url in RSS_URLS:
        try:
            try:
                # UserAgentを設定してリクエスト
                response = requests.get(rss_url, headers=HEADERS, timeout=5)
                if response.status_code != 200:
                    print(f"Skipping {rss_url}: Status {response.status_code}")
                    continue
                feed = feedparser.parse(response.content)
            except Exception as e:
                print(f"Connection Failed: {rss_url} ({e})")
                continue

            if len(feed.entries) == 0:
                continue

            clean_journal_title = clean_source_name(feed.feed.get('title', 'Unknown Journal'))
            
            for entry in feed.entries:
                if entry.link in history: continue
                if entry.link in seen_links_in_session: continue
                seen_links_in_session.add(entry.link)

                date_obj, display_date_str, is_month_only = get_entry_info(entry)
                if not is_target_date(date_obj, is_month_only): continue
                
                iso_date_str = date_obj.strftime('%Y-%m-%d') if date_obj else datetime.now().strftime('%Y-%m-%d')

                all_candidates.append({
                    "source": clean_journal_title,
                    "title": clean_text(entry.title),
                    "link": entry.link,
                    "date": display_date_str, 
                    "iso_date": iso_date_str, 
                    "author": get_first_author(entry),
                    "abstract": getattr(entry, 'summary', 'No abstract')
                })
                processed_links.append(entry.link)
            
            # アクセス間隔のゆらぎ
            interval = random.uniform(REQUEST_INTERVAL_MIN, REQUEST_INTERVAL_MAX)
            time.sleep(interval)

        except Exception as e:
            print(f"Error processing {rss_url}: {e}")
            continue

    total_count = len(all_candidates)
    print(f"AIチェック対象: {total_count} 件")
    if total_count == 0: return

    # AIによるバッチ分析
    for i in range(0, total_count, BATCH_SIZE):
        batch = all_candidates[i : i + BATCH_SIZE]
        
        batch_for_ai = []
        for idx, item in enumerate(batch):
            batch_for_ai.append({
                "id": idx, 
                "source": item["source"], 
                "title": item["title"], 
                "abstract": item["abstract"]
            })
        
        # PROMPT_TEMPLATE と {papers_text} を使って分析
        result = analyze_papers_batch(batch_for_ai)
        
        for rel in result.get('relevant_papers', []):
            rel_id = rel.get('id')
            if rel_id is not None and 0 <= rel_id < len(batch):
                target_paper = batch[rel_id]
                print(f"  [HIT] {target_paper['source']}: {target_paper['title'][:30]}...")
                send_to_slack(target_paper, rel)
        
        time.sleep(2)

    save_history(processed_links)
    print("完了。")

if __name__ == "__main__":
    main()
