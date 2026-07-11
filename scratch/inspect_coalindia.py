import os
import sys
import datetime
import requests
import xml.etree.ElementTree as ET
import email.utils
import pandas as pd
from datetime import timezone, timedelta

# Ensure we can load modules from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

CSV_PATH = os.path.join(project_root, "data/processed/curated_historical_sentiment.csv")

def convert_to_ist(pub_date_str: str) -> str:
    try:
        dt = email.utils.parsedate_to_datetime(pub_date_str)
        ist_offset = timezone(timedelta(hours=5, minutes=30))
        dt_ist = dt.astimezone(ist_offset)
        return dt_ist.strftime("%Y-%m-%d")
    except Exception:
        return None

def safe_print(msg: str):
    """Prints a message by encoding/decoding as ASCII with replace errors to avoid Windows charmap crashes."""
    try:
        print(msg.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
    except Exception:
        print(msg.encode('ascii', errors='replace').decode('ascii'))

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    coal_df = df[df['Ticker'] == 'COALINDIA.NS']
    
    total = len(coal_df)
    available = coal_df['Sentiment_Available'].sum()
    missing = total - available
    missing_pct = (missing / total) * 100 if total > 0 else 0
    
    safe_print(f"Coal India Total Rows: {total}")
    safe_print(f"Coal India Sentiment Available: {available} ({100 - missing_pct:.2f}%)")
    safe_print(f"Coal India Sentiment Missing: {missing} ({missing_pct:.2f}%)")
    
    missing_dates = coal_df[coal_df['Sentiment_Available'] == 0]['Date'].tolist()
    safe_print(f"\nFound {len(missing_dates)} dates with 0 headlines after filtering.")
    
    # Pick 8 sample dates spread out across the date range to get a good sample
    if len(missing_dates) >= 8:
        indices = [
            0, 
            len(missing_dates)//7, 
            2*len(missing_dates)//7, 
            3*len(missing_dates)//7, 
            4*len(missing_dates)//7, 
            5*len(missing_dates)//7, 
            6*len(missing_dates)//7, 
            len(missing_dates)-1
        ]
        sample_dates = [missing_dates[i] for i in indices]
    else:
        sample_dates = missing_dates
        
    safe_print(f"Sample dates chosen: {sample_dates}")
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    aliases = ["coal india", "coalindia"]
    query_term = "Coal+India"
    
    for date_str in sample_dates:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        day_before = dt - datetime.timedelta(days=1)
        day_after = dt + datetime.timedelta(days=1)
        
        query_date_start = day_before.strftime("%Y-%m-%d")
        query_date_end = day_after.strftime("%Y-%m-%d")
        
        url = f"https://news.google.com/rss/search?q={query_term}+after:{query_date_start}+before:{query_date_end}&hl=en-IN&gl=IN&ceid=IN:en"
        
        safe_print("\n" + "="*80)
        safe_print(f"DATE: {date_str} (Query range: {query_date_start} to {query_date_end})")
        safe_print(f"URL: {url}")
        
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                safe_print(f"HTTP Error: {resp.status_code}")
                continue
                
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            raw_count = len(items)
            safe_print(f"RAW Headline Count (returned by Google News RSS): {raw_count}")
            
            if raw_count == 0:
                safe_print("No headlines returned in the search query (Genuine zero coverage).")
                continue
                
            filtered_by_date = 0
            filtered_by_alias = 0
            kept = 0
            
            for idx, item in enumerate(items, 1):
                title = item.find("title").text
                pub_date_raw = item.find("pubDate").text
                ist_date = convert_to_ist(pub_date_raw)
                
                title_lower = title.lower()
                has_alias = any(alias in title_lower for alias in aliases)
                
                is_date_match = (ist_date == date_str)
                
                status = "KEPT"
                if not is_date_match:
                    status = f"FILTERED (Date Mismatch: IST date is {ist_date})"
                    filtered_by_date += 1
                elif not has_alias:
                    status = f"FILTERED (Alias Mismatch: title does not contain 'coal india' or 'coalindia')"
                    filtered_by_alias += 1
                else:
                    kept += 1
                    
                safe_print(f"  {idx}. {title} | {pub_date_raw} | Status: {status}")
                
            safe_print(f"\nSummary for {date_str}:")
            safe_print(f"  Total Raw: {raw_count}")
            safe_print(f"  Filtered by Date: {filtered_by_date}")
            safe_print(f"  Filtered by Alias: {filtered_by_alias}")
            safe_print(f"  Kept: {kept}")
            
        except Exception as e:
            safe_print(f"Error querying URL: {e}")

if __name__ == "__main__":
    main()
