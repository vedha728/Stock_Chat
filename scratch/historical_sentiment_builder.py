import os
import sys
import argparse
import datetime
import random
import time
import requests
import xml.etree.ElementTree as ET
import email.utils
import pandas as pd
from datetime import timezone, timedelta

# Reconfigure stdout to use UTF-8 to avoid UnicodeEncodeError in Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

# Ensure we can load modules from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from sentiment import analyze_news_sentiment

# Curated list of 29 stocks with their query terms and title-anchor aliases
CURATED_STOCKS = [
    # IT & Tech (3)
    ("TCS.NS", "TCS", ["tcs", "tata consultancy"]),
    ("INFY.NS", "Infosys", ["infosys", "infy"]),
    ("WIPRO.NS", "Wipro", ["wipro"]),
    
    # Banking & Financials (4)
    ("HDFCBANK.NS", "HDFC+Bank", ["hdfc bank", "hdfcbank", "hdfc"]),
    ("SBIN.NS", "SBI", ["sbin", "state bank", "sbi"]), # FIX: Changed query term from SBIN to SBI
    ("ICICIBANK.NS", "ICICI+Bank", ["icici bank", "icicibank", "icici"]),
    ("BAJFINANCE.NS", "Bajaj+Finance", ["bajaj finance", "bajajfinance", "bajaj fin"]),
    
    # Energy, Oil & Commodities (3)
    ("RELIANCE.NS", "RELIANCE", ["reliance", "ril"]),
    ("ONGC.NS", "ONGC", ["ongc", "oil and natural gas"]),
    ("COALINDIA.NS", "Coal+India", ["coal india", "coalindia"]),
    
    # Auto & Components (3)
    ("MARUTI.NS", "Maruti", ["maruti"]),
    ("M&M.NS", "M%26M", ["m&m", "mahindra"]),
    ("TVSMOTOR.NS", "TVS+Motor", ["tvs motor", "tvsmotor", "tvs"]),
    
    # FMCG & Consumer (3)
    ("HINDUNILVR.NS", "Hindustan+Unilever", ["hindunilvr", "hindustan unilever", "hul"]),
    ("ITC.NS", "ITC", ["itc"]),
    ("NESTLEIND.NS", "Nestle", ["nestle"]),
    
    # Pharma & Healthcare (3)
    ("SUNPHARMA.NS", "Sun+Pharma", ["sun pharma", "sunpharma"]),
    ("CIPLA.NS", "Cipla", ["cipla"]),
    ("APOLLOHOSP.NS", "Apollo+Hospitals", ["apollo hospital", "apollo hospitals"]),
    
    # Infra & Capital Goods (3)
    ("LT.NS", "L%26T", ["l&t", "larsen", "toubro"]),
    ("BHEL.NS", "BHEL", ["bhel"]),
    ("BEL.NS", "BEL", ["bel", "bharat electronics"]),
    
    # Adani Group (2)
    ("ADANIPORTS.NS", "Adani+Ports", ["adani ports", "adaniports"]),
    ("ADANIENT.NS", "Adani+Enterprises", ["adani ent", "adanient", "adani enterprises"]),
    
    # Metals & Mining (2)
    ("TATASTEEL.NS", "Tata+Steel", ["tata steel", "tatasteel"]),
    ("HINDALCO.NS", "Hindalco", ["hindalco"]),
    
    # Retail, Real Estate & Telecom (3)
    ("TITAN.NS", "Titan", ["titan"]),
    ("DLF.NS", "DLF", ["dlf"]),
    ("BHARTIARTL.NS", "Bharti+Airtel", ["bharti airtel", "airtel"])
]

CSV_PATH = os.path.join(project_root, "data/processed/curated_historical_sentiment.csv")

def convert_to_ist(pub_date_str: str) -> str:
    """Parses GMT RFC 2822 pubDate string and converts to YYYY-MM-DD in IST timezone."""
    try:
        dt = email.utils.parsedate_to_datetime(pub_date_str)
        ist_offset = timezone(timedelta(hours=5, minutes=30))
        dt_ist = dt.astimezone(ist_offset)
        return dt_ist.strftime("%Y-%m-%d")
    except Exception:
        return None

def load_existing_data() -> set:
    """Loads existing processed keys (Ticker, Date) to support resume feature."""
    if not os.path.exists(CSV_PATH):
        return set()
    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return set()
        df['Key'] = df['Ticker'] + "_" + df['Date']
        return set(df['Key'].tolist())
    except Exception as e:
        print(f"[Warning] Failed to load existing cache CSV: {e}")
        return set()

def main():
    parser = argparse.ArgumentParser(description="Curated Sentiment Scraper")
    parser.add_argument(
        "--session", "-s",
        type=int,
        choices=[1, 2],
        help="Session number: 1 (stocks 1-15) or 2 (stocks 16-29)"
    )
    parser.add_argument(
        "--tickers", "-t",
        type=str,
        help="Comma-separated list of specific ticker symbols to scrape (e.g., SBIN.NS,ICICIBANK.NS)"
    )
    args = parser.parse_args()
    
    if not args.session and not args.tickers:
        parser.error("Either --session or --tickers must be specified.")
        
    # Define stocks to process
    if args.tickers:
        ticker_list = [t.strip().upper() for t in args.tickers.split(",")]
        stocks = [item for item in CURATED_STOCKS if item[0] in ticker_list]
        print(f"[*] Targeting {len(stocks)} specific ticker(s): {[item[0] for item in stocks]}")
    elif args.session == 1:
        stocks = CURATED_STOCKS[:15]
        print(f"[*] Starting Session 1: processing 15 stocks (from {stocks[0][0]} to {stocks[-1][0]})")
    else:
        stocks = CURATED_STOCKS[15:]
        print(f"[*] Starting Session 2: processing 14 stocks (from {stocks[0][0]} to {stocks[-1][0]})")
        
    # Generate business days in target range: 2024-06-01 to 2025-06-01
    dates = pd.date_range(start="2024-06-01", end="2025-06-01", freq='B').strftime("%Y-%m-%d").tolist()
    print(f"[*] Total dates to process per stock: {len(dates)} business days.")
    
    # Load cache keys
    processed_keys = load_existing_data()
    print(f"[*] Loaded {len(processed_keys)} already processed records from cache.")
    
    # Prepare output file headers if it doesn't exist
    if not os.path.exists(CSV_PATH):
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        pd.DataFrame(columns=[
            "Ticker", "Date", "Sentiment_Score", "Positive_Headlines", "Negative_Headlines", "Sentiment_Available"
        ]).to_csv(CSV_PATH, index=False)
        
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    total_records = len(stocks) * len(dates)
    counter = 0
    skipped_count = 0
    saved_count = 0
    
    # Start main loops
    for ticker, query_term, aliases in stocks:
        print(f"\n[*] Processing ticker: {ticker}...")
        
        for date_str in dates:
            counter += 1
            key = f"{ticker}_{date_str}"
            
            # Check cache
            if key in processed_keys:
                skipped_count += 1
                continue
                
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            day_before = dt - datetime.timedelta(days=1)
            day_after = dt + datetime.timedelta(days=1)
            
            query_date_start = day_before.strftime("%Y-%m-%d")
            query_date_end = day_after.strftime("%Y-%m-%d")
            
            # Build search URL
            url = f"https://news.google.com/rss/search?q={query_term}+after:{query_date_start}+before:{query_date_end}&hl=en-IN&gl=IN&ceid=IN:en"
            
            headlines = []
            
            try:
                # Scrape with retry and exponential backoff
                max_retries = 3
                backoff_time = 10.0
                success = False
                
                for attempt in range(1, max_retries + 1):
                    resp = requests.get(url, headers=headers, timeout=10)
                    if resp.status_code == 200:
                        root = ET.fromstring(resp.content)
                        items = root.findall(".//item")
                        
                        for item in items:
                            title = item.find("title").text
                            pub_date_raw = item.find("pubDate").text
                            
                            # Date boundary timezone filter (IST)
                            ist_date = convert_to_ist(pub_date_raw)
                            if ist_date != date_str:
                                continue
                                
                            # Relevance title keyword filter
                            title_lower = title.lower()
                            has_alias = any(alias in title_lower for alias in aliases)
                            if not has_alias:
                                continue
                                
                            headlines.append(title)
                        success = True
                        break
                    elif resp.status_code in [429, 503]:
                        print(f"  [Warning] Rate limited (HTTP {resp.status_code}) on {ticker} for {date_str}. Attempt {attempt}/{max_retries}. Backing off {backoff_time}s...")
                        time.sleep(backoff_time)
                        backoff_time *= 2.0  # Exponential backoff
                    else:
                        print(f"  [Warning] HTTP {resp.status_code} on {ticker} for {date_str}. Attempt {attempt}/{max_retries}. Backing off {backoff_time}s...")
                        time.sleep(backoff_time)
                        backoff_time *= 1.5
                        
                if not success:
                    print(f"  [Error] Failed to scrape {ticker} for {date_str} after {max_retries} attempts.")
                    
            except Exception as e:
                print(f"  [Warning] Network error on {ticker} for {date_str}: {e}")
                
            # Score sentiment
            if headlines:
                sentiment_score, pos_count, neg_count = analyze_news_sentiment(headlines)
                sentiment_available = 1
            else:
                # Default fallback
                sentiment_score = 0.0
                pos_count = 0
                neg_count = 0
                sentiment_available = 0
                
            # Write incrementally to CSV
            row_df = pd.DataFrame([{
                "Ticker": ticker,
                "Date": date_str,
                "Sentiment_Score": round(sentiment_score, 4),
                "Positive_Headlines": pos_count,
                "Negative_Headlines": neg_count,
                "Sentiment_Available": sentiment_available
            }])
            row_df.to_csv(CSV_PATH, mode='a', header=False, index=False)
            
            saved_count += 1
            processed_keys.add(key)
            
            # Print periodic status
            if saved_count % 20 == 0 or saved_count == 1:
                pct = (counter / total_records) * 100
                print(f"  [{pct:.1f}%] Progress: {counter}/{total_records} | Saved: {saved_count} | Skipped: {skipped_count} | Last: {date_str} -> Score: {sentiment_score:.2f} (Available: {sentiment_available})")
                
            # Slower randomized pacing delay (3.0s - 5.0s)
            time.sleep(random.uniform(3.0, 5.0))
            
    print(f"\n[+] Session complete! Processed: {counter} | Saved: {saved_count} | Skipped: {skipped_count}")

if __name__ == "__main__":
    main()
