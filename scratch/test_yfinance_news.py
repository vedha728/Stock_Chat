import yfinance as yf
import json

ticker_symbol = "TCS.NS"
print(f"[*] Fetching native yfinance news for {ticker_symbol}...")

try:
    ticker = yf.Ticker(ticker_symbol)
    news_items = ticker.news
    print(f"Total articles returned: {len(news_items)}\n")
    
    for i, item in enumerate(news_items):
        content = item.get("content", {})
        title = content.get("title")
        publisher = content.get("provider", {}).get("displayName")
        pub_date = content.get("pubDate")
        link = content.get("clickThroughUrl", {}).get("url")
        
        print(f"{i+1}. Publisher: {publisher} | Date: {pub_date}")
        print(f"   Title: {title}")
        print(f"   Link: {link}\n")
except Exception as e:
    print(f"[Error] Failed to fetch yfinance news: {e}")
