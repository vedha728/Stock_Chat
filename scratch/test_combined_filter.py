import os
import requests
import datetime
import urllib.parse
import re
from dotenv import load_dotenv

load_dotenv()

now = datetime.datetime.now()
five_days_ago = now - datetime.timedelta(days=5)
from_date_str = five_days_ago.strftime("%Y-%m-%d")
api_key = os.getenv("NEWS_API_KEY")
query = '"Tata Consultancy Services" OR "TCS"'

# Root domains for NewsAPI query (NewsAPI matches subdomains when root is passed)
root_domains = [
    "indiatimes.com",
    "moneycontrol.com",
    "livemint.com",
    "thehindubusinessline.com",
    "financialexpress.com",
    "business-standard.com",
    "reuters.com",
    "bloomberg.com",
    "cnbctv18.com",
    "ndtvprofit.com",
    "yahoo.com"
]

# Strict subdomains/domains allowed in Python post-filtering
ALLOWED_SUBDOMAINS = {
    "economictimes.indiatimes.com",
    "m.economictimes.com",
    "moneycontrol.com",
    "www.moneycontrol.com",
    "livemint.com",
    "www.livemint.com",
    "thehindubusinessline.com",
    "www.thehindubusinessline.com",
    "financialexpress.com",
    "www.financialexpress.com",
    "business-standard.com",
    "www.business-standard.com",
    "reuters.com",
    "www.reuters.com",
    "bloomberg.com",
    "www.bloomberg.com",
    "cnbctv18.com",
    "www.cnbctv18.com",
    "ndtvprofit.com",
    "www.ndtvprofit.com",
    "finance.yahoo.com"
}

# Crime/legal keywords to exclude
EXCLUDE_KEYWORDS = ["harassment", "sexual", "assault", "bail", "arrest", "accused", "court", "nida khan"]

url = (
    f"https://newsapi.org/v2/everything?"
    f"qInTitle={urllib.parse.quote_plus(query)}&"
    f"domains={','.join(root_domains)}&"
    f"from={from_date_str}&"
    f"language=en&"
    f"sortBy=relevance&"
    f"pageSize=50&"
    f"apiKey={api_key}"
)

print(f"[*] Requesting NewsAPI...")
response = requests.get(url)
data = response.json()
articles = data.get("articles", [])
print(f"Raw articles returned: {len(articles)}")

filtered_articles = []
for a in articles:
    title = a.get("title", "")
    url_str = a.get("url", "")
    domain = url_str.split("/")[2] if url_str else "Unknown"
    
    # 1. Domain Check
    if domain not in ALLOWED_SUBDOMAINS:
        print(f"[X] Filtered (Domain not allowed): {domain} | Title: {title}")
        continue
        
    # 2. Keyword Check
    title_lower = title.lower()
    has_exclude = any(w in title_lower for w in EXCLUDE_KEYWORDS)
    if has_exclude:
        print(f"[X] Filtered (Exclusion keyword): {domain} | Title: {title}")
        continue
        
    # Article passed all filters
    filtered_articles.append(a)

print(f"\n--- PASSED ARTICLES ({len(filtered_articles)}) ---")
for i, a in enumerate(filtered_articles):
    url_str = a.get("url", "")
    domain = url_str.split("/")[2] if url_str else "Unknown"
    print(f"[OK] {i+1}. {domain} | {a.get('title')}")
