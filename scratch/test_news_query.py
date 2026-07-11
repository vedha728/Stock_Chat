import os
import requests
import datetime
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

now = datetime.datetime.now()
five_days_ago = now - datetime.timedelta(days=5)
from_date_str = five_days_ago.strftime("%Y-%m-%d")
api_key = os.getenv("NEWS_API_KEY")
query = '"Tata Consultancy Services" OR "TCS"'

def test_query(domains_list):
    domains_str = ",".join(domains_list)
    url = (
        f"https://newsapi.org/v2/everything?"
        f"qInTitle={urllib.parse.quote_plus(query)}&"
        f"domains={domains_str}&"
        f"from={from_date_str}&"
        f"language=en&"
        f"sortBy=relevance&"
        f"pageSize=50&"
        f"apiKey={api_key}"
    )
    r = requests.get(url).json()
    articles = r.get("articles", [])
    print(f"--- Testing domains: {domains_list} ---")
    print(f"Results count: {len(articles)}")
    for i, a in enumerate(articles):
        print(f"  {i+1}. Source: {a.get('source', {}).get('name')} | Domain: {a.get('url').split('/')[2]} | Title: {a.get('title')}")
    print()

# Test 1: Our current domains list
test_query([
    "economictimes.indiatimes.com", "moneycontrol.com", "livemint.com",
    "financialexpress.com", "business-standard.com", "reuters.com",
    "bloomberg.com", "cnbctv18.com", "ndtvprofit.com", "finance.yahoo.com"
])

# Test 2: Adding 'thehindubusinessline.com', 'livemint.com' subdomains, and others
test_query([
    "economictimes.indiatimes.com", "moneycontrol.com", "livemint.com", "www.livemint.com",
    "financialexpress.com", "business-standard.com", "reuters.com",
    "bloomberg.com", "cnbctv18.com", "ndtvprofit.com", "finance.yahoo.com",
    "thehindubusinessline.com", "www.thehindubusinessline.com"
])

# Test 3: Broad root domains
test_query([
    "indiatimes.com", "moneycontrol.com", "livemint.com", "thehindubusinessline.com",
    "financialexpress.com", "business-standard.com", "reuters.com", "bloomberg.com"
])
