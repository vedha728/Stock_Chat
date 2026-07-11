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
url = (
    f"https://newsapi.org/v2/everything?"
    f"qInTitle={urllib.parse.quote_plus(query)}&"
    f"from={from_date_str}&"
    f"language=en&"
    f"sortBy=relevance&"
    f"pageSize=50&"
    f"apiKey={api_key}"
)

print(f"[*] Requesting: {url.replace(api_key, 'HIDDEN')}\n")
response = requests.get(url)
data = response.json()

print(f"Status: {data.get('status')}")
print(f"Total Results: {data.get('totalResults', 0)}\n")

articles = data.get("articles", [])
for i, article in enumerate(articles):
    title = article.get("title")
    source_name = article.get("source", {}).get("name")
    url_str = article.get("url")
    domain = url_str.split("/")[2] if url_str else "Unknown"
    print(f"{i+1}. Source: {source_name} | Domain: {domain}")
    print(f"   Title: {title}")
    print(f"   URL: {url_str}\n")
