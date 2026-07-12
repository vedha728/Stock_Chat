from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import os
import sys

# Link to src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from gemini_explain import explain_educational_concept

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        query = params.get("query", [""])[0].strip()

        if not query:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Question is required for learning module"}).encode('utf-8'))
            return

        try:
            explanation = explain_educational_concept(query)
            response_payload = {
                "query": query,
                "explanation": explanation
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        except Exception:
            fallback = (
                "Educational Guide:\n\n"
                "To get started with Indian stock markets, here are the key indicators to watch:\n\n"
                "• **RSI (Relative Strength Index):** Measures momentum on a scale of 0-100. Below 30 is oversold, above 70 is overbought.\n"
                "• **Moving Averages (MA50 & MA200):** Track intermediate and long-term price trends.\n"
                "• **FII & DII Flows:** Foreign and Domestic Institutional Investors whose buying and selling activity heavily impacts price trends.\n"
                "• **P/E (Price-to-Earnings) Ratio:** Valuates the stock price relative to its earnings per share.\n\n"
                "Please query a specific stock indicator or topic again to fetch its detailed description."
            )
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "query": query,
                "explanation": fallback
            }).encode('utf-8'))
