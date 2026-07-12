from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import os
import sys

# Link to src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from data_collector import extract_ticker
from backtest import run_backtest

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
            self.wfile.write(json.dumps({"error": "Stock name or symbol is required for backtest"}).encode('utf-8'))
            return

        try:
            ticker, name = extract_ticker(query)
            if not ticker:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Could not find matching stock symbol for '{query}'."}).encode('utf-8'))
                return

            # Run 1-year trade simulation backtest
            report = run_backtest(ticker)

            response_payload = {
                "ticker": ticker,
                "company_name": name,
                "initial_capital": float(report.get("Initial_Capital", 100000.0)),
                "final_value": float(report.get("Final_Value", 100000.0)),
                "strategy_return_pct": float(report.get("Strategy_Return_Pct", 0.0)),
                "benchmark_return_pct": float(report.get("Benchmark_Return_Pct", 0.0)),
                "total_trades": int(report.get("Total_Trades", 0)),
                "win_rate_pct": float(report.get("Win_Rate_Pct", 0.0))
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        except Exception:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Unable to run backtest simulation for '{query}' due to data availability. Please try another stock."}).encode('utf-8'))
