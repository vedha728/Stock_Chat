from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import os
import sys
import math

# Link to src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

# Import modules
from data_collector import (
    extract_multiple_tickers, 
    fetch_stock_price, 
    fetch_news_headlines, 
    fetch_fundamentals, 
    get_latest_macro_returns
)
from indicators import calculate_technical_indicators
from sentiment import analyze_news_sentiment
from institutional import fetch_latest_fii_dii, analyze_institutional_signals
from feature_engineering import prepare_inference_row
from predict import predict_stock_action
from gemini_explain import generate_comparison_explanation

def clean_nan(val):
    if val is None:
        return None
    try:
        import pandas as pd
        if pd.isna(val) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return None
    except:
        pass
    return val

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
            self.wfile.write(json.dumps({"error": "Stocks list (comma separated) is required"}).encode('utf-8'))
            return

        try:
            tickers = extract_multiple_tickers(query)
            if not tickers:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Could not find any matching stock symbols in our database."}).encode('utf-8'))
                return

            if len(tickers) < 2:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Please enter at least two stocks to perform a comparison."}).encode('utf-8'))
                return

            stocks_data = []
            macro_returns = get_latest_macro_returns()

            for ticker, company_name in tickers:
                # Fetch price data & calculate technicals
                price_df = fetch_stock_price(ticker)
                price_indicators_df = calculate_technical_indicators(price_df)
                latest_row = price_indicators_df.iloc[-1]
                current_price = float(latest_row['Close'])

                # News headlines & sentiment
                try:
                    headlines = fetch_news_headlines(ticker, company_name)
                    titles_list = [h["title"] for h in headlines]
                    sent_score, pos, neg, _ = analyze_news_sentiment(titles_list)
                    sent_avail = 1 if headlines else 0
                except Exception:
                    sent_score, pos, neg, sent_avail = 0.0, 0, 0, 0

                # Institutional flows
                try:
                    fii_dii_df = fetch_latest_fii_dii()
                    fii_net_list = fii_dii_df['FII_Net'].tolist()
                    dii_net_list = fii_dii_df['DII_Net'].tolist()
                    fii_dii_summary = analyze_institutional_signals(fii_net_list, dii_net_list)
                except Exception:
                    fii_dii_summary = {"FII_10d_Net": 0.0, "DII_10d_Net": 0.0}

                # Fundamentals
                try:
                    fundamentals = fetch_fundamentals(ticker)
                except Exception:
                    fundamentals = {}

                # ML Prediction
                feature_row = prepare_inference_row(price_indicators_df, fii_dii_summary, (sent_score, pos, neg, sent_avail), macro_returns)
                ml_result = predict_stock_action(feature_row)

                stocks_data.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "current_price": clean_nan(current_price),
                    "recommendation": ml_result.get("Recommendation", "HOLD"),
                    "confidence": clean_nan(ml_result.get("Confidence", 0.5) * 100),
                    "fundamentals": {k: clean_nan(v) for k, v in fundamentals.items()} if isinstance(fundamentals, dict) else {},
                    "sentiment_score": clean_nan(sent_score),
                    "tech_above_50": bool(latest_row.get('Price_Above_MA50', 0)),
                    "tech_above_200": bool(latest_row.get('Price_Above_MA200', 0)),
                    "fii_net": clean_nan(fii_dii_summary.get('FII_10d_Net', 0.0)),
                    "dii_net": clean_nan(fii_dii_summary.get('DII_10d_Net', 0.0))
                })

            # Generate side-by-side AI explanation
            try:
                ai_summary = generate_comparison_explanation(stocks_data)
            except Exception as e:
                ai_summary = f"Comparison summary unavailable. Loaded standard indicators.\n\nRecommended: {', '.join([f'{s['company_name']}: {s['recommendation']}' for s in stocks_data])}."

            response_payload = {
                "stocks": stocks_data,
                "ai_summary": ai_summary
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Server error: {str(e)}"}).encode('utf-8'))
