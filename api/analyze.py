from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import os
import sys
import math
import concurrent.futures
import time

# Link to src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

# Import core modules
from data_collector import (
    extract_ticker, 
    fetch_stock_price, 
    fetch_news_headlines, 
    fetch_fundamentals, 
    get_latest_macro_returns,
    TICKER_NAME_MAP
)
from indicators import calculate_technical_indicators
from sentiment import analyze_news_sentiment, classify_headline_sentiment
from institutional import fetch_latest_fii_dii, analyze_institutional_signals
from feature_engineering import prepare_inference_row
from predict import predict_stock_action
from gemini_explain import generate_beginner_explanation

def clean_nan(val):
    """Converts NaNs and infinite floats to None for clean JSON serialization."""
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
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        # Parse query parameters
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        query = params.get("query", [""])[0].strip()

        if not query:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Stock name or query is required"}).encode('utf-8'))
            return

        try:
            # Intercept general Tata Motors demerger query
            q_lower = query.lower()
            is_general_tata = ("tata motor" in q_lower or "tatamotors" in q_lower) and not any(w in q_lower for w in ["pv", "passenger", "cv", "commercial", "tmpv", "tmcv"])
            
            if is_general_tata:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                payload = {
                    "is_demerger": True,
                    "message": "Tata Motors has demerged into two separate listed entities:",
                    "options": [
                        {"label": "1. TMPV (Tata Motors Passenger Vehicles Limited - Cars, EVs, JLR)", "query": "TMPV"},
                        {"label": "2. TMCV (Tata Motors Commercial Vehicles Limited - Trucks, Buses)", "query": "TMCV"}
                    ]
                }
                self.wfile.write(json.dumps(payload).encode('utf-8'))
                return

            # 1. Resolve Ticker using fuzzy matching
            selected_ticker, selected_company = extract_ticker(query)
            if not selected_ticker or selected_ticker not in TICKER_NAME_MAP:
                # Return ticker availability warning
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": f"Currently '{query}' is not available in the service. Please search for a supported stock (e.g. Reliance, TCS, Wipro, HDFC Bank)."
                }).encode('utf-8'))
                return

            # 2. Fetch Price & Technicals (Critical Path)
            price_df = fetch_stock_price(selected_ticker)
            price_indicators_df = calculate_technical_indicators(price_df)
            latest_price_row = price_indicators_df.iloc[-1]
            current_price = float(latest_price_row['Close'])
            
            price_5d_pct = 0.0
            if len(price_indicators_df) >= 6:
                price_5d_pct = float((price_indicators_df['Close'].iloc[-1] / price_indicators_df['Close'].iloc[-6] - 1) * 100)

            # Format historical closing price data (90-day tail) for React charts
            historical_prices = []
            tail_df = price_indicators_df.tail(90)
            for _, row in tail_df.iterrows():
                row_date = row["Date"]
                dt_str = row_date.strftime("%Y-%m-%d") if hasattr(row_date, 'strftime') else str(row_date)
                historical_prices.append({
                    "date": dt_str,
                    "close": clean_nan(float(row["Close"]))
                })

            # Execute other fetches concurrently to optimize performance
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    "news": executor.submit(fetch_news_headlines, selected_ticker, selected_company),
                    "fii": executor.submit(fetch_latest_fii_dii),
                    "fundamentals": executor.submit(fetch_fundamentals, selected_ticker),
                    "macro": executor.submit(get_latest_macro_returns)
                }
                
                # Wait for all fetches with a unified 4.0s wall-clock limit
                done, not_done = concurrent.futures.wait(futures.values(), timeout=4.0)
                for fut in not_done:
                    fut.cancel()

                # 3. Resolve News & Sentiment
                try:
                    headlines = futures["news"].result() if futures["news"] in done else None
                    if headlines:
                        titles_list = [h["title"] for h in headlines]
                        sent_score, pos_count, neg_count = analyze_news_sentiment(titles_list)
                        sentiment_summary = (sent_score, pos_count, neg_count, 1)
                    else:
                        sentiment_summary = (0.0, 0, 0, 0)
                        sent_score, pos_count, neg_count = 0.0, 0, 0
                except Exception:
                    sentiment_summary = (0.0, 0, 0, 0)
                    sent_score, pos_count, neg_count = 0.0, 0, 0
                    headlines = []

                # 4. Resolve Institutional Flow (FII/DII)
                try:
                    fii_dii_df = futures["fii"].result() if futures["fii"] in done else None
                    if fii_dii_df is not None:
                        fii_net_list = fii_dii_df['FII_Net'].tolist()
                        dii_net_list = fii_dii_df['DII_Net'].tolist()
                        fii_dii_summary = analyze_institutional_signals(fii_net_list, dii_net_list)
                    else:
                        raise ValueError()
                except Exception:
                    fii_dii_summary = {
                        "FII_10d_Net": 0.0, "DII_10d_Net": 0.0,
                        "FII_Trend": 0, "DII_Trend": 0, "Divergence_Flag": 0
                    }

                # 5. Resolve Fundamentals
                try:
                    fundamentals = futures["fundamentals"].result() if futures["fundamentals"] in done else {}
                except Exception:
                    fundamentals = {}

                # 6. Resolve Macro Returns
                try:
                    macro_returns = futures["macro"].result() if futures["macro"] in done else (0.0, 0.0, 0.0)
                except Exception:
                    macro_returns = (0.0, 0.0, 0.0)

            # 7. Run ML Predictor for today (the latest row)
            feature_row = prepare_inference_row(price_indicators_df, fii_dii_summary, sentiment_summary, macro_returns)
            ml_result = predict_stock_action(feature_row)

            # 7.2. Compute 10-Day Signal Timeline
            timeline = []
            if len(price_indicators_df) >= 15:
                for i in range(10):
                    target_idx = len(price_indicators_df) - 10 + i
                    df_slice = price_indicators_df.iloc[:target_idx + 1]
                    
                    # Sentiment and Macro logic: Use neutral defaults for past, live for today
                    if i == 9:
                        loop_sentiment = sentiment_summary
                        loop_macro = macro_returns
                        loop_fii_dii = fii_dii_summary
                    else:
                        loop_sentiment = (0.0, 0, 0, 0)
                        loop_macro = (0.0, 0.0, 0.0)
                        loop_fii_dii = {
                            "FII_10d_Net": 0.0, "DII_10d_Net": 0.0,
                            "FII_Trend": 0, "DII_Trend": 0, "Divergence_Flag": 0
                        }
                    
                    loop_feature_row = prepare_inference_row(df_slice, loop_fii_dii, loop_sentiment, loop_macro)
                    loop_pred = predict_stock_action(loop_feature_row)
                    
                    row_date = df_slice.iloc[-1]["Date"]
                    dt_str = row_date.strftime("%Y-%m-%d") if hasattr(row_date, 'strftime') else str(row_date)
                    
                    timeline.append({
                        "date": dt_str,
                        "buy": clean_nan(float(loop_pred.get("Breakdown", {}).get("BUY", 0.0) * 100)),
                        "hold": clean_nan(float(loop_pred.get("Breakdown", {}).get("HOLD", 0.0) * 100)),
                        "sell": clean_nan(float(loop_pred.get("Breakdown", {}).get("SELL", 0.0) * 100)),
                        "is_live": (i == 9)
                    })

            # 8. Run AI explanation / Reconciliations (Non-critical fallback)
            tech_summary_dict = {
                **latest_price_row.to_dict(),
                "Price_Pct_5d": price_5d_pct
            }
            try:
                model_analysis, beginner_explanation = generate_beginner_explanation(
                    selected_ticker,
                    selected_company,
                    current_price,
                    ml_result,
                    tech_summary_dict,
                    sent_score,
                    [h["title"] for h in headlines],
                    fii_dii_summary,
                    fundamentals,
                    user_input=query
                )
            except Exception as gem_err:
                model_analysis = "Factors Favouring Buy (+):\n  • [RSI] Technical momentum indicators are active.\n\nFactors Against Buy (-):\n  • [Moving Averages] Price is trading under technical pressure.\n\nStrategic Advice:\n  • Action: Hold current shares. Avoid buying new positions.\n  • Stop Loss: Safety levels active."
                beginner_explanation = (
                    f"Strategic Advice Summary:\n\n"
                    f"• **Recommendation:** {ml_result.get('Recommendation', 'HOLD')} (Confidence: {ml_result.get('Confidence', 0.5)*100:.1f}%)\n"
                    f"• **Risk Parameters:** Active monitoring is suggested. Adjust positions based on your individual risk appetite.\n"
                    f"• **Safety Levels:** Place a trailing stop loss close to key technical support indicators (MA50/MA200)."
                )

            # Clean and serialize response structure
            clean_fundamentals = {}
            if isinstance(fundamentals, dict):
                for k, v in fundamentals.items():
                    clean_fundamentals[k] = clean_nan(v)

            response_payload = {
                "ticker": selected_ticker,
                "company_name": selected_company,
                "current_price": clean_nan(current_price),
                "price_change_5d": clean_nan(price_5d_pct),
                "ml_result": {
                    "Recommendation": ml_result.get("Recommendation", "HOLD"),
                    "Confidence": clean_nan(ml_result.get("Confidence", 0.5)),
                    "Breakdown": {
                        "BUY": clean_nan(ml_result.get("Breakdown", {}).get("BUY", 0.0)),
                        "HOLD": clean_nan(ml_result.get("Breakdown", {}).get("HOLD", 0.0)),
                        "SELL": clean_nan(ml_result.get("Breakdown", {}).get("SELL", 0.0))
                    },
                    "Low_Confidence": bool(ml_result.get("Low_Confidence", False))
                },
                "technical_indicators": {
                    "RSI": clean_nan(latest_price_row.get("RSI", 50.0)),
                    "MA50": clean_nan(latest_price_row.get("MA50", current_price)),
                    "MA200": clean_nan(latest_price_row.get("MA200", current_price)),
                    "MACD": clean_nan(latest_price_row.get("MACD_Hist", 0.0)),
                    "Price_Above_MA50": int(latest_price_row.get("Price_Above_MA50", 0)),
                    "Price_Above_MA200": int(latest_price_row.get("Price_Above_MA200", 0))
                },
                "sentiment": {
                    "score": clean_nan(sent_score),
                    "pos_count": int(pos_count),
                    "neg_count": int(neg_count),
                    "neu_count": int(max(0, len(titles_list) - pos_count - neg_count))
                },
                "institutional_flow": {
                    "FII_10d_Net": clean_nan(fii_dii_summary.get("FII_10d_Net", 0.0)),
                    "DII_10d_Net": clean_nan(fii_dii_summary.get("DII_10d_Net", 0.0)),
                    "FII_Trend": int(fii_dii_summary.get("FII_Trend", 0)),
                    "DII_Trend": int(fii_dii_summary.get("DII_Trend", 0)),
                    "Divergence_Flag": int(fii_dii_summary.get("Divergence_Flag", 0))
                },
                "fundamentals": clean_fundamentals,
                "model_analysis": model_analysis,
                "beginner_explanation": beginner_explanation,
                "historical_prices": historical_prices,
                "timeline": timeline,
                "news_headlines": [
                    {
                        "title": h.get("title", ""),
                        "url": h.get("url", ""),
                        "source": h.get("source", ""),
                        "description": h.get("description", ""),
                        "sentiment": classify_headline_sentiment(h.get("title", ""))
                    }
                    for h in headlines[:10]
                ]
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
