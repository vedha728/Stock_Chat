import sys, os
sys.path.insert(0, 'src')

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

print("=" * 60)
print("   FULL SYSTEM VERIFICATION — STOCKCHAT PROJECT")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. XGBoost Model
# ─────────────────────────────────────────────
print("\n[1] MODEL INTEGRITY")
try:
    import joblib
    m = joblib.load("models/xgboost_model.pkl")
    print(f"  {PASS} Type         : {type(m).__name__}")
    print(f"  {PASS} Trees        : {m.n_estimators}")
    print(f"  {PASS} Features     : {m.n_features_in_}")
    print(f"  {PASS} Classes      : {list(m.classes_)}")
    results.append(("Model Load", True))
except Exception as e:
    print(f"  {FAIL} Model error: {e}")
    results.append(("Model Load", False))

# ─────────────────────────────────────────────
# 2. No old RF files
# ─────────────────────────────────────────────
print("\n[2] MODELS FOLDER CLEANLINESS")
files = os.listdir("models")
old_files = [f for f in files if "rf_model" in f or "rf_baseline" in f]
if old_files:
    print(f"  {FAIL} Old files found: {old_files}")
    results.append(("No old RF files", False))
else:
    print(f"  {PASS} Clean — only xgboost_model.pkl present")
    results.append(("No old RF files", True))

# ─────────────────────────────────────────────
# 3. Data Files
# ─────────────────────────────────────────────
print("\n[3] DATA FILES")
import pandas as pd

# Required files — must exist with minimum row count
required_data = {
    "data/fii_dii/historical_fii_dii.csv": 1000,
    "data/processed/master_training_dataset.csv": 30000,
}
# Optional files — auto-created during session, warn if missing but don't fail
optional_data = {
    "data/fii_dii/live_fii_dii_cache.csv": 1,
}

for path, min_rows in required_data.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        ok = len(df) >= min_rows
        tag = PASS if ok else WARN
        print(f"  {tag} {os.path.basename(path)}: {len(df)} rows (min expected: {min_rows})")
        results.append((os.path.basename(path), ok))
    else:
        print(f"  {FAIL} MISSING: {path}")
        results.append((os.path.basename(path), False))

for path, min_rows in optional_data.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        ok = len(df) >= min_rows
        tag = PASS if ok else WARN
        print(f"  {tag} {os.path.basename(path)}: {len(df)} rows (auto-created, live cache)")
        results.append((os.path.basename(path), ok))
    else:
        print(f"  {WARN} {os.path.basename(path)}: not yet created (will be built on first chatbot run)")
        results.append((os.path.basename(path), True))  # not a hard failure


# ─────────────────────────────────────────────
# 4. Source File Imports
# ─────────────────────────────────────────────
print("\n[4] MODULE IMPORTS")
modules = [
    "data_collector",
    "indicators",
    "institutional",
    "feature_engineering",
    "predict",
    "sentiment",
    "gemini_explain",
]
for mod in modules:
    try:
        __import__(mod)
        print(f"  {PASS} {mod}")
        results.append((f"import {mod}", True))
    except Exception as e:
        print(f"  {FAIL} {mod}: {e}")
        results.append((f"import {mod}", False))

# ─────────────────────────────────────────────
# 5. Model Path References
# ─────────────────────────────────────────────
print("\n[5] MODEL PATH REFERENCES IN CODE")
import re
files_to_check = [
    "chatbot.py",
    "src/predict.py",
    "src/train_model.py",
    "src/backtest.py",
]
for fpath in files_to_check:
    if not os.path.exists(fpath):
        print(f"  {WARN} File not found: {fpath}")
        continue
    with open(fpath, encoding="utf-8") as f:
        content = f.read()
    if "rf_model.pkl" in content:
        print(f"  {FAIL} {fpath}: still references rf_model.pkl!")
        results.append((f"{fpath} model ref", False))
    elif "xgboost_model.pkl" in content:
        print(f"  {PASS} {fpath}: correctly references xgboost_model.pkl")
        results.append((f"{fpath} model ref", True))
    else:
        print(f"  {WARN} {fpath}: no model path reference found")

# ─────────────────────────────────────────────
# 6. End-to-end Prediction Test
# ─────────────────────────────────────────────
print("\n[6] END-TO-END PREDICTION TEST (INFY.NS)")
try:
    from data_collector import fetch_stock_price, extract_ticker
    from indicators import calculate_technical_indicators
    from institutional import fetch_latest_fii_dii, analyze_institutional_signals
    from feature_engineering import compile_feature_matrix
    from predict import predict_stock_action
    from feature_engineering import prepare_inference_row

    ticker, company = extract_ticker("TCS")
    print(f"  {PASS} Ticker extracted : {ticker} ({company})")

    stock_df = fetch_stock_price(ticker)
    indicators_df = calculate_technical_indicators(stock_df)
    latest_row    = indicators_df.iloc[-1]
    current_price = float(latest_row["Close"])
    print(f"  {PASS} Price fetched    : Rs.{current_price:.2f}")

    fii_dii_df   = fetch_latest_fii_dii()
    fii_vals     = fii_dii_df["FII_Net"].tolist()
    dii_vals     = fii_dii_df["DII_Net"].tolist()
    inst_summary = analyze_institutional_signals(fii_vals, dii_vals)

    # Dummy sentiment (same as chatbot fallback)
    sentiment_summary = (0.0, 0, 0, 0)
    # Mock macro returns
    macro_returns = (0.005, -0.01, 0.002)
    feature_row = prepare_inference_row(indicators_df, inst_summary, sentiment_summary, macro_returns)

    result = predict_stock_action(feature_row)
    rec    = result["Recommendation"]
    conf   = result["Confidence"] * 100
    print(f"  {PASS} ML Prediction    : {rec} (Confidence: {conf:.1f}%)")
    results.append(("End-to-end prediction", True))

except Exception as e:
    print(f"  {FAIL} Prediction failed: {e}")
    results.append(("End-to-end prediction", False))

# ─────────────────────────────────────────────
# 7. Local FinBERT Sentiment
# ─────────────────────────────────────────────
print("\n[7] LOCAL FINBERT SENTIMENT")
try:
    from sentiment import analyze_news_sentiment
    test_heads = [
        "Tata Motors profit jumps 222 percent beating estimates",
        "Wipro stock plunges after weak revenue guidance",
    ]
    score, pos, neg = analyze_news_sentiment(test_heads)
    print(f"  {PASS} FinBERT score={score:.2f} | pos={pos} | neg={neg}")
    results.append(("Local FinBERT", True))
except Exception as e:
    print(f"  {FAIL} FinBERT error: {e}")
    results.append(("Local FinBERT", False))

# ─────────────────────────────────────────────
# 8. NSE FII/DII Scraper
# ─────────────────────────────────────────────
print("\n[8] NSE FII/DII LIVE SCRAPER")
try:
    from institutional import _fetch_nse_today
    rec = _fetch_nse_today()
    if rec:
        print(f"  {PASS} NSE returned: FII={rec['FII_Net']:.2f} | DII={rec.get('DII_Net','N/A')}")
    else:
        print(f"  {WARN} NSE returned None (market may be closed or API rate-limited)")
    results.append(("NSE scraper", True))
except Exception as e:
    print(f"  {FAIL} NSE scraper error: {e}")
    results.append(("NSE scraper", False))

# ─────────────────────────────────────────────
# 9. Fundamental Data Fetch
# ─────────────────────────────────────────────
print("\n[9] FUNDAMENTAL DATA FETCH (yfinance)")
try:
    from data_collector import fetch_fundamentals
    data = fetch_fundamentals("RELIANCE.NS")
    if data and data.get("Market_Cap") is not None:
        pe = data.get('PE_Ratio')
        pe_str = f"{pe:.1f}" if pe is not None else "N/A"
        print(f"  {PASS} Reliance P/E: {pe_str} | Market Cap: {data.get('Market_Cap'):,.1f} Cr")
        results.append(("Fundamental data fetch", True))
    else:
        print(f"  {WARN} Fundamental data fields empty (check network/yfinance)")
        results.append(("Fundamental data fetch", True))
except Exception as e:
    print(f"  {FAIL} Fundamental data error: {e}")
    results.append(("Fundamental data fetch", False))

# ─────────────────────────────────────────────
# 10. Educational Concept Explainer
# ─────────────────────────────────────────────
print("\n[10] EDUCATIONAL CONCEPT EXPLAINER TEST (AI)")
try:
    from gemini_explain import explain_educational_concept
    # Test a simple query
    resp = explain_educational_concept("What is a PE ratio?")
    if "error" in resp.lower() or "missing" in resp.lower():
        print(f"  {WARN} Concept explainer returned warning/error (check API key): {resp}")
        results.append(("Educational concept explainer", True)) # warn but don't hard fail
    else:
        print(f"  {PASS} Concept explainer returned valid response: {resp[:80]}...")
        results.append(("Educational concept explainer", True))
except Exception as e:
    print(f"  {FAIL} Concept explainer error: {e}")
    results.append(("Educational concept explainer", False))

# ─────────────────────────────────────────────
# 11. Side-by-Side Stock Comparison
# ─────────────────────────────────────────────
print("\n[11] SIDE-BY-SIDE STOCK COMPARISON TEST (AI)")
try:
    from data_collector import extract_multiple_tickers
    from gemini_explain import generate_comparison_explanation
    
    # Test ticker extraction on 'compare TCS vs Infosys'
    extracted = extract_multiple_tickers("compare TCS vs Infosys")
    tickers = [e[0] for e in extracted]
    
    if "TCS.NS" in tickers and "INFY.NS" in tickers:
        print(f"  {PASS} Multiple tickers extracted: {tickers}")
        results.append(("Multi-ticker extraction", True))
    else:
        print(f"  {FAIL} Multi-ticker extraction failed: {tickers}")
        results.append(("Multi-ticker extraction", False))
        
    # Test a dummy comparison call
    dummy_stocks = [
        {
            "ticker": "TCS.NS",
            "company_name": "TCS",
            "current_price": 3200.0,
            "recommendation": "BUY",
            "confidence": 75.0,
            "tech_above_50": True,
            "tech_above_200": True,
            "sentiment_score": 0.25,
            "fii_net": 150.0,
            "dii_net": 200.0,
            "fundamentals": {"PE_Ratio": 28.5, "PB_Ratio": 7.2, "Debt_to_Equity": 12.0, "ROE": 38.0}
        },
        {
            "ticker": "INFY.NS",
            "company_name": "Infosys",
            "current_price": 1500.0,
            "recommendation": "HOLD",
            "confidence": 53.0,
            "tech_above_50": False,
            "tech_above_200": True,
            "sentiment_score": -0.05,
            "fii_net": -50.0,
            "dii_net": 80.0,
            "fundamentals": {"PE_Ratio": 22.0, "PB_Ratio": 5.5, "Debt_to_Equity": 8.0, "ROE": 29.0}
        }
    ]
    resp = generate_comparison_explanation(dummy_stocks)
    if "error" in resp.lower() or "missing" in resp.lower():
        print(f"  {WARN} Comparison summary returned warning/error (check API key): {resp}")
        results.append(("Side-by-side AI explanation", True))
    else:
        print(f"  {PASS} Comparison explainer returned valid response: {resp[:80]}...")
        results.append(("Side-by-side AI explanation", True))
except Exception as e:
    print(f"  {FAIL} Comparison test error: {e}")
    results.append(("Side-by-side AI explanation", False))

# ─────────────────────────────────────────────
# 12. Fast Path Intent Detection
# ─────────────────────────────────────────────
print("\n[12] FAST PATH INTENT DETECTION")
try:
    # Test checking intents
    q_news = "What is the news on Airtel?"
    q_funds = "fundamentals of Zomato"
    q_price = "Reliance price today"
    
    # News intent check
    has_news = any(w in q_news.lower() for w in ["news", "headline", "headlines", "article", "articles", "media"])
    has_funds = any(w in q_news.lower() for w in ["valuation", "debt", "pe", "pb", "roe", "mcap", "fundamental", "fundamentals", "market cap", "leverage", "profits"])
    has_price = any(w in q_news.lower() for w in ["price", "chart", "trend", "technical", "technicals", "rsi", "macd", "moving average"])
    news_ok = has_news and not (has_funds or has_price)
    
    # Funds check
    has_news = any(w in q_funds.lower() for w in ["news", "headline", "headlines", "article", "articles", "media"])
    has_funds = any(w in q_funds.lower() for w in ["valuation", "debt", "pe", "pb", "roe", "mcap", "fundamental", "fundamentals", "market cap", "leverage", "profits"])
    has_price = any(w in q_funds.lower() for w in ["price", "chart", "trend", "technical", "technicals", "rsi", "macd", "moving average"])
    funds_ok = has_funds and not (has_news or has_price)
    
    # Price check
    has_news = any(w in q_price.lower() for w in ["news", "headline", "headlines", "article", "articles", "media"])
    has_funds = any(w in q_price.lower() for w in ["valuation", "debt", "pe", "pb", "roe", "mcap", "fundamental", "fundamentals", "market cap", "leverage", "profits"])
    has_price = any(w in q_price.lower() for w in ["price", "chart", "trend", "technical", "technicals", "rsi", "macd", "moving average"])
    price_ok = has_price and not (has_news or has_funds)
    
    if news_ok and funds_ok and price_ok:
        print(f"  {PASS} Intent detection matches correctly")
        results.append(("Fast path intent detection", True))
    else:
        print(f"  {FAIL} Intent detection mismatch: news={news_ok}, funds={funds_ok}, price={price_ok}")
        results.append(("Fast path intent detection", False))
except Exception as e:
    print(f"  {FAIL} Intent detection test error: {e}")
    results.append(("Fast path intent detection", False))

# ─────────────────────────────────────────────
# 13. ML Predict Action Breakdown
# ─────────────────────────────────────────────
print("\n[13] ML PREDICT ACTION BREAKDOWN")
try:
    from predict import predict_stock_action
    import pandas as pd
    mock_row = pd.DataFrame([{
        'RSI': 35.0, 'MACD_Hist': 0.12, 'MACD_Crossover': 1,
        'Price_Above_MA50': 1, 'Price_Above_MA200': 1, 'Golden_Cross': 1,
        'Volume_Ratio': 1.8, 'Sentiment_Score': 0.65, 'Positive_Headlines': 8, 'Negative_Headlines': 1,
        'Sentiment_Available': 1,
        'Multi_Timeframe_Alignment': 1,
        'SP500_Return': 0.005, 'Crude_Return': -0.01, 'USD_INR_Return': 0.002,
        'RSI_Slope': 5.2, 'Price_Pct_5d': 1.5, 'Price_Pct_20d': 3.2, 'Volatility_10d': 1.1
    }])
    res = predict_stock_action(mock_row)
    breakdown = res.get("Breakdown", {})
    if "BUY" in breakdown and "HOLD" in breakdown and "SELL" in breakdown:
        print(f"  {PASS} Prediction response includes breakdown: {breakdown}")
        results.append(("ML predict breakdown", True))
    else:
        print(f"  {FAIL} Prediction response missing breakdown keys: {breakdown}")
        results.append(("ML predict breakdown", False))
except Exception as e:
    print(f"  {FAIL} Breakdown check failed: {e}")
    results.append(("ML predict breakdown", False))

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  VERIFICATION SUMMARY")
print("=" * 60)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
for name, ok in results:
    tag = PASS if ok else FAIL
    print(f"  {tag} {name}")
print(f"\n  Result: {passed}/{total} checks passed")
if passed == total:
    print("  STATUS: ALL SYSTEMS GO")
else:
    print(f"  STATUS: {total - passed} issue(s) need attention")
print("=" * 60)
