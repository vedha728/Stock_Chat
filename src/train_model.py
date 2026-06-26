import os
import time
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import joblib

# Import modules from our package
from data_collector import fetch_stock_price
from indicators import calculate_technical_indicators
from institutional import generate_historical_fii_dii_csv
from feature_engineering import compile_feature_matrix

# 100 stocks across Nifty 100 sectors for broad, generalizable training
TRAINING_STOCKS = [
    # --- IT / Technology (9 stocks) ---
    "INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS",
    
    # --- Banking & Finance (20 stocks) ---
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BANKBARODA.NS", "PNB.NS", "INDUSINDBK.NS", 
    "FEDERALBNK.NS", "YESBANK.NS", "CANBK.NS", "UNIONBANK.NS", "IOB.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "SHRIRAMFIN.NS", 
    "MUTHOOTFIN.NS", "CHOLAFIN.NS", "PFC.NS", "RECLTD.NS",
    
    # --- Oil/Energy & Commodities (11 stocks) ---
    "RELIANCE.NS", "ONGC.NS", "OIL.NS", "BPCL.NS", "IOC.NS", "HINDPETRO.NS", "GAIL.NS", "PETRONET.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
    
    # --- Auto (11 stocks - including demerged divisions) ---
    "MARUTI.NS", "M&M.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "BOSCHLTD.NS", "MOTHERSON.NS", "TMPV.NS", "TMCV.NS",
    
    # --- FMCG & Consumer (10 stocks) ---
    "HINDUNILVR.NS", "NESTLEIND.NS", "ITC.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS", "EMAMILTD.NS", "GODREJCP.NS", "COLPAL.NS", "PGHH.NS",
    
    # --- Pharma & Healthcare (10 stocks) ---
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS", "LUPIN.NS", "TORNTPHARM.NS", "APOLLOHOSP.NS", "MAXHEALTH.NS", "FORTIS.NS",
    
    # --- Capital Goods & Infrastructure (10 stocks) ---
    "LT.NS", "BHEL.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "CGPOWER.NS", "BEL.NS", "IRFC.NS", "IRB.NS", "GMRAIRPORT.NS",
    
    # --- Adani Group (9 stocks) ---
    "ADANIPORTS.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "ATGL.NS", "AWL.NS", "NDTV.NS", "AMBUJACEM.NS", "ACC.NS",
    
    # --- Metals & Mining (6 stocks) ---
    "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS", "NMDC.NS", "TATASTEEL.NS",
    
    # --- Retail, Real Estate, Cement & Telecom (15 stocks) ---
    "ASIANPAINT.NS", "TITAN.NS", "DMART.NS", "TRENT.NS", "VMART.NS", "PAGEIND.NS", "ABFRL.NS", "MRF.NS", "APOLLOTYRE.NS", "BHARTIARTL.NS", "IDEA.NS", 
    "DLF.NS", "GODREJPROP.NS", "ULTRACEMCO.NS", "SHREECEM.NS"
]

def generate_forward_labels(df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.02) -> pd.DataFrame:
    """
    Creates labels based on forward returns:
    - BUY (2): price increased > +2% in 5 days
    - SELL (0): price decreased < -2% in 5 days
    - HOLD (1): price moved between -2% and +2%
    Discards the last 'forward_days' rows because they don't have future prices.
    """
    df = df.copy()
    labels = []
    
    for idx in range(len(df)):
        if idx + forward_days < len(df):
            price_current = df['Close'].iloc[idx]
            price_future = df['Close'].iloc[idx + forward_days]
            
            # If price is multi-indexed or pandas Series (handling edge cases)
            if isinstance(price_current, pd.Series):
                price_current = price_current.iloc[0]
            if isinstance(price_future, pd.Series):
                price_future = price_future.iloc[0]
                
            ret = (price_future - price_current) / price_current
            
            if ret > threshold:
                labels.append(2)  # BUY
            elif ret < -threshold:
                labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD
        else:
            labels.append(np.nan)  # Discard later
            
    df['Label'] = labels
    # Drop rows where Label is NaN
    df = df.dropna(subset=['Label'])
    df['Label'] = df['Label'].astype(int)
    return df


def run_training_pipeline():
    """
    Downloads historical data, calculates indicators, engineers features,
    trains XGBoost model on 23 features, prints evaluation metrics, and saves model.
    """
    print("=========================================================================")
    print("[*] Starting Machine Learning Training Pipeline")
    print("=========================================================================")
    
    # 1. Generate historical FII/DII data
    # 5 years: Jan 2020 to June 2025 — covers COVID crash, bear market, and bull run
    start_date = "2020-01-01"
    end_date   = "2025-06-01"
    fii_dii_csv = generate_historical_fii_dii_csv(start_date, end_date)
    fii_dii_df = pd.read_csv(fii_dii_csv)
    
    # Download global macro data
    from data_collector import fetch_global_macro_data
    print("[*] Fetching historical global macro data (S&P 500, Crude, USD/INR)...")
    macro_df = fetch_global_macro_data(start_date, end_date)
    
    combined_datasets = []
    succeeded_tickers = []   # Tickers processed successfully
    skipped_tickers   = []   # Tickers that failed (with reason)
    
    # 2. Process each training stock — with retry logic for network failures
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds between retries

    for ticker in TRAINING_STOCKS:
        success = False
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if attempt == 1:
                    print(f"\n[*] Processing training dataset for {ticker}...")
                else:
                    print(f"\n[*] Retrying {ticker} (attempt {attempt}/{MAX_RETRIES})...")

                # Download 5 years of daily data
                stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False)

                # Clean column multi-index if present (yfinance sometimes outputs multi-index columns)
                if isinstance(stock_df.columns, pd.MultiIndex):
                    stock_df.columns = stock_df.columns.get_level_values(0)
                stock_df = stock_df.reset_index()

                if stock_df.empty:
                    # Empty data means delisted or wrong ticker — no point retrying
                    print(f"[Skip] No price data found for {ticker} (possibly delisted or invalid ticker).")
                    skipped_tickers.append((ticker, "No data returned (delisted or invalid ticker)"))
                    break  # Exit retry loop for this ticker

                # Save raw training CSV — sanitize ticker for filename
                import re as _re
                safe_ticker = _re.sub(r'[^a-zA-Z0-9]', '_', ticker)
                os.makedirs("data/raw", exist_ok=True)
                stock_df.to_csv(f"data/raw/{safe_ticker}_5y.csv", index=False)

                # Compute technical indicators
                price_indicators_df = calculate_technical_indicators(stock_df)

                # Merge with FII/DII flow and generate sentiment proxy
                feature_df = compile_feature_matrix(price_indicators_df, fii_dii_df, macro_df=macro_df, is_training=True, ticker=ticker)

                # Apply BUY/HOLD/SELL labels
                labeled_df = generate_forward_labels(feature_df, forward_days=5, threshold=0.02)

                combined_datasets.append(labeled_df)
                succeeded_tickers.append(ticker)
                print(f"[+] Loaded {len(labeled_df)} training rows for {ticker}")
                success = True
                break  # Exit retry loop — success

            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    print(f"[Warning] Attempt {attempt} failed for {ticker}: {e}")
                    print(f"          Waiting {RETRY_DELAY}s before retry...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"[Error] All {MAX_RETRIES} attempts failed for {ticker}: {e}")
                    skipped_tickers.append((ticker, f"Error after {MAX_RETRIES} attempts: {last_error[:80]}"))
            
    # ── Data Collection Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print(f"  [OK] Succeeded : {len(succeeded_tickers)}/{len(TRAINING_STOCKS)} stocks")
    if skipped_tickers:
        print(f"  [WARN] Skipped  : {len(skipped_tickers)} stock(s)")
        for t, reason in skipped_tickers:
            print(f"     -> {t}: {reason}")
    print("=" * 60)

        
    # Abort if no data was collected at all
    if not combined_datasets:
        print("\n[FATAL] No training data could be collected for any stock! Training aborted.")
        print("        Check your internet connection and Yahoo Finance availability.")
        return

    # Combine all stocks into one big dataset
    master_df = pd.concat(combined_datasets, ignore_index=True)
    print(f"\n[+] Combined Dataset Size: {master_df.shape[0]} rows, {master_df.shape[1]} columns")

    
    # Save processed master dataset for verification
    os.makedirs("data/processed", exist_ok=True)
    master_df.to_csv("data/processed/master_training_dataset.csv", index=False)
    print("[+] Saved master training dataset to data/processed/master_training_dataset.csv")
    
    # 3. Define features (X) and labels (y) — 19 features total
    feature_cols = [
        'RSI', 'MACD_Hist', 'MACD_Crossover',
        'Price_Above_MA50', 'Price_Above_MA200', 'Golden_Cross',
        'Volume_Ratio', 'Sentiment_Score', 'Positive_Headlines', 'Negative_Headlines',
        'Sentiment_Available',
        'Multi_Timeframe_Alignment',
        'SP500_Return', 'Crude_Return', 'USD_INR_Return',
        # Rolling window features (Issue #3 — sequence context for XGBoost)
        'RSI_Slope', 'Price_Pct_5d', 'Price_Pct_20d', 'Volatility_10d'
    ]
    
    X = master_df[feature_cols]
    y = master_df['Label']
    
    # 4. Train-Test Split (80% train, 20% validation — stratified to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[+] Train size: {len(X_train)} rows | Test size: {len(X_test)} rows")

    # =========================================================
    # 5. Main Model: XGBoost Classifier (tuned for stock data)
    # =========================================================
    print("\n[*] Training XGBoost Classifier on CPU...")
    # Label encode: XGBoost needs labels 0,1,2 (already the case: SELL=0, HOLD=1, BUY=2)
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    t0 = time.time()
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False              # suppress per-iteration output
    )
    xgb_time = time.time() - t0
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc  = accuracy_score(y_test, xgb_pred)
    print(f"[+] XGBoost done in {xgb_time:.2f}s | Accuracy: {xgb_acc*100:.2f}%")

    # =========================================================
    # 7. Detailed XGBoost Report
    # =========================================================
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, xgb_pred, target_names=["SELL", "HOLD", "BUY"]))
    print("Confusion Matrix (SELL=0, HOLD=1, BUY=2):")
    print(confusion_matrix(y_test, xgb_pred))

    # Feature Importance
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nTop Feature Importances (XGBoost):")
    for rank, idx in enumerate(indices):
        bar = '#' * int(importances[idx] * 100)
        print(f"  {rank+1:>2}. {feature_cols[idx]:<30} {importances[idx]*100:>5.2f}%  {bar}")
    print("="*60)

    # 5-fold Cross Validation for robustness check
    print("\n[*] Running 5-fold Cross Validation on XGBoost...")
    cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"[+] CV Scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"[+] Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    # =========================================================
    # 8. Save Best Model
    # =========================================================
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgboost_model.pkl"
    joblib.dump(xgb_model, model_path)
    print(f"\n[+] Saved XGBoost model to {model_path}")
    print(f"[DONE] Training complete. XGBoost is now the active prediction model.")
    return {
        "stocks_processed": len(succeeded_tickers),
        "training_rows": int(master_df.shape[0]),
        "test_accuracy_pct": f"{xgb_acc*100:.2f}%",
        "cv_mean_accuracy_pct": f"{cv_scores.mean()*100:.2f}%",
        "notes": "Trained with 19-feature XGBoost. Correct active tickers included in training."
    }
    
if __name__ == "__main__":
    run_training_pipeline()
