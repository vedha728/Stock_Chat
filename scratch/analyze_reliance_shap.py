import os
import sys
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from scipy.special import softmax

# Add src folder to python path
project_root = "d:/Stockchat"
sys.path.insert(0, os.path.join(project_root, "src"))

from predict import predict_stock_action
from data_collector import fetch_stock_price, get_latest_macro_returns
from indicators import calculate_technical_indicators
from institutional import fetch_latest_fii_dii, analyze_institutional_signals
from feature_engineering import prepare_inference_row

def main():
    ticker = "RELIANCE.NS"
    company_name = "Reliance"
    
    print(f"[*] Fetching price data and technical indicators for {ticker}...")
    price_df = fetch_stock_price(ticker)
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)
    price_indicators_df = calculate_technical_indicators(price_df)
    
    print("[*] Fetching FII/DII data and global macro returns...")
    fii_dii_df = fetch_latest_fii_dii()
    fii_dii_summary = analyze_institutional_signals(fii_dii_df['FII_Net'].tolist(), fii_dii_df['DII_Net'].tolist())
    macro_returns = get_latest_macro_returns()
    
    # Let's set a positive news sentiment proxy for Reliance (+0.40 sentiment score, news available)
    sentiment_summary = (0.40, 2, 0, 1) # (score, positive_count, negative_count, available)
    
    # Prepare the feature matrix row
    row = prepare_inference_row(price_indicators_df, fii_dii_summary, sentiment_summary, macro_returns)
    
    # Load model and get feature list
    model = joblib.load("models/xgboost_model.pkl")
    booster = model.get_booster()
    expected_features = booster.feature_names
    
    # Align row columns to the exact sequence the model expects
    row_aligned = row[expected_features]
    
    # 1. Predict raw margin values (log-odds output before softmax)
    dmat = xgb.DMatrix(row_aligned)
    margins = booster.predict(dmat, output_margin=True)[0]
    
    # 2. Predict SHAP contributions
    # Shape of contribs is (num_rows, num_classes, num_features + 1)
    # The last element in the feature axis (+1) is the base/bias value.
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    
    # 3. Get probabilities via softmax
    probs = softmax(margins)
    
    print("\n" + "="*90)
    print(f" SHAP CONTRIBUTION ANALYSIS: {company_name} ({ticker})")
    print("="*90)
    print(f"Current Stock Price      : Rs.{price_indicators_df['Close'].iloc[-1]:,.2f}")
    print(f"Simulated Sentiment Score: +0.40 (News Available)")
    print("-" * 90)
    
    # Map predictions
    rec_labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
    predicted_class = np.argmax(probs)
    print(f"Model Action Recommendation: {rec_labels[predicted_class]}")
    print(f"Class Probabilities        : SELL={probs[0]*100:.1f}%, HOLD={probs[1]*100:.1f}%, BUY={probs[2]*100:.1f}%")
    print("-" * 90)
    
    # Verify SHAP math: sum(SHAP) + Bias must equal Margin for each class
    math_checks_passed = True
    print("Mathematical Correctness Verification:")
    for c in range(3):
        class_name = rec_labels[c]
        class_contribs = contribs[c]
        bias = class_contribs[-1]
        feature_shap_sum = sum(class_contribs[:-1])
        calculated_margin = feature_shap_sum + bias
        expected_margin = margins[c]
        
        diff = abs(calculated_margin - expected_margin)
        status = "PASSED" if diff < 1e-4 else "FAILED"
        if diff >= 1e-4:
            math_checks_passed = False
        print(f"  • Class {class_name:<4} : Sum of SHAP ({feature_shap_sum:+.4f}) + Bias ({bias:+.4f}) = Margin ({calculated_margin:+.4f}) [Expected: {expected_margin:+.4f}] -> {status}")
    
    print("-" * 90)
    
    # Print the detailed SHAP contribution table
    print(f"{'Feature Name':<28} | {'Raw Value':<12} | {'SELL SHAP':<12} | {'HOLD SHAP':<12} | {'BUY SHAP':<12}")
    print("-" * 90)
    
    for idx, feature in enumerate(expected_features):
        raw_val = row_aligned[feature].iloc[0]
        sell_shap = contribs[0, idx]
        hold_shap = contribs[1, idx]
        buy_shap = contribs[2, idx]
        
        # Format raw values nicely based on type
        if isinstance(raw_val, float):
            if abs(raw_val) < 0.01:
                val_str = f"{raw_val*100:+.3f}%" if "Return" in feature or "Pct" in feature else f"{raw_val:.5f}"
            else:
                val_str = f"{raw_val:.2f}"
        else:
            val_str = str(int(raw_val))
            
        print(f"{feature:<28} | {val_str:>12} | {sell_shap:>+12.4f} | {hold_shap:>+12.4f} | {buy_shap:>+12.4f}")
        
    print("-" * 90)
    # Print Base Value / Bias
    sell_bias = contribs[0, -1]
    hold_bias = contribs[1, -1]
    buy_bias = contribs[2, -1]
    print(f"{'[Model Base Value / Bias]':<28} | {'N/A':>12} | {sell_bias:>+12.4f} | {hold_bias:>+12.4f} | {buy_bias:>+12.4f}")
    
    print("=" * 90)
    if math_checks_passed:
        print("  >> [STATUS] ALL SHAP VERIFICATIONS SUCCESSFUL. Features contribute correctly to class logits.")
    else:
        print("  >> [STATUS] WARNING: SHAP value sum mismatch detected.")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
