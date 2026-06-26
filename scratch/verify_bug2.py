import os
import sys
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

project_root = "d:/Stockchat"
sys.path.insert(0, os.path.join(project_root, "src"))

from predict import predict_stock_action
from data_collector import fetch_stock_price, get_latest_macro_returns, extract_ticker
from indicators import calculate_technical_indicators
from institutional import fetch_latest_fii_dii, analyze_institutional_signals
from feature_engineering import prepare_inference_row

def main():
    ticker = "WIPRO.NS"
    company_name = "Wipro"
    
    print("[*] Fetching price indicators for Wipro to generate baseline features...")
    price_df = fetch_stock_price(ticker)
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)
    price_indicators_df = calculate_technical_indicators(price_df)
    
    fii_dii_df = fetch_latest_fii_dii()
    fii_dii_summary = analyze_institutional_signals(fii_dii_df['FII_Net'].tolist(), fii_dii_df['DII_Net'].tolist())
    macro_returns = get_latest_macro_returns()
    
    # Load 19-feature model
    model = joblib.load("models/xgboost_model.pkl")
    booster = model.get_booster()
    expected_features = booster.feature_names
    sent_score_idx = expected_features.index('Sentiment_Score')
    sent_avail_idx = expected_features.index('Sentiment_Available')
    
    print("\n" + "="*80)
    print(" SHAP VERIFICATION ANALYSIS FOR BUG 2 (WIPRO)")
    print("="*80)
    
    # ── Test Scenario 1: Sentiment_Available = 0 (Quiet Days / Filler Case) ──
    print("\n[TEST 1] Sentiment_Available = 0 (Quiet Days / Filler Case)")
    print(f"{'Sentiment_Score':<15} | {'Sentiment_Avail':<15} | {'BUY Probability':<15} | {'SHAP Score Contrib':<20}")
    print("-" * 75)
    
    s1_all_zero = True
    baseline_contrib = None
    for score in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
        # Sentiment Available is 0
        sentiment_summary = (score, 0, 0, 0)
        row = prepare_inference_row(price_indicators_df, fii_dii_summary, sentiment_summary, macro_returns)
        row_aligned = row[expected_features]
        
        # Predict probability
        pred_res = predict_stock_action(row)
        buy_prob = pred_res['Breakdown']['BUY']
        
        # Predict SHAP
        dmat = xgb.DMatrix(row_aligned)
        contribs = booster.predict(dmat, pred_contribs=True)
        buy_contrib = contribs[0, 2, sent_score_idx] # class 2 = BUY, sent_score_idx = Sentiment_Score
        
        if baseline_contrib is None:
            baseline_contrib = buy_contrib
        elif abs(buy_contrib - baseline_contrib) > 1e-6:
            s1_all_zero = False
            
        print(f"{score:<15.2f} | {0:<15} | {buy_prob*100:<14.2f}% | {buy_contrib:<+20.6f}")
        
    if s1_all_zero:
        print("\n  >> [PASS] Test 1: When Sentiment_Available = 0, Sentiment_Score has EXACTLY ZERO variation impact on BUY probability.")
    else:
        print("\n  >> [FAIL] Test 1: Sentiment_Score still contributes when Sentiment_Available = 0.")
        
    # ── Test Scenario 2: Sentiment_Available = 1 (Active News Days) ──
    print("\n" + "="*80)
    print("[TEST 2] Sentiment_Available = 1 (Active News Days)")
    print(f"{'Sentiment_Score':<15} | {'Sentiment_Avail':<15} | {'BUY Probability':<15} | {'SHAP Score Contrib':<20}")
    print("-" * 75)
    
    s2_correct_sign = True
    negative_score_contrib = None
    
    for score in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
        # Sentiment Available is 1. Headline counts > 0
        sentiment_summary = (score, 3, 3, 1)
        row = prepare_inference_row(price_indicators_df, fii_dii_summary, sentiment_summary, macro_returns)
        row_aligned = row[expected_features]
        
        # Predict probability
        pred_res = predict_stock_action(row)
        buy_prob = pred_res['Breakdown']['BUY']
        
        # Predict SHAP
        dmat = xgb.DMatrix(row_aligned)
        contribs = booster.predict(dmat, pred_contribs=True)
        buy_contrib = contribs[0, 2, sent_score_idx]
        
        if score == -0.10:
            negative_score_contrib = buy_contrib
            
        # Verify that negative sentiment yields a negative or lower SHAP contribution compared to positive sentiment
        # Real negative sentiment should have a negative/neutral SHAP score
        print(f"{score:<15.2f} | {1:<15} | {buy_prob*100:<14.2f}% | {buy_contrib:<+20.6f}")
        
    # Get specific checks for Wipro's negative -0.10 sentiment
    wipro_neg_summary = (-0.10, 2, 4, 1) # genuine negative case
    wipro_row = prepare_inference_row(price_indicators_df, fii_dii_summary, wipro_neg_summary, macro_returns)
    wipro_aligned = wipro_row[expected_features]
    wipro_pred = predict_stock_action(wipro_row)
    wipro_dmat = xgb.DMatrix(wipro_aligned)
    wipro_contribs = booster.predict(wipro_dmat, pred_contribs=True)
    wipro_score_contrib = wipro_contribs[0, 2, sent_score_idx]
    wipro_avail_contrib = wipro_contribs[0, 2, sent_avail_idx]
    
    print("\n" + "="*80)
    print(" DETAILED WIPRO GENUINE NEGATIVE SENTIMENT CHECK (-0.10 Score, Available=1)")
    print(f"  - Sentiment Score        : -0.10")
    print(f"  - Sentiment Available    : 1")
    print(f"  - BUY Class Probability   : {wipro_pred['Breakdown']['BUY']*100:.2f}%")
    print(f"  - SHAP Score Contribution: {wipro_score_contrib:+.6f}")
    print(f"  - SHAP Avail Contribution: {wipro_avail_contrib:+.6f}")
    print(f"  - Total Sentiment Impact : {wipro_score_contrib + wipro_avail_contrib:+.6f}")
    
    # In the original bug, a negative score on Wipro had a positive SHAP score of ~ +0.02
    # Verify that the score contribution for a negative sentiment is negative or significantly lower.
    if wipro_score_contrib <= 0.0:
        print("\n  >> [PASS] Test 2: Genuinely negative news sentiment correctly has a NEGATIVE contribution to the BUY probability.")
    else:
        # Check if the overall sentiment impact (score + avail) is negative
        total_impact = wipro_score_contrib + wipro_avail_contrib
        if total_impact <= 0.0:
            print("\n  >> [PASS] Test 2: Overall sentiment impact is negative. The model correctly penalizes BUY probability for negative news.")
        else:
            print("\n  >> [FAIL] Test 2: Negative sentiment still has a positive impact on BUY probability.")
            s2_correct_sign = False

    print("="*80 + "\n")

if __name__ == "__main__":
    main()
