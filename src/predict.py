import os
import joblib
import pandas as pd
import numpy as np

CONFIDENCE_THRESHOLD = 0.55   # signals below this are suppressed to HOLD

def predict_stock_action(feature_row: pd.DataFrame) -> dict:
    """
    Loads the XGBoost model and predicts the stock action.
    Applies a 55% confidence threshold — signals weaker than this
    are overridden to HOLD to avoid low-quality BUY/SELL calls.
    Returns: dict with Recommendation, Confidence, Breakdown, Low_Confidence flag.
    """
    model_path_json = "models/xgboost_model.json"
    model_path_pkl = "models/xgboost_model.pkl"
    class_map  = {0: "SELL", 1: "HOLD", 2: "BUY"}

    if os.path.exists(model_path_json):
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(model_path_json)
        expected_features = bst.feature_names
        feature_row_aligned = feature_row[expected_features]
        dtrain = xgb.DMatrix(feature_row_aligned)
        probs = bst.predict(dtrain)[0]
        pred_class = int(np.argmax(probs))
    elif os.path.exists(model_path_pkl):
        model = joblib.load(model_path_pkl)
        expected_features = model.get_booster().feature_names
        feature_row_aligned = feature_row[expected_features]
        probs      = model.predict_proba(feature_row_aligned)[0]
        pred_class = model.predict(feature_row_aligned)[0]
    else:
        raise FileNotFoundError(
            f"Trained model not found at '{model_path_json}' or '{model_path_pkl}'. "
            f"Please run: python src/train_model.py"
        )

    prediction = class_map[pred_class]
    confidence = float(probs[pred_class])

    sell_prob = float(probs[0])
    hold_prob = float(probs[1])
    buy_prob  = float(probs[2])

    # ── Confidence Threshold Filter ──────────────────────────────
    # If the model's top prediction confidence is below 55%, the
    # signal is too weak to act on — override to HOLD.
    low_confidence = False
    override_reason = None
    original_prediction = prediction
    original_confidence = confidence

    if confidence < CONFIDENCE_THRESHOLD and prediction != "HOLD":
        override_reason = (
            f"Original signal was {prediction} ({confidence*100:.1f}% confidence), "
            f"but this is below the 55% threshold. "
            f"Signal is too weak — defaulting to HOLD."
        )
        prediction     = "HOLD"
        low_confidence = True
        # print(f"[*] Threshold filter: {override_reason}")

    return {
        "Recommendation": prediction,
        "Confidence":     confidence,
        "Low_Confidence": low_confidence,
        "Override_Reason": override_reason,
        "Original_Recommendation": original_prediction,
        "Original_Confidence":     original_confidence,
        "Breakdown": {
            "BUY":  buy_prob,
            "HOLD": hold_prob,
            "SELL": sell_prob
        }
    }


if __name__ == "__main__":
    # Test prediction with mock features (19 features — updated to match current model)
    feature_cols = [
        'RSI', 'MACD_Hist', 'MACD_Crossover',
        'Price_Above_MA50', 'Price_Above_MA200', 'Golden_Cross',
        'Volume_Ratio', 'Sentiment_Score', 'Positive_Headlines', 'Negative_Headlines',
        'Sentiment_Available',
        'Multi_Timeframe_Alignment',
        'SP500_Return', 'Crude_Return', 'USD_INR_Return',
        'RSI_Slope', 'Price_Pct_5d', 'Price_Pct_20d', 'Volatility_10d'
    ]

    # Mock bullish row
    mock_row = pd.DataFrame([{
        'RSI': 35.0,
        'MACD_Hist': 0.12,
        'MACD_Crossover': 1,
        'Price_Above_MA50': 1,
        'Price_Above_MA200': 1,
        'Golden_Cross': 1,
        'Volume_Ratio': 1.8,
        'Sentiment_Score': 0.05,
        'Positive_Headlines': 4,
        'Negative_Headlines': 3,
        'Sentiment_Available': 1,
        'Multi_Timeframe_Alignment': 1,
        'SP500_Return': 0.005,
        'Crude_Return': -0.01,
        'USD_INR_Return': 0.002,
        'RSI_Slope': 5.2,
        'Price_Pct_5d': 1.5,
        'Price_Pct_20d': 3.2,
        'Volatility_10d': 1.1
    }])

    try:
        res = predict_stock_action(mock_row)
        print("Test Prediction:", res)
    except Exception as e:
        print("Cannot run test prediction yet:", e)
