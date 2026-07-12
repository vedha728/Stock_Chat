import json
import math
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

def predict_pure_python(model_json_path, feature_dict):
    with open(model_json_path, "r") as f:
        model_data = json.load(f)
    
    # 1. Get feature names and order
    feature_names = model_data["learner"]["feature_names"]
    
    # Map feature values to index positions
    X = [feature_dict.get(name, float("nan")) for name in feature_names]
    
    # 2. Get trees
    trees = model_data["learner"]["gradient_booster"]["model"]["trees"]
    tree_info = model_data["learner"]["gradient_booster"]["model"]["tree_info"]
    
    # 3. Get base score margin
    base_score_str = model_data["learner"].get("learner_model_param", {}).get("base_score", "0.5")
    if base_score_str.startswith("["):
        base_margin = json.loads(base_score_str)
    else:
        # Fallback to float base_score
        val = float(base_score_str)
        # If it's a probability (like 0.5), margin is log(p)
        margin_val = math.log(val) if val > 0 else 0.0
        base_margin = [margin_val, margin_val, margin_val]

    # Initialize margins for 3 classes
    raw_scores = [float(m) for m in base_margin]
    
    for t_idx, tree in enumerate(trees):
        class_id = tree_info[t_idx]
        
        # Traverse this tree
        left_children = tree["left_children"]
        right_children = tree["right_children"]
        split_indices = tree["split_indices"]
        split_conditions = tree["split_conditions"]
        default_left = tree["default_left"]
        base_weights = tree["base_weights"]
        
        node = 0
        while left_children[node] != -1:
            f_idx = split_indices[node]
            threshold = split_conditions[node]
            val = X[f_idx]
            
            # Check for NaN/missing
            if val is None or math.isnan(val):
                node = left_children[node] if default_left[node] == 1 else right_children[node]
            else:
                node = left_children[node] if val < threshold else right_children[node]
                
        # Leaf value
        raw_scores[class_id] += base_weights[node]
        
    # 4. Apply Softmax to raw scores to get probabilities
    exp_scores = [math.exp(score) for score in raw_scores]
    sum_exp = sum(exp_scores)
    probs = [score / sum_exp for score in exp_scores]
    
    return probs

# Test it
if __name__ == "__main__":
    # Load mock features
    mock_features = {
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
    }
    
    # Get actual XGBoost predictions
    pkl_model = joblib.load("models/xgboost_model.pkl")
    df = pd.DataFrame([mock_features])
    
    expected_features = pkl_model.get_booster().feature_names
    df_aligned = df[expected_features]
    
    actual_probs = pkl_model.predict_proba(df_aligned)[0]
    
    # Get pure Python predictions
    pure_probs = predict_pure_python("models/xgboost_model.json", mock_features)
    
    print("Actual XGBoost Probs:", actual_probs)
    print("Pure Python Probs:  ", pure_probs)
    print("Match?", np.allclose(actual_probs, pure_probs, atol=1e-5))
