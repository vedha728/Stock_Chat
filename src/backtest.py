import os
import yfinance as yf
import pandas as pd
import numpy as np

from indicators import calculate_technical_indicators
from feature_engineering import compile_feature_matrix

# Must match the threshold in predict.py — signals weaker than 55% are suppressed to HOLD
BACKTEST_CONFIDENCE_THRESHOLD = 0.55

def calculate_max_drawdown(portfolio_values: list[float]) -> float:
    """
    Calculates the maximum drawdown (worst peak-to-trough drop) of the portfolio.
    """
    vals = np.array(portfolio_values)
    if len(vals) == 0:
        return 0.0
    peaks = np.maximum.accumulate(vals)
    drawdowns = (peaks - vals) / np.where(peaks == 0, 1, peaks)
    return float(np.max(drawdowns))


def run_backtest(ticker: str) -> dict:
    """
    Runs a historical backtest of the model's predictions over the last 1 year.
    Initial Capital: Rs. 100,000.
    Strategy:
      - If BUY signal is generated (above 55% confidence) and not in a trade: Buy with 100% capital.
      - Exit trade (sell) after 5 trading days OR if a SELL signal is generated.

    IMPORTANT — Confidence Threshold:
      The backtest applies the same 55% confidence threshold as the live prediction engine
      (predict.py). Only BUY/SELL signals with probability >= 55% trigger a trade.
      Weaker signals are treated as HOLD, keeping the backtest realistic and consistent
      with real-world chatbot behaviour.

    Returns: A summary dictionary of performance metrics.
    """
    # Model loading and predictions are deferred to the step-by-step parser below

    # 1. Fetch last 1 year of daily prices for the stock FIRST
    #    (we need the actual date range before generating FII/DII)
    print(f"[*] Downloading 1-year historical data for {ticker} backtest...")
    stock_df = yf.download(ticker, period="1y", progress=False)
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)
    stock_df = stock_df.reset_index()

    if stock_df.empty:
        raise ValueError(f"No price data found for {ticker}. Check that the ticker symbol is correct.")

    print(f"[+] Downloaded {len(stock_df)} trading days for {ticker}.")

    # Determine exact date range of the price data
    start_dt_str = pd.to_datetime(stock_df['Date'].min()).strftime("%Y-%m-%d")
    end_dt_str   = pd.to_datetime(stock_df['Date'].max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    # 2. Generate FII/DII data scoped exactly to the backtest window
    #    CRITICAL FIX: do NOT reuse the training FII/DII file (covers 2020-2025).
    #    A 1-year backtest run today needs 2025-2026 data. If we INNER JOIN
    #    price data (2025-2026) with training FII/DII (2020-2025), the result
    #    is 0 rows → empty feature_df → "No portfolio data" error.
    #    Solution: always regenerate for the actual backtest window dates.
    from institutional import generate_historical_fii_dii_csv
    base_dir = "/tmp/data/fii_dii" if os.environ.get("VERCEL") == "1" else "data/fii_dii"
    os.makedirs(base_dir, exist_ok=True)
    backtest_fii_csv = os.path.join(base_dir, f"backtest_fii_dii_{start_dt_str}_{end_dt_str}.csv")
    if not os.path.exists(backtest_fii_csv):
        print(f"[*] Generating FII/DII data for backtest window ({start_dt_str} to {end_dt_str})...")
        generate_historical_fii_dii_csv(start_dt_str, end_dt_str, output_path=backtest_fii_csv)
    else:
        print(f"[*] Reusing cached backtest FII/DII for window {start_dt_str} to {end_dt_str}.")

    fii_dii_df = pd.read_csv(backtest_fii_csv)
    print(f"[+] Loaded {len(fii_dii_df)} rows of FII/DII data for backtest window.")



    from data_collector import fetch_global_macro_data
    print("[*] Fetching global macro returns for backtest window...")
    try:
        macro_df = fetch_global_macro_data(start_dt_str, end_dt_str)
        print(f"[+] Macro data loaded: {len(macro_df)} rows.")
    except Exception as e:
        print(f"[Warning] Could not fetch macro data: {e}. Using zeros as fallback.")
        # Build zero-filled macro dataframe so backtest can continue
        date_range = pd.date_range(start=start_dt_str, end=end_dt_str, freq='B')
        macro_df = pd.DataFrame({
            'Date': date_range,
            'SP500_Return': 0.0,
            'Crude_Return': 0.0,
            'USD_INR_Return': 0.0
        })

    price_indicators_df = calculate_technical_indicators(stock_df)
    feature_df = compile_feature_matrix(price_indicators_df, fii_dii_df, macro_df=macro_df, is_training=True, ticker=ticker)

    # 4. Predict signals day-by-day — WITH confidence threshold applied (same as live chatbot)
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

    # Verify all required features are present
    missing_cols = [c for c in feature_cols if c not in feature_df.columns]
    if missing_cols:
        raise ValueError(
            f"Backtest feature mismatch — the following columns are missing from the data: {missing_cols}. "
            "This may mean the model was trained with different features. Retrain with: python src/train_model.py"
        )

    X = feature_df[feature_cols]

    model_path_json = "models/xgboost_model.json"
    model_path_pkl = "models/xgboost_model.pkl"

    if os.path.exists(model_path_json):
        import json
        import math
        
        with open(model_path_json, "r") as f:
            model_data = json.load(f)
            
        feature_names = model_data["learner"]["feature_names"]
        trees = model_data["learner"]["gradient_booster"]["model"]["trees"]
        tree_info = model_data["learner"]["gradient_booster"]["model"]["tree_info"]
        
        base_score_str = model_data["learner"].get("learner_model_param", {}).get("base_score", "0.5")
        if base_score_str.startswith("["):
            base_margin = json.loads(base_score_str)
        else:
            val = float(base_score_str)
            margin_val = math.log(val) if val > 0 else 0.0
            base_margin = [margin_val, margin_val, margin_val]
            
        base_margin = [float(m) for m in base_margin]
        
        raw_predictions = []
        probabilities = []
        
        # Convert X to list of dicts for extremely fast row-by-row iteration
        rows = X.to_dict(orient="records")
        
        for row_dict in rows:
            row_X = [row_dict.get(name, float("nan")) for name in feature_names]
            raw_scores = list(base_margin)
            
            for t_idx, tree in enumerate(trees):
                class_id = tree_info[t_idx]
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
                    val = row_X[f_idx]
                    
                    if val is None or math.isnan(val):
                        node = left_children[node] if default_left[node] == 1 else right_children[node]
                    else:
                        node = left_children[node] if val < threshold else right_children[node]
                        
                raw_scores[class_id] += base_weights[node]
                
            exp_scores = [math.exp(score) for score in raw_scores]
            sum_exp = sum(exp_scores)
            probs = [score / sum_exp for score in exp_scores]
            pred_class = int(np.argmax(probs))
            
            raw_predictions.append(pred_class)
            probabilities.append(probs)
            
    elif os.path.exists(model_path_pkl):
        import joblib
        model = joblib.load(model_path_pkl)
        raw_predictions = model.predict(X)
        probabilities   = model.predict_proba(X)
    else:
        raise FileNotFoundError("Model file not found! Train model first.")

    # Apply threshold: if top confidence < 55%, override to HOLD (1)
    thresholded_signals = []
    threshold_overrides = 0
    for i, pred_class in enumerate(raw_predictions):
        confidence = float(probabilities[i][pred_class])
        if confidence < BACKTEST_CONFIDENCE_THRESHOLD and pred_class != 1:
            thresholded_signals.append(1)   # Override to HOLD
            threshold_overrides += 1
        else:
            thresholded_signals.append(int(pred_class))

    feature_df = feature_df.copy()
    feature_df['Signal'] = thresholded_signals

    print(f"[+] Predictions computed. Threshold overrides (weak signals -> HOLD): {threshold_overrides}/{len(X)}")

    # 5. Trading Simulation
    initial_capital  = 100000.0
    cash             = initial_capital
    position         = 0.0   # Number of shares held
    portfolio_values = []

    in_trade     = False
    entry_price  = 0.0
    entry_idx    = 0
    entry_capital = 0.0  # Cash held before entering the trade
    trade_returns = []  # Track returns of individual trades

    total_trades   = 0
    winning_trades = 0
    losing_trades  = 0

    fee_rate        = 0.001   # 0.1% transaction fee (brokerage/tax)
    stop_loss_pct   = -0.025  # -2.5% Stop-loss
    take_profit_pct = 0.05    # +5.0% Take-profit

    for idx in range(len(feature_df)):
        price  = float(feature_df['Close'].iloc[idx])
        signal = int(feature_df['Signal'].iloc[idx])

        # If in trade, check exit conditions
        if in_trade:
            days_held = idx - entry_idx
            current_return = (price - entry_price) / entry_price
            
            # Exit conditions: 
            # 1. Stop-loss hit (<= -2.5%)
            # 2. Take-profit hit (>= 5.0%)
            # 3. Time-limit reached (>= 5 days)
            # 4. Model issued a SELL signal (signal == 0)
            # 5. Last day of backtest
            is_stop_loss = current_return <= stop_loss_pct
            is_take_profit = current_return >= take_profit_pct
            is_time_exit = days_held >= 5
            is_sell_signal = signal == 0
            is_last_day = idx == len(feature_df) - 1

            if is_stop_loss or is_take_profit or is_time_exit or is_sell_signal or is_last_day:
                # Sell shares
                gross_sale   = position * price
                sell_fee     = gross_sale * fee_rate
                cash         = gross_sale - sell_fee
                
                # Trade return accounts for both entry and exit fees
                trade_return = (cash - entry_capital) / entry_capital
                trade_returns.append(trade_return)

                if trade_return > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1

                position = 0.0
                in_trade = False
                total_trades += 1

        # If not in trade, check entry condition
        elif signal == 2 and idx < len(feature_df) - 1:  # Buy signal
            entry_capital = cash
            buy_fee       = cash * fee_rate
            net_buy       = cash - buy_fee
            
            position    = net_buy / price
            entry_price = price
            entry_idx   = idx
            cash        = 0.0
            in_trade    = True

        # Record current portfolio value
        curr_val = cash + (position * price)
        portfolio_values.append(curr_val)

    if not portfolio_values:
        raise ValueError("No portfolio data was generated. The backtest window may be too short.")

    # Final metrics calculation
    final_portfolio_value = portfolio_values[-1]
    strategy_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100

    # Benchmark return (Buy and Hold the stock)
    start_price      = float(feature_df['Close'].iloc[0])
    end_price        = float(feature_df['Close'].iloc[-1])
    benchmark_return = ((end_price - start_price) / start_price) * 100

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0

    wins   = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]

    avg_win  = np.mean(wins)   * 100 if wins   else 0.0
    avg_loss = np.mean(losses) * 100 if losses else 0.0

    max_dd = calculate_max_drawdown(portfolio_values) * 100

    return {
        "Ticker":                ticker,
        "Initial_Capital":       initial_capital,
        "Final_Value":           final_portfolio_value,
        "Strategy_Return_Pct":   strategy_return,
        "Benchmark_Return_Pct":  benchmark_return,
        "Total_Trades":          total_trades,
        "Win_Rate_Pct":          win_rate,
        "Avg_Win_Pct":           avg_win,
        "Avg_Loss_Pct":          avg_loss,
        "Max_Drawdown_Pct":      max_dd,
        "Threshold_Overrides":   threshold_overrides,   # extra diagnostic info
    }

if __name__ == "__main__":
    try:
        report = run_backtest("TCS.NS")
        print("\nBacktest Summary Report:")
        for k, v in report.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print("Cannot run backtest yet:", e)
