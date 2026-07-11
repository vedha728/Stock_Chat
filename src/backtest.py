import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

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
    model_path = "models/xgboost_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not trained yet! Run: python src/train_model.py")

    model = joblib.load(model_path)

    # Verify the model has the expected number of features (19)
    expected_features = 19
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != expected_features:
        raise ValueError(
            f"Model has {model.n_features_in_} features, but backtest expects {expected_features}. "
            "Please retrain the model: python src/train_model.py"
        )

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
    backtest_fii_csv = f"data/fii_dii/backtest_fii_dii_{start_dt_str}_{end_dt_str}.csv"
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

    # Get both hard predictions AND raw probabilities
    # Apply confidence threshold to match live chatbot logic
    raw_predictions = model.predict(X)          # Array: 0 (SELL), 1 (HOLD), 2 (BUY)
    probabilities   = model.predict_proba(X)    # Array: [sell_prob, hold_prob, buy_prob] per row

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
    trade_returns = []  # Track returns of individual trades

    total_trades   = 0
    winning_trades = 0
    losing_trades  = 0

    for idx in range(len(feature_df)):
        price  = float(feature_df['Close'].iloc[idx])
        signal = int(feature_df['Signal'].iloc[idx])

        # If in trade, check exit conditions
        if in_trade:
            days_held = idx - entry_idx
            # Exit conditions: 5 days passed OR sell signal generated OR last day
            if days_held >= 5 or signal == 0 or idx == len(feature_df) - 1:
                # Sell shares
                cash         = position * price
                trade_return = (price - entry_price) / entry_price
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
            position    = cash / price
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
