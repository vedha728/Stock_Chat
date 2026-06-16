import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from indicators import calculate_technical_indicators
from feature_engineering import compile_feature_matrix

def calculate_max_drawdown(portfolio_values: list[float]) -> float:
    """
    Calculates the maximum drawdown (worst peak-to-trough drop) of the portfolio.
    """
    vals = np.array(portfolio_values)
    peaks = np.maximum.accumulate(vals)
    drawdowns = (peaks - vals) / peaks
    return float(np.max(drawdowns))


def run_backtest(ticker: str) -> dict:
    """
    Runs a historical backtest of the model's predictions over the last 1 year.
    Initial Capital: Rs. 100,000.
    Strategy:
      - If BUY signal is generated and not in a trade: Buy the stock with 100% capital.
      - Exit trade (sell) after 5 trading days OR if a SELL signal is generated.
    Returns: A summary dictionary of performance metrics.
    """
    model_path = "models/xgboost_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not trained yet! Run: python src/train_model.py")
        
    model = joblib.load(model_path)
    
    # 1. Fetch historical FII/DII data for the 1-year window
    # We use the generated historical FII/DII data or fetch the latest 10 days + mock the rest.
    # Since we need 1 year of daily FII/DII, we read from the historical csv
    fii_dii_csv = "data/fii_dii/historical_fii_dii.csv"
    if not os.path.exists(fii_dii_csv):
        # Generate it if it doesn't exist
        from institutional import generate_historical_fii_dii_csv
        generate_historical_fii_dii_csv("2024-06-01", "2026-06-01")
        
    fii_dii_df = pd.read_csv(fii_dii_csv)
    
    # 2. Fetch last 1 year of daily prices for the stock
    print(f"[*] Downloading 1-year historical data for {ticker} backtest...")
    stock_df = yf.download(ticker, period="1y")
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)
    stock_df = stock_df.reset_index()
    
    if stock_df.empty:
        raise ValueError(f"No price data found for {ticker}")
        
    # 3. Calculate indicators and compile features
    start_dt_str = pd.to_datetime(stock_df['Date'].min()).strftime("%Y-%m-%d")
    end_dt_str = pd.to_datetime(stock_df['Date'].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    from data_collector import fetch_global_macro_data
    macro_df = fetch_global_macro_data(start_dt_str, end_dt_str)
    
    price_indicators_df = calculate_technical_indicators(stock_df)
    feature_df = compile_feature_matrix(price_indicators_df, fii_dii_df, macro_df=macro_df, is_training=True)
    
    # 4. Predict signals day-by-day
    feature_cols = [
        'RSI', 'MACD_Hist', 'MACD_Crossover', 
        'Price_Above_MA50', 'Price_Above_MA200', 'Golden_Cross', 
        'Volume_Ratio', 'Sentiment_Score', 'Positive_Headlines', 'Negative_Headlines',
        'FII_10d_Net', 'DII_10d_Net', 'FII_Trend', 'DII_Trend', 
        'Divergence_Flag', 'Multi_Timeframe_Alignment',
        'SP500_Return', 'Crude_Return', 'USD_INR_Return'
    ]
    
    X = feature_df[feature_cols]
    predictions = model.predict(X) # Array of predictions: 0 (SELL), 1 (HOLD), 2 (BUY)
    
    # Add predictions back to dataframe
    feature_df['Signal'] = predictions
    
    # 5. Trading Simulation
    initial_capital = 100000.0
    cash = initial_capital
    position = 0.0  # Number of shares held
    portfolio_value = initial_capital
    portfolio_values = []
    
    in_trade = False
    entry_price = 0.0
    entry_idx = 0
    trade_returns = [] # Track returns of individual trades
    
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    
    for idx in range(len(feature_df)):
        price = float(feature_df['Close'].iloc[idx])
        signal = int(feature_df['Signal'].iloc[idx])
        
        # If in trade, check exit conditions
        if in_trade:
            days_held = idx - entry_idx
            # Exit conditions: 5 days passed OR sell signal generated
            if days_held >= 5 or signal == 0 or idx == len(feature_df) - 1:
                # Sell shares
                cash = position * price
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
        elif signal == 2 and idx < len(feature_df) - 1: # Buy signal
            position = cash / price
            entry_price = price
            entry_idx = idx
            cash = 0.0
            in_trade = True
            
        # Record current portfolio value
        curr_val = cash + (position * price)
        portfolio_values.append(curr_val)

    # Final metrics calculation
    final_portfolio_value = portfolio_values[-1]
    strategy_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100
    
    # Benchmark return (Buy and Hold the stock)
    start_price = float(feature_df['Close'].iloc[0])
    end_price = float(feature_df['Close'].iloc[-1])
    benchmark_return = ((end_price - start_price) / start_price) * 100
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
    
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    
    avg_win = np.mean(wins) * 100 if wins else 0.0
    avg_loss = np.mean(losses) * 100 if losses else 0.0
    
    max_dd = calculate_max_drawdown(portfolio_values) * 100
    
    return {
        "Ticker": ticker,
        "Initial_Capital": initial_capital,
        "Final_Value": final_portfolio_value,
        "Strategy_Return_Pct": strategy_return,
        "Benchmark_Return_Pct": benchmark_return,
        "Total_Trades": total_trades,
        "Win_Rate_Pct": win_rate,
        "Avg_Win_Pct": avg_win,
        "Avg_Loss_Pct": avg_loss,
        "Max_Drawdown_Pct": max_dd
    }

if __name__ == "__main__":
    try:
        report = run_backtest("TATAMOTORS.NS")
        print("\nBacktest Summary Report:")
        for k, v in report.items():
            if isinstance(v, float):
                print(f"{k}: {v:.2f}")
            else:
                print(f"{k}: {v}")
    except Exception as e:
        print("Cannot run backtest yet:", e)
