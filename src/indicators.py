import pandas as pd
import numpy as np
import ta

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators on the OHLCV stock price DataFrame.
    Required indicators:
    - RSI (14-day)
    - MACD (Line, Signal, Histogram, and Crossover flag)
    - Moving Averages (50-day SMA, 200-day SMA, Above flags, Golden Cross flag)
    - Volume ratio (Today's volume / 20-day average volume)
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # 1. RSI (14-day)
    rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()

    # 2. MACD
    macd_indicator = ta.trend.MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD_Line'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    df['MACD_Hist'] = macd_indicator.macd_diff()  # Histogram

    # MACD Crossover occurred today: 
    # Defined as MACD line crossing the Signal line today (crossing 0 in histogram)
    df['MACD_Crossover'] = 0
    # Compute crossovers: sign change in histogram from yesterday to today
    for i in range(1, len(df)):
        hist_prev = df.loc[i - 1, 'MACD_Hist']
        hist_curr = df.loc[i, 'MACD_Hist']
        if pd.notna(hist_prev) and pd.notna(hist_curr):
            # Check if they crossed (multiplication is negative means opposite signs)
            if hist_prev * hist_curr < 0 or (hist_prev == 0 and hist_curr != 0):
                df.loc[i, 'MACD_Crossover'] = 1

    # 3. Moving Averages
    df['MA50'] = ta.trend.sma_indicator(close=df['Close'], window=50)
    df['MA200'] = ta.trend.sma_indicator(close=df['Close'], window=200)

    # Boolean flags (0 or 1)
    df['Price_Above_MA50'] = (df['Close'] > df['MA50']).astype(int)
    df['Price_Above_MA200'] = (df['Close'] > df['MA200']).astype(int)
    
    # Golden Cross present (50 MA > 200 MA)
    df['Golden_Cross'] = (df['MA50'] > df['MA200']).astype(int)

    # 4. Volume Analysis
    # Calculate 20-day average volume
    df['Vol_Avg20'] = df['Volume'].rolling(window=20).mean()
    # Volume ratio (today's volume vs 20-day average)
    df['Volume_Ratio'] = df['Volume'] / df['Vol_Avg20']
    
    # Handle NaNs (e.g. at the beginning of the series)
    df['RSI'] = df['RSI'].fillna(50.0)  # Default neutral RSI
    df['MACD_Line'] = df['MACD_Line'].fillna(0.0)
    df['MACD_Signal'] = df['MACD_Signal'].fillna(0.0)
    df['MACD_Hist'] = df['MACD_Hist'].fillna(0.0)
    df['MA50'] = df['MA50'].bfill().ffill()  # Backfill then forward fill for missing values
    df['MA200'] = df['MA200'].bfill().ffill()
    df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0) # Default neutral volume ratio
    
    return df

if __name__ == "__main__":
    # Test indicators calculation
    import yfinance as yf
    print("[*] Fetching test data...")
    test_df = yf.download("TATAMOTORS.NS", period="1y")
    # Reset multi-index if yfinance downloaded columns are multi-indexed
    if isinstance(test_df.columns, pd.MultiIndex):
        test_df.columns = test_df.columns.get_level_values(0)
    test_df = test_df.reset_index()
    
    print("[*] Calculating indicators...")
    result_df = calculate_technical_indicators(test_df)
    print(result_df[['Date', 'Close', 'RSI', 'MACD_Hist', 'MACD_Crossover', 'MA50', 'MA200', 'Golden_Cross', 'Volume_Ratio']].tail(3))
