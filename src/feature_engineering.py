import pandas as pd
import numpy as np

def calculate_multi_timeframe_alignment(price_above_50: int, price_above_200: int, golden_cross: int, macd_hist: float) -> int:
    """
    Computes multi-timeframe bullish alignment:
    Returns 1 if:
      - Price is above 50 MA
      - Price is above 200 MA
      - Golden Cross is active (50 MA > 200 MA)
      - MACD momentum is positive (MACD Hist > 0)
    Else returns 0.
    """
    if price_above_50 == 1 and price_above_200 == 1 and golden_cross == 1 and macd_hist > 0:
        return 1
    return 0


def generate_historical_sentiment_proxy(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a realistic historical news sentiment proxy based on daily price changes.
    Used for 2-year training data since historical NewsAPI news is unavailable.
    - If daily return > 1.5%: positive sentiment score and high positive headlines count.
    - If daily return < -1.5%: negative sentiment score and high negative headlines count.
    - Otherwise: neutral sentiment score and balanced headlines.
    Adds gaussian noise to avoid perfect correlations (so the model does not cheat).
    """
    df = price_df.copy()
    np.random.seed(42)
    
    # Calculate daily percent change
    df['Return'] = df['Close'].pct_change()
    
    sentiment_scores = []
    pos_counts = []
    neg_counts = []
    
    for ret in df['Return']:
        if pd.isna(ret):
            sentiment_scores.append(0.0)
            pos_counts.append(0)
            neg_counts.append(0)
            continue
            
        # Add random noise to returns to prevent absolute correlation
        noise = np.random.normal(0, 0.005)
        noisy_return = ret + noise
        
        if noisy_return > 0.015:
            # Bullish day
            score = np.random.uniform(0.3, 0.8)
            pos = int(np.random.randint(5, 15))
            neg = int(np.random.randint(0, 3))
        elif noisy_return < -0.015:
            # Bearish day
            score = np.random.uniform(-0.8, -0.3)
            pos = int(np.random.randint(0, 3))
            neg = int(np.random.randint(5, 15))
        else:
            # Neutral day
            score = np.random.uniform(-0.2, 0.2)
            pos = int(np.random.randint(1, 5))
            neg = int(np.random.randint(1, 5))
            
        sentiment_scores.append(score)
        pos_counts.append(pos)
        neg_counts.append(neg)
        
    df['Sentiment_Score'] = sentiment_scores
    df['Positive_Headlines'] = pos_counts
    df['Negative_Headlines'] = neg_counts
    
    return df.drop(columns=['Return'])


def compile_feature_matrix(price_df: pd.DataFrame, fii_dii_df: pd.DataFrame, macro_df: pd.DataFrame = None, is_training: bool = False) -> pd.DataFrame:
    """
    Merges price, indicators, sentiment, and FII/DII data on 'Date'.
    Prepares the final features for the model.
    """
    # Ensure Date columns are datetime objects or matching strings
    price_df = price_df.copy()
    fii_dii_df = fii_dii_df.copy()
    
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    fii_dii_df['Date'] = pd.to_datetime(fii_dii_df['Date']).dt.date
    
    # 1. Handle news sentiment
    if is_training:
        # Generate proxy sentiment historically
        price_df = generate_historical_sentiment_proxy(price_df)
        
    # 2. Merge Price + Sentiment with FII/DII flow data
    # FII/DII data is macro-level daily data. We merge on Date.
    merged_df = pd.merge(price_df, fii_dii_df, on='Date', how='inner')
    
    # Merge with Global Macro returns if provided
    if macro_df is not None:
        macro_df = macro_df.copy()
        macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.date
        merged_df = pd.merge(merged_df, macro_df, on='Date', how='left')
        merged_df[['SP500_Return', 'Crude_Return', 'USD_INR_Return']] = merged_df[['SP500_Return', 'Crude_Return', 'USD_INR_Return']].ffill().fillna(0.0)
        
    # 3. Calculate rolling 10-day flows and trends for FII/DII
    merged_df['FII_10d_Net'] = merged_df['FII_Net'].rolling(window=10).sum()
    merged_df['DII_10d_Net'] = merged_df['DII_Net'].rolling(window=10).sum()
    
    # Fill NaN values for rolling sums at the start of the series
    merged_df['FII_10d_Net'] = merged_df['FII_10d_Net'].bfill().fillna(0.0)
    merged_df['DII_10d_Net'] = merged_df['DII_10d_Net'].bfill().fillna(0.0)

    # Compute FII/DII trend and divergence signals row by row
    fii_trends = []
    dii_trends = []
    div_flags = []
    
    # Use 10-day window for trends
    for idx in range(len(merged_df)):
        if idx < 9:
            # Default values for the first 9 rows
            fii_trends.append(0)
            dii_trends.append(0)
            div_flags.append(0)
            continue
            
        fii_window = merged_df['FII_Net'].iloc[idx-9:idx+1].tolist()
        dii_window = merged_df['DII_Net'].iloc[idx-9:idx+1].tolist()
        
        # Calculate trend direction (1 if last 3 days > first 3 days, else 0)
        f_trend = 1 if (fii_window[-1] + fii_window[-2]) > (fii_window[0] + fii_window[1]) else 0
        d_trend = 1 if (dii_window[-1] + dii_window[-2]) > (dii_window[0] + dii_window[1]) else 0
        
        fii_trends.append(f_trend)
        dii_trends.append(d_trend)
        
        # Divergence Flag (1 if FII and DII are moving in opposite directions, else 0)
        fii_sum = sum(fii_window)
        dii_sum = sum(dii_window)
        d_flag = 1 if (fii_sum * dii_sum < 0) else 0
        div_flags.append(d_flag)
        
    merged_df['FII_Trend'] = fii_trends
    merged_df['DII_Trend'] = dii_trends
    merged_df['Divergence_Flag'] = div_flags

    # 4. Multi-Timeframe Bulkish Alignment
    alignments = []
    for idx in range(len(merged_df)):
        align = calculate_multi_timeframe_alignment(
            merged_df.loc[idx, 'Price_Above_MA50'],
            merged_df.loc[idx, 'Price_Above_MA200'],
            merged_df.loc[idx, 'Golden_Cross'],
            merged_df.loc[idx, 'MACD_Hist']
        )
        alignments.append(align)
    merged_df['Multi_Timeframe_Alignment'] = alignments

    # Define features to output
    features = [
        'RSI', 'MACD_Hist', 'MACD_Crossover', 
        'Price_Above_MA50', 'Price_Above_MA200', 'Golden_Cross', 
        'Volume_Ratio', 'Sentiment_Score', 'Positive_Headlines', 'Negative_Headlines',
        'FII_10d_Net', 'DII_10d_Net', 'FII_Trend', 'DII_Trend', 
        'Divergence_Flag', 'Multi_Timeframe_Alignment',
        'SP500_Return', 'Crude_Return', 'USD_INR_Return'
    ]
    
    # If training, keep 'Date', 'Close', features, and label (which will be added later)
    # If inference, we just want the final row
    output_cols = ['Date', 'Close'] + features
    
    # Keep columns that are in merged_df
    final_cols = [col for col in output_cols if col in merged_df.columns]
    
    return merged_df[final_cols]


def prepare_inference_row(latest_price_row: pd.Series, fii_dii_summary: dict, sentiment_summary: tuple[float, int, int], macro_returns: tuple[float, float, float]) -> pd.DataFrame:
    """
    Constructs a single 19-feature row for real-time model inference.
    """
    sent_score, pos_count, neg_count = sentiment_summary
    sp500_ret, crude_ret, usdinr_ret = macro_returns
    
    # Compute multi-timeframe alignment
    mt_alignment = calculate_multi_timeframe_alignment(
        int(latest_price_row['Price_Above_MA50']),
        int(latest_price_row['Price_Above_MA200']),
        int(latest_price_row['Golden_Cross']),
        float(latest_price_row['MACD_Hist'])
    )
    
    # Construct dict of exactly 19 features matching training dataset
    feature_row = {
        'RSI': float(latest_price_row['RSI']),
        'MACD_Hist': float(latest_price_row['MACD_Hist']),
        'MACD_Crossover': int(latest_price_row['MACD_Crossover']),
        'Price_Above_MA50': int(latest_price_row['Price_Above_MA50']),
        'Price_Above_MA200': int(latest_price_row['Price_Above_MA200']),
        'Golden_Cross': int(latest_price_row['Golden_Cross']),
        'Volume_Ratio': float(latest_price_row['Volume_Ratio']),
        'Sentiment_Score': float(sent_score),
        'Positive_Headlines': int(pos_count),
        'Negative_Headlines': int(neg_count),
        'FII_10d_Net': float(fii_dii_summary['FII_10d_Net']),
        'DII_10d_Net': float(fii_dii_summary['DII_10d_Net']),
        'FII_Trend': int(fii_dii_summary['FII_Trend']),
        'DII_Trend': int(fii_dii_summary['DII_Trend']),
        'Divergence_Flag': int(fii_dii_summary['Divergence_Flag']),
        'Multi_Timeframe_Alignment': int(mt_alignment),
        'SP500_Return': float(sp500_ret),
        'Crude_Return': float(crude_ret),
        'USD_INR_Return': float(usdinr_ret)
    }
    
    return pd.DataFrame([feature_row])
