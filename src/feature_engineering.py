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
    Generates a neutral, price-independent sentiment proxy for training data.
    Used because real historical news data (NewsAPI) is unavailable for past years.

    FIX (Option C — Neutral Random Noise):
    Previously, sentiment was derived directly from daily price returns
    (price up → positive sentiment, price down → negative sentiment).
    This caused circular data leakage — the model was learning price direction
    disguised as sentiment, inflating training accuracy artificially.

    Now: sentiment values are randomly sampled from a narrow neutral distribution,
    completely uncorrelated to price. This means:
      - No fake patterns are learned during training
      - The feature slot stays open (same 23-feature architecture)
      - At inference, real FinBERT scores from NewsAPI will stand out as
        genuine signal against this neutral baseline, and the model
        will respond to them meaningfully.

    Distribution used:
      Sentiment_Score    : Normal(mean=0, std=0.15), clamped to [-0.4, 0.4]
      Positive_Headlines : Uniform integer [1, 6]
      Negative_Headlines : Uniform integer [1, 6]
    These ranges reflect a genuinely uncertain, noisy news environment —
    not one that always agrees with price direction.
    """
    df = price_df.copy()
    np.random.seed(42)
    n = len(df)

    # Random neutral sentiment scores — no connection to price at all
    raw_scores = np.random.normal(loc=0.0, scale=0.15, size=n)
    sentiment_scores = np.clip(raw_scores, -0.4, 0.4).tolist()

    # Random headline counts — balanced, no directional bias
    pos_counts = np.random.randint(1, 7, size=n).tolist()   # 1 to 6
    neg_counts = np.random.randint(1, 7, size=n).tolist()   # 1 to 6

    df['Sentiment_Score']    = sentiment_scores
    df['Positive_Headlines'] = pos_counts
    df['Negative_Headlines'] = neg_counts

    return df


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

    # 4. Multi-Timeframe Bullish Alignment
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

    # 5. Rolling Window Features (Issue #3 fix — sequence context for XGBoost)
    # RSI_Slope: direction of RSI momentum (positive = RSI recovering, negative = RSI weakening)
    merged_df['RSI_Slope'] = merged_df['RSI'].diff(5).fillna(0.0)

    # Price % change over 5 and 20 trading days (short and medium term momentum)
    merged_df['Price_Pct_5d']  = merged_df['Close'].pct_change(5).fillna(0.0)  * 100
    merged_df['Price_Pct_20d'] = merged_df['Close'].pct_change(20).fillna(0.0) * 100

    # 10-day rolling volatility: std of daily returns (measures risk/uncertainty)
    daily_returns = merged_df['Close'].pct_change()
    merged_df['Volatility_10d'] = daily_returns.rolling(window=10).std().fillna(0.0) * 100

    # Define features to output (18 total: 14 original + 4 new rolling features)
    features = [
        'RSI', 'MACD_Hist', 'MACD_Crossover',
        'Price_Above_MA50', 'Price_Above_MA200', 'Golden_Cross',
        'Volume_Ratio', 'Sentiment_Score', 'Positive_Headlines', 'Negative_Headlines',
        'Multi_Timeframe_Alignment',
        'SP500_Return', 'Crude_Return', 'USD_INR_Return',
        'RSI_Slope', 'Price_Pct_5d', 'Price_Pct_20d', 'Volatility_10d'
    ]
    
    # If training, keep 'Date', 'Close', features, and label (which will be added later)
    # If inference, we just want the final row
    output_cols = ['Date', 'Close'] + features
    
    # Keep columns that are in merged_df
    final_cols = [col for col in output_cols if col in merged_df.columns]
    
    return merged_df[final_cols]


def prepare_inference_row(
    price_indicators_df: pd.DataFrame,
    fii_dii_summary: dict,
    sentiment_summary: tuple,
    macro_returns: tuple
) -> pd.DataFrame:
    """
    Constructs a single 23-feature row for real-time model inference.
    Accepts the full price_indicators_df (not just latest row) so that
    rolling window features (RSI_Slope, Price_Pct_5d, Price_Pct_20d,
    Volatility_10d) can be accurately computed from recent history.
    """
    sent_score, pos_count, neg_count = sentiment_summary
    sp500_ret, crude_ret, usdinr_ret = macro_returns

    # Use the last row for point-in-time values
    latest = price_indicators_df.iloc[-1]

    # Compute multi-timeframe alignment
    mt_alignment = calculate_multi_timeframe_alignment(
        int(latest['Price_Above_MA50']),
        int(latest['Price_Above_MA200']),
        int(latest['Golden_Cross']),
        float(latest['MACD_Hist'])
    )

    # ── Rolling window features (Issue #3) ───────────────────────────────
    # RSI_Slope: RSI today minus RSI 5 days ago — captures momentum direction
    rsi_series = price_indicators_df['RSI']
    rsi_slope = float(rsi_series.iloc[-1] - rsi_series.iloc[-6]) if len(rsi_series) >= 6 else 0.0

    # Price % change over 5 and 20 trading days
    close_series = price_indicators_df['Close']
    price_pct_5d  = float((close_series.iloc[-1] / close_series.iloc[-6]  - 1) * 100) if len(close_series) >= 6  else 0.0
    price_pct_20d = float((close_series.iloc[-1] / close_series.iloc[-21] - 1) * 100) if len(close_series) >= 21 else 0.0

    # 10-day rolling volatility of daily returns
    if len(close_series) >= 11:
        daily_rets = close_series.pct_change().dropna()
        volatility_10d = float(daily_rets.iloc[-10:].std() * 100)
    else:
        volatility_10d = 0.0
    # ─────────────────────────────────────────────────────────────────────

    # Construct dict of exactly 18 features matching training dataset
    feature_row = {
        'RSI':                      float(latest['RSI']),
        'MACD_Hist':                float(latest['MACD_Hist']),
        'MACD_Crossover':           int(latest['MACD_Crossover']),
        'Price_Above_MA50':         int(latest['Price_Above_MA50']),
        'Price_Above_MA200':        int(latest['Price_Above_MA200']),
        'Golden_Cross':             int(latest['Golden_Cross']),
        'Volume_Ratio':             float(latest['Volume_Ratio']),
        'Sentiment_Score':          float(sent_score),
        'Positive_Headlines':       int(pos_count),
        'Negative_Headlines':       int(neg_count),
        'Multi_Timeframe_Alignment':int(mt_alignment),
        'SP500_Return':             float(sp500_ret),
        'Crude_Return':             float(crude_ret),
        'USD_INR_Return':           float(usdinr_ret),
        'RSI_Slope':                rsi_slope,
        'Price_Pct_5d':             price_pct_5d,
        'Price_Pct_20d':            price_pct_20d,
        'Volatility_10d':           volatility_10d,
    }

    return pd.DataFrame([feature_row])
