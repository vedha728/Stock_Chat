import streamlit as st
import os
import sys
import pandas as pd
import numpy as np

# Reconfigure stdout to use UTF-8 to avoid encoding issues in logs
sys.stdout.reconfigure(encoding='utf-8')

# Diagnostics to debug the import error
print("--- PYTHON DIAGNOSTICS ---")
print("Python executable:", sys.executable)
print("Python path:", sys.path)
import subprocess
try:
    res = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
    print("--- INSTALLED PACKAGES ---")
    print(res.stdout)
except Exception as e:
    print("Failed to list packages:", e)

try:
    import google
    print("Google namespace path:", getattr(google, "__path__", "No __path__ attribute"))
except Exception as e:
    print("Failed to import google:", e)

# Link to src directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "src"))

# Imports from src modules
from data_collector import extract_ticker, extract_multiple_tickers, fetch_stock_price, fetch_news_headlines, fetch_fundamentals, get_latest_macro_returns
from indicators import calculate_technical_indicators
from sentiment import analyze_news_sentiment, deduplicate_headlines
from institutional import fetch_latest_fii_dii, analyze_institutional_signals
from feature_engineering import prepare_inference_row
from predict import predict_stock_action
from backtest import run_backtest
from gemini_explain import generate_beginner_explanation, explain_educational_concept, generate_comparison_explanation
from score_engine import compute_signal_scores

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockChat AI Advisory Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark CSS styling injection
st.markdown("""
<style>
    /* Custom main background and fonts */
    .stApp {
        background-color: #0E1117;
        color: #E0E2E7;
    }
    
    /* Title bar styling */
    .title-banner {
        padding: 20px;
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 12px;
        border: 1px solid #1E293B;
        text-align: center;
        margin-bottom: 25px;
    }
    
    /* Custom Glassmorphism Card styling */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }
    
    /* Glow cards based on recommendation */
    .verdict-card-buy {
        border-left: 6px solid #10B981;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.1);
        background: rgba(16, 185, 129, 0.05);
    }
    .verdict-card-sell {
        border-left: 6px solid #EF4444;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.1);
        background: rgba(239, 68, 68, 0.05);
    }
    .verdict-card-hold {
        border-left: 6px solid #F59E0B;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.1);
        background: rgba(245, 158, 11, 0.05);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 14px;
    }
    .badge-buy { background-color: #10B981; color: white; }
    .badge-sell { background-color: #EF4444; color: white; }
    .badge-hold { background-color: #F59E0B; color: black; }
    
    /* Metrics grids styling */
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 12px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* News list */
    .news-item {
        padding: 8px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 14px;
    }
    .news-item:last-child {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/nolan/96/combo-chart.png", width=60)
st.sidebar.title("StockChat AI")
st.sidebar.caption("Multi-Signal Indian Stock Advisor")

# Let users supply Gemini API key to override environment if needed
user_gemini_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password", help="If left blank, the app will use the server's default GEMINI_API_KEY secret.")
if user_gemini_key:
    os.environ["GEMINI_API_KEY"] = user_gemini_key

# Main Mode Selector in Sidebar
app_mode = st.sidebar.radio(
    "Choose Mode",
    ["🔍 Stock Analysis & Chat", "📊 Side-by-Side Comparison", "📈 Strategy Backtesting", "📚 Learn Basics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Supported Stocks:
TCS, Reliance, Wipro, Infosys, SBI, HDFC Bank, Adani Ports, Tata Steel, Tata Power, Maruti Suzuki, L&T, Airtel, and all other major NSE indices (via auto-Yahoo Finance fetch).
""")

# ─────────────────────────────────────────────────────────────
# APP HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-banner">
    <h1 style="margin: 0; color: #38BDF8;">📊 Multi-Signal AI Stock Advisory</h1>
    <p style="margin: 5px 0 0 0; color: #94A3B8; font-size: 15px;">Leverages Technical Charts, News Sentiment, Institutional Flows, and Fundamental Ratios via XGBoost & Gemini</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# UTILITY HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def get_verdict_style(rec: str, low_conf: bool):
    if low_conf or rec == "HOLD":
        return "verdict-card-hold", "badge-hold", "⚖️ HOLD"
    elif rec == "BUY":
        return "verdict-card-buy", "badge-buy", "📈 BUY"
    else:
        return "verdict-card-sell", "badge-sell", "📉 SELL"

# ─────────────────────────────────────────────────────────────
# MODE 1: STOCK ANALYSIS & CHAT
# ─────────────────────────────────────────────────────────────
if app_mode == "🔍 Stock Analysis & Chat":
    st.write("### 🔍 Search Stock or Ask a Question")
    
    # Initialize session state for the query
    if "chat_active_query" not in st.session_state:
        st.session_state.chat_active_query = ""

    # Predefined buttons that update active query and trigger rerun
    cols = st.columns(4)
    with cols[0]:
        if st.button("Should I buy TCS?"):
            st.session_state.chat_active_query = "Should I buy TCS?"
            st.rerun()
    with cols[1]:
        if st.button("Analyze Maruti Suzuki"):
            st.session_state.chat_active_query = "Analyze Maruti Suzuki"
            st.rerun()
    with cols[2]:
        if st.button("News about Wipro"):
            st.session_state.chat_active_query = "News about Wipro"
            st.rerun()
    with cols[3]:
        if st.button("Is L&T overextended?"):
            st.session_state.chat_active_query = "Is L&T overextended?"
            st.rerun()

    # Text input box displaying the persistent active query
    user_input_val = st.text_input(
        "Type your query below:",
        value=st.session_state.chat_active_query,
        placeholder="e.g., Should I buy TCS? or analyze Reliance, or what is the news on Wipro?",
        help="Type any stock name or natural query like: 'Is L&T stock good to buy?'"
    )

    # Sync text input changes to active_query
    if user_input_val != st.session_state.chat_active_query:
        st.session_state.chat_active_query = user_input_val
        st.rerun()

    # Use the session state query for execution
    chat_input = st.session_state.chat_active_query

    if chat_input:
        # Determine intent tags
        q_lower = chat_input.lower()
        has_news_intent = any(w in q_lower for w in ["news", "headline", "headlines", "article", "articles", "media"])
        has_fundamentals_intent = any(w in q_lower for w in ["valuation", "debt", "pe", "pb", "roe", "mcap", "fundamental", "fundamentals", "market cap", "leverage", "profits"])
        has_price_intent = any(w in q_lower for w in ["price", "chart", "trend", "technical", "technicals", "rsi", "macd", "moving average"])
        
        # Intercept general Tata Motors demerger query
        is_general_tata = ("tata motor" in q_lower or "tatamotors" in q_lower) and not any(w in q_lower for w in ["pv", "passenger", "cv", "commercial", "tmpv", "tmcv"])
        
        selected_ticker = None
        selected_company = None
        
        if is_general_tata:
            st.warning("💡 **Tata Motors has demerged into two separate listed entities:**")
            tata_choice = st.radio(
                "Which company are you asking about?",
                ["Tata Motors Passenger Vehicles (TMPV)", "Tata Motors Commercial Vehicles (TMCV)"]
            )
            if "Passenger" in tata_choice:
                selected_ticker, selected_company = "TMPV.NS", "Tata Motors Passenger Vehicles"
            else:
                selected_ticker, selected_company = "TMCV.NS", "Tata Motors Commercial Vehicles"
        else:
            with st.spinner("Extracting stock symbol..."):
                selected_ticker, selected_company = extract_ticker(chat_input)
        
        if not selected_ticker:
            st.error(f"❌ **Stock Not Found:** Could not identify a stock symbol matching '{chat_input}'.")
            st.info("💡 **Tip**: Try typing the exact company name (e.g. 'Infosys', 'Adani Ports') or the ticker symbol directly.")
        else:
            st.success(f"📌 **Target Stock**: {selected_company} (`{selected_ticker}`)")
            
            # --- ROUTE TO FAST PATHS IF ONLY SPECIFIC METRIC REQUESTED ---
            
            # FAST PATH 1: News only
            if has_news_intent and not (has_fundamentals_intent or has_price_intent):
                with st.spinner(f"Retrieving news for {selected_company}..."):
                    try:
                        headlines = fetch_news_headlines(selected_ticker, selected_company)
                        headlines = deduplicate_headlines(headlines)
                        st.subheader(f"📰 Latest News: {selected_company}")
                        if not headlines:
                            st.info("No recent news headlines found.")
                        else:
                            for idx, h in enumerate(headlines[:10], 1):
                                st.markdown(f"**{idx}.** {h}")
                    except Exception as e:
                        st.error(f"Failed to fetch news: {e}")
            
            # FAST PATH 2: Fundamentals only
            elif has_fundamentals_intent and not (has_news_intent or has_price_intent):
                with st.spinner(f"Retrieving fundamentals for {selected_company}..."):
                    try:
                        fundamentals = fetch_fundamentals(selected_ticker)
                        price_df = fetch_stock_price(selected_ticker)
                        current_price = float(price_df.iloc[-1]['Close'])
                        
                        st.subheader(f"📊 Fundamentals Dashboard: {selected_company}")
                        if not fundamentals:
                            st.info("Fundamentals metrics are currently unavailable.")
                        else:
                            f_cols = st.columns(3)
                            with f_cols[0]:
                                pe = fundamentals.get("PE_Ratio")
                                pe_str = f"{pe:.2f}" if pe is not None else "N/A"
                                st.metric("P/E Ratio", pe_str, help="Price to Earnings Ratio")
                                st.metric("Return on Equity (ROE)", f"{fundamentals.get('ROE', 0.0):.2f}%")
                            with f_cols[1]:
                                pb = fundamentals.get("PB_Ratio")
                                pb_str = f"{pb:.2f}" if pb is not None else "N/A"
                                st.metric("P/B Ratio", pb_str, help="Price to Book Ratio")
                                st.metric("Debt to Equity Ratio", f"{fundamentals.get('Debt_to_Equity', 0.0):.2f}%")
                            with f_cols[2]:
                                eps = fundamentals.get("EPS")
                                eps_str = f"₹{eps:.2f}" if eps is not None else "N/A"
                                st.metric("EPS (Earnings Per Share)", eps_str)
                                mcap = fundamentals.get("Market_Cap")
                                mcap_str = f"₹{mcap:,.0f} Cr" if mcap is not None else "N/A"
                                st.metric("Market Cap", mcap_str)
                    except Exception as e:
                        st.error(f"Failed to fetch fundamentals: {e}")
            
            # FAST PATH 3: Price/Technicals only
            elif has_price_intent and not (has_news_intent or has_fundamentals_intent):
                with st.spinner(f"Retrieving technical chart data for {selected_company}..."):
                    try:
                        price_df = fetch_stock_price(selected_ticker)
                        price_indicators_df = calculate_technical_indicators(price_df)
                        latest_row = price_indicators_df.iloc[-1]
                        current_price = float(latest_row['Close'])
                        
                        st.subheader(f"📈 Technical Indicators: {selected_company}")
                        
                        # Display chart
                        st.line_chart(price_df['Close'].tail(90), height=250)
                        
                        t_cols = st.columns(4)
                        t_cols[0].metric("Current Close", f"₹{current_price:,.2f}")
                        t_cols[1].metric("RSI (14d)", f"{latest_row.get('RSI', 50.0):.2f}")
                        t_cols[2].metric("50-Day Moving Average", f"₹{latest_row.get('MA50', current_price):,.2f}")
                        t_cols[3].metric("200-Day Moving Average", f"₹{latest_row.get('MA200', current_price):,.2f}")
                    except Exception as e:
                        st.error(f"Failed to fetch technical indicators: {e}")
            
            # FULL STANDARD REPORT
            else:
                with st.spinner(f"🔄 Running Multi-Signal AI Analysis for {selected_company} (fetching stock prices, technical indicators, media news sentiment, institutional flows, and global macro return factors)..."):
                    # ── Step 1: Fetch Price & Technicals (Critical) ──
                    try:
                        price_df = fetch_stock_price(selected_ticker)
                        price_indicators_df = calculate_technical_indicators(price_df)
                        latest_price_row = price_indicators_df.iloc[-1]
                        current_price = float(latest_price_row['Close'])
                        
                        # Calculate 5-day return for the strategy note checks
                        price_5d_pct = float((price_indicators_df['Close'].iloc[-1] / price_indicators_df['Close'].iloc[-6] - 1) * 100) if len(price_indicators_df) >= 6 else 0.0
                    except Exception as e:
                        st.error(f"Error fetching stock data: {e}")
                        st.stop()
                    
                    # ── Step 2: Fetch News Sentiment (Non-critical) ──
                    try:
                        headlines = fetch_news_headlines(selected_ticker, selected_company)
                        if headlines:
                            sent_score, pos_count, neg_count = analyze_news_sentiment(headlines)
                            sentiment_summary = (sent_score, pos_count, neg_count, 1)
                        else:
                            sentiment_summary = (0.0, 0, 0, 0)
                            sent_score, pos_count, neg_count = 0.0, 0, 0
                    except Exception as e:
                        st.warning(f"Could not analyze news sentiment: {e}")
                        sentiment_summary = (0.0, 0, 0, 0)
                        sent_score, pos_count, neg_count = 0.0, 0, 0
                        headlines = []
                    
                    # ── Step 3: Fetch Institutional Flow (Non-critical) ──
                    try:
                        fii_dii_df = fetch_latest_fii_dii()
                        fii_net_list = fii_dii_df['FII_Net'].tolist()
                        dii_net_list = fii_dii_df['DII_Net'].tolist()
                        fii_dii_summary = analyze_institutional_signals(fii_net_list, dii_net_list)
                    except Exception as e:
                        st.warning(f"FII/DII flow analysis unavailable: {e}")
                        fii_dii_summary = {
                            "FII_10d_Net": 0.0, "DII_10d_Net": 0.0,
                            "FII_Trend": 0, "DII_Trend": 0, "Divergence_Flag": 0
                        }
                    
                    # ── Step 4: Fetch Fundamentals (Non-critical) ──
                    try:
                        fundamentals = fetch_fundamentals(selected_ticker)
                    except Exception as e:
                        st.warning(f"Could not load fundamentals: {e}")
                        fundamentals = {}
                    
                    # ── Step 5: Fetch Macro Returns (Non-critical) ──
                    try:
                        macro_returns = get_latest_macro_returns()
                    except Exception as e:
                        st.warning(f"Global macro returns unavailable: {e}")
                        macro_returns = (0.0, 0.0, 0.0)
                    
                    # ── Step 6: ML Prediction ──
                    try:
                        feature_row = prepare_inference_row(price_indicators_df, fii_dii_summary, sentiment_summary, macro_returns)
                        ml_result = predict_stock_action(feature_row)
                    except Exception as e:
                        st.error(f"XGBoost model inference failed: {e}")
                        st.stop()
                    
                    # ── Step 7: Explain via Gemini ──
                    # Build unified technical summary dictionary containing Price_Pct_5d
                    tech_summary_dict = {
                        **latest_price_row.to_dict(),
                        "Price_Pct_5d": price_5d_pct
                    }
                    try:
                        model_analysis, beginner_explanation = generate_beginner_explanation(
                            selected_ticker,
                            selected_company,
                            current_price,
                            ml_result,
                            tech_summary_dict,
                            sent_score,
                            headlines,
                            fii_dii_summary,
                            fundamentals,
                            user_input=chat_input
                        )
                    except Exception as gem_err:
                        model_analysis = "Alternative local technical analysis loaded."
                        beginner_explanation = (
                            f"AI Explanation unavailable. Details: {gem_err}\n\n"
                            f"**ML Recommendation**: {ml_result.get('Recommendation')} "
                            f"({ml_result.get('Confidence', 0.0)*100:.1f}% confidence)"
                        )
                
                # ─────────────────────────────────────────────────────
                # DISPLAY COMPILATION REPORT
                # ─────────────────────────────────────────────────────
                rec = ml_result["Recommendation"]
                conf = ml_result["Confidence"] * 100
                low_conf = ml_result.get("Low_Confidence", False)
                buy_pct = ml_result.get("Breakdown", {}).get("BUY", 0.0) * 100
                hold_pct = ml_result.get("Breakdown", {}).get("HOLD", 0.0) * 100
                sell_pct = ml_result.get("Breakdown", {}).get("SELL", 0.0) * 100
                
                # Setup layout columns
                col_left, col_right = st.columns([1.2, 1.8])
                
                with col_left:
                    # Glassmorphism Card for recommendation
                    card_class, badge_class, icon_text = get_verdict_style(rec, low_conf)
                    
                    st.markdown(f"""
                    <div class="glass-card {card_class}">
                        <div class="metric-label">AI Signal Verdict</div>
                        <div style="font-size: 28px; font-weight: bold; margin-top: 5px;">
                            <span class="badge {badge_class}">{icon_text}</span>
                        </div>
                        <div style="font-size: 14px; margin-top: 10px; color: #94A3B8;">
                            Top signal confidence: <b>{conf:.1f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability meter breakdown
                    st.markdown("### 📊 ML Signal Probabilities")
                    st.write("BUY Probability")
                    st.progress(buy_pct / 100.0)
                    st.caption(f"{buy_pct:.1f}% chance of price increase (+2% target in 5 days)")
                    
                    st.write("HOLD Probability")
                    st.progress(hold_pct / 100.0)
                    st.caption(f"{hold_pct:.1f}% chance of sideways consolidations")
                    
                    st.write("SELL Probability")
                    st.progress(sell_pct / 100.0)
                    st.caption(f"{sell_pct:.1f}% chance of price drop")
                    
                    # Signals summary
                    st.markdown("### 🔍 Core Signals Summary")
                    # Technical outlook
                    p50 = latest_price_row.get("Price_Above_MA50", 0)
                    p200 = latest_price_row.get("Price_Above_MA200", 0)
                    if p50 and p200:
                        st.success("📈 **Technical Chart**: Price moving UP strongly")
                    elif p50 or p200:
                        st.warning("⚠️ **Technical Chart**: Price moving UP weakly")
                    else:
                        st.error("📉 **Technical Chart**: Price moving DOWN")
                        
                    # News Outlook
                    if sent_score > 0.15:
                        st.success(f"📰 **News Sentiment**: Positive (Score: {sent_score:+.2f})")
                    elif sent_score < -0.15:
                        st.error(f"📰 **News Sentiment**: Negative (Score: {sent_score:+.2f})")
                    else:
                        st.warning(f"⚖️ **News Sentiment**: Neutral / Flat (Score: {sent_score:+.2f})")
                        
                    # FII/DII Outlook
                    fii_net = fii_dii_summary.get("FII_10d_Net", 0)
                    dii_net = fii_dii_summary.get("DII_10d_Net", 0)
                    if fii_net > 0 and dii_net > 0:
                        st.success("💼 **Smart Money**: Institutions buying heavily")
                    elif fii_net < 0 and dii_net < 0:
                        st.error("💼 **Smart Money**: Institutions pulling money out")
                    else:
                        st.warning("💼 **Smart Money**: Diverging institutional flow")
                        
                with col_right:
                    st.markdown(f"### 📈 90-Day Closing Price: {selected_company}")
                    st.line_chart(price_df['Close'].tail(90), height=200)
                    
                    # Metrics Grid
                    st.markdown("### 📐 Metric scorecard")
                    grid_cols = st.columns(3)
                    with grid_cols[0]:
                        st.metric("RSI (14d)", f"{latest_price_row.get('RSI', 50.0):.2f}")
                        st.metric("P/E Ratio", f"{fundamentals.get('PE_Ratio', 0.0):.2f}" if fundamentals.get('PE_Ratio') else "N/A")
                    with grid_cols[1]:
                        st.metric("50-Day MA", f"₹{latest_price_row.get('MA50', current_price):,.2f}")
                        st.metric("P/B Ratio", f"{fundamentals.get('PB_Ratio', 0.0):.2f}" if fundamentals.get('PB_Ratio') else "N/A")
                    with grid_cols[2]:
                        st.metric("200-Day MA", f"₹{latest_price_row.get('MA200', current_price):,.2f}")
                        st.metric("Return on Equity", f"{fundamentals.get('ROE', 0.0):.2f}%" if fundamentals.get('ROE') else "N/A")
                
                # AI Explanation & Strategy Guide
                st.markdown("---")
                
                # Check for overextended stock note triggers
                is_rising_hold = (
                    latest_price_row.get("Price_Above_MA50", 0) == 1 and
                    latest_price_row.get("RSI", 50.0) >= 60.0 and
                    price_5d_pct > 0.0 and
                    buy_pct < 30.0
                )
                if is_rising_hold:
                    note = (
                        "\n\n💡 **Strategy Note:** This stock has already moved up recently and looks a bit extended. "
                        "Our model favors buying during pullbacks (mean-reversion / buy-the-dip) rather than chasing stocks that have already risen, "
                        "since there is statistically less room left for a further 2%+ gain in the next 5 days."
                    )
                    if note not in beginner_explanation:
                        beginner_explanation += note
                
                st.write("### 🧠 AI Explanation & Strategy Guide (For Beginners)")
                st.markdown(beginner_explanation)
                
                st.write("### 🤖 Model Analysis & Strategy")
                st.markdown(model_analysis)
                
                # Display individual news articles
                if headlines:
                    st.write("### 📰 Recent News Articles & Headlines")
                    for i, h in enumerate(headlines[:5], 1):
                        st.markdown(f"* {h}")

# ─────────────────────────────────────────────────────────────
# MODE 2: SIDE-BY-SIDE COMPARISON
# ─────────────────────────────────────────────────────────────
elif app_mode == "📊 Side-by-Side Comparison":
    st.write("### 📊 Compare Multiple Stocks Side-by-Side")
    compare_input = st.text_input(
        "Enter stock names separated by commas (e.g. Wipro, Infosys, TCS):",
        placeholder="Wipro, Infosys, TCS"
    )
    
    if compare_input:
        with st.spinner("Extracting and analyzing stocks..."):
            tickers = extract_multiple_tickers(compare_input)
            
            if not tickers:
                st.error("Could not find any matching stock symbols.")
            elif len(tickers) < 2:
                st.warning("Please enter at least two stocks to perform a comparison.")
            else:
                stocks_data = []
                try:
                    macro_returns = get_latest_macro_returns()
                    for ticker, company_name in tickers:
                        # Fetch price, indicators, news, fii/dii, and fundamentals
                        price_df = fetch_stock_price(ticker)
                        price_indicators_df = calculate_technical_indicators(price_df)
                        latest_row = price_indicators_df.iloc[-1]
                        current_price = float(latest_row['Close'])
                        
                        headlines = fetch_news_headlines(ticker, company_name)
                        sent_score, pos, neg = analyze_news_sentiment(headlines)
                        sent_avail = 1 if headlines else 0
                        sentiment_summary = (sent_score, pos, neg, sent_avail)
                        
                        fii_dii_df = fetch_latest_fii_dii()
                        fii_net_list = fii_dii_df['FII_Net'].tolist()
                        dii_net_list = fii_dii_df['DII_Net'].tolist()
                        fii_dii_summary = analyze_institutional_signals(fii_net_list, dii_net_list)
                        
                        fundamentals = fetch_fundamentals(ticker)
                        
                        feature_row = prepare_inference_row(price_indicators_df, fii_dii_summary, sentiment_summary, macro_returns)
                        ml_result = predict_stock_action(feature_row)
                        
                        stocks_data.append({
                            "ticker": ticker,
                            "company_name": company_name,
                            "current_price": current_price,
                            "recommendation": ml_result["Recommendation"],
                            "confidence": ml_result["Confidence"] * 100,
                            "fundamentals": fundamentals,
                            "sentiment_score": sent_score,
                            "tech_above_50": bool(latest_row.get('Price_Above_MA50', 0)),
                            "tech_above_200": bool(latest_row.get('Price_Above_MA200', 0)),
                            "fii_net": fii_dii_summary.get('FII_10d_Net', 0.0),
                            "dii_net": fii_dii_summary.get('DII_10d_Net', 0.0)
                        })
                    
                    # Generate side-by-side AI explanation
                    ai_summary = generate_comparison_explanation(stocks_data)
                    
                    # Render comparison table
                    comp_df = pd.DataFrame(columns=["Metric"] + [f"{s['company_name']} ({s['ticker']})" for s in stocks_data])
                    
                    rows = [
                        ("Signal Verdict", [f"{s['recommendation']} ({s['confidence']:.1f}%)" for s in stocks_data]),
                        ("Current Price", [f"₹{s['current_price']:,.2f}" for s in stocks_data]),
                        ("P/E Ratio", [f"{s['fundamentals'].get('PE_Ratio', 0.0):.2f}" if s['fundamentals'].get('PE_Ratio') else "N/A" for s in stocks_data]),
                        ("P/B Ratio", [f"{s['fundamentals'].get('PB_Ratio', 0.0):.2f}" if s['fundamentals'].get('PB_Ratio') else "N/A" for s in stocks_data]),
                        ("Debt to Equity", [f"{s['fundamentals'].get('Debt_to_Equity', 0.0):.1f}%" if s['fundamentals'].get('Debt_to_Equity') else "N/A" for s in stocks_data]),
                        ("Return on Equity", [f"{s['fundamentals'].get('ROE', 0.0):.1f}%" if s['fundamentals'].get('ROE') else "N/A" for s in stocks_data]),
                        ("Sentiment Score", [f"{s['sentiment_score']:.2f}" for s in stocks_data]),
                        ("Above 50d MA?", ["Yes" if s['tech_above_50'] else "No" for s in stocks_data]),
                        ("Above 200d MA?", ["Yes" if s['tech_above_200'] else "No" for s in stocks_data]),
                    ]
                    
                    for label, vals in rows:
                        comp_df.loc[len(comp_df)] = [label] + vals
                        
                    st.write("### ⚔️ Side-by-Side Stock Comparison")
                    st.dataframe(comp_df, use_container_width=True)
                    
                    st.write("### 🧠 AI Side-by-Side Comparison Summary")
                    st.markdown(ai_summary)
                    
                except Exception as err:
                    st.error(f"Comparison Error: {err}")

# ─────────────────────────────────────────────────────────────
# MODE 3: STRATEGY BACKTESTING
# ─────────────────────────────────────────────────────────────
elif app_mode == "📈 Strategy Backtesting":
    st.write("### 📈 Run 1-Year Trade Simulation Backtest")
    backtest_query = st.text_input(
        "Enter stock name or symbol to backtest:",
        placeholder="e.g. TCS"
    )
    
    if backtest_query:
        with st.spinner(f"Running historical simulation backtest for '{backtest_query}'..."):
            try:
                ticker, name = extract_ticker(backtest_query)
                if not ticker:
                    st.error(f"Could not resolve stock ticker for '{backtest_query}'.")
                else:
                    report = run_backtest(ticker)
                    
                    st.subheader(f"📈 1-Year Backtest Report for {name} ({ticker})")
                    
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        st.markdown(f"""
                        <div class="glass-card">
                            <div class="metric-label">Initial Investment</div>
                            <div class="metric-value">₹{report['Initial_Capital']:,.2f}</div>
                        </div>
                        <div class="glass-card">
                            <div class="metric-label">AI Strategy Return</div>
                            <div class="metric-value" style="color: {'#10B981' if report['Strategy_Return_Pct'] >= 0 else '#EF4444'};">
                                {report['Strategy_Return_Pct']:.2f}%
                            </div>
                        </div>
                        <div class="glass-card">
                            <div class="metric-label">Total Simulated Trades</div>
                            <div class="metric-value">{report['Total_Trades']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_b2:
                        st.markdown(f"""
                        <div class="glass-card">
                            <div class="metric-label">Final Portfolio Value</div>
                            <div class="metric-value">₹{report['Final_Value']:,.2f}</div>
                        </div>
                        <div class="glass-card">
                            <div class="metric-label">Benchmark Return (Buy & Hold)</div>
                            <div class="metric-value" style="color: {'#10B981' if report['Benchmark_Return_Pct'] >= 0 else '#EF4444'};">
                                {report['Benchmark_Return_Pct']:.2f}%
                            </div>
                        </div>
                        <div class="glass-card">
                            <div class="metric-label">Winning Trades %</div>
                            <div class="metric-value">{report['Win_Rate_Pct']:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    # Visual representation of performance difference
                    strat_ret = report['Strategy_Return_Pct']
                    bench_ret = report['Benchmark_Return_Pct']
                    if strat_ret > bench_ret:
                        st.success(f"🏆 **AI Strategy outperformed the stock's Buy & Hold return by {(strat_ret - bench_ret):.2f}%**! The multi-signal model successfully avoided downswings and entered on key momentum.")
                    else:
                        st.info(f"ℹ️ **Benchmark Buy & Hold outperformed the AI strategy by {(bench_ret - strat_ret):.2f}%**. This stock had a persistent, strong uptrend where trading signals were less effective than simply holding.")
            except Exception as e:
                st.error(f"Backtesting simulation failed: {e}")

# ─────────────────────────────────────────────────────────────
# MODE 4: LEARN BASICS
# ─────────────────────────────────────────────────────────────
elif app_mode == "📚 Learn Basics":
    st.write("### 📚 AI Financial Glossary & Learning Center")
    learn_query = st.text_input(
        "Ask any question about financial markets, terms, or how indicators work:",
        placeholder="e.g. What is RSI? or How does debt-to-equity ratio affect stock safety?"
    )
    
    if learn_query:
        with st.spinner("Consulting Gemini AI educational model..."):
            try:
                explanation = explain_educational_concept(learn_query)
                st.markdown("### 🧠 AI Financial Guide (For Beginners)")
                st.markdown(explanation)
            except Exception as e:
                st.error(f"Could not fetch explanation: {e}")
