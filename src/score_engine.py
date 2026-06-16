"""
score_engine.py — Signal Score Breakdown Engine
================================================
Computes a per-domain scorecard from the same data the ML model uses.
This is NOT the ML prediction — it's a human-readable explanation of
what each signal domain is saying, scored on a simple -2 to +2 scale.

Designed for --verbose mode / project reviewers / interviewers.
"""


def compute_signal_scores(technical: dict, sentiment_score: float,
                           fii_dii_summary: dict, fundamentals: dict) -> dict:
    """
    Computes scores for each signal domain.

    Returns a dict with:
        scores       : {domain: int}     → raw -2..+2 scores per domain
        details      : {domain: [str]}   → list of reasons behind each score
        net_score    : int               → sum of all domain scores
        net_label    : str               → human-readable net verdict
    """
    scores  = {}
    details = {}

    # ─────────────────────────────────────────────
    # 1. TECHNICAL SCORE  (-4 to +4 possible, then clamped to -2..+2)
    # ─────────────────────────────────────────────
    tech_score = 0
    tech_reasons = []

    # Moving averages (biggest weight)
    above_50  = int(technical.get("Price_Above_MA50",  0))
    above_200 = int(technical.get("Price_Above_MA200", 0))
    if above_50 and above_200:
        tech_score += 2
        tech_reasons.append("✅ Price above both MA50 & MA200 (strong uptrend)")
    elif above_50 or above_200:
        tech_score += 1
        tech_reasons.append("⚠️  Price above one moving average (weak uptrend)")
    else:
        tech_score -= 2
        tech_reasons.append("❌ Price below both MA50 & MA200 (downtrend)")

    # RSI
    rsi = float(technical.get("RSI", 50))
    if rsi >= 60:
        tech_score += 1
        tech_reasons.append(f"✅ RSI {rsi:.1f} — bullish momentum")
    elif rsi <= 35:
        tech_score -= 1
        tech_reasons.append(f"⚠️  RSI {rsi:.1f} — oversold (potential bounce or continued decline)")
    elif 40 <= rsi <= 55:
        tech_reasons.append(f"⚖️  RSI {rsi:.1f} — neutral zone")
    else:
        tech_reasons.append(f"⚖️  RSI {rsi:.1f} — slightly weak")

    # MACD histogram direction
    macd_hist = float(technical.get("MACD_Hist", 0))
    if macd_hist > 0:
        tech_score += 1
        tech_reasons.append(f"✅ MACD histogram positive (+{macd_hist:.3f}) — bullish momentum")
    elif macd_hist < 0:
        tech_score -= 1
        tech_reasons.append(f"❌ MACD histogram negative ({macd_hist:.3f}) — bearish momentum")

    # Volume confirmation
    vol_ratio = float(technical.get("Volume_Ratio", 1.0))
    if vol_ratio >= 1.3:
        tech_reasons.append(f"✅ Volume {vol_ratio:.2f}x above average — strong conviction")
    elif vol_ratio <= 0.7:
        tech_reasons.append(f"⚠️  Volume {vol_ratio:.2f}x below average — low conviction")
    else:
        tech_reasons.append(f"⚖️  Volume ratio {vol_ratio:.2f}x — normal trading activity")

    # Clamp final technical score to -2..+2
    tech_score = max(-2, min(2, tech_score))
    scores["Technical"]  = tech_score
    details["Technical"] = tech_reasons

    # ─────────────────────────────────────────────
    # 2. NEWS SENTIMENT SCORE  (-2 to +2)
    # ─────────────────────────────────────────────
    sent_score   = 0
    sent_reasons = []

    if sentiment_score >= 0.35:
        sent_score = 2
        sent_reasons.append(f"✅ Sentiment score {sentiment_score:+.2f} — strongly positive news")
    elif sentiment_score >= 0.10:
        sent_score = 1
        sent_reasons.append(f"✅ Sentiment score {sentiment_score:+.2f} — mildly positive news")
    elif sentiment_score <= -0.35:
        sent_score = -2
        sent_reasons.append(f"❌ Sentiment score {sentiment_score:+.2f} — strongly negative news")
    elif sentiment_score <= -0.10:
        sent_score = -1
        sent_reasons.append(f"⚠️  Sentiment score {sentiment_score:+.2f} — mildly negative news")
    else:
        sent_score = 0
        sent_reasons.append(f"⚖️  Sentiment score {sentiment_score:+.2f} — neutral / no strong news")

    scores["News Sentiment"]  = sent_score
    details["News Sentiment"] = sent_reasons

    # ─────────────────────────────────────────────
    # 3. FII/DII INSTITUTIONAL SCORE  (-2 to +2)
    # ─────────────────────────────────────────────
    inst_score   = 0
    inst_reasons = []

    fii_net = float(fii_dii_summary.get("FII_10d_Net", 0))
    dii_net = float(fii_dii_summary.get("DII_10d_Net", 0))
    net_flow = fii_net + dii_net  # combined net institutional flow

    if fii_net > 0 and dii_net > 0:
        inst_score = 2
        inst_reasons.append(f"✅ Both FII (+₹{fii_net:,.0f} Cr) & DII (+₹{dii_net:,.0f} Cr) buying")
    elif fii_net < 0 and dii_net < 0:
        inst_score = -2
        inst_reasons.append(f"❌ Both FII ({fii_net:,.0f} Cr) & DII ({dii_net:,.0f} Cr) selling")
    elif net_flow > 0:
        inst_score = 1
        inst_reasons.append(f"⚖️  DII buying (+₹{dii_net:,.0f} Cr) outweighs FII outflow ({fii_net:,.0f} Cr)")
        inst_reasons.append(f"   Net institutional flow: +₹{net_flow:,.0f} Cr (net positive)")
    elif net_flow < 0:
        inst_score = -1
        inst_reasons.append(f"⚠️  FII outflow ({fii_net:,.0f} Cr) outweighs DII buying (+₹{dii_net:,.0f} Cr)")
        inst_reasons.append(f"   Net institutional flow: {net_flow:,.0f} Cr (net negative)")
    else:
        inst_score = 0
        inst_reasons.append("⚖️  Institutional flows roughly balanced")

    scores["FII/DII Institutions"]  = inst_score
    details["FII/DII Institutions"] = inst_reasons

    # ─────────────────────────────────────────────
    # 4. FUNDAMENTALS SCORE  (-2 to +2)
    # ─────────────────────────────────────────────
    fund_score   = 0
    fund_reasons = []

    if not fundamentals:
        fund_score = 0
        fund_reasons.append("⚠️  No fundamental data available")
    else:
        # P/E Ratio
        pe = fundamentals.get("PE_Ratio")
        if pe is not None:
            if pe < 20:
                fund_score += 1
                fund_reasons.append(f"✅ P/E {pe:.1f} — undervalued relative to market")
            elif pe <= 30:
                fund_reasons.append(f"⚖️  P/E {pe:.1f} — fairly valued")
            else:
                fund_score -= 1
                fund_reasons.append(f"⚠️  P/E {pe:.1f} — premium valuation")

        # Debt/Equity
        de = fundamentals.get("Debt_to_Equity")
        if de is not None:
            if de < 30:
                fund_score += 1
                fund_reasons.append(f"✅ D/E {de:.1f}% — very low debt, financially strong")
            elif de <= 80:
                fund_reasons.append(f"⚖️  D/E {de:.1f}% — manageable debt levels")
            else:
                fund_score -= 1
                fund_reasons.append(f"⚠️  D/E {de:.1f}% — high leverage, watch out")

        # ROE
        roe = fundamentals.get("ROE")
        if roe is not None:
            if roe >= 15:
                fund_score += 1
                fund_reasons.append(f"✅ ROE {roe:.1f}% — highly efficient use of capital")
            elif roe >= 8:
                fund_reasons.append(f"⚖️  ROE {roe:.1f}% — moderate return on equity")
            else:
                fund_score -= 1
                fund_reasons.append(f"⚠️  ROE {roe:.1f}% — poor capital efficiency")

    # Clamp
    fund_score = max(-2, min(2, fund_score))
    scores["Fundamentals"]  = fund_score
    details["Fundamentals"] = fund_reasons

    # ─────────────────────────────────────────────
    # 5. NET SCORE + LABEL
    # ─────────────────────────────────────────────
    net = sum(scores.values())

    if net >= 5:
        label = "Strong BUY"
    elif net >= 2:
        label = "Weak BUY"
    elif net >= -1:
        label = "HOLD / Neutral"
    elif net >= -3:
        label = "Weak SELL"
    else:
        label = "Strong SELL"

    return {
        "scores":    scores,
        "details":   details,
        "net_score": net,
        "net_label": label,
    }


def score_bar(score: int) -> str:
    """Returns a compact visual bar for a -2..+2 score."""
    bars = {
        -2: "[bold red]██░░░ -2[/bold red]",
        -1: "[red]███░░ -1[/red]",
         0: "[yellow]░███░  0[/yellow]",
         1: "[green]░░███ +1[/green]",
         2: "[bold green]░░░██ +2[/bold green]",
    }
    return bars.get(score, f"{score:+d}")
