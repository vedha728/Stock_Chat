import os
import sys
import argparse
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console, Group as RichGroup
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich.status import Status

# Add src/ to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_collector import extract_ticker, extract_multiple_tickers, fetch_stock_price, fetch_news_headlines, fetch_fundamentals, get_latest_macro_returns

from indicators import calculate_technical_indicators
from sentiment import analyze_news_sentiment, deduplicate_headlines
from institutional import fetch_latest_fii_dii, analyze_institutional_signals
from feature_engineering import prepare_inference_row
from predict import predict_stock_action
from backtest import run_backtest
from gemini_explain import generate_beginner_explanation, explain_educational_concept, generate_comparison_explanation
from score_engine import compute_signal_scores, score_bar

# Initialize Rich Console
console = Console()
load_dotenv()

def print_header():
    """Prints a styled startup banner."""
    console.print()
    banner = Text(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        " 📊  MULTI-SIGNAL AI STOCK ADVISORY CHATBOT (INDIAN MARKET BEGINNERS)  📊 \n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        style="bold cyan",
        justify="center"
    )
    console.print(banner)
    console.print(
        "[dim]  Designed for Intel i3 CPU | CPU-friendly ML Classifier | Explainable AI (Gemini 1.5)  [/dim]\n",
        justify="center"
    )
    console.print(
        "  [bold green]Commands:[/bold green]\n"
        "  • Ask a question: [bold white]'Should I buy Tata Motors today?'[/bold white] or [bold white]'Reliance'[/bold white]\n"
        "  • Run a Backtest: [bold white]'/backtest TCS'[/bold white] or type [bold white]'backtest TCS'[/bold white]\n"
        "  • Exit: [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red]\n"
    )
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


def display_backtest_report(ticker: str, report: dict):
    """Renders a beautiful rich table summarizing the backtest results."""
    # Recommendation color formatting
    ret_pct = report['Strategy_Return_Pct']
    bench_pct = report['Benchmark_Return_Pct']
    
    ret_style = "bold green" if ret_pct >= 0 else "bold red"
    bench_style = "bold green" if bench_pct >= 0 else "bold red"
    
    table = Table(title=f"📈 1-Year Backtest Report for {ticker}", show_header=True, header_style="bold cyan")
    table.add_column("Metric Description", style="dim")
    table.add_column("Value", justify="right")
    
    table.add_row("Initial Investment Capital", f"₹{report['Initial_Capital']:,.2f}")
    table.add_row("Final Portfolio Value", f"₹{report['Final_Value']:,.2f}")
    table.add_row("AI Strategy Cumulative Return %", f"[{ret_style}]{ret_pct:.2f}%[/{ret_style}]")
    table.add_row(f"Benchmark Buy & Hold Return %", f"[{bench_style}]{bench_pct:.2f}%[/{bench_style}]")
    table.add_row("Total Simulated Trades Executed", str(report['Total_Trades']))
    table.add_row("Winning Trades % (Win Rate)", f"{report['Win_Rate_Pct']:.2f}%")
    table.add_row("Average Profit per Win", f"+{report['Avg_Win_Pct']:.2f}%")
    table.add_row("Average Loss per Draw", f"{report['Avg_Loss_Pct']:.2f}%")
    table.add_row("Maximum Drawdown (Worst Loss Streak)", f"[bold red]{report['Max_Drawdown_Pct']:.2f}%[/bold red]")
    
    console.print(Panel(table, expand=False, border_style="cyan"))
    
    # Simple takeaway comment
    if ret_pct > bench_pct:
        console.print("🏆 [bold green]Takeaway:[/bold green] The AI strategy [bold green]beat[/bold green] the benchmark Buy & Hold strategy. This shows multi-signal filtering helped avoid market downturns!")
    else:
        console.print("ℹ️ [bold yellow]Takeaway:[/bold yellow] The benchmark stock performed very strongly, or the market was in a persistent uptrend where Buy & Hold is hard to beat.")
    console.print()


def get_direct_answer(user_input: str, rec: str, company: str) -> tuple[str, str, str, str, str]:
    """
    Compares what the user asked (buy/sell/hold intent) with the ML recommendation
    and returns a tuple of (direct answer, style color, panel border color, rec_icon, rec_color).
    """
    q = user_input.lower()
    asked_buy  = any(w in q for w in ["buy", "purchase", "invest", "get", "enter"])
    asked_sell = any(w in q for w in ["sell", "exit", "quit", "offload"])
    asked_hold = any(w in q for w in ["hold", "keep", "wait", "stay"])

    if rec == "BUY":
        if asked_buy:
            return (f"✅ Yes! Our analysis says it's a good time to consider buying {company}.", "bold green", "green", "📈 BUY", "bold green")
        elif asked_sell:
            return (f"❌ No, do not sell {company} right now. Selling is not recommended because key signals (like big investors and long-term trends) show strong upward support. Instead, conditions are highly favorable to BUY or HOLD.", "bold red", "red", "📈 BUY", "bold green")
        elif asked_hold:
            return (f"📈 Our analysis leans toward BUY — holding is safe, but conditions are favorable to add more shares of {company}.", "bold green", "green", "📈 BUY", "bold green")
        else:
            return (f"✅ Our analysis recommends BUY for {company}.", "bold green", "green", "📈 BUY", "bold green")

    elif rec == "SELL":
        if asked_buy:
            return (f"❌ No, do not buy {company} right now. Buying is not recommended because technical trends and big institutional flows are pointing downwards. Instead, you should SELL or AVOID.", "bold red", "red", "📉 SELL", "bold red")
        elif asked_sell:
            return (f"✅ Yes, our analysis supports selling {company} at this time.", "bold green", "green", "📉 SELL", "bold green")
        elif asked_hold:
            return (f"⚠️  No, holding is not recommended. Our analysis suggests you SELL {company} because downward pressure is building. Keeping it carries risk.", "bold red", "red", "📉 SELL", "bold red")
        else:
            return (f"❌ Our analysis recommends SELL for {company}.", "bold red", "red", "📉 SELL", "bold red")

    else:  # HOLD
        if asked_buy:
            return (f"⚖️  No, buying is not recommended right now. Market signals for {company} are mixed and no strong trend exists, so it is safer to wait.", "bold yellow", "yellow", "⚖️ HOLD", "bold yellow")
        elif asked_sell:
            return (f"⚠️  No, selling is not necessary yet. The trend is stable and neutral, so you can safely HOLD your shares of {company} for now.", "bold yellow", "yellow", "⚖️ HOLD", "bold yellow")
        elif asked_hold:
            return (f"✅ Yes, our analysis agrees — HOLD {company} and wait for a better entry point.", "bold green", "green", "⚖️ HOLD", "bold green")
        else:
            return (f"⚖️  Our analysis recommends HOLD for {company} — no strong buy or sell signal yet.", "bold blue", "blue", "⚖️ HOLD", "bold blue")


def display_score_breakdown(technical: dict, sentiment_score: float,
                            fii_dii_summary: dict, fundamentals: dict,
                            ml_result: dict):
    """
    Renders the verbose Signal Score Breakdown panel.
    Shows per-domain scores on a -2..+2 scale and the net verdict.
    Only shown when chatbot is run with --verbose flag.
    """
    breakdown = compute_signal_scores(technical, sentiment_score, fii_dii_summary, fundamentals)
    scores    = breakdown["scores"]
    details   = breakdown["details"]
    net       = breakdown["net_score"]
    net_label = breakdown["net_label"]

    # ML model info
    ml_conf  = ml_result.get("Original_Confidence", ml_result.get("Confidence", 0)) * 100
    ml_rec   = ml_result.get("Original_Recommendation", ml_result.get("Recommendation", "?"))
    override = ml_result.get("Low_Confidence", False)

    # Score table
    score_table = Table(
        title="📐 Signal Score Breakdown  [dim](Verbose Mode)[/dim]",
        show_header=True,
        header_style="bold dim",
        box=None,
        padding=(0, 1)
    )
    score_table.add_column("Domain",        style="bold white", min_width=24)
    score_table.add_column("Score",         justify="center",  min_width=12)
    score_table.add_column("Key Reasons",   style="dim")

    domain_icons = {
        "Technical":            "📈",
        "News Sentiment":       "📰",
        "FII/DII Institutions": "💼",
        "Fundamentals":         "📊",
    }

    for domain, s in scores.items():
        icon    = domain_icons.get(domain, "")
        reasons = "  |  ".join(details.get(domain, []))
        score_table.add_row(f"{icon} {domain}", score_bar(s), reasons)

    # Separator + net row
    score_table.add_section()
    net_color = "bold green" if net > 1 else ("bold red" if net < -1 else "bold yellow")
    score_table.add_row(
        "[bold white]NET SCORE[/bold white]",
        f"[{net_color}]{net:+d}[/{net_color}]",
        f"[{net_color}]Scorecard says: {net_label}[/{net_color}]"
    )

    # ML model line
    ml_line = (
        f"[dim]ML Model raw signal → {ml_rec} ({ml_conf:.1f}% confidence)"
        + (" → overridden to HOLD (below 55% threshold)" if override else "")
        + "[/dim]"
    )

    console.print(Panel(
        score_table,
        subtitle=ml_line,
        border_style="magenta"
    ))


def display_prediction_report(ticker: str, company: str, price: float, result: dict, technical_summary: dict, sentiment_score: float, fii_dii_summary: dict, model_analysis: str, beginner_explanation: str, user_input: str = "", fundamentals: dict = None, verbose: bool = False):
    """Renders the AI prediction report and LLM explanation on the terminal."""
    rec = result["Recommendation"]
    conf = result["Confidence"] * 100
    
    # 1. Generate direct answer and layout styles based on what user asked vs what model recommends
    direct_answer, answer_style, panel_border, rec_icon, rec_color = get_direct_answer(user_input, rec, company)

    # Extract probabilities for intent breakdown
    breakdown = result.get("Breakdown", {})
    buy_pct = breakdown.get("BUY", 0.0) * 100
    hold_pct = breakdown.get("HOLD", 0.0) * 100
    sell_pct = breakdown.get("SELL", 0.0) * 100

    q = user_input.lower()
    asked_buy  = any(w in q for w in ["buy", "purchase", "invest", "get", "enter"])
    asked_sell = any(w in q for w in ["sell", "exit", "quit", "offload"])
    asked_hold = any(w in q for w in ["hold", "keep", "wait", "stay"])

    # 3. Main recommendation panel
    info_table = Table(show_header=False, box=None, padding=(0, 1))
    info_table.add_column("Label", style="bold white", width=11)
    info_table.add_column("Content")
    
    info_table.add_row(" 📊 Stock:", f"[bold white]{company} ({ticker})[/bold white]")
    info_table.add_row(" 💵 Price:", f"[bold white]₹{price:,.2f}[/bold white]")
    info_table.add_row("", "")  # Spacer
    
    # Check if near-buy watch zone
    is_near_buy = False
    if result.get("Low_Confidence"):
        orig_rec = result.get("Original_Recommendation", "?")
        if orig_rec == "BUY" and 50.0 <= buy_pct <= 54.9:
            is_near_buy = True

    verdict_text = Text()
    if result.get("Low_Confidence"):
        if is_near_buy:
            verdict_text.append("⚖️ HOLD (Near-Buy Watch Zone)", style="bold yellow")
        else:
            verdict_text.append("⚖️ HOLD (Below threshold)", style=rec_color)
    else:
        verdict_text.append(f"{rec_icon} ({conf:.1f}% Confidence)", style=rec_color)
    info_table.add_row(" Verdict:", verdict_text)
    
    if result.get("Low_Confidence") or rec == "HOLD":
        info_table.add_row("", "")  # Spacer
        
        if result.get("Low_Confidence"):
            orig_rec = result.get("Original_Recommendation", "?")
            orig_conf = result.get("Original_Confidence", 0.0) * 100
            
            if asked_sell:
                if is_near_buy:
                    reason_text = Text(
                        f"BUY signal ({buy_pct:.1f}%) is close to 55.0% threshold. No sell signal exists.\n"
                        f"Defaulting to HOLD.",
                        style="bold yellow"
                    )
                elif orig_rec == "BUY":
                    reason_text = Text(
                        f"BUY signal ({orig_conf:.1f}%) below 55.0% threshold. No strong sell signal exists.\n"
                        f"Defaulting to HOLD.",
                        style="bold yellow"
                    )
                elif orig_rec == "SELL":
                    reason_text = Text(
                        f"SELL signal ({orig_conf:.1f}%) below 55.0% threshold. Trend is not weak enough to sell.\n"
                        f"Defaulting to HOLD.",
                        style="bold yellow"
                    )
                else:
                    reason_text = Text(
                        f"Market trend is neutral ({orig_conf:.1f}%). Defaulting to HOLD.",
                        style="bold yellow"
                    )
            else: # Buying/neutral perspective
                if is_near_buy:
                    reason_text = Text(
                        f"BUY signal ({buy_pct:.1f}%) is very close to 55.0% safety threshold.\n"
                        f"Watch closely for breakout.",
                        style="bold yellow"
                    )
                else:
                    reason_text = Text(
                        f"{orig_rec} signal ({orig_conf:.1f}%) below safety threshold of 55.0%.\n"
                        f"Defaulting to HOLD.",
                        style="bold yellow"
                    )
        else: # Natural HOLD (Low_Confidence is False)
            if asked_sell:
                reason_text = Text(
                    f"SELL chance is low ({sell_pct:.1f}%). Neutral market trend.\n"
                    f"Selling is not recommended.",
                    style="bold yellow"
                )
            elif asked_buy:
                reason_text = Text(
                    f"BUY chance is low ({buy_pct:.1f}%). Neutral market trend.\n"
                    f"Buying is not recommended.",
                    style="bold yellow"
                )
            else:
                reason_text = Text(
                    f"Market trend is neutral. Recommendation is to HOLD.",
                    style="bold yellow"
                )
        info_table.add_row(" Reason:", reason_text)
        
    chances_text = Text()
    chances_text.append(f"\n SIGNAL CHANCES:\n", style="bold white")
    
    max_val = max(buy_pct, hold_pct, sell_pct)
    orig_rec = result.get("Original_Recommendation", "?")
    
    # BUY
    chances_text.append("  • 📈 BUY Chance  ", style="bold white")
    if result.get("Low_Confidence") and orig_rec == "BUY":
        chances_text.append(f": {buy_pct:.1f}% (Highest, but below threshold)\n", style="bold yellow")
    elif buy_pct == max_val:
        chances_text.append(f": {buy_pct:.1f}% (Strong Buy Signal ✅)\n", style="bold green")
    else:
        chances_text.append(f": {buy_pct:.1f}%\n", style="white")
        
    # HOLD
    chances_text.append("  • ⚖️ HOLD Chance ", style="bold white")
    if result.get("Low_Confidence") and orig_rec == "HOLD":
        chances_text.append(f": {hold_pct:.1f}% (Highest, but below threshold)\n", style="bold yellow")
    elif hold_pct == max_val:
        if result.get("Low_Confidence"):
            chances_text.append(f": {hold_pct:.1f}% (Defaulted HOLD ⚖️)\n", style="bold yellow")
        else:
            chances_text.append(f": {hold_pct:.1f}% (Neutral Trend ⚖️)\n", style="bold yellow")
    else:
        chances_text.append(f": {hold_pct:.1f}%\n", style="white")
        
    # SELL
    chances_text.append("  • 📉 SELL Chance ", style="bold white")
    if result.get("Low_Confidence") and orig_rec == "SELL":
        chances_text.append(f": {sell_pct:.1f}% (Highest, but below threshold)\n", style="bold yellow")
    elif sell_pct == max_val:
        chances_text.append(f": {sell_pct:.1f}% (Strong Sell Signal 🛑)\n", style="bold red")
    else:
        chances_text.append(f": {sell_pct:.1f}%\n", style="white")
        
    rec_panel = Panel(
        RichGroup(info_table, chances_text), 
        title=f"📊 [bold white]Analysis Report[/bold white]",
        border_style=panel_border
    )
    console.print(rec_panel)
    
    # 2. Model Analysis & Strategy
    console.print()
    console.print(Panel(model_analysis, title="🤖 Model Analysis & Strategy", border_style="cyan"))
    
    # 3. Core Signals Table
    # Determine signal directions
    # Technical
    price_above_50  = technical_summary.get('Price_Above_MA50',  0)
    price_above_200 = technical_summary.get('Price_Above_MA200', 0)
    if price_above_50 and price_above_200:
        tech_status = "[green]Price moving UP strongly ✅[/green]"
    elif price_above_50 or price_above_200:
        tech_status = "[yellow]Price moving UP weakly ⚠️[/yellow]"
    else:
        tech_status = "[red]Price moving DOWN ⚠️[/red]"

    # Sentiment
    sent_status = "[green]Good news in media ✅[/green]" if sentiment_score > 0.15 else \
                  ("[red]Negative news in media ⚠️[/red]" if sentiment_score < -0.15 else "[yellow]No strong news either way ⚖️[/yellow]")

    # Institutional (Smart Money)
    fii_net = fii_dii_summary.get('FII_10d_Net', 0)
    dii_net = fii_dii_summary.get('DII_10d_Net', 0)
    if fii_net > 0 and dii_net > 0:
        money_status = "[green]Big investors buying heavily ✅[/green]"
    elif fii_net < 0 and dii_net < 0:
        money_status = "[red]Big investors pulling money out ⚠️[/red]"
    elif fii_net < 0 < dii_net:
        money_status = "[yellow]Foreign investors out, Indian investors buying ⚖️[/yellow]"
    else:
        money_status = "[green]Foreign investors buying, Indian investors reducing ✅[/green]"

    signal_table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2), width=80)
    signal_table.add_column("Signal Domain", style="bold white", min_width=25)
    signal_table.add_column("Current Outlook", min_width=50)
    
    signal_table.add_row("📈 Technical Chart Trend", tech_status)
    signal_table.add_row("📰 Public News Sentiment", sent_status)
    signal_table.add_row("💼 Smart Money (Institutions)", money_status)

    # Technical sub-table (RSI, MA50, MA200, MACD)
    tech_detail_table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2), width=65)
    tech_detail_table.add_column("Indicator", style="bold white", min_width=20)
    tech_detail_table.add_column("Value", justify="right", min_width=10)
    tech_detail_table.add_column("Status", min_width=28)

    rsi = technical_summary.get("RSI", None)
    if rsi is not None:
        if rsi >= 70:
            rsi_status = "[red]Overbought ⚠️[/red]"
        elif rsi <= 30:
            rsi_status = "[green]Oversold — Potential Buy 🟢[/green]"
        else:
            rsi_status = "[yellow]Neutral ⚖️[/yellow]"
        tech_detail_table.add_row("RSI (14-day)", f"{rsi:.1f}", rsi_status)

    ma50 = technical_summary.get("MA50", None)
    current_close = technical_summary.get("Close", None)
    if ma50 is not None and current_close is not None:
        ma50_status = "[green]Above (Uptrend) ✅[/green]" if current_close >= ma50 else "[red]Below (Downtrend) ⚠️[/red]"
        tech_detail_table.add_row("50-Day Moving Avg", f"₹{ma50:,.2f}", ma50_status)

    ma200 = technical_summary.get("MA200", None)
    if ma200 is not None and current_close is not None:
        ma200_status = "[green]Above (Long-term Up) ✅[/green]" if current_close >= ma200 else "[red]Below (Long-term Down) ⚠️[/red]"
        tech_detail_table.add_row("200-Day Moving Avg", f"₹{ma200:,.2f}", ma200_status)

    macd_hist = technical_summary.get("MACD_Hist", None)
    if macd_hist is not None:
        macd_status = "[green]Bullish momentum 📈[/green]" if macd_hist > 0 else "[red]Bearish momentum 📉[/red]"
        tech_detail_table.add_row("MACD Histogram", f"{macd_hist:+.2f}", macd_status)

    console.print()
    console.print(Panel(signal_table, title="🔍 Core Signal Breakdown", border_style="dim"))
    console.print()
    console.print(Panel(tech_detail_table, title="📐 Technical Details", border_style="dim"))

    # 3.5 Fundamentals Table (if available)
    if fundamentals:
        fund_table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2), width=95)
        fund_table.add_column("Key Metric", style="bold white", min_width=30)
        fund_table.add_column("Value", justify="right", min_width=20)
        fund_table.add_column("Benchmark / Interpretation", min_width=40)
        
        # P/E Ratio — label type (Trailing / Forward / Computed)
        pe = fundamentals.get("PE_Ratio")
        pe_type = fundamentals.get("PE_Type", "Trailing")
        eps = fundamentals.get("EPS")
        if pe is not None:
            pe_str = f"{pe:.2f}  [{pe_type} P/E]"
        elif eps is not None and eps < 0:
            pe_str = f"N/A  (Negative EPS: ₹{eps:.2f})"
        else:
            pe_str = "N/A"
        pe_interp = "[green]Undervalued / Healthy (< 25)[/green]" if pe is not None and pe < 25 else \
                    ("[yellow]Fairly Valued (25-40)[/yellow]" if pe is not None and pe <= 40 else \
                     "[red]Premium / High Value (> 40)[/red]" if pe is not None else \
                     "[dim]Not applicable — company has negative earnings[/dim]")
        fund_table.add_row("📊 P/E Ratio (Price/Earnings)", pe_str, pe_interp)

        # EPS (Earnings Per Share)
        if eps is not None:
            eps_str = f"₹{eps:.2f}"
            eps_interp = "[green]Profitable ✅[/green]" if eps > 0 else "[red]Loss-making ⚠️[/red]"
            fund_table.add_row("📉 EPS (Earnings Per Share)", eps_str, eps_interp)
        
        # P/B Ratio
        pb = fundamentals.get("PB_Ratio")
        pb_str = f"{pb:.2f}" if pb is not None else "N/A"
        pb_interp = "[green]Value Buy (< 3.0)[/green]" if pb is not None and pb < 3.0 else \
                    ("[yellow]Moderate (3.0-6.0)[/yellow]" if pb is not None and pb <= 6.0 else "[red]Growth Premium (> 6.0)[/red]" if pb is not None else "N/A")
        fund_table.add_row("📖 P/B Ratio (Price/Book)", pb_str, pb_interp)
        
        # Debt to Equity
        de = fundamentals.get("Debt_to_Equity")
        de_str = f"{de:.2f}%" if de is not None else "N/A"
        de_interp = "[green]Low Debt / Low Risk (< 50%)[/green]" if de is not None and de < 50 else \
                    ("[yellow]Leveraged (50-100%)[/yellow]" if de is not None and de <= 100 else "[red]High Debt / High Risk (> 100%)[/red]" if de is not None else "N/A")
        fund_table.add_row("🛡️ Debt to Equity Ratio", de_str, de_interp)
        
        # ROE
        roe = fundamentals.get("ROE")
        roe_str = f"{roe:.2f}%" if roe is not None else "N/A"
        roe_interp = "[green]High Return / Efficient (> 15%)[/green]" if roe is not None and roe >= 15 else \
                     ("[yellow]Moderate Return (8-15%)[/yellow]" if roe is not None and roe >= 8 else "[red]Low Return (< 8%)[/red]" if roe is not None else "N/A")
        fund_table.add_row("💰 Return on Equity (ROE)", roe_str, roe_interp)
        
        # Market Cap
        mcap = fundamentals.get("Market_Cap")
        mcap_str = f"₹{mcap:,.0f} Cr" if mcap is not None else "N/A"
        mcap_interp = "[cyan]Mega Cap (> 1,00,000 Cr)[/cyan]" if mcap is not None and mcap > 100000 else \
                      ("[blue]Large Cap (20,000 - 1,00,000 Cr)[/blue]" if mcap is not None and mcap >= 20000 else "[dim]Mid/Small Cap (< 20,000 Cr)[/dim]" if mcap is not None else "N/A")
        fund_table.add_row("🏛️ Market Capitalization", mcap_str, mcap_interp)
        
        console.print()
        console.print(Panel(fund_table, title="📊 Fundamental Valuation (Long-Term Health)", border_style="dim"))

    # 4. Verbose Score Breakdown (only when --verbose flag is set)
    if verbose:
        console.print()
        display_score_breakdown(technical_summary, sentiment_score, fii_dii_summary, fundamentals, result)

    # 5. Gemini Explanation Output
    is_rising_hold = (
        technical_summary.get("Price_Above_MA50", 0) == 1 and
        technical_summary.get("RSI", 50.0) >= 60.0 and
        technical_summary.get("Price_Pct_5d", 0.0) > 0.0 and
        buy_pct < 30.0
    )
    if is_rising_hold:
        note_text = (
            "\n\n💡 [bold yellow]Strategy Note:[/bold yellow] This stock has already moved up recently and looks a bit extended. "
            "Our model favors buying during pullbacks (mean-reversion / buy-the-dip) rather than chasing stocks that have already risen, "
            "since there is statistically less room left for a further 2%+ gain in the next 5 days."
        )
        beginner_explanation += note_text

    console.print()
    console.print(Panel(beginner_explanation, title="🧠 AI Explanation & Strategy Guide (For Beginners)", border_style="cyan"))


def display_comparison_report(stocks: list[dict], ai_summary: str):
    """Renders a side-by-side comparison table of multiple stocks."""
    table = Table(title="⚔️ Side-by-Side Stock Comparison", show_header=True, header_style="bold cyan", box=None)
    
    table.add_column("Metric / Indicator", style="bold white")
    for s in stocks:
        table.add_column(f"{s['company_name']} ({s['ticker']})", justify="center")

    # Signal
    sig_row = ["Signal Verdict"]
    for s in stocks:
        rec = s["recommendation"]
        conf = s["confidence"]
        if rec == "BUY":
            sig_row.append(f"[green]✅ BUY ({conf:.1f}%)[/green]")
        elif rec == "SELL":
            sig_row.append(f"[red]❌ SELL ({conf:.1f}%)[/red]")
        else:
            sig_row.append(f"[blue]⚖️ HOLD ({conf:.1f}%)[/blue]")
    table.add_row(*sig_row)
    
    # Current Price
    price_row = ["Current Price"]
    for s in stocks:
        price_row.append(f"₹{s['current_price']:,.2f}")
    table.add_row(*price_row)

    # PE Ratio
    pe_row = ["P/E Ratio"]
    for s in stocks:
        pe = s["fundamentals"].get("PE_Ratio")
        if pe is None:
            pe_row.append("N/A")
        else:
            style = "green" if pe < 25 else ("red" if pe > 40 else "yellow")
            pe_row.append(f"[{style}]{pe:.2f}[/{style}]")
    table.add_row(*pe_row)

    # PB Ratio
    pb_row = ["P/B Ratio"]
    for s in stocks:
        pb = s["fundamentals"].get("PB_Ratio")
        if pb is None:
            pb_row.append("N/A")
        else:
            style = "green" if pb < 3.0 else ("red" if pb > 6.0 else "yellow")
            pb_row.append(f"[{style}]{pb:.2f}[/{style}]")
    table.add_row(*pb_row)

    # Debt to Equity
    de_row = ["Debt to Equity"]
    for s in stocks:
        de = s["fundamentals"].get("Debt_to_Equity")
        if de is None:
            de_row.append("N/A")
        else:
            style = "green" if de < 50 else ("red" if de > 100 else "yellow")
            de_row.append(f"[{style}]{de:.2f}%[/{style}]")
    table.add_row(*de_row)

    # ROE
    roe_row = ["Return on Equity (ROE)"]
    for s in stocks:
        roe = s["fundamentals"].get("ROE")
        if roe is None:
            roe_row.append("N/A")
        else:
            style = "green" if roe >= 15 else ("red" if roe < 8 else "yellow")
            roe_row.append(f"[{style}]{roe:.2f}%[/{style}]")
    table.add_row(*roe_row)

    # Technical trend
    tech_row = ["Price Trend"]
    for s in stocks:
        above_50 = s.get("tech_above_50")
        above_200 = s.get("tech_above_200")
        if above_50 and above_200:
            tech_row.append("[green]Upward (Strong) ✅[/green]")
        elif above_50 or above_200:
            tech_row.append("[yellow]Mixed Outlook ⚠️[/yellow]")
        else:
            tech_row.append("[red]Downward Trend ⚠️[/red]")
    table.add_row(*tech_row)

    # Sentiment score
    sent_row = ["News Sentiment Mood"]
    for s in stocks:
        score = s.get("sentiment_score", 0.0)
        style = "green" if score > 0.15 else ("red" if score < -0.15 else "yellow")
        label = "Positive ✅" if score > 0.15 else ("Negative ⚠️" if score < -0.15 else "Neutral ⚖️")
        sent_row.append(f"[{style}]{label} ({score:+.2f})[/{style}]")
    table.add_row(*sent_row)

    # Big investors
    big_row = ["Big Investors (FII/DII)"]
    for s in stocks:
        fii = s.get("fii_net", 0.0)
        dii = s.get("dii_net", 0.0)
        if fii > 0 and dii > 0:
            big_row.append("[green]Buying heavily ✅[/green]")
        elif fii < 0 and dii < 0:
            big_row.append("[red]Pulling money out ⚠️[/red]")
        else:
            big_row.append("[yellow]Mixed / Diverging ⚖️[/yellow]")
    table.add_row(*big_row)

    console.print(Panel(table, border_style="cyan"))
    console.print(Panel(ai_summary, title="🧠 AI Side-by-Side Comparison (For Beginners)", border_style="magenta"))


def main_loop(verbose: bool = False):
    """Main chatbot interactive terminal loop."""
    print_header()
    if verbose:
        console.print("  [bold magenta]🔬 Verbose Mode ON[/bold magenta] — Signal Score Breakdown panel will be shown after each analysis.\n")
    
    # Pre-check model existence
    model_path = "models/xgboost_model.pkl"
    if not os.path.exists(model_path):
        console.print("[bold red]Model Warning:[/bold red] The XGBoost model `models/xgboost_model.pkl` is missing.")
        console.print("[dim]Run: [bold white]python src/train_model.py[/bold white] to train it.[/dim]\n")
        
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]Chatbot[/bold cyan]")
            user_input = user_input.strip()
            
            # Intercept general Tata Motors queries
            q_clean = user_input.lower().strip()
            is_general_tata = ("tata motor" in q_clean or "tatamotors" in q_clean) and \
                              not any(w in q_clean for w in ["pv", "passenger", "cv", "commercial", "tmpv", "tmcv"])
            
            if is_general_tata:
                console.print("\n[bold yellow]💡 Tata Motors has demerged into two separate listed entities:[/bold yellow]")
                console.print("  1. [bold cyan]TMPV[/bold cyan] (Tata Motors Passenger Vehicles Limited - Cars, EVs, JLR)")
                console.print("  2. [bold cyan]TMCV[/bold cyan] (Tata Motors Commercial Vehicles Limited - Trucks, Buses)")
                console.print()
                
                choice = Prompt.ask(
                    "[bold white]Which company are you asking about? [1/2/Cancel][/bold white]", 
                    choices=["1", "2", "PV", "CV", "TMPV", "TMCV", "pv", "cv", "tmpv", "tmcv", "Cancel", "cancel"],
                    show_choices=False,
                    default="Cancel"
                )
                
                if choice.upper() in ["1", "PV", "TMPV"]:
                    for term in ["tata motors", "tatamotors", "tata motor"]:
                        if term in user_input.lower():
                            idx = user_input.lower().find(term)
                            user_input = user_input[:idx] + "TMPV" + user_input[idx+len(term):]
                    console.print("[*] Re-routing query to [bold cyan]Tata Motors Passenger Vehicles (TMPV)[/bold cyan]...")
                elif choice.upper() in ["2", "CV", "TMCV"]:
                    for term in ["tata motors", "tatamotors", "tata motor"]:
                        if term in user_input.lower():
                            idx = user_input.lower().find(term)
                            user_input = user_input[:idx] + "TMCV" + user_input[idx+len(term):]
                    console.print("[*] Re-routing query to [bold cyan]Tata Motors Commercial Vehicles (TMCV)[/bold cyan]...")
                else:
                    console.print("[*] Query cancelled.")
                    continue
            
            # Check exit
            if user_input.lower() in ['exit', 'quit']:
                console.print("\n[bold yellow]👋 Thank you for using the AI Stock Advisor! Happy Investing![/bold yellow]\n")
                break
                
            if not user_input:
                continue
                
            # Check if user wants a Backtest
            if user_input.lower().startswith('/backtest ') or user_input.lower().startswith('backtest '):
                # Extract stock query
                query = user_input.replace('/backtest', '').replace('backtest', '').strip()
                
                with console.status("[bold cyan]Running historical simulation backtest (1 year)...[/bold cyan]") as status:
                    try:
                        ticker, name = extract_ticker(query)
                        if ticker is None:
                            console.print(f"[bold red]❌ Stock Not Found:[/bold red] Could not find a stock matching '[bold white]{query}[/bold white]'.")
                            console.print("[dim]Try using the full company name. Example: 'backtest Tata Motors' or 'backtest SBI'[/dim]")
                            continue
                        report = run_backtest(ticker)
                        display_backtest_report(ticker, report)
                    except Exception as err:
                        console.print(f"[bold red]❌ Backtesting Error:[/bold red] {err}")
                continue
                
            # Regular query -> extract tickers
            tickers_found = extract_multiple_tickers(user_input)
            
            # Guard: stock not found
            if not tickers_found:
                clean_input = user_input.strip()
                words = clean_input.split()
                is_question = (
                    len(words) >= 2 or 
                    clean_input.endswith("?") or 
                    any(w in clean_input.lower() for w in ["what", "how", "why", "who", "define", "explain", "meaning", "is", "are"])
                )
                
                if is_question:
                    with console.status("[bold green]Consulting Gemini on your question...[/bold green]") as status:
                        try:
                            explanation = explain_educational_concept(user_input)
                            # Print beautiful panel with custom title
                            console.print(Panel(
                                explanation,
                                title="🧠 AI Financial Guide (For Beginners)",
                                border_style="cyan"
                            ))
                        except Exception as ex:
                            console.print(f"[bold red]❌ Error: [/bold red] {ex}")
                else:
                    console.print(f"\n[bold red]❌ Stock Not Found:[/bold red] I couldn't identify a stock from '[bold white]{user_input}[/bold white]'.")
                    console.print("[bold yellow]💡 Tips:[/bold yellow]")
                    console.print("  • Use the company name: [bold white]'Should I buy Zomato?'[/bold white]")
                    console.print("  • Or the short name:    [bold white]'SBI stock analysis'[/bold white]")
                    console.print("  • Supported stocks include: Reliance, TCS, Infosys, SBI, HDFC Bank, Zomato, L&T, Airtel, and more.\n")
                continue
            
            # Single-stock analysis
            # Single-stock analysis
            if len(tickers_found) == 1:
                ticker, company_name = tickers_found[0]
                q_lower = user_input.lower()
                
                # Check specific intents
                has_news_intent = any(w in q_lower for w in ["news", "headline", "headlines", "article", "articles", "media"])
                has_fundamentals_intent = any(w in q_lower for w in ["valuation", "debt", "pe", "pb", "roe", "mcap", "fundamental", "fundamentals", "market cap", "leverage", "profits"])
                has_price_intent = any(w in q_lower for w in ["price", "chart", "trend", "technical", "technicals", "rsi", "macd", "moving average"])
                
                # 1. News Intent only (Fast path)
                if has_news_intent and not (has_fundamentals_intent or has_price_intent):
                    with console.status(f"[bold green]Fetching news for {company_name}...[/bold green]") as status:
                        try:
                            headlines = fetch_news_headlines(ticker, company_name)
                            headlines = deduplicate_headlines(headlines)
                            console.print(f"\n📰 [bold cyan]Latest News for {company_name} ({ticker})[/bold cyan]\n")
                            if not headlines:
                                console.print("No recent news found.")
                            else:
                                for i, h in enumerate(headlines[:10], 1):
                                    console.print(f"  {i}. {h}")
                            console.print()
                        except Exception as err:
                            console.print(f"[bold red]❌ News Fetch Error:[/bold red] {err}")
                    continue
                
                # 2. Fundamentals Intent only (Fast path)
                elif has_fundamentals_intent and not (has_news_intent or has_price_intent):
                    with console.status(f"[bold green]Fetching fundamentals for {company_name}...[/bold green]") as status:
                        try:
                            fundamentals = fetch_fundamentals(ticker)
                            
                            # Fetch current price as reference
                            price_df = fetch_stock_price(ticker)
                            if isinstance(price_df.columns, pd.MultiIndex):
                                price_df.columns = price_df.columns.get_level_values(0)
                            current_price = float(price_df.iloc[-1]['Close'])
                            
                            console.print(f"\n📊 [bold cyan]Fundamental Valuation: {company_name} ({ticker}) | Price: ₹{current_price:,.2f}[/bold cyan]\n")
                            if not fundamentals:
                                console.print("Fundamentals data unavailable.")
                            else:
                                fund_table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2), width=95)
                                fund_table.add_column("Key Metric", style="bold white", min_width=30)
                                fund_table.add_column("Value", justify="right", min_width=20)
                                fund_table.add_column("Benchmark / Interpretation", min_width=40)
                                
                                pe = fundamentals.get("PE_Ratio")
                                pe_type = fundamentals.get("PE_Type", "Trailing")
                                eps = fundamentals.get("EPS")
                                if pe is not None:
                                    pe_str = f"{pe:.2f}  [{pe_type} P/E]"
                                elif eps is not None and eps < 0:
                                    pe_str = f"N/A  (Negative EPS: ₹{eps:.2f})"
                                else:
                                    pe_str = "N/A"
                                pe_interp = "[green]Undervalued / Healthy (< 25)[/green]" if pe is not None and pe < 25 else \
                                            ("[yellow]Fairly Valued (25-40)[/yellow]" if pe is not None and pe <= 40 else \
                                             "[red]Premium / High Value (> 40)[/red]" if pe is not None else \
                                             "[dim]Not applicable — company has negative earnings[/dim]")
                                fund_table.add_row("📊 P/E Ratio (Price/Earnings)", pe_str, pe_interp)

                                if eps is not None:
                                    eps_str = f"₹{eps:.2f}"
                                    eps_interp = "[green]Profitable ✅[/green]" if eps > 0 else "[red]Loss-making ⚠️[/red]"
                                    fund_table.add_row("📉 EPS (Earnings Per Share)", eps_str, eps_interp)
                                
                                pb = fundamentals.get("PB_Ratio")
                                pb_str = f"{pb:.2f}" if pb is not None else "N/A"
                                pb_interp = "[green]Value Buy (< 3.0)[/green]" if pb is not None and pb < 3.0 else \
                                            ("[yellow]Moderate (3.0-6.0)[/yellow]" if pb is not None and pb <= 6.0 else "[red]Growth Premium (> 6.0)[/red]" if pb is not None else "N/A")
                                fund_table.add_row("📖 P/B Ratio (Price/Book)", pb_str, pb_interp)
                                
                                de = fundamentals.get("Debt_to_Equity")
                                de_str = f"{de:.2f}%" if de is not None else "N/A"
                                de_interp = "[green]Low Debt / Low Risk (< 50%)[/green]" if de is not None and de < 50 else \
                                            ("[yellow]Leveraged (50-100%)[/yellow]" if de is not None and de <= 100 else "[red]High Debt / High Risk (> 100%)[/red]" if de is not None else "N/A")
                                fund_table.add_row("🛡️ Debt to Equity Ratio", de_str, de_interp)
                                
                                roe = fundamentals.get("ROE")
                                roe_str = f"{roe:.2f}%" if roe is not None else "N/A"
                                roe_interp = "[green]High Return / Efficient (> 15%)[/green]" if roe is not None and roe >= 15 else \
                                             ("[yellow]Moderate Return (8-15%)[/yellow]" if roe is not None and roe >= 8 else "[red]Low Return (< 8%)[/red]" if roe is not None else "N/A")
                                fund_table.add_row("💰 Return on Equity (ROE)", roe_str, roe_interp)
                                
                                mcap = fundamentals.get("Market_Cap")
                                mcap_str = f"₹{mcap:,.0f} Cr" if mcap is not None else "N/A"
                                mcap_interp = "[cyan]Mega Cap (> 1,00,000 Cr)[/cyan]" if mcap is not None and mcap > 100000 else \
                                              ("[blue]Large Cap (20,000 - 1,00,000 Cr)[/blue]" if mcap is not None and mcap >= 20000 else "[dim]Mid/Small Cap (< 20,000 Cr)[/dim]" if mcap is not None else "N/A")
                                fund_table.add_row("🏛️ Market Capitalization", mcap_str, mcap_interp)
                                
                                console.print(Panel(fund_table, title="📊 Fundamentals Dashboard", border_style="cyan"))
                        except Exception as err:
                            console.print(f"[bold red]❌ Fundamentals Fetch Error:[/bold red] {err}")
                    continue
                
                # 3. Price/Technical Intent only (Fast path)
                elif has_price_intent and not (has_news_intent or has_fundamentals_intent):
                    with console.status(f"[bold green]Fetching price and technicals for {company_name}...[/bold green]") as status:
                        try:
                            price_df = fetch_stock_price(ticker)
                            if isinstance(price_df.columns, pd.MultiIndex):
                                price_df.columns = price_df.columns.get_level_values(0)
                            price_indicators_df = calculate_technical_indicators(price_df)
                            latest_price_row = price_indicators_df.iloc[-1]
                            current_price = float(latest_price_row['Close'])
                            
                            console.print(f"\n📈 [bold cyan]Technicals & Price: {company_name} ({ticker}) | Current: ₹{current_price:,.2f}[/bold cyan]\n")
                            
                            price_above_50  = latest_price_row.get('Price_Above_MA50',  0)
                            price_above_200 = latest_price_row.get('Price_Above_MA200', 0)
                            if price_above_50 and price_above_200:
                                tech_status = "[green]Price moving UP strongly ✅[/green]"
                            elif price_above_50 or price_above_200:
                                tech_status = "[yellow]Price moving UP weakly ⚠️[/yellow]"
                            else:
                                tech_status = "[red]Price moving DOWN ⚠️[/red]"
                                
                            tech_table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2), width=70)
                            tech_table.add_column("Indicator", style="bold white", min_width=25)
                            tech_table.add_column("Value / Outlook", min_width=40)
                            
                            tech_table.add_row("Price Trend Status", tech_status)
                            tech_table.add_row("RSI (14-day momentum)", f"{latest_price_row.get('RSI', 50.0):.2f}")
                            tech_table.add_row("50-Day Moving Average", f"₹{latest_price_row.get('MA50', current_price):,.2f}")
                            tech_table.add_row("200-Day Moving Average", f"₹{latest_price_row.get('MA200', current_price):,.2f}")
                            
                            console.print(Panel(tech_table, title="📈 Technical Indicators", border_style="cyan"))
                        except Exception as err:
                            console.print(f"[bold red]❌ Technicals Fetch Error:[/bold red] {err}")
                    continue
                
                # 4. Standard full analysis
                with console.status("[bold green]Analyzing stock market signals...[/bold green]") as status:
                    try:
                        status.update(f"[bold green]Fetching stock price history for {ticker}...[/bold green]")

                        # Step 1: Download price data (critical — cannot continue without this)
                        price_df = fetch_stock_price(ticker)
                        if isinstance(price_df.columns, pd.MultiIndex):
                            price_df.columns = price_df.columns.get_level_values(0)

                        # Step 2: Calculate indicators (critical — required for ML features)
                        price_indicators_df = calculate_technical_indicators(price_df)
                        latest_price_row = price_indicators_df.iloc[-1]
                        current_price = float(latest_price_row['Close'])

                        # Step 3: Fetch recent headlines (non-critical — fallback to empty list)
                        status.update(f"[bold green]Reading headlines for {company_name}...[/bold green]")
                        headlines_available = True
                        try:
                            headlines = fetch_news_headlines(ticker, company_name)
                        except Exception as news_err:
                            console.print(f"[dim yellow]⚠️  News headlines unavailable ({news_err}). Continuing with technical & ML analysis.[/dim yellow]")
                            headlines = []
                            headlines_available = False

                        # Step 4: Fetch FII/DII data (non-critical — fallback to neutral zeros)
                        status.update("[bold green]Checking institutional cash flows...[/bold green]")
                        try:
                            fii_dii_df = fetch_latest_fii_dii()
                            fii_net_list = fii_dii_df['FII_Net'].tolist()
                            dii_net_list = fii_dii_df['DII_Net'].tolist()
                            fii_dii_summary = analyze_institutional_signals(fii_net_list, dii_net_list)
                        except Exception as fii_err:
                            console.print(f"[dim yellow]⚠️  FII/DII data unavailable ({fii_err}). Using neutral values.[/dim yellow]")
                            fii_dii_summary = {
                                "FII_10d_Net": 0.0, "DII_10d_Net": 0.0,
                                "FII_Trend": 0, "DII_Trend": 0, "Divergence_Flag": 0
                            }

                        # Step 5: Fetch fundamental valuation metrics (non-critical)
                        status.update(f"[bold green]Fetching fundamental valuation data for {company_name}...[/bold green]")
                        try:
                            fundamentals = fetch_fundamentals(ticker)
                        except Exception as fund_err:
                            console.print(f"[dim yellow]⚠️  Fundamentals unavailable ({fund_err}). Skipping valuation panel.[/dim yellow]")
                            fundamentals = {}

                        # Step 6: Analyze News Sentiment (non-critical — fallback to neutral)
                        status.update("[bold green]Running sentiment analyzer on headlines...[/bold green]")
                        try:
                            if headlines:
                                sent_score, pos_count, neg_count = analyze_news_sentiment(headlines)
                                sentiment_summary = (sent_score, pos_count, neg_count, 1)
                            else:
                                sentiment_summary = (0.0, 0, 0, 0)
                                sent_score, pos_count, neg_count = 0.0, 0, 0
                        except Exception as sent_err:
                            console.print(f"[dim yellow]⚠️  Sentiment analysis failed ({sent_err}). Using neutral sentiment.[/dim yellow]")
                            sentiment_summary = (0.0, 0, 0, 0)
                            sent_score = 0.0

                        # Step 7: Fetch global macro returns (non-critical — fallback to zeros)
                        status.update("[bold green]Fetching latest global macro returns...[/bold green]")
                        try:
                            macro_returns = get_latest_macro_returns()
                        except Exception as macro_err:
                            console.print(f"[dim yellow]⚠️  Macro data unavailable ({macro_err}). Using neutral macro values.[/dim yellow]")
                            macro_returns = (0.0, 0.0, 0.0)

                        # Step 8: Merge features and run ML Prediction
                        status.update("[bold green]Running Machine Learning Classifier...[/bold green]")
                        feature_row = prepare_inference_row(price_indicators_df, fii_dii_summary, sentiment_summary, macro_returns)
                        ml_result = predict_stock_action(feature_row)

                        # Define unified technical summary dictionary containing Price_Pct_5d
                        tech_summary_dict = {
                            **latest_price_row.to_dict(),
                            "Price_Pct_5d": float((price_indicators_df['Close'].iloc[-1] / price_indicators_df['Close'].iloc[-6] - 1) * 100) if len(price_indicators_df) >= 6 else 0.0
                        }

                        # Step 9: Explain via Gemini
                        status.update("[bold green]Translating findings via Gemini AI...[/bold green]")
                        try:
                            model_analysis, beginner_explanation = generate_beginner_explanation(
                                ticker,
                                company_name,
                                current_price,
                                ml_result,
                                tech_summary_dict,
                                sent_score,
                                headlines,
                                fii_dii_summary,
                                fundamentals,
                                user_input=user_input
                            )
                        except Exception as gemini_err:
                            console.print(f"[dim yellow]⚠️  Gemini AI explanation unavailable ({gemini_err}). Showing raw analysis.[/dim yellow]")
                            model_analysis = (
                                "Factors Favouring Buy (+):\n"
                                "  • [RSI] Momentum is neutral.\n\n"
                                "Factors Against Buy / Favouring Hold (-):\n"
                                "  • [Moving Averages] Price is under pressure.\n\n"
                                "Strategic Advice:\n"
                                f"  • Action: Avoid buying new shares. Hold current shares.\n"
                                f"  • Stop Loss (Safety Net): ₹{current_price * 0.95:,.2f} (Current price: ₹{current_price:,.2f}) — limit losses."
                            )
                            beginner_explanation = (
                                f"Technical analysis complete for {company_name} ({ticker}).\n"
                                f"Current price: ₹{current_price:,.2f}\n"
                                f"ML Signal: {ml_result.get('Recommendation', 'N/A')} "
                                f"({ml_result.get('Confidence', 0)*100:.1f}% confidence)\n"
                                f"[AI explanation unavailable — Gemini API may be down or key missing]"
                            )

                        # Step 10: Output beautifully formatted dashboard
                        display_prediction_report(
                            ticker,
                            company_name,
                            current_price,
                            ml_result,
                            tech_summary_dict,
                            sent_score,
                            fii_dii_summary,
                            model_analysis,
                            beginner_explanation,
                            user_input,   # pass the original question for direct answer
                            fundamentals,
                            verbose=verbose
                        )

                    except Exception as err:
                        console.print(f"[bold red]❌ Analysis Error:[/bold red] {err}")
                        console.print("[dim]Could not fetch price data. Check that the stock ticker is valid and your internet connection is active.[/dim]")


            
            # Side-by-Side stock comparison
            else:
                with console.status("[bold green]Comparing stock market signals...[/bold green]") as status:
                    try:
                        stocks_data = []
                        status.update("[bold green]Fetching latest global macro returns...[/bold green]")
                        macro_returns = get_latest_macro_returns()
                        for ticker, company_name in tickers_found:
                            status.update(f"[bold green]Analyzing {company_name} ({ticker})...[/bold green]")
                            # Fetch and predict for this stock
                            price_df = fetch_stock_price(ticker)
                            if isinstance(price_df.columns, pd.MultiIndex):
                                price_df.columns = price_df.columns.get_level_values(0)
                            price_indicators_df = calculate_technical_indicators(price_df)
                            latest_price_row = price_indicators_df.iloc[-1]
                            current_price = float(latest_price_row['Close'])
                            
                            headlines = fetch_news_headlines(ticker, company_name)
                            fii_dii_df = fetch_latest_fii_dii()
                            fii_net_list = fii_dii_df['FII_Net'].tolist()
                            dii_net_list = fii_dii_df['DII_Net'].tolist()
                            fii_dii_summary = analyze_institutional_signals(fii_net_list, dii_net_list)
                            
                            fundamentals = fetch_fundamentals(ticker)
                            sent_score, pos_count, neg_count = analyze_news_sentiment(headlines)
                            sent_avail = 1 if headlines else 0
                            sentiment_summary = (sent_score, pos_count, neg_count, sent_avail)
                            
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
                                "tech_above_50": bool(latest_price_row.get('Price_Above_MA50', 0)),
                                "tech_above_200": bool(latest_price_row.get('Price_Above_MA200', 0)),
                                "fii_net": fii_dii_summary.get('FII_10d_Net', 0.0),
                                "dii_net": fii_dii_summary.get('DII_10d_Net', 0.0)
                            })
                            
                        status.update("[bold green]Generating side-by-side AI explanation...[/bold green]")
                        ai_summary = generate_comparison_explanation(stocks_data)
                        display_comparison_report(stocks_data, ai_summary)
                    except Exception as err:
                        console.print(f"[bold red]❌ Comparison Error:[/bold red] {err}")
                    
        except KeyboardInterrupt:
            console.print("\n[bold yellow]👋 Session interrupted. Goodbye![/bold yellow]\n")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Stock Advisory Chatbot")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show Signal Score Breakdown panel (for reviewers/debugging)"
    )
    args = parser.parse_args()
    main_loop(verbose=args.verbose)
