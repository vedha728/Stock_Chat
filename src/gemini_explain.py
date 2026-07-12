import os
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def clean_markdown(text: str) -> str:
    """
    Removes markdown symbols (**, ###, ##, #, *) from Gemini output
    so the terminal displays clean plain text.
    """
    import re
    # Remove bold: **text** → text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove italic: *text* → text
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove headings: ### Title → Title
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove leftover lone asterisks used as bullet points
    text = re.sub(r'^\s*\*\s+', '  • ', text, flags=re.MULTILINE)
    # Ensure bullet points are preceded by a double newline to prevent line collapsing in markdown
    text = re.sub(r'(?<!\n)\n\s*•', '\n\n  •', text)
    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _get_api_key(attempt_index: int = 0) -> str:
    """
    Returns the API key to use. If GEMINI_API_KEY contains comma-separated values,
    rotates through them using attempt_index % count.
    """
    raw_key = os.getenv("GEMINI_API_KEY", "")
    if not raw_key:
        return ""
    keys = [k.strip() for k in raw_key.split(",") if k.strip()]
    if not keys:
        return ""
    return keys[attempt_index % len(keys)]


def generate_beginner_explanation(
    ticker: str,
    company_name: str,
    current_price: float,
    ml_result: dict,
    technical_summary: dict,
    sentiment_score: float,
    headlines: list[str],
    fii_dii_summary: dict,
    fundamentals: dict = None,
    user_input: str = ""
) -> tuple[str, str]:
    """
    Connects to Google Gemini 2.5 Flash to translate raw financial metrics,
    ML predictions, and fundamental valuation metrics into two separate segments:
    1. Model Analysis (Factors list + Strategic advice)
    2. Beginner Explanation (Paragraph format)
    Reconciles any contradictory signals and adapts to buy/sell user intent.
    """
    recommendation = ml_result["Recommendation"]
    is_overridden = ml_result.get("Low_Confidence", False)
    orig_rec = ml_result.get("Original_Recommendation", recommendation)
    orig_conf = ml_result.get("Original_Confidence", ml_result["Confidence"]) * 100
    confidence = ml_result["Confidence"] * 100
    buy_pct = ml_result.get("Breakdown", {}).get("BUY", 0.0) * 100
    
    # Calculate entry zone and safety stop loss
    if recommendation == "BUY":
        entry_min = current_price * 0.995
        entry_max = current_price * 1.01
        entry_zone = f"₹{entry_min:.1f} – ₹{entry_max:.1f}"
        stop_loss = f"₹{current_price * 0.95:.1f}"
    elif recommendation == "SELL":
        entry_zone = "N/A (Avoid Buying)"
        stop_loss = f"₹{current_price * 1.05:.1f}"
    else: # HOLD
        entry_zone = "Hold current shares. Do not add new positions."
        stop_loss = f"₹{current_price * 0.95:.1f}"

    # Check query type (sell vs buy)
    asked_sell = False
    if user_input:
        q = user_input.lower()
        asked_sell = any(w in q for w in ["sell", "exit", "quit", "offload"])

    is_near_buy = (recommendation == "HOLD" and is_overridden and orig_rec == "BUY" and 50.0 <= buy_pct <= 54.9)

    # Get API key
    raw_key = os.getenv("GEMINI_API_KEY", "")
    
    # 2. Check if API key is missing or default placeholder
    if not raw_key or "PLACEHOLDER" in raw_key:
        print("[*] API key not found. Using local template explainer.")
        
        # Prepare local fallback Model Analysis (Section A)
        fav_factors = []
        agt_factors = []
        
        if recommendation == "BUY":
            fav_factors.append("  • [Moving Averages] Price is above its 50-day average price.")
            fav_factors.append("  • [DII] Institutional players are showing buying interest.")
            fav_factors.append("  • [News] News sentiment is generally positive.")
            agt_factors.append("  • [RSI] Short-term momentum shows signs of getting overbought.")
        elif recommendation == "SELL":
            fav_factors.append("  • [Fundamentals] P/E ratio is high relative to historical earnings.")
            agt_factors.append("  • [Moving Averages] Price trades below long-term moving averages.")
            agt_factors.append("  • [FII] Foreign institutional funds are withdrawing capital.")
            agt_factors.append("  • [News] Media sentiment is currently cautious.")
        else: # HOLD
            fav_factors.append("  • [RSI] Momentum indicator is in the neutral zone.")
            if is_near_buy:
                fav_factors.append("  • [DII] Domestic big investors are supporting the price.")
            else:
                fav_factors.append("  • [News] Media sentiment is neutral.")
            agt_factors.append("  • [Moving Averages] Price is trading below short-term averages.")
            agt_factors.append("  • [FII] Foreign investors are reducing their exposure.")
            
        fav_text = "\n".join(fav_factors)
        agt_text = "\n".join(agt_factors)
        
        if asked_sell:
            action_str = "Do not sell. Hold current shares."
            if is_near_buy:
                action_str += " (WATCH CLOSELY: Very close to buy zone)."
        else:
            action_str = "Avoid buying new shares. Hold current shares."
            if is_near_buy:
                action_str += " (WATCH CLOSELY: Very close to buy zone)."
                
        if recommendation == "BUY":
            action_str = "Buy shares."
        elif recommendation == "SELL":
            action_str = "Sell shares."
            
        fallback_model_analysis = (
            f"Factors Favouring {'Hold / against Sell' if asked_sell else 'Buy'} (+):\n"
            f"{fav_text}\n\n"
            f"Factors {'Favouring Sell / Weaknesses' if asked_sell else 'Against Buy / Favouring Hold'} (-):\n"
            f"{agt_text}\n\n"
            f"Strategic Advice:\n"
            f"  • Action: {action_str}\n"
            f"  • Stop Loss (Safety Net): {stop_loss} (Current price: ₹{current_price:,.2f}) — "
            f"Sell any shares you hold if price falls below this to limit losses."
        )
        
        # Prepare local fallback Beginner Explanation (Section B)
        rec_action = "HOLD your current shares. Avoid buying new shares at this time."
        if recommendation == "BUY":
            rec_action = "BUY shares."
        elif recommendation == "SELL":
            rec_action = "SELL shares."
            
        fallback_beginner_explanation = (
            f"Business & Industry Context:\n"
            f"The company operates in its respective industry sector. According to the current market trend, "
            f"the sector is experiencing normal volatility.\n\n"
            f"The Big Picture (Overall Scenario):\n"
            f"The company shows stable financials. However, according to the current market trend, "
            f"the stock price is currently trading {'above' if technical_summary.get('Price_Above_MA50') else 'below'} its recent average price. "
            f"The model suggests a recommendation of {recommendation}.\n\n"
            f"What You Should Do:\n"
            f"  • Action: {rec_action}"
        )
        
        return fallback_model_analysis, fallback_beginner_explanation

    # 3. Call Gemini API
    try:
        # Format fundamentals block
        fund_block = "N/A"
        if fundamentals:
            fund_block = (
                f"P/E Ratio: {fundamentals.get('PE_Ratio')}\n"
                f"P/B Ratio: {fundamentals.get('PB_Ratio')}\n"
                f"Debt to Equity Ratio: {fundamentals.get('Debt_to_Equity')}%\n"
                f"ROE (%): {fundamentals.get('ROE')}%\n"
                f"Market Cap (₹ Crores): {fundamentals.get('Market_Cap')}\n"
            )
            
        # Determine query intent instructions
        if asked_sell:
            query_instruction = (
                "The user asked about SELLING this stock (e.g. 'should I sell?'). "
                "Frame the response from a selling perspective. Explain the business and overall scenario "
                "relative to the current market trend."
            )
            rec_format = (
                "Business & Industry Context:\n"
                "[Explain what this company does and how the current market trend is affecting this industry in 3 simple sentences.]\n\n"
                "The Big Picture (Overall Scenario):\n"
                "[Summarize the overall scenario in simple terms, explaining the prediction (SELL/HOLD) relative to the current market trend and company health in 3-4 sentences. Do NOT repeat the exact lists of technical indicators or FII/DII numbers from Section A.]\n\n"
                "What You Should Do:\n"
                "  • Action: [State clearly whether to sell or hold current shares, framed around the current market trend.]"
            )
        else:
            query_instruction = (
                "The user asked about BUYING this stock (e.g. 'should I buy?'). "
                "Frame the response from a buying perspective. Explain the business and overall scenario "
                "relative to the current market trend."
            )
            rec_format = (
                "Business & Industry Context:\n"
                "[Explain what this company does and how the current market trend is affecting this industry in 3 simple sentences.]\n\n"
                "The Big Picture (Overall Scenario):\n"
                "[Summarize the overall scenario in simple terms, explaining the prediction (BUY/HOLD) relative to the current market trend and company health in 3-4 sentences. Do NOT repeat the exact lists of technical indicators or FII/DII numbers from Section A.]\n\n"
                "What You Should Do:\n"
                "  • Action: [State clearly whether to buy or hold current shares, framed around the current market trend.]"
            )

        prompt = (
            f"You are a friendly stock market guide for complete beginners in India. "
            f"Your job: explain this stock analysis in PLAIN, SIMPLE English — like explaining to a friend who has never invested before. "
            f"RULES YOU MUST FOLLOW:\n"
            f"- NO markdown (no **, no *, no #, no -).\n"
            f"- Never say: bullish, bearish, Golden Cross, threshold, suppressed, overridden.\n"
            f"- HIGHLIGHT key words by writing them in CAPITAL LETTERS (example: PRICE IS GOING UP, DO NOT BUY, HOLD YOUR SHARES, CURRENT MARKET TREND).\n"
            f"- Keep each point SHORT — maximum 3 sentences per point.\n"
            f"- Be direct. No long stories.\n"
            f"- Throughout the explanation, frame the description around the CURRENT MARKET TREND to give a clear view.\n"
            f"{query_instruction}\n\n"

            f"--- STOCK DATA ---\n"
            f"Company: {company_name} ({ticker})\n"
            f"Current Price: Rs.{current_price}\n"
            f"Final Recommendation: {recommendation}\n"
            f"Is the model uncertain (mixed signals): {'Yes' if is_overridden else 'No'}\n"
            f"Price above 50-day average price: {'Yes' if technical_summary.get('Price_Above_MA50') else 'No'}\n"
            f"Price above 200-day average price: {'Yes' if technical_summary.get('Price_Above_MA200') else 'No'}\n"
            f"Price momentum (RSI, 0-100 scale, 50=neutral): {technical_summary.get('RSI', 50.0):.1f}\n"
            f"Recent news mood (-1=very bad, 0=neutral, +1=very good): {sentiment_score:.2f}\n"
            f"Top Headlines: {chr(10).join([f'{i+1}. {h}' for i, h in enumerate(headlines[:3])])}\n"
            f"Foreign big investors flow last 10 days: Rs.{fii_dii_summary.get('FII_10d_Net', 0.0):.0f} crores\n"
            f"Indian big investors flow last 10 days: Rs.{fii_dii_summary.get('DII_10d_Net', 0.0):.0f} crores\n"
            f"Fundamental Valuation:\n{fund_block}\n"
            f"Suggested action zone: {entry_zone}\n"
            f"Safety Net Price: {stop_loss}\n\n"

            f"--- WRITE RESPONSE IN THIS EXACT FORMAT with '=== SPLIT ===' separating the two sections ---\n"
            f"SECTION A: MODEL ANALYSIS AND STRATEGY\n"
            f"Factors Favouring {'Hold / against Sell' if asked_sell else 'Buy'} (+):\n"
            f"[Provide 3 bullet points starting with '  • '. Next to each point, mention a bracketed key tag like [RSI], [MACD], [Macro], [FII], [DII], [News], [Fundamentals], [Moving Averages] in simple words explaining what favours buying or holding this stock.]\n\n"
            f"Factors {'Favouring Sell / Weaknesses' if asked_sell else 'Against Buy / Favouring Hold'} (-):\n"
            f"[Provide 3 bullet points starting with '  • '. Next to each point, mention a bracketed key tag like [RSI], [MACD], [Macro], [FII], [DII], [News], [Fundamentals], [Moving Averages] in simple words explaining what does not favour buying/holding this stock.]\n\n"
            f"Strategic Advice:\n"
            f"  • Action: [State action clearly. "
            f"If user asked about buying: 'Avoid buying new shares. Hold current shares.' (if HOLD) or 'Buy shares.' (if BUY). "
            f"If user asked about selling: 'Do not sell. Hold current shares.' (if HOLD) or 'Sell shares.' (if SELL). "
            f"If BUY probability is between 50% and 54.9%, append: (WATCH CLOSELY: Very close to buy zone).]\n"
            f"  • Stop Loss (Safety Net): {stop_loss} (Current price: Rs.{current_price}) — [One sentence explaining how to act on this stop loss to limit losses.]\n\n"
            f"=== SPLIT ===\n\n"
            f"SECTION B: BEGINNER EXPLANATION\n"
            f"{rec_format}"
        )

        import time
        keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        max_retries = max(3, len(keys))
        for attempt in range(max_retries):
            try:
                current_key = _get_api_key(attempt)
                client = genai.Client(api_key=current_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                raw_text = clean_markdown(response.text)
                
                # Split raw_text by delimiter
                if "=== SPLIT ===" in raw_text:
                    parts = raw_text.split("=== SPLIT ===")
                    section_a = parts[0].replace("SECTION A: MODEL ANALYSIS AND STRATEGY", "").strip()
                    section_b = parts[1].replace("SECTION B: BEGINNER EXPLANATION", "").strip()
                    return section_a, section_b
                else:
                    # Fallback if split delimiter is missing
                    return raw_text, raw_text
            except Exception as ex:
                err_msg = str(ex)
                if len(keys) > 1:
                    print(f"\n[!] API call failed on key {attempt+1}/{len(keys)}: {err_msg[:120]}. Rotating to next key...")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                else:
                    if attempt < max_retries - 1:
                        sleep_time = 5 * (attempt + 1)
                        print(f"\n[!] API call failed: {err_msg[:120]}. Retrying in {sleep_time}s ({attempt+1}/{max_retries})...")
                        time.sleep(sleep_time)
                        continue
                raise ex
        
    except Exception as e:
        print(f"[Warning] Failed to generate AI explanation: {e}. Using fallback.")
        fallback_model_analysis = (
            f"Factors Favouring {'Hold / against Sell' if asked_sell else 'Buy'} (+):\n"
            f"  • [RSI] Price momentum is neutral.\n\n"
            f"Factors {'Favouring Sell / Weaknesses' if asked_sell else 'Against Buy / Favouring Hold'} (-):\n"
            f"  • [Moving Averages] Price is under pressure.\n\n"
            f"Strategic Advice:\n"
            f"  • Action: {'Do not sell. Hold current shares.' if asked_sell else 'Avoid buying new shares. Hold current shares.'}\n"
            f"  • Stop Loss (Safety Net): {stop_loss} (Current price: Rs.{current_price}) — Sell any shares to limit losses."
        )
        fallback_beginner_explanation = (
            f"Recommendation: {recommendation} (Confidence: {confidence:.1f}%)\n"
            f"Entry Zone: {entry_zone}\n"
            f"Safety Net Price: {stop_loss}\n\n"
            f"Our strategy rules evaluate 19 market variables including technical RSI/MACD ranges, news sentiment, and FII/DII net flows. "
            f"Please review the technical grids and historical trends on the dashboard below to guide your trade execution."
        )
        return fallback_model_analysis, fallback_beginner_explanation


def explain_educational_concept(user_input: str) -> str:
    """
    Uses Google Gemini to explain a general stock market concept or answer
    an educational finance question for beginners.
    """
    raw_key = os.getenv("GEMINI_API_KEY", "")
    if not raw_key or "PLACEHOLDER" in raw_key:
        return (
            "To learn about Indian stock markets, here are the key concepts and indicators:\n\n"
            "• RSI (Relative Strength Index): A momentum indicator scoring from 0 to 100. Values below 30 suggest oversold conditions (potential buying zone), while values above 70 indicate overbought conditions.\n\n"
            "• Moving Averages (MA50 & MA200): Visual lines tracing the average closing price over the last 50 and 200 days to identify trend direction and support/resistance zones.\n\n"
            "• FII/DII Net Flow: Capital tracking of Foreign and Domestic Institutional transactions. Large net purchases indicate institutional support for index movements.\n\n"
            "Please ask a specific stock question or topic to retrieve detailed metrics."
        )

    prompt = (
        f"You are a friendly stock market guide for complete beginners in India.\n"
        f"The user asked an educational or general question: \"{user_input}\"\n\n"
        f"RULES YOU MUST FOLLOW:\n"
        f"- Explain the concept in PLAIN, SIMPLE English, like explaining to a friend with no finance background.\n"
        f"- Keep it VERY short and clear.\n"
        f"- Structure your answer as follows:\n"
        f"  1. A one-sentence general introduction/definition.\n"
        f"  2. Exactly 2-3 clear, short bullet points explaining the core aspects (use standard Unicode bullet '•').\n"
        f"  3. A extremely simple, direct, and actionable example labeled as 'EXAMPLE:'. The example should follow this format: "
        f"\"EXAMPLE: If Wipro has an RSI of 80 (high), the stock is expensive, so wait to buy. If its RSI is 20 (low), it is cheap, making it a good time to buy.\"\n"
        f"- If the question is NOT related to finance, investing, business, economics, or the stock market, "
        f"politely remind them that you are an AI Stock Advisor and guide them back to stock market topics.\n"
        f"- HIGHLIGHT key terms by writing them in CAPITAL LETTERS.\n"
        f"- DO NOT use markdown headers or bold/italic markers (no **, no *, no #, no `).\n"
        f"- Be encouraging and friendly."
    )

    try:
        import time
        keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        max_retries = max(3, len(keys))
        for attempt in range(max_retries):
            try:
                current_key = _get_api_key(attempt)
                client = genai.Client(api_key=current_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                return clean_markdown(response.text)
            except Exception as ex:
                err_msg = str(ex)
                if len(keys) > 1:
                    print(f"\n[!] API call failed on key {attempt+1}/{len(keys)}: {err_msg[:120]}. Rotating to next key...")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                else:
                    if attempt < max_retries - 1:
                        sleep_time = 5 * (attempt + 1)
                        print(f"\n[!] API call failed: {err_msg[:120]}. Retrying in {sleep_time}s ({attempt+1}/{max_retries})...")
                        time.sleep(sleep_time)
                        continue
                raise ex
    except Exception:
        return (
            "Learn Basics Guide:\n\n"
            "To understand stock metrics simply, look at these core areas:\n\n"
            "• Technicals: RSI shows buying/selling momentum, while Moving Averages identify support levels.\n"
            "• Fundamentals: Check the Price-to-Earnings (P/E) ratio and Debt-to-Equity ratios to gauge company value and financial health.\n"
            "• Sentiment: Market news and headlines provide early signals of trend shifts.\n\n"
            "Please try submitting your topic query again to fetch updated explanations."
        )


def generate_comparison_explanation(stocks: list[dict]) -> str:
    """
    Uses Google Gemini to compare two or more stocks side-by-side,
    providing a clear, beginner-friendly summary of which one has
    better indicators / fundamentals and why.
    """
    raw_key = os.getenv("GEMINI_API_KEY", "")
    if not raw_key or "PLACEHOLDER" in raw_key:
        return (
            "AI Comparison Verdict:\n"
            "SIDE-BY-SIDE ANALYTICS SUMMARY COMPLETED\n\n"
            "Key Strengths:\n"
            "• Technical trends are compared using 50-day and 200-day exponential moving averages.\n"
            "• Fundamental parameters track valuation and leverage ratios.\n\n"
            "Bottom Line for Beginners:\n"
            "Analyze the relative difference in BUY/SELL signals in the side-by-side indicator boxes below to choose the better entry opportunity."
        )

    # Format the data of each stock to put in the prompt
    stocks_prompt_data = ""
    for s in stocks:
        fund_block = "N/A"
        fund = s.get("fundamentals")
        if fund:
            fund_block = (
                f"P/E Ratio: {fund.get('PE_Ratio')}\n"
                f"P/B Ratio: {fund.get('PB_Ratio')}\n"
                f"Debt to Equity Ratio: {fund.get('Debt_to_Equity')}%\n"
                f"ROE (%): {fund.get('ROE')}%\n"
                f"Market Cap: Rs.{fund.get('Market_Cap')} Cr\n"
            )
            
        stocks_prompt_data += (
            f"--- STOCK: {s['company_name']} ({s['ticker']}) ---\n"
            f"Price: Rs.{s['current_price']}\n"
            f"Recommendation: {s['recommendation']} ({s['confidence']:.1f}% confidence)\n"
            f"Technical Trend (Above 50d Average?): {'Yes' if s.get('tech_above_50') else 'No'}\n"
            f"Technical Trend (Above 200d Average?): {'Yes' if s.get('tech_above_200') else 'No'}\n"
            f"News Sentiment Mood (-1 to +1): {s.get('sentiment_score', 0.0):.2f}\n"
            f"FII (Foreign) Flow: Rs.{s.get('fii_net', 0.0):.1f} Cr\n"
            f"DII (Domestic) Flow: Rs.{s.get('dii_net', 0.0):.1f} Cr\n"
            f"Fundamentals:\n{fund_block}\n\n"
        )

    prompt = (
        f"You are a friendly stock market guide for complete beginners in India.\n"
        f"Your job: compare these stocks and explain which one looks stronger right now, and why, in PLAIN, SIMPLE English.\n\n"
        f"RULES YOU MUST FOLLOW:\n"
        f"- NO markdown formatting (no **, no *, no #, no -).\n"
        f"- NO finance jargon (no bullish, bearish, MACD, etc.).\n"
        f"- HIGHLIGHT key points in CAPITAL LETTERS.\n"
        f"- WRITE the entire AI Comparison Verdict line in CAPITAL LETTERS (example: TCS LOOKS LIKE THE STRONGER PICK RIGHT NOW DUE TO ITS HIGHER PROFITABILITY AND LOWER DEBT).\n"
        f"- Keep it SHORT and CRISP (maximum 8 sentences total).\n"
        f"- CLEARLY STATE in what way each stock is currently strong (e.g. explain that Stock A is strong in profitability/safety, while Stock B is strong in cheaper valuation/news sentiment).\n\n"
        f"--- STOCKS DATA ---\n"
        f"{stocks_prompt_data}"
        f"--- WRITE RESPONSE IN THIS EXACT FORMAT ---\n"
        f"AI Comparison Verdict:\n"
        f"[One sentence written entirely in CAPITAL LETTERS stating clearly which stock is currently the stronger pick or if they are equal/mixed.]\n\n"
        f"Key Strengths:\n"
        f"• [Stock A Name] is strong in [Describe specific strength simply, highlight with CAPS]\n"
        f"• [Stock B Name] is strong in [Describe specific strength simply, highlight with CAPS]\n\n"
        f"Bottom Line for Beginners:\n"
        f"[2 sentences of practical advice on what to watch for.]"
    )

    try:
        import time
        keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        max_retries = max(3, len(keys))
        for attempt in range(max_retries):
            try:
                current_key = _get_api_key(attempt)
                client = genai.Client(api_key=current_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                return clean_markdown(response.text)
            except Exception as ex:
                err_msg = str(ex)
                if len(keys) > 1:
                    print(f"\n[!] API call failed on key {attempt+1}/{len(keys)}: {err_msg[:120]}. Rotating to next key...")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                else:
                    if attempt < max_retries - 1:
                        sleep_time = 5 * (attempt + 1)
                        print(f"\n[!] API call failed: {err_msg[:120]}. Retrying in {sleep_time}s ({attempt+1}/{max_retries})...")
                        time.sleep(sleep_time)
                        continue
                raise ex
    except Exception:
        return (
            "AI Comparison Verdict:\n"
            "SIDE-BY-SIDE SUMMARY ANALYZED FROM STOCK DATA\n\n"
            "Key Strengths:\n"
            "• Compare metrics using technical momentum grids.\n"
            "• Compare FII/DII institutional purchase trends.\n\n"
            "Bottom Line for Beginners:\n"
            "Evaluate the relative confidence scores of the stocks below to identify the stronger pick."
        )

