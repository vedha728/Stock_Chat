import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


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
    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def generate_beginner_explanation(
    ticker: str,
    company_name: str,
    current_price: float,
    ml_result: dict,
    technical_summary: dict,
    sentiment_score: float,
    headlines: list[str],
    fii_dii_summary: dict,
    fundamentals: dict = None
) -> str:
    """
    Connects to Google Gemini 2.5 Flash to translate raw financial metrics,
    ML predictions, and fundamental valuation metrics into an easy-to-understand
    response for beginners, reconciling any contradictory signals.
    """
    # 1. Calculate Entry Zone and Safety Price mathematically in Python
    recommendation = ml_result["Recommendation"]
    # If overridden, use the original signal details to explain the original logic
    is_overridden = ml_result.get("Low_Confidence", False)
    orig_rec = ml_result.get("Original_Recommendation", recommendation)
    orig_conf = ml_result.get("Original_Confidence", ml_result["Confidence"]) * 100
    confidence = ml_result["Confidence"] * 100
    
    if recommendation == "BUY":
        entry_min = current_price * 0.995
        entry_max = current_price * 1.01
        entry_zone = f"₹{entry_min:.1f} – ₹{entry_max:.1f}"
        stop_loss = f"₹{current_price * 0.95:.1f} (Safety price)"
    elif recommendation == "SELL":
        entry_zone = "N/A (Avoid Buying)"
        stop_loss = f"₹{current_price * 1.05:.1f} (Safety price)"
    else: # HOLD
        entry_zone = "Hold current shares. Do not add new positions."
        stop_loss = f"₹{current_price * 0.95:.1f} (Safety price)"

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    
    # 2. Check if API key is missing or default placeholder
    if not api_key or "PLACEHOLDER" in api_key:
        print("[*] Gemini API key not found. Using local template explainer.")
        
        # Prepare a highly detailed local fallback explanation
        reasons = []
        if recommendation == "BUY":
            reasons.append(f"The stock price is in an uptrend (currently above its 50-day average price of ₹{technical_summary.get('MA50', current_price*0.95):.1f}).")
            reasons.append("Institutional players (Foreign and Domestic) are buying the stock, which gives strong capital backing.")
            reasons.append(f"Recent news sentiment is positive (Score: {sentiment_score:.2f}) with support from public media.")
        elif recommendation == "SELL":
            reasons.append("The stock shows technical weakness, trading below its moving average lines.")
            reasons.append("Institutional funds are withdrawing capital, indicating 'smart money' is leaving.")
            reasons.append("News sentiment is neutral to negative, creating overhead pressure.")
        else:
            reasons.append("Technical indicators (like RSI) are in the neutral 40-60 zone, indicating no clear direction.")
            reasons.append("FII and DII buying activity is mixed or diverging, showing indecision among big players.")
            reasons.append("No major news catalyst has occurred in the last 48 hours to drive the price.")
            
        fund_text = ""
        if fundamentals and fundamentals.get("PE_Ratio") is not None:
            fund_text = f"\nFundamentals: P/E ratio is {fundamentals['PE_Ratio']:.1f}, ROE is {fundamentals['ROE']:.1f}%, and Debt/Equity is {fundamentals['Debt_to_Equity']:.1f}%. "
            
        fallback_text = (
            f"Why {recommendation}:\n"
            f"1. {reasons[0]}\n"
            f"2. {reasons[1]}\n"
            f"3. {reasons[2]}\n"
            f"{fund_text}\n"
            f"Entry Zone: {entry_zone}\n"
            f"Safety Net Price: {stop_loss}\n\n"
            f"Recommendation: {recommendation} (Confidence: {confidence:.1f}%)\n"
            f"Disclaimer: Educational tool only. Always do your own research before trading."
        )
        return fallback_text

    # 3. Call Gemini API
    try:
        client = genai.Client(api_key=api_key)
        
        # Format fundamentals block
        fund_block = "N/A"
        if fundamentals:
            fund_block = (
                f"P/E Ratio: {fundamentals.get('PE_Ratio')}\n"
                f"P/B Ratio: {fundamentals.get('PB_Ratio')}\n"
                f"Debt to Equity Ratio: {fundamentals.get('Debt_to_Equity')}\n"
                f"ROE (%): {fundamentals.get('ROE')}\n"
                f"Market Cap (₹ Crores): {fundamentals.get('Market_Cap')}\n"
            )
            
        prompt = (
            f"You are a friendly stock market guide for complete beginners in India. "
            f"Your job: explain this stock analysis in PLAIN, SIMPLE English — like explaining to a friend who has never invested before. "
            f"RULES YOU MUST FOLLOW:\n"
            f"- NO markdown (no **, no *, no #, no -).\n"
            f"- NO finance jargon. Never say: bullish, bearish, MACD, Golden Cross, RSI, FII, DII, threshold, suppressed, overridden.\n"
            f"- HIGHLIGHT key words by writing them in CAPITAL LETTERS (example: PRICE IS GOING UP, DO NOT BUY, HOLD YOUR SHARES).\n"
            f"- Keep each point SHORT — maximum 3 sentences per point.\n"
            f"- Be direct. No long stories. No analogies unless they make it simpler in one line.\n"
            f"- Final answer should be crystal clear to someone who has never invested before.\n\n"

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

            f"--- WRITE RESPONSE IN THIS EXACT FORMAT ---\n"
            f"Our Recommendation:\n"
            f"[One sentence. State clearly what the person should do: BUY / HOLD / DO NOT BUY. "
            f"If signals were mixed, say 'signals were unclear' — do NOT mention SELL or any raw model output.]\n\n"
            f"Why:\n"
            f"1. Price Movement: [In 2-3 simple sentences, is the price going up or down recently? "
            f"Use CAPITAL LETTERS to highlight key facts like PRICE IS RISING or PRICE IS BELOW RECENT AVERAGE.]\n"
            f"2. Big Investors: [In 2 sentences, are big investors putting money in or taking money out? "
            f"Say 'foreign big investors' and 'Indian big investors' — not FII/DII.]\n"
            f"3. News: [In 1-2 sentences, what does recent news say about this company? "
            f"Mention the actual headline if relevant.]\n"
            f"4. Company Health: [In 2-3 sentences, is this a financially strong company? "
            f"Use simple terms: 'the company earns well for its size', 'the company carries high debt', etc. "
            f"HIGHLIGHT key verdict in CAPS like FINANCIALLY STRONG or HIGH DEBT RISK.]\n\n"
            f"What You Should Do:\n"
            f"Action: [state the entry zone or hold instruction clearly]\n"
            f"Safety Net Price: {stop_loss} — [one sentence explaining what this means in plain words]\n\n"
            f"Important:\n"
            f"[One sentence disclaimer, plain English, under 15 words.]"
        )

        
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                return clean_markdown(response.text)
            except Exception as ex:
                err_msg = str(ex)
                if "429" in err_msg or "quota" in err_msg.lower() or "limit" in err_msg.lower():
                    if attempt < max_retries - 1:
                        sleep_time = 15 * (attempt + 1)
                        print(f"\n[!] Gemini rate limit hit. Retrying in {sleep_time}s ({attempt+1}/{max_retries})...")
                        time.sleep(sleep_time)
                        continue
                raise ex
        
    except Exception as e:
        print(f"[Warning] Failed to generate explanation via Gemini API: {e}. Using fallback.")
        return f"Recommendation: {recommendation} (Confidence: {confidence:.1f}%)\n" \
               f"Entry Zone: {entry_zone}\n" \
               f"Safety Net Price: {stop_loss}\n" \
               f"Error generating LLM explanation. Please verify your internet connection and API keys."


def explain_educational_concept(user_input: str) -> str:
    """
    Uses Google Gemini to explain a general stock market concept or answer
    an educational finance question for beginners.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "PLACEHOLDER" in api_key:
        return (
            "I couldn't identify a specific stock in your query, and the Gemini API key is missing "
            "to answer general financial questions. Please ask about a supported stock (e.g., 'SBI') "
            "or set your GEMINI_API_KEY in the .env file."
        )

    prompt = (
        f"You are a friendly stock market guide for complete beginners in India.\n"
        f"The user asked an educational or general question: \"{user_input}\"\n\n"
        f"RULES YOU MUST FOLLOW:\n"
        f"- Explain the concept in PLAIN, SIMPLE English, like explaining to a friend with no finance background.\n"
        f"- NO markdown formatting (no **, no *, no #, no -).\n"
        f"- If the question is NOT related to finance, investing, business, economics, or the stock market, "
        f"politely remind them that you are an AI Stock Advisor and guide them back to stock market topics.\n"
        f"- HIGHLIGHT key terms by writing them in CAPITAL LETTERS.\n"
        f"- Keep the explanation SHORT and CRISP (maximum 5 sentences total).\n"
        f"- Be encouraging and friendly."
    )

    try:
        client = genai.Client(api_key=api_key)
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                return clean_markdown(response.text)
            except Exception as ex:
                err_msg = str(ex)
                if "429" in err_msg or "quota" in err_msg.lower() or "limit" in err_msg.lower():
                    if attempt < max_retries - 1:
                        sleep_time = 15 * (attempt + 1)
                        time.sleep(sleep_time)
                        continue
                raise ex
    except Exception as e:
        return f"Error generating explanation: {e}"


def generate_comparison_explanation(stocks: list[dict]) -> str:
    """
    Uses Google Gemini to compare two or more stocks side-by-side,
    providing a clear, beginner-friendly summary of which one has
    better indicators / fundamentals and why.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "PLACEHOLDER" in api_key:
        return "Comparison analysis completed. Configure GEMINI_API_KEY in the .env file to get a detailed AI comparison summary."

    # Format the data of each stock to put in the prompt
    stocks_prompt_data = ""
    for s in stocks:
        fund_block = "N/A"
        fund = s.get("fundamentals")
        if fund:
            fund_block = (
                f"P/E Ratio: {fund.get('PE_Ratio')}\n"
                f"P/B Ratio: {fund.get('PB_Ratio')}\n"
                f"Debt to Equity Ratio: {fund.get('Debt_to_Equity')}\n"
                f"ROE (%): {fund.get('ROE')}\n"
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
        f"- WRITE the entire AI Comparison Verdict line in CAPITAL LETTERS (example: RELIANCE INDUSTRIES LOOKS LIKE THE STRONGER PICK RIGHT NOW DUE TO ITS STABLE PROFITS AND LOWER DEBT).\n"
        f"- Keep it SHORT and CRISP (maximum 8 sentences total).\n"
        f"- Provide a clear takeaway comparison: which one is better for value/growth, or are both mixed?\n\n"
        f"--- STOCKS DATA ---\n"
        f"{stocks_prompt_data}"
        f"--- WRITE RESPONSE IN THIS EXACT FORMAT ---\n"
        f"AI Comparison Verdict:\n"
        f"[One sentence written entirely in CAPITAL LETTERS stating clearly which stock is currently the stronger pick or if they are equal/mixed.]\n\n"
        f"Key Differences:\n"
        f"1. Valuation & Profits: [Explain PE/ROE differences simply, highlight with CAPS]\n"
        f"2. Trend & Big Investors: [Explain who has better support from big institutions / price trend]\n\n"
        f"Bottom Line for Beginners:\n"
        f"[2 sentences of practical advice on what to watch for.]"
    )

    try:
        client = genai.Client(api_key=api_key)
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                return clean_markdown(response.text)
            except Exception as ex:
                err_msg = str(ex)
                if "429" in err_msg or "quota" in err_msg.lower() or "limit" in err_msg.lower():
                    if attempt < max_retries - 1:
                        sleep_time = 15 * (attempt + 1)
                        time.sleep(sleep_time)
                        continue
                raise ex
    except Exception as e:
        return f"Error generating comparison summary: {e}"

