import os
import re
import datetime
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def fetch_latest_fii_dii() -> pd.DataFrame:
    """
    Fetches the last 10 days of daily FII and DII net cash flow data.
    Strategy (3-tier):
      Tier 1 — NSE India API (real data for today, stored in rolling cache)
      Tier 2 — Moneycontrol scraper (backup multi-day source, used to backfill cache)
      Tier 3 — Simulation fallback (if both sources are unavailable)
    Returns: pd.DataFrame with columns ['Date', 'FII_Net', 'DII_Net']
    """
    # print("[*] Fetching latest FII/DII data...")
    cache_path = "data/fii_dii/live_fii_dii_cache.csv"
    os.makedirs("data/fii_dii", exist_ok=True)

    # 1. Try today's real FII/DII from NSE India API
    today_record = _fetch_nse_today()
    if today_record:
        _append_to_rolling_cache(today_record, cache_path)
        # print(f"[+] NSE India: Real FII/DII for {today_record['Date']} fetched.")

    # Load cache if it exists
    cache_df = pd.DataFrame(columns=["Date", "FII_Net", "DII_Net"])
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
        cache_df["Date"] = cache_df["Date"].astype(str)
        cache_df = cache_df.drop_duplicates(subset="Date").sort_values("Date")

    # If cache has fewer than 10 real days, try to backfill using Moneycontrol
    if len(cache_df) < 10:
        # print("[*] Cache has fewer than 10 days of real data. Backfilling via Moneycontrol...")
        mc_df = _fetch_moneycontrol()
        if mc_df is not None and len(mc_df) >= 3:
            mc_df["Date"] = mc_df["Date"].astype(str)
            # Combine Moneycontrol data and cache, prioritizing cached/NSE data
            combined = pd.concat([cache_df, mc_df], ignore_index=True)
            combined = combined.drop_duplicates(subset="Date").sort_values("Date")
            combined.to_csv(cache_path, index=False)
            cache_df = combined
            # print(f"[+] Moneycontrol: Successfully backfilled. Cache now has {len(cache_df)} real days.")
        else:
            print("[Warning] Moneycontrol backfill failed or returned empty.")

    # Now, check the size of cache again
    if len(cache_df) >= 1:
        if len(cache_df) < 10:
            # Only pad with simulation if we still have fewer than 10 days
            sim_df = _simulate_fii_dii(days=10 - len(cache_df))
            sim_df["Date"] = sim_df["Date"].astype(str)
            sim_df = sim_df[~sim_df["Date"].isin(cache_df["Date"])]
            cache_df = pd.concat([sim_df, cache_df], ignore_index=True).sort_values("Date")
            print(f"[+] FII/DII: {len(cache_df) - len(sim_df)} real day(s) + {len(sim_df)} simulated day(s).")
        else:
            print(f"[+] FII/DII: 100% real data loaded (10/10 days from NSE/Moneycontrol).")
        return cache_df.tail(10).reset_index(drop=True)

    # 3. Last-resort fallback simulation
    print("[*] All real sources unavailable. Using simulated FII/DII data.")
    return _simulate_fii_dii(days=10)



def _fetch_nse_today() -> dict | None:
    """
    Calls NSE India API to get today's real FII/DII net flows.
    Returns a dict with Date, FII_Net, DII_Net or None on failure.
    """
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept": "application/json, text/plain, */*",
        }
        # Must visit homepage first to get valid session cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        api_headers = {**headers, "Referer": "https://www.nseindia.com/"}
        resp = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers=api_headers, timeout=8
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        fii_net, dii_net, date_str = None, None, None

        for record in data:
            cat = record.get("category", "").upper()
            net = record.get("netValue", "0").replace(",", "")
            dt  = record.get("date", "")
            try:
                net_val = float(net)
                parsed_date = datetime.datetime.strptime(dt, "%d-%b-%Y").date()
            except Exception:
                continue

            if "FII" in cat or "FPI" in cat:
                fii_net  = net_val
                date_str = parsed_date
            elif "DII" in cat:
                dii_net = net_val

        if fii_net is not None and dii_net is not None and date_str:
            return {"Date": date_str, "FII_Net": fii_net, "DII_Net": dii_net}
    except Exception as e:
        print(f"[Warning] NSE API error: {e}")
    return None


def _append_to_rolling_cache(record: dict, cache_path: str):
    """
    Appends one day's FII/DII record to the local rolling cache CSV.
    Keeps only the last 30 days. Avoids duplicate dates.
    """
    # Normalize date to string for consistent comparison
    record = dict(record)
    record["Date"] = str(record["Date"])
    new_row = pd.DataFrame([record])
    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path)
        existing["Date"] = existing["Date"].astype(str)
        combined = pd.concat([existing, new_row], ignore_index=True)
        combined = combined.drop_duplicates(subset="Date").sort_values("Date")
        combined = combined.tail(30)
    else:
        combined = new_row
    combined.to_csv(cache_path, index=False)


def _fetch_moneycontrol() -> pd.DataFrame | None:
    """
    Scrapes Moneycontrol FII/DII activity page as a backup data source.
    Extracts structured JSON from the __NEXT_DATA__ script tag.
    Returns DataFrame with Date, FII_Net, DII_Net or None on failure.
    """
    url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        import json
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        html = resp.text
        
        # Search for __NEXT_DATA__ block
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
        if not match:
            return None
            
        data = json.loads(match.group(1))
        fii_dii_list = data.get("props", {}).get("pageProps", {}).get("FiiDiiData", {}).get("fiiDiiData", [])
        
        parsed = []
        for item in fii_dii_list:
            dt_str = item.get("date")
            fii_val = item.get("fiiCM")
            dii_val = item.get("diiCM")
            
            if dt_str and fii_val and dii_val:
                try:
                    # Parse date: format is YYYY-MM-DD
                    parsed_date = datetime.datetime.strptime(dt_str, "%Y-%m-%d").date()
                    # Clean commas and parse float
                    fii_net = float(fii_val.replace(",", ""))
                    dii_net = float(dii_val.replace(",", ""))
                    parsed.append({"Date": parsed_date, "FII_Net": fii_net, "DII_Net": dii_net})
                except Exception:
                    continue
                    
        if parsed:
            return pd.DataFrame(parsed).sort_values("Date").reset_index(drop=True)
    except Exception as e:
        print(f"[Warning] Moneycontrol JSON parser error: {e}")
    return None



def _simulate_fii_dii(days: int = 10) -> pd.DataFrame:
    """
    Generates realistic simulated FII/DII data as last-resort fallback.
    Seed changes daily so values feel fresh each day.
    """
    today = datetime.date.today()
    dates = []
    curr = today
    while len(dates) < days:
        if curr.weekday() < 5:
            dates.append(curr)
        curr -= datetime.timedelta(days=1)
    dates.reverse()
    np.random.seed(int(today.strftime("%Y%m%d")) % 1000)
    fii_flows = np.random.normal(loc=200,  scale=1500, size=days)
    dii_flows = np.random.normal(loc=1200, scale=800,  size=days)
    df = pd.DataFrame({"Date": dates, "FII_Net": fii_flows, "DII_Net": dii_flows})
    print(f"[+] Created {days} days of simulated FII/DII data (fallback).")
    return df



def generate_historical_fii_dii_csv(start_date: str, end_date: str, output_path: str = None) -> str:
    """
    Generates historical FII/DII net flow daily data for training.
    Always regenerates to match the requested date range.
    Saves to: data/fii_dii/historical_fii_dii.csv (default) or output_path if provided.

    FIX (Issue #2 — Neutral Random FII/DII):
    Previously, flows were correlated with a simulated market return proxy
    (positive market day → FII buys, negative → FII sells). This created
    circular leakage similar to Issue #1 — the model learned market direction
    disguised as institutional flow signal.

    Now: FII and DII flows are sampled independently from realistic
    distributions based on actual NSE historical statistics, with NO
    correlation to any price or market direction signal.

    Distributions used (based on observed NSE FII/DII patterns):
      FII_Net : Normal(mean=200,  std=1500) Cr/day  — net buyer on average, high variance
      DII_Net : Normal(mean=800,  std=700)  Cr/day  — consistent domestic buying, lower variance
    These are statistically plausible but contain no directional market signal.

    Args:
        start_date:  Start of date range (YYYY-MM-DD)
        end_date:    End of date range (YYYY-MM-DD)
        output_path: Optional custom path to save CSV. Defaults to
                     data/fii_dii/historical_fii_dii.csv (training file).
                     Pass a different path (e.g. for backtest windows) to avoid
                     overwriting the main training FII/DII file.
    """
    os.makedirs("data/fii_dii", exist_ok=True)
    csv_path = output_path if output_path else "data/fii_dii/historical_fii_dii.csv"

    print(f"[*] Generating historical FII/DII dataset ({start_date} to {end_date})...")

    # Create date range (business days only)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(dates)

    # Neutral random flows — no correlation to market direction
    np.random.seed(42)
    fii_net = np.random.normal(loc=200,  scale=1500, size=n)   # FII: net buyer, high volatility
    dii_net = np.random.normal(loc=800,  scale=700,  size=n)   # DII: consistent buyer, lower vol

    df = pd.DataFrame({
        "Date":    dates.date,
        "FII_Net": fii_net,
        "DII_Net": dii_net
    })

    df.to_csv(csv_path, index=False)
    print(f"[+] Saved {len(df)} rows of neutral FII/DII training data to {csv_path}")
    return csv_path



def analyze_institutional_signals(fii_10d: list[float], dii_10d: list[float]) -> dict:
    """
    Processes FII/DII flow arrays to compute features:
    - 10-day Net cumulative flow
    - Trend direction (1 for increasing/buying, 0 for decreasing/selling)
    - Divergence Flag (1 if both buying/selling, 0 if opposite)
    - Plain English label for explanation
    """
    fii_sum = sum(fii_10d)
    dii_sum = sum(dii_10d)
    
    # Trend: calculate linear slope or simple difference between last 3 days and first 3 days of the 10-day period
    fii_trend = 1 if (fii_10d[-1] + fii_10d[-2]) > (fii_10d[0] + fii_10d[1]) else 0
    dii_trend = 1 if (dii_10d[-1] + dii_10d[-2]) > (dii_10d[0] + dii_10d[1]) else 0
    
    # Divergence pattern classification:
    # 1. Both buying heavily
    if fii_sum > 0 and dii_sum > 0:
        signal = "Strong Buying Support"
        divergence_flag = 0
    # 2. Both selling heavily
    elif fii_sum < 0 and dii_sum < 0:
        signal = "Strong Selling Pressure"
        divergence_flag = 0
    # 3. FII selling, DII buying (Divergence / stabilizing)
    elif fii_sum < 0 < dii_sum:
        signal = "FII Selling, DII Buying (Market stabilized by domestic funds)"
        divergence_flag = 1
    # 4. FII buying, DII selling
    else:
        signal = "FII Buying, DII Profit Booking"
        divergence_flag = 1
        
    return {
        "FII_10d_Net": fii_sum,
        "DII_10d_Net": dii_sum,
        "FII_Trend": fii_trend,
        "DII_Trend": dii_trend,
        "Divergence_Flag": divergence_flag,
        "Description": signal
    }

if __name__ == "__main__":
    # Test scraping / mock
    df = fetch_latest_fii_dii()
    print(df)
    
    # Test historical generator
    csv = generate_historical_fii_dii_csv("2024-06-01", "2026-06-01")
    hist_df = pd.read_csv(csv)
    print(f"Historical head:\n{hist_df.head(2)}")
