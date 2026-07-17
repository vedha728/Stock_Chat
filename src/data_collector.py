import os
import requests
import pandas as pd
import datetime
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Dictionary of Nifty 50 + important Indian stocks for instant mapping
# Multiple aliases per stock so users can type anything and get matched
POPULAR_STOCKS = {
    # ── Tata Group ────────────────────────────────────────────────────────────
    "tmpv":                      "TMPV.NS",
    "tata motors passenger vehicles": "TMPV.NS",
    "tata pv":                   "TMPV.NS",
    "tmcv":                      "TMCV.NS",
    "tata motors commercial vehicles": "TMCV.NS",
    "tata cv":                   "TMCV.NS",
    "tata consultancy":     "TCS.NS",
    "tata consultancy services": "TCS.NS",
    "tcs":                  "TCS.NS",
    "tata steel":           "TATASTEEL.NS",
    "tatasteel":            "TATASTEEL.NS",
    "tata power":           "TATAPOWER.NS",
    "tata chemicals":       "TATACHEM.NS",
    "tata consumer":        "TATACONSUM.NS",
    "tata elxsi":           "TATAELXSI.NS",
    "tata communications":  "TATACOMM.NS",
    # ── IT Sector ────────────────────────────────────────────────────────────
    "infosys":              "INFY.NS",
    "infy":                 "INFY.NS",
    "wipro":                "WIPRO.NS",
    "tech mahindra":        "TECHM.NS",
    "techmahindra":         "TECHM.NS",
    "tech mah":             "TECHM.NS",
    "hcl":                  "HCLTECH.NS",
    "hcl tech":             "HCLTECH.NS",
    "hcl technologies":     "HCLTECH.NS",
    "ltimindtree":          "LTM.NS",
    "lti mindtree":         "LTM.NS",
    "lti":                  "LTM.NS",
    "mphasis":              "MPHASIS.NS",
    "persistent":           "PERSISTENT.NS",
    "persistent systems":   "PERSISTENT.NS",
    "coforge":              "COFORGE.NS",
    # ── Banking ──────────────────────────────────────────────────────────────
    "hdfc":                 "HDFCBANK.NS",
    "hdfc bank":            "HDFCBANK.NS",
    "hdfcbank":             "HDFCBANK.NS",
    "icici":                "ICICIBANK.NS",
    "icici bank":           "ICICIBANK.NS",
    "icicibank":            "ICICIBANK.NS",
    "sbi":                  "SBIN.NS",
    "state bank":           "SBIN.NS",
    "state bank of india":  "SBIN.NS",
    "axis bank":            "AXISBANK.NS",
    "axis":                 "AXISBANK.NS",
    "axisbank":             "AXISBANK.NS",
    "kotak":                "KOTAKBANK.NS",
    "kotak mahindra":       "KOTAKBANK.NS",
    "kotak bank":           "KOTAKBANK.NS",
    "kotakbank":            "KOTAKBANK.NS",
    "bank of baroda":       "BANKBARODA.NS",
    "bob":                  "BANKBARODA.NS",
    "punjab national bank": "PNB.NS",
    "pnb":                  "PNB.NS",
    "indusind":             "INDUSINDBK.NS",
    "indusind bank":        "INDUSINDBK.NS",
    "federal bank":         "FEDERALBNK.NS",
    "yes bank":             "YESBANK.NS",
    "canara bank":          "CANBK.NS",
    "canara":               "CANBK.NS",
    "union bank":           "UNIONBANK.NS",
    "iob":                  "IOB.NS",
    "indian overseas bank": "IOB.NS",
    # ── Finance / NBFC ───────────────────────────────────────────────────────
    "bajaj finance":        "BAJFINANCE.NS",
    "bajajfinance":         "BAJFINANCE.NS",
    "bajaj fin":            "BAJFINANCE.NS",
    "bajaj finserv":        "BAJAJFINSV.NS",
    "bajajfinserv":         "BAJAJFINSV.NS",
    "shriram finance":      "SHRIRAMFIN.NS",
    "shriram":              "SHRIRAMFIN.NS",
    "muthoot finance":      "MUTHOOTFIN.NS",
    "muthoot":              "MUTHOOTFIN.NS",
    "cholafin":             "CHOLAFIN.NS",
    "chola":                "CHOLAFIN.NS",
    "cholamandalam":        "CHOLAFIN.NS",
    "pfc":                  "PFC.NS",
    "power finance":        "PFC.NS",
    "power finance corporation": "PFC.NS",
    "rec":                  "RECLTD.NS",
    "rec limited":          "RECLTD.NS",
    # ── Reliance / Oil & Gas ─────────────────────────────────────────────────
    "reliance":             "RELIANCE.NS",
    "reliance industries":  "RELIANCE.NS",
    "ril":                  "RELIANCE.NS",
    "ongc":                 "ONGC.NS",
    "oil and natural gas":  "ONGC.NS",
    "oil india":            "OIL.NS",
    "bpcl":                 "BPCL.NS",
    "bharat petroleum":     "BPCL.NS",
    "ioc":                  "IOC.NS",
    "indian oil":           "IOC.NS",
    "hpcl":                 "HINDPETRO.NS",
    "hindustan petroleum":  "HINDPETRO.NS",
    "gail":                 "GAIL.NS",
    "petronet":             "PETRONET.NS",
    "petronet lng":         "PETRONET.NS",
    # ── Auto ─────────────────────────────────────────────────────────────────
    "maruti":               "MARUTI.NS",
    "maruti suzuki":        "MARUTI.NS",
    "msil":                 "MARUTI.NS",
    "mahindra":             "M&M.NS",
    "m&m":                  "M&M.NS",
    "mahindra and mahindra":"M&M.NS",
    "m and m":              "M&M.NS",
    "hero motocorp":        "HEROMOTOCO.NS",
    "hero":                 "HEROMOTOCO.NS",
    "hero moto":            "HEROMOTOCO.NS",
    "bajaj auto":           "BAJAJ-AUTO.NS",
    "bajaj":                "BAJAJ-AUTO.NS",
    "eicher":               "EICHERMOT.NS",
    "eicher motors":        "EICHERMOT.NS",
    "royal enfield":        "EICHERMOT.NS",
    "ashok leyland":        "ASHOKLEY.NS",
    "ashok":                "ASHOKLEY.NS",
    "tvs motor":            "TVSMOTOR.NS",
    "tvs":                  "TVSMOTOR.NS",
    "bosch":                "BOSCHLTD.NS",
    "motherson":            "MOTHERSON.NS",
    # ── FMCG ─────────────────────────────────────────────────────────────────
    "hul":                  "HINDUNILVR.NS",
    "hindustan unilever":   "HINDUNILVR.NS",
    "unilever":             "HINDUNILVR.NS",
    "nestle":               "NESTLEIND.NS",
    "nestle india":         "NESTLEIND.NS",
    "itc":                  "ITC.NS",
    "britannia":            "BRITANNIA.NS",
    "dabur":                "DABUR.NS",
    "marico":               "MARICO.NS",
    "emami":                "EMAMILTD.NS",
    "godrej consumer":      "GODREJCP.NS",
    "godrej":               "GODREJCP.NS",
    "colgate":              "COLPAL.NS",
    "colgate palmolive":    "COLPAL.NS",
    "procter gamble":       "PGHH.NS",
    "p&g":                  "PGHH.NS",
    # ── Pharma / Healthcare ──────────────────────────────────────────────────
    "sun pharma":           "SUNPHARMA.NS",
    "sunpharma":            "SUNPHARMA.NS",
    "sun pharmaceutical":   "SUNPHARMA.NS",
    "sun pharmacy":         "SUNPHARMA.NS",
    "dr reddy":             "DRREDDY.NS",
    "dr reddys":            "DRREDDY.NS",
    "dr. reddy":            "DRREDDY.NS",
    "cipla":                "CIPLA.NS",
    "divis":                "DIVISLAB.NS",
    "divi":                 "DIVISLAB.NS",
    "divi labs":            "DIVISLAB.NS",
    "divi laboratory":      "DIVISLAB.NS",
    "aurobindo":            "AUROPHARMA.NS",
    "aurobindo pharma":     "AUROPHARMA.NS",
    "lupin":                "LUPIN.NS",
    "torrent pharma":       "TORNTPHARM.NS",
    "torrent":              "TORNTPHARM.NS",
    "apollo hospitals":     "APOLLOHOSP.NS",
    "apollo":               "APOLLOHOSP.NS",
    "max healthcare":       "MAXHEALTH.NS",
    "fortis":               "FORTIS.NS",
    # ── Infrastructure / Capital Goods ───────────────────────────────────────
    "l&t":                  "LT.NS",
    "l & t":                "LT.NS",
    "larsen":               "LT.NS",
    "larsen toubro":        "LT.NS",
    "larsen and toubro":    "LT.NS",
    "larsen & toubro":      "LT.NS",
    "ntpc":                 "NTPC.NS",
    "power grid":           "POWERGRID.NS",
    "powergrid":            "POWERGRID.NS",
    "bhel":                 "BHEL.NS",
    "siemens":              "SIEMENS.NS",
    "abb":                  "ABB.NS",
    "abb india":            "ABB.NS",
    "havells":              "HAVELLS.NS",
    "cg power":             "CGPOWER.NS",
    "bharat electronics":   "BEL.NS",
    "bel":                  "BEL.NS",
    "irfc":                 "IRFC.NS",
    "indian railway finance": "IRFC.NS",
    "irb":                  "IRB.NS",
    "gmr":                  "GMRAIRPORT.NS",
    # ── Adani Group ──────────────────────────────────────────────────────────
    "adani port":           "ADANIPORTS.NS",   # singular
    "adani ports":          "ADANIPORTS.NS",   # plural
    "adaniports":           "ADANIPORTS.NS",
    "adani port sez":       "ADANIPORTS.NS",
    "adani enterprises":    "ADANIENT.NS",
    "adani enterprise":     "ADANIENT.NS",
    "adanient":             "ADANIENT.NS",
    "adani green":          "ADANIGREEN.NS",
    "adani green energy":   "ADANIGREEN.NS",
    "adani power":          "ADANIPOWER.NS",
    "adani total gas":      "ATGL.NS",
    "adani gas":            "ATGL.NS",
    "adani wilmar":         "AWL.NS",
    "ndtv":                 "NDTV.NS",
    "ambuja cement":        "AMBUJACEM.NS",
    "ambuja":               "AMBUJACEM.NS",
    "acc":                  "ACC.NS",
    "acc cement":           "ACC.NS",
    # ── Metals / Mining ──────────────────────────────────────────────────────
    "jsw steel":            "JSWSTEEL.NS",
    "jsw":                  "JSWSTEEL.NS",
    "hindalco":             "HINDALCO.NS",
    "vedanta":              "VEDL.NS",
    "vedl":                 "VEDL.NS",
    "coal india":           "COALINDIA.NS",
    "coalindia":            "COALINDIA.NS",
    "sail":                 "SAIL.NS",
    "steel authority":      "SAIL.NS",
    "nmdc":                 "NMDC.NS",
    "national mineral":     "NMDC.NS",
    "hindustan zinc":       "HINDZINC.NS",
    "hindzinc":             "HINDZINC.NS",
    "nalco":                "NATIONALUM.NS",
    # ── Consumer / Retail ────────────────────────────────────────────────────
    "asian paints":         "ASIANPAINT.NS",
    "asianpaints":          "ASIANPAINT.NS",
    "asian paint":          "ASIANPAINT.NS",
    "titan":                "TITAN.NS",
    "titan company":        "TITAN.NS",
    "tanishq":              "TITAN.NS",
    "dmart":                "DMART.NS",
    "avenue supermarts":    "DMART.NS",
    "avenue supermart":     "DMART.NS",
    "trent":                "TRENT.NS",
    "westside":             "TRENT.NS",
    "v-mart":               "VMART.NS",
    "vmart":                "VMART.NS",
    "page industries":      "PAGEIND.NS",
    "page":                 "PAGEIND.NS",
    "jockey":               "PAGEIND.NS",
    "abfrl":                "ABFRL.NS",
    "aditya birla fashion": "ABFRL.NS",
    "mrf":                  "MRF.NS",
    "mrf tyres":            "MRF.NS",
    "apollo tyres":         "APOLLOTYRE.NS",
    "apollo tyre":          "APOLLOTYRE.NS",
    # ── Telecom ──────────────────────────────────────────────────────────────
    "bharti airtel":        "BHARTIARTL.NS",
    "airtel":               "BHARTIARTL.NS",
    "bharti":               "BHARTIARTL.NS",
    "bhartiartl":           "BHARTIARTL.NS",
    "vodafone idea":        "IDEA.NS",
    "vi":                   "IDEA.NS",
    "idea":                 "IDEA.NS",
    # ── New Age / Tech ───────────────────────────────────────────────────────
    "zomato":               "ETERNAL.NS",
    "eternal":              "ETERNAL.NS",
    "zomato eternal":       "ETERNAL.NS",
    "nykaa":                "NYKAA.NS",
    "fss nykaa":            "NYKAA.NS",
    "paytm":                "PAYTM.NS",
    "one97":                "PAYTM.NS",
    "policybazaar":         "POLICYBZR.NS",
    "policy bazaar":        "POLICYBZR.NS",
    "delhivery":            "DELHIVERY.NS",
    "swiggy":               "SWIGGY.NS",
    "ola electric":         "OLAELEC.NS",
    "ola":                  "OLAELEC.NS",
    # ── Real Estate ──────────────────────────────────────────────────────────
    "dlf":                  "DLF.NS",
    "godrej properties":    "GODREJPROP.NS",
    "godrej prop":          "GODREJPROP.NS",
    "prestige":             "PRESTIGE.NS",
    "prestige estates":     "PRESTIGE.NS",
    "oberoi realty":        "OBEROIRLTY.NS",
    "oberoi":               "OBEROIRLTY.NS",
    # ── Cement ───────────────────────────────────────────────────────────────
    "ultratech":            "ULTRACEMCO.NS",
    "ultratech cement":     "ULTRACEMCO.NS",
    "shree cement":         "SHREECEM.NS",
    "shree":                "SHREECEM.NS",
    "jk cement":            "JKCEMENT.NS",
    "dalmia":               "DALBHARAT.NS",
    "dalmia bharat":        "DALBHARAT.NS",
    # ── Insurance ────────────────────────────────────────────────────────────
    "sbi life":             "SBILIFE.NS",
    "hdfc life":            "HDFCLIFE.NS",
    "icici prudential":     "ICICIPRULI.NS",
    "icici pru":            "ICICIPRULI.NS",
    "lic":                  "LICI.NS",
    "life insurance corporation": "LICI.NS",
    "star health":          "STARHEALTH.NS",
    "new india assurance":  "NIACL.NS",
    # ── Exchanges / AMC ──────────────────────────────────────────────────────
    "nse":                  "NSEI",         # index, not tradeable but common query
    "bse":                  "BSE.NS",
    "bombay stock exchange":"BSE.NS",
    "cams":                 "CAMS.NS",
    "kfintech":             "KFINTECH.NS",
    "cdsl":                 "CDSL.NS",
    "nsdl":                 "NSDL.NS",
    "hdfc amc":             "HDFCAMC.NS",
    "nippon amc":           "NAM-INDIA.NS",
    "nippon":               "NAM-INDIA.NS",
    # ── Miscellaneous Nifty 50 ───────────────────────────────────────────────
    "upl":                  "UPL.NS",
    "upl limited":          "UPL.NS",
    "srf":                  "SRF.NS",
    "pidilite":             "PIDILITIND.NS",
    "fevicol":              "PIDILITIND.NS",
    "info edge":            "NAUKRI.NS",
    "naukri":               "NAUKRI.NS",
    "indigo":               "INDIGO.NS",
    "interglobe":           "INDIGO.NS",
    "interglobe aviation":  "INDIGO.NS",
    "spicejet":             "SPICEJET.NS",
    "irctc":                "IRCTC.NS",
    "indian railway catering": "IRCTC.NS",
    "container corporation": "CONCOR.NS",
    "concor":               "CONCOR.NS",
    "motherson sumi":       "MOTHERSON.NS",
    "dixon":                "DIXON.NS",
    "dixon technologies":   "DIXON.NS",
    "kaynes":               "KAYNES.NS",
    "amber":                "AMBER.NS",
    "amber enterprises":    "AMBER.NS",
}

# Reverse map: ticker → friendly display name
TICKER_NAME_MAP = {
    "TMPV.NS":       "Tata Motors PV",
    "TMCV.NS":       "Tata Motors CV",
    "TCS.NS":        "TCS",
    "INFY.NS":       "Infosys",
    "WIPRO.NS":      "Wipro",
    "TECHM.NS":      "Tech Mahindra",
    "HCLTECH.NS":    "HCL Technologies",
    "LTM.NS":            "LTIMindtree",
    "MPHASIS.NS":    "Mphasis",
    "PERSISTENT.NS": "Persistent Systems",
    "COFORGE.NS":    "Coforge",
    "HDFCBANK.NS":   "HDFC Bank",
    "ICICIBANK.NS":  "ICICI Bank",
    "SBIN.NS":       "State Bank of India",
    "AXISBANK.NS":   "Axis Bank",
    "KOTAKBANK.NS":  "Kotak Mahindra Bank",
    "BANKBARODA.NS": "Bank of Baroda",
    "PNB.NS":        "Punjab National Bank",
    "INDUSINDBK.NS": "IndusInd Bank",
    "FEDERALBNK.NS": "Federal Bank",
    "YESBANK.NS":    "Yes Bank",
    "CANBK.NS":      "Canara Bank",
    "UNIONBANK.NS":  "Union Bank",
    "IOB.NS":        "Indian Overseas Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "MUTHOOTFIN.NS": "Muthoot Finance",
    "CHOLAFIN.NS":   "Cholamandalam Finance",
    "PFC.NS":        "Power Finance Corporation",
    "RECLTD.NS":     "REC Limited",
    "RELIANCE.NS":   "Reliance Industries",
    "ONGC.NS":       "ONGC",
    "OIL.NS":        "Oil India",
    "BPCL.NS":       "BPCL",
    "IOC.NS":        "Indian Oil",
    "HINDPETRO.NS":  "Hindustan Petroleum (HPCL)",
    "GAIL.NS":       "GAIL India",
    "PETRONET.NS":   "Petronet LNG",
    "MARUTI.NS":     "Maruti Suzuki",
    "M&M.NS":        "Mahindra & Mahindra",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "EICHERMOT.NS":  "Eicher Motors",
    "ASHOKLEY.NS":   "Ashok Leyland",
    "TVSMOTOR.NS":   "TVS Motor",
    "BOSCHLTD.NS":   "Bosch",
    "MOTHERSON.NS":  "Motherson Sumi",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "NESTLEIND.NS":  "Nestle India",
    "ITC.NS":        "ITC",
    "BRITANNIA.NS":  "Britannia",
    "DABUR.NS":      "Dabur",
    "MARICO.NS":     "Marico",
    "EMAMILTD.NS":   "Emami",
    "GODREJCP.NS":   "Godrej Consumer",
    "COLPAL.NS":     "Colgate-Palmolive India",
    "PGHH.NS":       "Procter & Gamble India",
    "SUNPHARMA.NS":  "Sun Pharma",
    "DRREDDY.NS":    "Dr. Reddy's",
    "CIPLA.NS":      "Cipla",
    "DIVISLAB.NS":   "Divi's Laboratories",
    "AUROPHARMA.NS": "Aurobindo Pharma",
    "LUPIN.NS":      "Lupin",
    "TORNTPHARM.NS": "Torrent Pharma",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "MAXHEALTH.NS":  "Max Healthcare",
    "FORTIS.NS":     "Fortis Healthcare",
    "LT.NS":         "Larsen & Toubro",
    "NTPC.NS":       "NTPC",
    "POWERGRID.NS":  "Power Grid",
    "BHEL.NS":       "BHEL",
    "SIEMENS.NS":    "Siemens India",
    "ABB.NS":        "ABB India",
    "HAVELLS.NS":    "Havells",
    "CGPOWER.NS":    "CG Power",
    "BEL.NS":        "Bharat Electronics",
    "IRFC.NS":       "IRFC",
    "IRB.NS":        "IRB Infrastructure",
    "GMRAIRPORT.NS":  "GMR Airports Infrastructure",
    "ADANIPORTS.NS": "Adani Ports & SEZ",
    "ADANIENT.NS":   "Adani Enterprises",
    "ADANIGREEN.NS": "Adani Green Energy",
    "ADANIPOWER.NS": "Adani Power",
    "ATGL.NS":       "Adani Total Gas",
    "AWL.NS":        "Adani Wilmar",
    "NDTV.NS":       "NDTV",
    "AMBUJACEM.NS":  "Ambuja Cements",
    "ACC.NS":        "ACC",
    "JSWSTEEL.NS":   "JSW Steel",
    "HINDALCO.NS":   "Hindalco",
    "VEDL.NS":       "Vedanta",
    "COALINDIA.NS":  "Coal India",
    "SAIL.NS":       "SAIL",
    "NMDC.NS":       "NMDC",
    "HINDZINC.NS":   "Hindustan Zinc",
    "NATIONALUM.NS": "NALCO",
    "TATASTEEL.NS":  "Tata Steel",
    "TATAPOWER.NS":  "Tata Power",
    "TATACHEM.NS":   "Tata Chemicals",
    "TATACONSUM.NS": "Tata Consumer Products",
    "TATAELXSI.NS":  "Tata Elxsi",
    "TATACOMM.NS":   "Tata Communications",
    "ASIANPAINT.NS": "Asian Paints",
    "TITAN.NS":      "Titan",
    "DMART.NS":      "DMart",
    "TRENT.NS":      "Trent",
    "VMART.NS":      "V-Mart Retail",
    "PAGEIND.NS":    "Page Industries",
    "ABFRL.NS":      "Aditya Birla Fashion",
    "MRF.NS":        "MRF",
    "APOLLOTYRE.NS": "Apollo Tyres",
    "BHARTIARTL.NS": "Bharti Airtel",
    "IDEA.NS":       "Vodafone Idea",
    "ETERNAL.NS":    "Zomato (Eternal)",
    "NYKAA.NS":      "Nykaa",
    "PAYTM.NS":      "Paytm",
    "POLICYBZR.NS":  "PolicyBazaar",
    "DELHIVERY.NS":  "Delhivery",
    "SWIGGY.NS":     "Swiggy",
    "OLAELEC.NS":    "Ola Electric",
    "DLF.NS":        "DLF",
    "GODREJPROP.NS": "Godrej Properties",
    "PRESTIGE.NS":   "Prestige Estates",
    "OBEROIRLTY.NS": "Oberoi Realty",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "SHREECEM.NS":   "Shree Cement",
    "JKCEMENT.NS":   "JK Cement",
    "DALBHARAT.NS":  "Dalmia Bharat",
    "SBILIFE.NS":    "SBI Life Insurance",
    "HDFCLIFE.NS":   "HDFC Life",
    "ICICIPRULI.NS": "ICICI Prudential Life",
    "LICI.NS":       "LIC",
    "STARHEALTH.NS": "Star Health Insurance",
    "NIACL.NS":      "New India Assurance",
    "BSE.NS":        "BSE Limited",
    "CAMS.NS":       "CAMS",
    "KFINTECH.NS":   "KFintech",
    "CDSL.NS":       "CDSL",
    "HDFCAMC.NS":    "HDFC AMC",
    "NAM-INDIA.NS":  "Nippon India AMC",
    "UPL.NS":        "UPL",
    "SRF.NS":        "SRF",
    "PIDILITIND.NS": "Pidilite Industries",
    "NAUKRI.NS":     "Info Edge (Naukri)",
    "INDIGO.NS":     "IndiGo (InterGlobe)",
    "SPICEJET.NS":   "SpiceJet",
    "IRCTC.NS":      "IRCTC",
    "CONCOR.NS":     "Container Corporation",
    "DIXON.NS":      "Dixon Technologies",
    "KAYNES.NS":     "Kaynes Technology",
    "AMBER.NS":      "Amber Enterprises",
}

def extract_ticker(query: str) -> tuple[str, str]:
    """
    Extracts the NSE stock ticker and company name from a user query.
    Strategy:
      1. Check POPULAR_STOCKS dictionary (fast, handles aliases)
      2. Yahoo Finance search API (dynamic fallback for any other stock)
    Returns: (ticker, company_name)
    """
    query_clean = query.lower().strip()

    # Remove common filler words so "adani port stock" → "adani port"
    # Note: do NOT strip "india" as it's part of names like "coal india", "oil india"
    for filler in [" stock", " share", " shares", " equity", " nse", " bse",
                   " ltd", " limited", " today", " price"]:
        query_clean = query_clean.replace(filler, "")
    query_clean = query_clean.strip()

    # 1. Exact or substring match in POPULAR_STOCKS
    # Sort by key length (longest first) so "bajaj finance" beats "bajaj"
    for key in sorted(POPULAR_STOCKS.keys(), key=len, reverse=True):
        if key in query_clean:
            ticker = POPULAR_STOCKS[key]
            name   = TICKER_NAME_MAP.get(ticker, ticker)
            return ticker, name

    # 1.5 Local Fuzzy Match Fallback
    import difflib
    # Try matching full cleaned query
    matches = difflib.get_close_matches(query_clean, POPULAR_STOCKS.keys(), n=1, cutoff=0.7)
    if matches:
        matched_key = matches[0]
        ticker = POPULAR_STOCKS[matched_key]
        name   = TICKER_NAME_MAP.get(ticker, ticker)
        print(f"[*] Stock '{query_clean}' not found exactly. Auto-correcting to {name} ({ticker})...")
        return ticker, name

    # Try matching individual words
    words = query_clean.split()
    for word in words:
        if len(word) < 3:
            continue
        matches = difflib.get_close_matches(word, POPULAR_STOCKS.keys(), n=1, cutoff=0.8)
        if matches:
            matched_key = matches[0]
            ticker = POPULAR_STOCKS[matched_key]
            name   = TICKER_NAME_MAP.get(ticker, ticker)
            print(f"[*] Stock '{word}' not found exactly. Auto-correcting to {name} ({ticker})...")
            return ticker, name

    # 2. Dynamic Yahoo Finance search fallback
    try:
        from urllib.parse import quote_plus
        url = (
            f"https://query2.finance.yahoo.com/v1/finance/search"
            f"?q={quote_plus(query_clean)}&newsCount=0&listsCount=0"
        )
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            quotes = response.json().get("quotes", [])
            # Prefer .NS (NSE) over .BO (BSE) over anything else
            for preferred_suffix in [".NS", ".BO"]:
                for quote in quotes:
                    symbol = quote.get("symbol", "")
                    if symbol.endswith(preferred_suffix):
                        name = quote.get("shortname") or quote.get("longname") or symbol
                        return symbol, name
    except Exception as e:
        print(f"[Warning] Dynamic ticker search failed: {e}")

    return None, None


def extract_multiple_tickers(query: str) -> list[tuple[str, str]]:
    """
    Parses the query and extracts multiple stock tickers.
    Supports splitters like 'vs', 'and', 'compare', 'or', and commas.
    Returns list of unique (ticker, company_name) tuples.
    """
    query_lower = query.lower()
    
    # Check if this is a comparison query
    is_comparison = any(w in query_lower for w in [" vs ", " vs. ", " and ", " compare ", " compared to ", " or ", ","])
    
    if not is_comparison:
        t, n = extract_ticker(query)
        if t:
            return [(t, n)]
        return []
        
    import re
    # Split on splitters: vs, vs., and, compare, compared to, or, commas
    parts = re.split(r'\bvs\b|\bvs\.\b|\band\b|\bcompare\b|\bcompared to\b|\bor\b|Layout\b|,', query_lower)
    
    tickers = []
    seen = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Run standard ticker extraction on the part
        t, n = extract_ticker(part)
        if t and t not in seen:
            seen.add(t)
            tickers.append((t, n))
            
    # If no tickers found via parts, try full string
    if not tickers:
        t, n = extract_ticker(query)
        if t:
            tickers = [(t, n)]
            
    return tickers



def fetch_stock_price(ticker: str) -> pd.DataFrame:
    """
    Fetches 1 year of daily OHLCV price data for a ticker using yfinance.
    Saves the raw data to data/raw/{ticker}.csv.
    """
    import yfinance as yf
    print(f"[*] Fetching historical price data for {ticker}...")
    
    # Download 1 year of historical data
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    if df.empty:
        raise ValueError(f"No stock data found for ticker: {ticker}")
        
    # Reset index to make Date a column and clean index
    df = df.reset_index()
    
    # Save raw CSV
    base_dir = "/tmp/data/raw" if os.environ.get("VERCEL") == "1" else "data/raw"
    os.makedirs(base_dir, exist_ok=True)
    raw_path = os.path.join(base_dir, f"{ticker.replace('.', '_')}.csv")
    df.to_csv(raw_path, index=False)
    print(f"[+] Saved raw price data to {raw_path}")
    
    return df


def clean_and_truncate_description(desc: str) -> str:
    """
    Cleans HTML tags and truncates the description to only complete sentences,
    ensuring a clean closure and a short, clear summary.
    """
    import re
    # Remove HTML tags
    desc_clean = re.sub(r'<[^<]+?>', '', desc).strip()
    if not desc_clean:
        return "Click the link to read the full article."
        
    # Split into sentences using a regex (split on period/question/exclamation followed by space or end)
    sentences = re.split(r'(?<=[.!?])\s+', desc_clean)
    
    complete_sentences = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # A sentence is complete if it ends with a punctuation mark (., ?, !) and doesn't end with ellipsis "..." or ".."
        if s[-1] in ['.', '?', '!'] and not s.endswith('..') and not s.endswith('...'):
            complete_sentences.append(s)
        else:
            break
            
    if complete_sentences:
        # Join the complete sentences. Limit total length to 220 chars.
        summary = " ".join(complete_sentences)
        if len(summary) > 220:
            if len(complete_sentences[0]) <= 220:
                summary = complete_sentences[0]
            else:
                summary = summary[:217] + "..."
        return summary
    else:
        # Fallback if no complete sentences found (remove trailing dots/metadata and append a single period)
        desc_clean = re.sub(r'\s*\.+\s*$', '', desc_clean)
        desc_clean = re.sub(r'\s*\[\+\d+\s+chars\]$', '', desc_clean)
        if desc_clean:
            return desc_clean.strip() + "."
        return "Click the link to read the full article."


def fetch_news_headlines(ticker: str, company_name: str) -> list[dict]:
    """
    Fetches news articles mentioning the stock from the last 5 days.
    Combines NewsAPI and Yahoo Finance RSS feed to ensure a rich list of 5-8 articles.
    Each article is returned as a dictionary:
      {"title": str, "url": str, "source": str, "description": str}
    """
    import xml.etree.ElementTree as ET
    import re
    
    headlines = []
    seen_titles = set()
    
    # Calculate timestamps for last 5 days
    now = datetime.datetime.now()
    five_days_ago = now - datetime.timedelta(days=5)
    from_date_str = five_days_ago.strftime("%Y-%m-%d")

    news_api_key = os.getenv("NEWS_API_KEY")
    
    def normalize_title(t: str) -> str:
        # Lowercase, keep only alphanumeric characters for comparison
        return re.sub(r'[^a-z0-9]', '', t.lower())
    
    # 1. Primary: NewsAPI (only if key is present and not default placeholder)
    if news_api_key and "PLACEHOLDER" not in news_api_key:
        try:
            print("[*] Fetching news from NewsAPI...")
            query = f'"{company_name}" OR "{ticker.split(".")[0]}"'
            
            root_domains = (
                "indiatimes.com,moneycontrol.com,livemint.com,thehindubusinessline.com,"
                "financialexpress.com,business-standard.com,reuters.com,bloomberg.com,"
                "cnbctv18.com,ndtvprofit.com,yahoo.com"
            )
            
            ALLOWED_SUBDOMAINS = {
                "economictimes.indiatimes.com",
                "m.economictimes.com",
                "moneycontrol.com",
                "www.moneycontrol.com",
                "livemint.com",
                "www.livemint.com",
                "thehindubusinessline.com",
                "www.thehindubusinessline.com",
                "financialexpress.com",
                "www.financialexpress.com",
                "business-standard.com",
                "www.business-standard.com",
                "reuters.com",
                "www.reuters.com",
                "bloomberg.com",
                "www.bloomberg.com",
                "cnbctv18.com",
                "www.cnbctv18.com",
                "ndtvprofit.com",
                "www.ndtvprofit.com",
                "finance.yahoo.com"
            }
            
            EXCLUDE_KEYWORDS = ["harassment", "sexual", "assault", "bail", "arrest", "accused", "court", "nida khan"]
            
            url = (
                f"https://newsapi.org/v2/everything?"
                f"qInTitle={quote_plus(query)}&"
                f"domains={root_domains}&"
                f"from={from_date_str}&"
                f"language=en&"
                f"sortBy=relevance&"
                f"pageSize=50&"
                f"apiKey={news_api_key}"
            )
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                for article in articles:
                    title = article.get("title")
                    url_str = article.get("url")
                    source_name = article.get("source", {}).get("name", "Financial News")
                    desc = article.get("description") or ""
                    
                    if title and "[Removed]" not in title:
                        # Parse domain from URL
                        domain = url_str.split("/")[2] if url_str else "Unknown"
                        
                        # Apply python domain filter
                        if domain not in ALLOWED_SUBDOMAINS:
                            continue
                            
                        # Apply python keyword filter
                        title_lower = title.lower()
                        if any(w in title_lower for w in EXCLUDE_KEYWORDS):
                            continue
                            
                        norm = normalize_title(title)
                        if norm not in seen_titles:
                            seen_titles.add(norm)
                            desc_clean = clean_and_truncate_description(desc)
                            headlines.append({
                                "title": title,
                                "url": url_str or "",
                                "source": source_name,
                                "description": desc_clean
                            })
                print(f"[+] Retrieved {len(headlines)} headlines from NewsAPI.")
        except Exception as e:
            print(f"[Warning] NewsAPI fetch failed: {e}. Falling back to Yahoo Finance RSS.")
            
    # 2. Supplement/Fallback: Yahoo Finance RSS (if NewsAPI returned fewer than 5 headlines)
    if len(headlines) < 5:
        try:
            print(f"[*] Supplementing with Yahoo Finance RSS (currently have {len(headlines)} articles)...")
            url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                ticker_clean = ticker.split(".")[0].lower()
                comp_clean = company_name.lower()
                
                raw_rss_count = 0
                added_rss_count = 0
                for item in root.findall(".//item"):
                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    desc_elem = item.find("description")
                    
                    if title_elem is not None and title_elem.text:
                        raw_rss_count += 1
                        title_text = title_elem.text
                        title_lower = title_text.lower()
                        
                        # Enforce matching using common stock aliases and abbreviations
                        aliases = {comp_clean, ticker_clean}
                        if ticker_clean == "sbin":
                            aliases.add("sbi")
                        elif ticker_clean.endswith("bank"):
                            aliases.add(ticker_clean.replace("bank", ""))
                        if "bank" in comp_clean:
                            words = comp_clean.split()
                            if len(words) >= 2:
                                aliases.add(" ".join(words[:2]))
                                
                        has_match = any((a in title_lower) for a in aliases)
                        if has_match:
                            norm = normalize_title(title_text)
                            if norm not in seen_titles:
                                url_text = link_elem.text if link_elem is not None else ""
                                desc_text = desc_elem.text if desc_elem is not None else ""
                                desc_clean = clean_and_truncate_description(desc_text)
                                    
                                seen_titles.add(norm)
                                headlines.append({
                                    "title": title_text,
                                    "url": url_text,
                                    "source": "Yahoo Finance",
                                    "description": desc_clean
                                })
                                added_rss_count += 1
                print(f"[+] Retrieved {raw_rss_count} raw RSS headlines, added {added_rss_count} unique matching articles.")
        except Exception as e:
            print(f"[Error] Failed to fetch news from RSS: {e}")
            
    return headlines[:15]

def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetches key fundamental valuation metrics for a stock ticker from yfinance.
    Returns a dictionary of metrics, or empty/None values if not found.
    Falls back to forwardPE when trailingPE is missing (e.g. negative earnings).
    """
    import yfinance as yf
    print(f"[*] Fetching fundamental valuation data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # --- P/E Ratio: trailing → forward → manual (price/EPS) ---
        trailing_pe = info.get("trailingPE")
        forward_pe  = info.get("forwardPE")
        trailing_eps = info.get("trailingEps")
        current_price = info.get("currentPrice") or info.get("previousClose")

        pe_ratio = None
        pe_type  = "Trailing"   # label shown in UI

        if trailing_pe is not None and trailing_pe > 0:
            pe_ratio = float(trailing_pe)
            pe_type  = "Trailing"
        elif forward_pe is not None and forward_pe > 0:
            pe_ratio = float(forward_pe)
            pe_type  = "Forward"
        elif trailing_eps is not None and trailing_eps > 0 and current_price:
            pe_ratio = float(current_price) / float(trailing_eps)
            pe_type  = "Computed"

        # --- ROE ---
        roe = info.get("returnOnEquity")
        if roe is not None:
            roe = float(roe) * 100   # convert fraction → %

        # --- Market Cap: rupees → crores ---
        mcap = info.get("marketCap")
        if mcap is not None:
            mcap = float(mcap) / 1e7   # 1 Crore = 10,000,000

        # --- EPS (raw, for display) ---
        eps = trailing_eps
        if eps is not None:
            eps = float(eps)

        return {
            "PE_Ratio":       pe_ratio,
            "PE_Type":        pe_type,           # "Trailing" / "Forward" / "Computed"
            "PB_Ratio":       info.get("priceToBook"),
            "Debt_to_Equity": info.get("debtToEquity"),
            "ROE":            roe,
            "Market_Cap":     mcap,
            "EPS":            eps,
        }
    except Exception as e:
        print(f"[Warning] Failed to fetch fundamental data: {e}")
        return {
            "PE_Ratio":       None,
            "PE_Type":        "Trailing",
            "PB_Ratio":       None,
            "Debt_to_Equity": None,
            "ROE":            None,
            "Market_Cap":     None,
            "EPS":            None,
        }

def fetch_global_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical S&P 500 return, Crude Oil return, and USD/INR exchange rate return.
    Aligns trading days using pandas forward fill (ffill).
    """
    import yfinance as yf
    
    # Download indexes
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)
    crude = yf.download("CL=F", start=start_date, end=end_date)
    usdinr = yf.download("INR=X", start=start_date, end=end_date)
    
    # Clean columns in case of multi-index
    for df in [sp500, crude, usdinr]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
    # Calculate daily percentage returns
    sp500_ret = sp500['Close'].pct_change().reset_index()
    sp500_ret.columns = ['Date', 'SP500_Return']
    
    crude_ret = crude['Close'].pct_change().reset_index()
    crude_ret.columns = ['Date', 'Crude_Return']
    
    usdinr_ret = usdinr['Close'].pct_change().reset_index()
    usdinr_ret.columns = ['Date', 'USD_INR_Return']
    
    # Merge datasets
    macro_df = pd.merge(sp500_ret, crude_ret, on='Date', how='outer')
    macro_df = pd.merge(macro_df, usdinr_ret, on='Date', how='outer')
    
    # Format Date
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.date
    
    # Sort and fill NaNs
    macro_df = macro_df.sort_values('Date').reset_index(drop=True)
    macro_df[['SP500_Return', 'Crude_Return', 'USD_INR_Return']] = macro_df[['SP500_Return', 'Crude_Return', 'USD_INR_Return']].ffill().fillna(0.0)
    
    # Shift macro return columns by 1 trading day to resolve timezone lag (2A)
    macro_df[['SP500_Return', 'Crude_Return', 'USD_INR_Return']] = macro_df[['SP500_Return', 'Crude_Return', 'USD_INR_Return']].shift(1).fillna(0.0)
    
    return macro_df


def get_latest_macro_returns() -> tuple[float, float, float]:
    """
    Returns the latest daily returns for: (SP500_Return, Crude_Return, USD_INR_Return)
    Uses a daily local CSV cache to avoid slow yfinance network calls on every chatbot query.
    """
    import os
    import pandas as pd
    import datetime
    
    cache_dir = "/tmp/data/macro" if os.environ.get("VERCEL") == "1" else "data/macro"
    cache_path = os.path.join(cache_dir, "global_macro_cache.csv")
    today = datetime.date.today()
    
    use_cache = False
    if os.path.exists(cache_path):
        mtime = datetime.date.fromtimestamp(os.path.getmtime(cache_path))
        if mtime == today:
            use_cache = True
            
    if use_cache:
        try:
            df = pd.read_csv(cache_path)
            if not df.empty:
                last_row = df.iloc[-1]
                return float(last_row['SP500_Return']), float(last_row['Crude_Return']), float(last_row['USD_INR_Return'])
        except Exception as e:
            print(f"[Warning] Error reading macro cache: {e}")
            
    # Fetch fresh data (last 30 days is enough to calculate latest daily returns safely)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        print("[*] Fetching latest global macro indicators (S&P 500, Crude, USD/INR)...")
        
        # We fetch last 30 days to ensure we have enough data to calculate pct_change even over long holiday periods
        import yfinance as yf
        sp500 = yf.download("^GSPC", period="1mo")
        crude = yf.download("CL=F", period="1mo")
        usdinr = yf.download("INR=X", period="1mo")
        
        for df in [sp500, crude, usdinr]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
        sp500_ret = sp500['Close'].pct_change().reset_index()
        sp500_ret.columns = ['Date', 'SP500_Return']
        
        crude_ret = crude['Close'].pct_change().reset_index()
        crude_ret.columns = ['Date', 'Crude_Return']
        
        usdinr_ret = usdinr['Close'].pct_change().reset_index()
        usdinr_ret.columns = ['Date', 'USD_INR_Return']
        
        # Merge
        macro_df = pd.merge(sp500_ret, crude_ret, on='Date', how='outer')
        macro_df = pd.merge(macro_df, usdinr_ret, on='Date', how='outer')
        
        macro_df = macro_df.sort_values('Date').reset_index(drop=True)
        macro_df[['SP500_Return', 'Crude_Return', 'USD_INR_Return']] = macro_df[['SP500_Return', 'Crude_Return', 'USD_INR_Return']].ffill().fillna(0.0)
        
        # Shift macro return columns by 1 trading day to resolve timezone lag (2A)
        macro_df[['SP500_Return', 'Crude_Return', 'USD_INR_Return']] = macro_df[['SP500_Return', 'Crude_Return', 'USD_INR_Return']].shift(1).fillna(0.0)
        
        # Save cache
        macro_df.to_csv(cache_path, index=False)
        
        last_row = macro_df.iloc[-1]
        return float(last_row['SP500_Return']), float(last_row['Crude_Return']), float(last_row['USD_INR_Return'])
    except Exception as e:
        print(f"[Warning] Failed to fetch live macro data: {e}. Defaulting to 0.0 returns.")
        return 0.0, 0.0, 0.0


if __name__ == "__main__":

    # Test execution
    ticker, name = extract_ticker("Should I buy Tata Motors today?")
    print(f"Extracted Ticker: {ticker}, Name: {name}")
    
    # Try fetching stock prices
    df = fetch_stock_price(ticker)
    print(df.head(2))
    
    # Try fetching headlines
    hl = fetch_news_headlines(ticker, name)
    print(f"Sample Headlines: {hl[:3]}")
