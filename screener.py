import argparse
import os
import re
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Avoid SQLite cache lock errors (such as KAYNES.NS)
YF_CACHE_DIR = "/tmp/yf_tz_cache"
os.makedirs(YF_CACHE_DIR, exist_ok=True)
try:
    yf.set_tz_cache_location(YF_CACHE_DIR)
except Exception:
    pass

# Normalizes renamed symbols automatically (no manual LTIM edit needed)
SYMBOL_MAP = {
    "LTIM": "LTM",
}

# 200+ NSE F&O stock list (deduplicated & normalized at runtime)
FNO_SYMBOLS = [
    "360ONE", "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ACC",
    "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER",
    "ALKEM", "AMBER", "AMBUJACEM", "ANANDRATHI", "ANGELONE", "APLAPOLLO",
    "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT", "ASTRAL",
    "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA",
    "BANKINDIA", "BATAINDIA", "BDL", "BEL", "BERGEPAINT", "BHARATFORG",
    "BHARTIARTL", "BHEL", "BIOCON", "BIRLASOFT", "BLUESTARCO", "BOSCHLTD",
    "BPCL", "BRITANNIA", "BSE", "CAMS", "CANBK", "CANFINHOME", "CDSL",
    "CENTRALBK", "CGPOWER", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "CUB",
    "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL",
    "CROMPTON", "CUMMINSIND", "DABUR", "DALBHARAT", "DEEPAKNTR", "DELHIVERY",
    "DELTACORP", "DIVISLAB", "DIXON", "DLF", "LALPATHLAB", "DRREDDY",
    "EICHERMOT", "ESCORTS", "ETERNAL", "FEDERALBNK", "FORTIS", "GAIL",
    "GLENMARK", "GMRAIRPORT", "GNFC", "GODREJCP", "GODREJPROP", "GRANULES",
    "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDPETRO", "HINDUNILVR", "HINDZINC", "HYUNDAI",
    "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX",
    "INDHOTEL", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY",
    "INOXWIND", "IOC", "IREDA", "IRFC", "ITC", "JINDALSTEL", "JIOFIN",
    "JSWENERGY", "JSWSTEEL", "JUBLFOOD", "KALYANKJIL", "KAYNES", "KEI",
    "KFINTECH", "KOTAKBANK", "KPITTECH", "LAURUSLABS", "LICHSGFIN", "LICI",
    "LODHA", "LT", "LTF", "LTM", "LUPIN", "M&M", "M&MFIN", "MAHABANK",
    "MANAPPURAM", "MARICO", "MARUTI", "MAXHEALTH", "MCX", "METROPOLIS",
    "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN", "NATIONALUM",
    "NAUKRI", "NBCC", "NCC", "NESTLEIND", "NHPC", "NMDC", "NTPC", "NYKAA",
    "OBEROREALTY", "OFSS", "OIL", "ONGC", "PAGEIND", "PATANJALI", "PAYTM",
    "PEL", "PERSISTENT", "PETRONET", "PFC", "PIDILITIND", "PIIND", "PNB",
    "PNBHOUSING", "POLICYBZR", "POLYCAB", "POONAWALLA", "PRESTIGE", "PVRINOX",
    "RAMCOCEM", "RBLBANK", "REC", "RELIANCE", "SAIL", "SBICARD", "SBILIFE",
    "SBIN", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SOLARINDS", "SONACOMS",
    "SRF", "SUNPHARMA", "SUNTV", "SUPREMEIND", "SUZLON", "SYNGENE", "TATACHEM",
    "TATACOMM", "TATACONSUM", "TATAELXSI", "TATAMOTORS", "TATAPOWER",
    "TATASTEEL", "TATATECH", "TCS", "TECHM", "TIINDIA", "TITAN", "TORNTPHARM",
    "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO", "UNIONBANK", "UNITDSPR", "UPL",
    "VBL", "VEDL", "VOLTAS", "WIPRO", "YESBANK", "ZEEL", "ZYDUSLIFE",
]

# Indices checked in isolation across timeframes for RSI(14) in 38-44
WATCH_INDICES = [
    {"name": "Nifty", "candidates": ["^NSEI"]},
    {"name": "Banknifty", "candidates": ["^NSEBANK"]},
    {"name": "Midcap Nifty", "candidates": ["^NSEMDCP50", "^NSMIDCP"]},
    {"name": "Finnifty", "candidates": ["^CNXFIN", "NIFTY_FIN_SERVICE.NS"]},
]


def get_secret(name):
    val = os.environ.get(name, "").strip()
    if val:
        return val
    try:
        from google.colab import userdata
        return (userdata.get(name) or "").strip()
    except Exception:
        return ""


def send_telegram(text):
    token = get_secret("TELEGRAM_BOT_TOKEN")
    chat_id = get_secret("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Skipping send.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, json=payload, timeout=20)
    print("Telegram status:", resp.status_code, resp.text)
    resp.raise_for_status()


def get_clean_fno_symbols():
    cleaned = []
    seen = set()
    for raw in FNO_SYMBOLS:
        s = str(raw or "").strip().upper()
        s = SYMBOL_MAP.get(s, s)
        if re.fullmatch(r"[A-Z0-9&\-]{2,20}", s) and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def compute_rsi(close_series, period=14):
    s = pd.to_numeric(close_series, errors="coerce").dropna()
    if len(s) < period + 2:
        return pd.Series(dtype=float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(100.0)


def extract_close_series(df, ticker):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            sub = df["Close"]
            if isinstance(sub, pd.DataFrame):
                if ticker in sub.columns:
                    return sub[ticker].dropna()
                return sub.iloc[:, 0].dropna()
            return sub.dropna()
    if "Close" in df.columns:
        return df["Close"].dropna()
    return pd.Series(dtype=float)


def fetch_series(ticker, interval, period):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return extract_close_series(df, ticker)
    except Exception as e:
        print(f"Download error for {ticker} ({interval}): {e}")
        return pd.Series(dtype=float)


def resolve_index_ticker(candidates):
    for t in candidates:
        s = fetch_series(t, "1d", "3mo")
        if not s.empty and len(s) >= 20:
            return t
    return None


def check_index_rsi_alerts():
    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %Y, %H:%M IST")
    isolated_triggers = []

    for idx in WATCH_INDICES:
        ticker = resolve_index_ticker(idx["candidates"])
        if not ticker:
            print(f"Could not resolve index ticker for {idx['name']}")
            continue

        s_15m = fetch_series(ticker, "15m", "30d")
        s_30m = fetch_series(ticker, "30m", "60d")
        s_1h = fetch_series(ticker, "1h", "60d")

        s_2h = pd.Series(dtype=float)
        s_4h = pd.Series(dtype=float)
        if not s_1h.empty and isinstance(s_1h.index, pd.DatetimeIndex):
            idx_ist = s_1h.index.tz_convert("Asia/Kolkata") if s_1h.index.tz else s_1h.index
            h_series = pd.Series(s_1h.values, index=idx_ist).dropna()
            s_2h = h_series.resample("2h").last().dropna()
            s_4h = h_series.resample("4h").last().dropna()

        s_1d = fetch_series(ticker, "1d", "5y")
        s_1wk = fetch_series(ticker, "1wk", "10y")
        s_1mo = fetch_series(ticker, "1mo", "15y")

        isolated_frames = [
            ("15m", s_15m),
            ("30m", s_30m),
            ("1h", s_1h),
            ("2h", s_2h),
            ("4h", s_4h),
            ("1d", s_1d),
            ("1w", s_1wk),
            ("1mo", s_1mo),
        ]

        for tf_label, series in isolated_frames:
            rsi_s = compute_rsi(series, period=14)
            if not rsi_s.empty:
                val = float(rsi_s.iloc[-1])
                if 38.0 <= val <= 44.0:
                    isolated_triggers.append(
                        f"• <b>{idx['name']}</b> ({ticker}) | <b>{tf_label}</b> RSI(14): <b>{val:.1f}</b>"
                    )
        time.sleep(0.25)

    if isolated_triggers:
        msg = (
            f"<b>📊 Index RSI(14) Trigger [38–44 Any Timeframe]</b>\n"
            f"<i>{ist_now}</i>\n\n" + "\n".join(isolated_triggers)
        )
        send_telegram(msg)
    else:
        print("No isolated index timeframe RSI(14) values in 38–44 range right now.")


def daily_rsi_in_target_bands(d_rsi):
    return (
        (39.0 <= d_rsi <= 43.0)
        or (49.0 <= d_rsi <= 53.0)
        or (59.0 <= d_rsi <= 63.0)
    )


def run_fno_screener():
    symbols = get_clean_fno_symbols()
    print(f"Screening {len(symbols)} F&O symbols...")
    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %Y, %H:%M IST")

    momentum_lines = []
    value_buy_lines = []

    for sym in symbols:
        ns_ticker = f"{sym}.NS"
        s_1d = fetch_series(ns_ticker, "1d", "2y")
        if len(s_1d) < 30:
            time.sleep(0.15)
            continue

        s_1wk = s_1d.resample("W-FRI").last().dropna()
        s_1mo = s_1d.resample("ME").last().dropna()

        rsi_d = compute_rsi(s_1d, 14)
        rsi_w = compute_rsi(s_1wk, 14)
        rsi_m = compute_rsi(s_1mo, 14)

        if len(rsi_d) < 2 or len(s_1wk) < 2 or rsi_w.empty or rsi_m.empty:
            time.sleep(0.15)
            continue

        ltp = float(s_1d.iloc[-1])
        d_prev = float(rsi_d.iloc[-2])
        d_curr = float(rsi_d.iloc[-1])
        w_curr = float(rsi_w.iloc[-1])
        m_curr = float(rsi_m.iloc[-1])

        w_close_prev = float(s_1wk.iloc[-2])
        w_close_curr = float(s_1wk.iloc[-1])

        in_daily_bands = daily_rsi_in_target_bands(d_curr)

        if (
            m_curr > 59.0
            and w_curr > 59.0
            and in_daily_bands
            and d_prev > d_curr
        ):
            momentum_lines.append(
                f"- <b>{sym}</b> (₹{ltp:.1f}) | D: {d_prev:.1f}→{d_curr:.1f} | W: {w_curr:.1f} | M: {m_curr:.1f}"
            )

        if (
            40.0 <= m_curr <= 43.0
            and w_close_curr > w_close_prev
            and in_daily_bands
            and d_prev < d_curr
        ):
            value_buy_lines.append(
                f"- <b>{sym}</b> (₹{ltp:.1f}) | D: {d_prev:.1f}→{d_curr:.1f} | W: {w_curr:.1f} (W↑) | M: {m_curr:.1f}"
            )

        time.sleep(0.15)

    if momentum_lines:
        send_telegram(
            f"<b>🚀 Momentum Screener</b>\n<i>{ist_now}</i>\n\n" + "\n".join(momentum_lines[:40])
        )
    else:
        print("No Momentum matches found.")

    if value_buy_lines:
        send_telegram(
            f"<b>💎 Value Buy Scanner</b>\n<i>{ist_now}</i>\n\n" + "\n".join(value_buy_lines[:40])
        )
    else:
        print("No Value Buy matches found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["indices", "fno", "all"],
        default="all",
    )
    args, _ = parser.parse_known_args()

    if args.mode in ("indices", "all"):
        check_index_rsi_alerts()
    if args.mode in ("fno", "all"):
        run_fno_screener()


if __name__ == "__main__":
    main()
