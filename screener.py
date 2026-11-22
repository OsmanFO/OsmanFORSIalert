import os
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import requests
import yfinance as yf

# ==========================================
# 1. TELEGRAM CONFIGURATION (@Osmanfnorsialertbot)
# ==========================================
BOT_USERNAME = "@Osmanfnorsialertbot"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# Replace/extend with the full current NSE F&O stock list
FNO_SYMBOLS = [
    "360ONE", "ABB", "ABCAPITAL", "ADANIENSOL", "ADANIENT", "ADANIGREEN",
    "ADANIPORTS", "ADANIPOWER", "ALKEM", "AMBER", "AMBUJACEM", "ANANDRATHI",
    "ANGELONE", "APLAPOLLO", "APOLLOHOSP", "ASHOKLEY", "ASIANPAINT", "ASTRAL",
    "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE",
    "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BDL", "BEL", "BHARATFORG",
    "BHARTIARTL", "BHEL", "BIOCON", "BLUESTARCO", "BOSCHLTD", "BPCL",
    "BRITANNIA", "BSE", "CAMS", "CANBK", "CDSL", "CGPOWER", "CHOLAFIN",
    "CIPLA", "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR",
    "CROMPTON", "CUMMINSIND", "DABUR", "DELHIVERY", "DIVISLAB", "DIXON",
    "DLF", "DMART", "DRREDDY", "EICHERMOT", "ETERNAL", "FEDERALBNK",
    "FORTIS", "GAIL", "GLENMARK", "GMRAIRPORT", "GODREJCP", "GODREJPROP",
    "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDPETRO", "HINDUNILVR", "HINDZINC", "HYUNDAI",
    "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX",
    "INDHOTEL", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY",
    "INOXWIND", "IOC", "IREDA", "IRFC", "ITC", "JINDALSTEL", "JIOFIN",
    "JSWENERGY", "JSWSTEEL", "JUBLFOOD", "KALYANKJIL", "KAYNES", "KEI",
    "KFINTECH", "KOTAKBANK", "KPITTECH", "LAURUSLABS", "LICHSGFIN", "LICI",
    "LODHA", "LT", "LTF", "LTIM"
]


# ==========================================
# 2. INDICATOR & RESAMPLING HELPERS
# ==========================================
def wilder_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(100.0)


def in_daily_rsi_zones(val: float) -> bool:
    return (39 <= val <= 43) or (49 <= val <= 53) or (59 <= val <= 63)


def resample_ohlc(daily_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return daily_df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(subset=["Close"])


# ==========================================
# 3. SCREENER ENGINE
# ==========================================
def run_screeners(symbols: list[str]) -> tuple[list[tuple], list[tuple]]:
    momentum_hits = []
    value_buy_hits = []

    yf_tickers = [f"{s}.NS" for s in symbols]
    data = yf.download(
        tickers=yf_tickers,
        period="5y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    for sym in symbols:
        yf_sym = f"{sym}.NS"
        try:
            df_d = (
                data[yf_sym].dropna(subset=["Close"]).copy()
                if isinstance(data.columns, pd.MultiIndex)
                else data.dropna(subset=["Close"]).copy()
            )
            if len(df_d) < 100:
                continue

            df_d.index = pd.to_datetime(df_d.index)
            if df_d.index.tz is not None:
                df_d.index = df_d.index.tz_localize(None)

            d_rsi = wilder_rsi(df_d["Close"], 14).dropna()
            if len(d_rsi) < 2:
                continue

            d_curr = float(d_rsi.iloc[-1])
            d_prev = float(d_rsi.iloc[-2])
            ltp = float(df_d["Close"].iloc[-1])

            df_w = resample_ohlc(df_d, "W-FRI")
            w_rsi_s = wilder_rsi(df_w["Close"], 14).dropna()
            if len(df_w) < 2 or w_rsi_s.empty:
                continue

            w_rsi = float(w_rsi_s.iloc[-1])
            w_close_curr = float(df_w["Close"].iloc[-1])
            w_close_prev = float(df_w["Close"].iloc[-2])

            df_m = resample_ohlc(df_d, "ME")
            m_rsi_s = wilder_rsi(df_m["Close"], 14).dropna()
            if m_rsi_s.empty:
                continue

            m_rsi = float(m_rsi_s.iloc[-1])
            zone_ok = in_daily_rsi_zones(d_curr)

            # Momentum Screener:
            # Monthly RSI > 59, Weekly RSI > 59, Daily RSI in [39-43, 49-53, 59-63], prev day RSI > today RSI
            if m_rsi > 59 and w_rsi > 59 and zone_ok and (d_prev > d_curr):
                momentum_hits.append((
                    sym, round(ltp, 2), round(d_prev, 1),
                    round(d_curr, 1), round(w_rsi, 1), round(m_rsi, 1)
                ))

            # Value Buy Scanner:
            # Monthly RSI in [40-43], Weekly close > prev weekly close, Daily RSI in [39-43, 49-53, 59-63], prev day RSI < today RSI
            if (40 <= m_rsi <= 43) and (w_close_curr > w_close_prev) and zone_ok and (d_prev < d_curr):
                value_buy_hits.append((
                    sym, round(ltp, 2), round(d_prev, 1),
                    round(d_curr, 1), round(w_rsi, 1), round(m_rsi, 1)
                ))

        except Exception as exc:
            print(f"[WARN] Skipping {sym}: {exc}")

    return momentum_hits, value_buy_hits


# ==========================================
# 4. TELEGRAM DISPATCH
# ==========================================
def send_telegram(msg: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables. "
            "Add them in GitHub Repository Secrets -> Actions."
        )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for i in range(0, len(msg), 4000):
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg[i : i + 4000],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        resp.raise_for_status()
        print(f"[INFO] Sent Telegram chunk via {BOT_USERNAME} (status {resp.status_code}).")


def trigger_scan() -> None:
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    mom, val = run_screeners(FNO_SYMBOLS)

    lines = [
        f"<b>F&amp;O RSI Scan ({BOT_USERNAME})</b>",
        f"<i>{now_ist.strftime('%d %b %Y, %H:%M IST')}</i>\n",
        "<b>🚀 Momentum Screener</b>",
    ]
    lines += (
        [f"• <b>{s}</b> (₹{p}) | D: {dp}→{dc} | W: {w} | M: {m}" for s, p, dp, dc, w, m in mom]
        if mom else ["• No matches."]
    )

    lines.append("\n<b>💎 Value Buy Scanner</b>")
    lines += (
        [f"• <b>{s}</b> (₹{p}) | D: {dp}→{dc} | W: {w} | M: {m}" for s, p, dp, dc, w, m in val]
        if val else ["• No matches."]
    )

    report = "\n".join(lines)
    print(report)
    send_telegram(report)


if __name__ == "__main__":
    trigger_scan()
