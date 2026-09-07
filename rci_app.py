"""
Cash Flow + Liquidity + Momentum Radar — Streamlit Web App
رادار التدفق النقدي والسيولة والزخم (متوافق مع الهواتف)
"""

import time
import re
import urllib.request
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ═══════════════════════════════════════════════
#  DEFAULT PARAMETERS
# ═══════════════════════════════════════════════
CMF_PERIOD         = 14   # فترة تدفق الأموال (Chaikin Money Flow)
RVOL_LOOKBACK      = 20   # متوسط الحجم للمقارنة
OBV_SLOPE_LEN      = 5    # قياس تسارع الزخم في آخر الشموع

CHUNK_SIZE = 50  
CACHE_TTL  = 80

# ═══════════════════════════════════════════════
#  جلب رموز NASDAQ 
# ═══════════════════════════════════════════════
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

FALLBACK_NASDAQ = sorted(set([
    "AAPL","TSLA","NVDA","AMD","MSFT","META","AMZN","GOOGL","NFLX","SMCI",
    "MARA","RIOT","COIN","PLTR","ARM","AVGO","QCOM","INTC","CRWD","PANW",
    "SOFI","ROKU","HOOD","UBER","MSTR","CVNA","AFRM","UPST","DKNG","RIVN"
]))

def _parse_nasdaq_text(text: str) -> list:
    symbols = []
    for line in text.strip().split("\n"):
        parts = line.strip().split("|")
        if len(parts) < 4: continue
        sym = parts[0].strip()
        if not sym or not sym[0].isalpha() or "File Creation Time" in sym or parts[3].strip() == "Y" or "$" in sym:
            continue
        if re.fullmatch(r"[A-Z][A-Z0-9.\-]*", sym):
            symbols.append(sym)
    return sorted(set(symbols))

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_all_nasdaq_symbols() -> tuple:
    try:
        req = urllib.request.Request(NASDAQ_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        symbols = _parse_nasdaq_text(text)
        return tuple(symbols), f"Official NASDAQ ({len(symbols)})", None
    except Exception as e:
        return tuple(FALLBACK_NASDAQ), f"Highly Liquid Top list ({len(FALLBACK_NASDAQ)})", str(e)

# ══════════════════════════════════════════════════════════
#  TIMEFRAME CONFIG
# ══════════════════════════════════════════════════════════
TF_CONFIG = {
    "1m" : {"yf_interval": "1m",  "yf_period": "1d", "label": "⚡ 1 Minute (Scalping)"},
    "5m" : {"yf_interval": "5m",  "yf_period": "5d", "label": "🔥 5 Minutes (Day Trade)"},
    "15m": {"yf_interval": "15m", "yf_period": "5d", "label": "⏱️ 15 Minutes (Intraday)"},
    "1h" : {"yf_interval": "1h",  "yf_period": "60d","label": "🕒 1 Hour (Hourly Trend)"},
}

# ═══════════════════════════════════════════════
#  DATA DOWNLOAD (BATCH)
# ═══════════════════════════════════════════════
def download_raw_batch(tickers: tuple, yf_interval: str, yf_period: str) -> dict:
    tickers = list(tickers)
    result = {}
    chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]

    for chunk in chunks:
        try:
            raw = yf.download(
                tickers=chunk, period=yf_period, interval=yf_interval,
                group_by="ticker", threads=True, progress=False, auto_adjust=True
            )
            if not raw.empty:
                for ticker in chunk:
                    try:
                        df = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
                        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                        result[ticker] = df if len(df) > 35 else None
                    except:
                        result[ticker] = None
        except:
            pass
        time.sleep(0.2)
    return result

# ═══════════════════════════════════════════════
#  INDICATORS & CHAIKIN MONEY FLOW (CMF)
# ═══════════════════════════════════════════════
def _rolling_sum(arr, period):
    n = len(arr)
    if n < period: return np.full(n, np.nan)
    cs = np.cumsum(arr)
    out = np.full(n, np.nan)
    out[period - 1] = cs[period - 1]
    if period < n: out[period:] = cs[period:] - cs[:-period]
    return out

def calculate_cmf(high, low, close, vol, period=14):
    n = len(close)
    if n < period: return np.zeros(n)
    hl_range = high - low
    hl_range[hl_range == 0] = 1e-10
    mf_multiplier = ((close - low) - (high - close)) / hl_range
    mf_volume = mf_multiplier * vol
    
    cmf = np.empty(n)
    for i in range(n):
        if i < period - 1:
            cmf[i] = 0
        else:
            sub_mf = np.sum(mf_volume[i - period + 1 : i + 1])
            sub_vol = np.sum(vol[i - period + 1 : i + 1])
            cmf[i] = sub_mf / (sub_vol + 1e-10)
    return cmf

def get_radar_signal(ticker, df, score_min, min_price, min_vol_avg):
    try:
        if df is None or len(df) < 35: return None
        
        close = df["Close"].values.astype(float)
        high  = df["High"].values.astype(float)
        low   = df["Low"].values.astype(float)
        vol   = np.maximum(df["Volume"].values.astype(float), 0.0)
        
        price = close[-1]
        
        # فلتر السعر والسيولة
        if price < min_price: return None
        avg_vol = vol[-RVOL_LOOKBACK:].mean()
        if avg_vol < min_vol_avg: return None

        i = len(close) - 1
        
        # 1. CASH FLOW (التدفق النقدي المؤسساتي - CMF & MFI) - [الوزن: 40%]
        cmf_arr = calculate_cmf(high, low, close, vol, CMF_PERIOD)
        cmf_val = cmf_arr[i]
        
        tp_price = (high + low + close) / 3.0
        mf = tp_price * vol
        pos = np.where(tp_price[1:] > tp_price[:-1], mf[1:], 0)
        neg = np.where(tp_price[1:] < tp_price[:-1], mf[1:], 0)
        rp = np.sum(pos[-10:])
        rn = np.sum(neg[-10:])
        mfi = 100.0 if rn < 1e-10 else 100.0 - (100.0 / (1.0 + rp / rn))
        
        s_cf_cmf = np.clip(cmf_val / 0.15, -1.0, 1.0)
        s_cf_mfi = np.clip((mfi - 50) / 30.0, -1.0, 1.0)
        score_cash_flow = (s_cf_cmf * 0.6) + (s_cf_mfi * 0.4)

        # 2. LIQUIDITY (السيولة اللحظية وانفجار الحجم - RVOL) - [الوزن: 35%]
        rvol = vol[i] / (avg_vol + 1e-10)
        score_liquidity = np.clip((rvol - 0.7) / 2.0, -0.5, 1.0)
        if rvol < 0.4: score_liquidity = -1.0

        # 3. MOMENTUM (الزخم والتسارع - OBV & Price ROC) - [الوزن: 25%]
        d = np.sign(np.diff(close))
        obv = np.empty(len(close))
        obv[0] = vol[0]
        for j in range(1, len(close)): obv[j] = obv[j - 1] + d[j - 1] * vol[j]
        obv_roc = (obv[i] - obv[i - OBV_SLOPE_LEN]) / (abs(obv[i - OBV_SLOPE_LEN]) + 1.0)
        prc_roc = (close[i] - close[i - OBV_SLOPE_LEN]) / (close[i - OBV_SLOPE_LEN] + 1e-10)
        
        score_momentum = np.clip((prc_roc * 25) + (obv_roc * 5), -1.0, 1.0)

        # ---- WEIGHTED EVIDENCE MODEL ----
        composite = float(
            (0.40 * score_cash_flow) + 
            (0.35 * score_liquidity) + 
            (0.25 * score_momentum)
        )

        sig = None
        if composite >= score_min: sig = "BUY (CASH FLOW+)"
        elif composite <= -score_min: sig = "SELL (OUTFLOW)"
        if sig is None: return None

        # ══════════════════════════════════════════════════════════════
        # منطق الـ SL و TP المبني على "تمركز السيولة" (Liquidity Pools)
        # ══════════════════════════════════════════════════════════════
        tpv = tp_price * vol
        r_tpv = _rolling_sum(tpv, 14)
        r_vol = _rolling_sum(vol, 14)
        vwap = r_tpv[i] / (r_vol[i] + 1e-10)

        liq_lookback = 15
        bsl_pool = np.max(high[max(0, i - liq_lookback):i]) # Buy-Side Liquidity
        ssl_pool = np.min(low[max(0, i - liq_lookback):i])  # Sell-Side Liquidity
        
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        buffer = tr * 0.2  

        if "BUY" in sig:
            sl = min(ssl_pool, vwap) - buffer if price > vwap else ssl_pool - buffer
            risk = price - sl
            tp = bsl_pool if (bsl_pool - price) > (risk * 1.0) else price + (risk * 2.0)
        else:
            sl = max(bsl_pool, vwap) + buffer if price < vwap else bsl_pool + buffer
            risk = sl - price
            tp = ssl_pool if (price - ssl_pool) > (risk * 1.0) else price - (risk * 2.0)

        return dict(
            Ticker=ticker, Signal=sig, Price=round(price, 2),
            CF=round(score_cash_flow, 2), LQ=round(score_liquidity, 2), MO=round(score_momentum, 2),
            RVOL=round(rvol, 2), CMF=round(cmf_val, 2),
            TP=round(tp, 2), SL=round(sl, 2),
            Score=round(composite, 3), _score=composite
        )
    except:
        return None

# ═══════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════
st.set_page_config(page_title="Cash Flow & Momentum Radar ⚡", page_icon="⚡", layout="centered")

st.markdown("""
<style>
.stButton > button { height: 3rem !important; font-size: 1.1rem !important; font-weight: 700 !important; border-radius: 8px !important; }
.card-buy { background: linear-gradient(135deg,#0a2e12,#11471d); border-left: 5px solid #00e676; border-radius: 8px; padding: 12px; margin: 8px 0; color: #e0ffe0; }
.card-sell { background: linear-gradient(135deg,#2e0a0a,#471111); border-left: 5px solid #ff5252; border-radius: 8px; padding: 12px; margin: 8px 0; color: #ffe0e0; }
.tag-buy  { background:#00e676; color:#000; border-radius:4px; padding:2px 8px; font-weight:bold; font-size:0.8rem; }
.tag-sell { background:#ff5252; color:#fff; border-radius:4px; padding:2px 8px; font-weight:bold; font-size:0.8rem; }
.metric-pill { background:#1e3a5f; color:#7dd3fc; border-radius:4px; padding:2px 6px; font-size:0.8rem; margin-right:5px;}
.score-high { color: #00e676; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Cash Flow + Liquidity + Momentum Radar")
st.caption("Advanced Institutional Flow & Liquidity Engine (CMF + RVOL + Order Flow)")
st.divider()

with st.spinner("📡 جلب القائمة الأساسية..."):
    nasdaq_tuple, source, err = fetch_all_nasdaq_symbols()
    nasdaq_all = list(nasdaq_tuple)

with st.expander("⚙️ إعدادات الرادار الثلاثي", expanded=True):
    tf = st.selectbox("⏱️ الإطار الزمني (Timeframe)", list(TF_CONFIG.keys()), index=2, format_func=lambda x: TF_CONFIG[x]["label"])
    
    c1, c2 = st.columns(2)
    with c1:
        score_min = st.slider("🎯 حساسـية الإشارة (Score Limit)", 0.20, 0.60, 0.35, 0.05, help="خفض الرقم يتيح ظهور الفرص بشكل أسرع وأسهل.")
        max_stocks = st.selectbox("📊 عدد الأسهم للمسح", [50, 100, 200, 500], index=1)
    with c2:
        min_price = st.number_input("💵 الحد الأدنى للسعر ($)", value=5.0, step=1.0)
        min_vol_avg = st.number_input("💧 أدنى سيولة متوسطة", value=5000, step=1000)

    search_input = st.text_input("🔑 إضافة أسهم مخصصة (مفصولة بفاصلة)", placeholder="مثال: AAPL, TSLA, NVDA")

custom_tickers = [x.strip().upper() for x in re.split(r'[,\s]+', search_input) if x.strip()]
scan_list = list(dict.fromkeys(custom_tickers + nasdaq_all[:max_stocks]))

st.info(f"سيتم فحص **{len(scan_list)}** سهم على فريم **{tf}**.")

if st.button("🔍 SCAN MARKET NOW", type="primary", use_container_width=True):
    start_time = time.time()
    results = []
    
    bar = st.progress(0)
    status_text = st.empty()
    
    chunks = [scan_list[i:i + CHUNK_SIZE] for i in range(0, len(scan_list), CHUNK_SIZE)]
    
    for idx, chunk in enumerate(chunks):
        status_text.text(f"📡 فحص تدفق السيولة والزخم... الحزمة {idx+1}/{len(chunks)}")
        
        cfg = TF_CONFIG[tf]
        raw_data = download_raw_batch(tuple(chunk), cfg["yf_interval"], cfg["yf_period"])
        
        for ticker, df in raw_data.items():
            sig = get_radar_signal(ticker, df, score_min, min_price, min_vol_avg)
            if sig:
                results.append(sig)
                
        bar.progress((idx + 1) / len(chunks))
        
    status_text.empty()
    bar.empty()
    
    results = sorted(results, key=lambda x: abs(x["_score"]), reverse=True)
    
    st.success(f"✅ اكتمل المسح في {time.time()-start_time:.1f} ثانية. تم رصد {len(results)} فرصة.")
    
    for r in results:
        is_buy = "BUY" in r['Signal']
        card_class = "card-buy" if is_buy else "card-sell"
        tag_class = "tag-buy" if is_buy else "tag-sell"
        
        html = f"""
        <div class="{card_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:1.3rem; font-weight:bold; letter-spacing:1px;">{r['Ticker']}</span>
                <span class="{tag_class}">{r['Signal']}</span>
            </div>
            <div style="font-size:0.95rem; margin-bottom:8px;">
                💵 Price: <b>${r['Price']}</b> &nbsp;|&nbsp; 
                🎯 TP: <b style="color:#00e676">${r['TP']}</b> &nbsp;|&nbsp; 
                🛡️ SL: <b style="color:#ff5252">${r['SL']}</b>
            </div>
            <div>
                <span class="metric-pill">CashFlow (CF): {r['CF']}</span>
                <span class="metric-pill">Liquidity (LQ): {r['LQ']}</span>
                <span class="metric-pill">Momentum (MO): {r['MO']}</span>
                <span class="metric-pill">CMF: {r['CMF']}</span>
                <span class="metric-pill">Score: <span class="score-high">{r['Score']}</span></span>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        
    if not results:
        st.warning("لم يتم اكتشاف أي فرص تتطابق مع معايير التدفق النقدي الحالية. جرّب خفض (Score Limit).")

st.caption("⚠️ **تنويه:** الرادار مصمم لاكتشاف مراكز التدفق النقدي المؤسساتي والسيولة الحية بناءً على عزم الأوامر Volume-Weighted.")
