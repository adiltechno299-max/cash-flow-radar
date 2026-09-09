"""
🕯️ Candle Radar 1.0 — Streamlit Web App
رادار الشموع الموحّدة (متوافق مع الهواتف)
--- تحويل بيانات OHLCV الخام إلى مقاييس Normalized بحيث تقارَن شمعة 5 دقائق بشمعة يومية ---
--- نسخة: فرص الشراء (BUY) فقط، تم تجاهل إشارات البيع (SELL) ---
--- فلتر القيمة السوقية: تجاهل الشركات الصغيرة (أقل من الحد المحدد) ---
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
#  DEFAULT PARAMETERS — Candle Radar
# ═══════════════════════════════════════════════
RVOL_LOOKBACK        = 20   # متوسط الحجم للمقارنة (لا يشمل الشمعة الحالية)
CANDLE_CONTEXT_LEN   = 5    # عدد الشموع الأخيرة لقياس استمرارية جودة الإغلاقات
VOLUME_DIR_LOOKBACK  = 20   # نافذة قياس اتجاه الحجم (تجميع/تصريف)
RANGE_POS_LOOKBACK   = 30   # نافذة موقع الإغلاق ضمن المدى السعري الأخير
SWING_LEFT           = 2    # عدد الشموع على يسار القمة/القاع لتأكيدها كـ Swing Point
SWING_RIGHT          = 2    # عدد الشموع على يمين القمة/القاع لتأكيدها
LIQUIDITY_LOOKBACK   = 50   # مدى البحث للخلف عن آخر تجمع سيولة غير ممتص
MIN_RISK_REWARD      = 1.2  # أدنى نسبة عائد/مخاطرة (يُمدَّد TP للوصول لها، لا رفض)

# ── عتبات تحويل المقاييس الموحّدة إلى درجات [-1, +1] ──
CLOSE_POS_MID  = 0.50   # الإغلاق بمنتصف المدى = محايد
CLOSE_POS_FULL = 0.25   # انحراف ±25% عن المنتصف = درجة كاملة (±1)
BODY_FULL      = 0.55   # جسم يشكّل 55% من المدى = حسم كامل
WICK_FULL      = 0.35   # فرق فتائل 35% من المدى = ضغط كامل باتجاه واحد

# ── Candle Radar 1.0 — أوزان النموذج المرجّح ──
# عند تفعيل فلتر الاتجاه: يُعاد توزيع الوزن (الاتجاه = 10% ناعمة، لا رفض لأي إشارة).
WEIGHT_CANDLE             = 0.40   # ثابت دائماً: تشريح الشمعة 40%
WEIGHT_VOLUME_WITH_TREND  = 0.30   # الحجم/النشاط عند تفعيل فلتر الاتجاه
WEIGHT_CONTEXT_WITH_TREND = 0.20   # السياق عند تفعيل فلتر الاتجاه
WEIGHT_TREND              = 0.10   # Trend Filter (1H/4H) وزن ناعم إضافي
WEIGHT_VOLUME_NO_TREND    = 0.35   # الحجم/النشاط عند تعطيل فلتر الاتجاه
WEIGHT_CONTEXT_NO_TREND   = 0.25   # السياق عند تعطيل فلتر الاتجاه

# أوزان داخلية للركائز
AN_W_CLOSE, AN_W_BODY, AN_W_WICK       = 0.50, 0.30, 0.20  # تشريح الشمعة
AC_W_RVOL, AC_W_DIR, AC_W_RANGE        = 0.55, 0.25, 0.20  # الحجم/النشاط
CX_W_SUSTAIN, CX_W_RANGEPOS, CX_W_ROOM = 0.40, 0.30, 0.30  # السياق

TREND_EMA_FAST    = 20
TREND_EMA_SLOW    = 50
TREND_SPREAD_NORM = 0.03

CHUNK_SIZE = 50
CACHE_TTL  = 60

DEFAULT_MIN_MARKET_CAP_B = 1.0

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
    "4h" : {"yf_interval": "1h",  "yf_period": "180d","label": "📈 4 Hours (Swing)", "resample": "4h"},
    "1d" : {"yf_interval": "1d",  "yf_period": "1y", "label": "📅 1 Day (Position/Swing)"},
}

# ═══════════════════════════════════════════════
#  DATA DOWNLOAD (BATCH)
# ═══════════════════════════════════════════════
MAX_RETRIES        = 4
BACKOFF_BASE        = 1.5
BACKOFF_MAX_SLEEP   = 20.0

def _download_chunk_with_backoff(chunk, yf_interval, yf_period, status_cb=None):
    attempt = 0
    while True:
        try:
            raw = yf.download(
                tickers=chunk, period=yf_period, interval=yf_interval,
                group_by="ticker", threads=True, progress=False, auto_adjust=True
            )
            return raw
        except Exception as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                if status_cb:
                    status_cb(f"⚠️ تعذر تحميل حزمة بعد {MAX_RETRIES} محاولات، سيتم تجاوزها.")
                return None
            sleep_time = min(BACKOFF_BASE ** attempt, BACKOFF_MAX_SLEEP)
            sleep_time += np.random.uniform(0, 0.5)
            if status_cb:
                status_cb(f"🌐 خطأ شبكة/تقييد معدل الطلبات، إعادة المحاولة {attempt}/{MAX_RETRIES} بعد {sleep_time:.1f}ث...")
            time.sleep(sleep_time)

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna()

def download_raw_batch(tickers: tuple, yf_interval: str, yf_period: str, status_cb=None, resample_rule: str = None) -> dict:
    tickers = list(tickers)
    result = {}
    chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]

    for chunk in chunks:
        raw = _download_chunk_with_backoff(chunk, yf_interval, yf_period, status_cb)
        if raw is not None and not raw.empty:
            for ticker in chunk:
                try:
                    df = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
                    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                    if resample_rule:
                        df = _resample_ohlcv(df, resample_rule)
                    result[ticker] = df if len(df) > 35 else None
                except:
                    result[ticker] = None
        time.sleep(0.2)
    return result

# ═══════════════════════════════════════════════
#  MARKET CAP FILTER (تجاهل الشركات الصغيرة)
# ═══════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_cap(ticker: str):
    try:
        fi = yf.Ticker(ticker).fast_info
        cap = fi.get("market_cap") if isinstance(fi, dict) else getattr(fi, "market_cap", None)
        if cap is None or cap <= 0:
            return None
        return float(cap)
    except Exception:
        return None

# ═══════════════════════════════════════════════
#  TREND FILTER (1H / 4H) — إضافة ناعمة وليست فلتر رفض
# ═══════════════════════════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)
def get_trend_score(ticker: str, trend_tf: str) -> float:
    try:
        cfg = TF_CONFIG[trend_tf]
        raw = yf.download(
            tickers=ticker, period=cfg["yf_period"], interval=cfg["yf_interval"],
            progress=False, auto_adjust=True
        )
        if raw is None or raw.empty:
            return 0.0
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if cfg.get("resample"):
            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df = _resample_ohlcv(df, cfg["resample"])
            close = df["Close"]
        if len(close) < TREND_EMA_SLOW + 5:
            return 0.0

        ema_fast = close.ewm(span=TREND_EMA_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=TREND_EMA_SLOW, adjust=False).mean()

        spread = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / (ema_slow.iloc[-1] + 1e-10)
        score = float(np.clip(spread / TREND_SPREAD_NORM, -1.0, 1.0))
        return score if np.isfinite(score) else 0.0
    except Exception:
        return 0.0

# ══════════════════════════════════════════════════════════════
#  NORMALIZED CANDLE METRICS — قلب رادار الشموع
# ══════════════════════════════════════════════════════════════
def compute_candle_metrics(open_, high, low, close):
    """
    يحوّل OHLC الخام إلى مقاييس موحّدة (0..1) تشريحية:
      Body %   = abs(Close - Open) / Range        → حسم الحركة أم تردد؟
      Upper %  = (High - max(C,O)) / Range        → الرفض البيعي بالأعلى
      Lower %  = (min(C,O) - Low) / Range         → دعم الثيران بالأسفل
      ClosePos = (Close - Low) / Range            → موقع الإغلاق (0=القاع، 1=القمة)
    كل القيم نسب من مدى الشمعة نفسها → شمعة 5 دقائق تقارَن بشمعة يومية مباشرة.
    الشمعة المسطحة (Range≈0) → قيم محايدة.
    """
    rng = np.asarray(high) - np.asarray(low)
    flat = rng <= 1e-12
    safe = np.where(flat, 1.0, rng)
    body_pct   = np.abs(close - open_) / safe
    upper_wick = (high - np.maximum(close, open_)) / safe
    lower_wick = (np.minimum(close, open_) - low) / safe
    close_pos  = (close - low) / safe
    if np.any(flat):
        body_pct   = np.where(flat, 0.0, body_pct)
        upper_wick = np.where(flat, 0.0, upper_wick)
        lower_wick = np.where(flat, 0.0, lower_wick)
        close_pos  = np.where(flat, 0.5, close_pos)
    return body_pct, upper_wick, lower_wick, close_pos

def classify_candle(o, c, prev_o, prev_c, body_pct, upper, lower, close_pos):
    """يصنّف الشمعة الحالية إلى نمط معروف (للعرض في الكارت فقط)."""
    bull = c > o
    if not bull:
        if upper >= 0.55 and body_pct <= 0.35:
            return "⬇️ نجمة هابطة (Shooting Star)"
        return "⬇️ شمعة هابطة"
    prev_bull = prev_c > prev_o
    if body_pct >= 0.75 and upper <= 0.12 and lower <= 0.12:
        return "🟢 Marubozu صاعد (حسم كامل)"
    if lower >= 0.55 and body_pct <= 0.35 and close_pos >= 0.60:
        return "🔨 مطرقة صاعدة (Hammer)"
    if (not prev_bull) and (c >= prev_o) and (o <= prev_c):
        return "🟢 ابتلاع صاعد (Bullish Engulfing)"
    if close_pos >= 0.80 and body_pct >= 0.40:
        return "🟢 إغلاق عند القمة (Close @ High)"
    if body_pct >= 0.40 and close_pos >= 0.65:
        return "🟢 شمعة قوية"
    return "🟢 شمعة صاعدة"

# ═══════════════════════════════════════════════
#  SWINGS & LIQUIDITY POOLS (للسياق + SL/TP)
# ═══════════════════════════════════════════════
def _rolling_sum(arr, period):
    n = len(arr)
    if n < period: return np.full(n, np.nan)
    cs = np.cumsum(arr)
    out = np.full(n, np.nan)
    out[period - 1] = cs[period - 1]
    if period < n: out[period:] = cs[period:] - cs[:-period]
    return out

def find_swing_points(high, low, left=2, right=2):
    n = len(high)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for k in range(left, n - right):
        window_h = high[k - left : k + right + 1]
        window_l = low[k - left : k + right + 1]
        if high[k] == window_h.max() and np.argmax(window_h) == left:
            swing_high[k] = True
        if low[k] == window_l.min() and np.argmin(window_l) == left:
            swing_low[k] = True
    return swing_high, swing_low

def find_liquidity_pools(high, low, swing_high, swing_low, i, lookback=50):
    start = max(0, i - lookback)
    bsl = None
    for k in range(i - 1, start - 1, -1):
        if swing_high[k]:
            level = high[k]
            if not np.any(high[k + 1 : i] > level):
                bsl = level
                break
    ssl = None
    for k in range(i - 1, start - 1, -1):
        if swing_low[k]:
            level = low[k]
            if not np.any(low[k + 1 : i] < level):
                ssl = level
                break
    if bsl is None:
        bsl = np.max(high[start:i]) if i > start else high[i]
    if ssl is None:
        ssl = np.min(low[start:i]) if i > start else low[i]
    return bsl, ssl

# ═══════════════════════════════════════════════
#  THE CANDLE RADAR SIGNAL ENGINE
# ═══════════════════════════════════════════════
def get_radar_signal(ticker, df, score_min, min_price, min_vol_avg, min_market_cap=0.0, trend_tf=None):
    """
    Candle Radar 1.0 — يرصد فرص الشراء (BUY) فقط.

    كل شمعة تُحوَّل إلى مقاييس موحّدة بالنسبة لمداها الخاص (Body/Upper/Lower/
    ClosePos) فتصبح شمعة 5 دقائق وشمعة يومية متكافئتين في التقييم تماماً.

    هيكل الأوزان:
      • Candle Anatomy 40%     (ClosePos 50% + Body 30% + Wicks 20%)
      • Volume/Activity 30-35% (RVOL 55% + اتجاه الحجم 25% + توسّع المدى 20%)
      • Context 20-25%         (استمرارية 40% + موقع المدى 30% + مساحة السيولة 30%)
      • Trend Filter 10%       — وزن ناعم إضافي فقط عند تفعيله، وليس شرط رفض.

    التدفق: Entry → SL → أقرب Liquidity TP → R:R (كما في الرادار الأول).
    """
    try:
        if df is None or len(df) < 35: return None

        open_ = df["Open"].values.astype(float)
        close = df["Close"].values.astype(float)
        high  = df["High"].values.astype(float)
        low   = df["Low"].values.astype(float)
        vol   = np.maximum(df["Volume"].values.astype(float), 0.0)

        price = close[-1]
        i = len(close) - 1

        # ═══ فلتر السعر والسيولة — RVOL يُحسب من الشموع السابقة فقط ═══
        if price < min_price: return None
        prior_window = vol[max(0, i - RVOL_LOOKBACK):i]
        avg_vol = prior_window.mean() if len(prior_window) > 0 else vol[i]
        if avg_vol < min_vol_avg: return None

        # شمعة بلا مدى فعلي (مسطحة): لا يمكن تشريحها → لا إشارة
        candle_range = high[i] - low[i]
        if not np.isfinite(candle_range) or candle_range <= 1e-9:
            return None

        # ══════════════════════════════════════════════════════
        # 1. CANDLE ANATOMY — 40%
        # ══════════════════════════════════════════════════════
        body_arr, upper_arr, lower_arr, cpos_arr = compute_candle_metrics(open_, high, low, close)
        body_pct   = float(body_arr[i])
        upper_wick = float(upper_arr[i])
        lower_wick = float(lower_arr[i])
        close_pos  = float(cpos_arr[i])
        direction  = 1.0 if close[i] > open_[i] else (-1.0 if close[i] < open_[i] else 0.0)

        # أهم مؤشر: أين أغلق السعر داخل مدى الشمعة؟
        s_close = np.clip((close_pos - CLOSE_POS_MID) / CLOSE_POS_FULL, -1.0, 1.0)
        # حسم الحركة: جسم كبير صاعد = +1، جسم كبير هابط = -1، دوجي = 0
        s_body  = direction * np.clip(body_pct / BODY_FULL, -1.0, 1.0)
        # توازن الفتائل: دعم شرائي سفلي مقابل رفض بيعي علوي
        s_wick  = np.clip((lower_wick - upper_wick) / WICK_FULL, -1.0, 1.0)

        score_anatomy = np.clip(
            (AN_W_CLOSE * s_close) + (AN_W_BODY * s_body) + (AN_W_WICK * s_wick), -1.0, 1.0
        )

        # Swings وتجمعات السيولة (تُستخدم في مساحة الصعود + SL/TP لاحقاً)
        swing_high, swing_low = find_swing_points(high, low, SWING_LEFT, SWING_RIGHT)
        bsl_pool, ssl_pool = find_liquidity_pools(high, low, swing_high, swing_low, i, LIQUIDITY_LOOKBACK)
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        tr = tr if np.isfinite(tr) and tr > 0 else price * 0.01

        # ══════════════════════════════════════════════════════
        # 2. VOLUME / ACTIVITY — 30-35%
        # ══════════════════════════════════════════════════════
        rvol = vol[i] / (avg_vol + 1e-10)
        s_rvol = np.clip((rvol - 0.7) / 2.0, -0.5, 1.0)
        if rvol < 0.4: s_rvol = -1.0

        # اتجاه الحجم: هل الشموع الصاعدة تحمل حجماً أكبر من الهابطة؟ (تجميع)
        vs = max(1, i - VOLUME_DIR_LOOKBACK)
        seg_vol = vol[vs:i+1]
        up_v = float(seg_vol[close[vs:i+1] > open_[vs:i+1]].sum())
        dn_v = float(seg_vol[close[vs:i+1] < open_[vs:i+1]].sum())
        s_voldir = np.clip((up_v - dn_v) / (up_v + dn_v + 1e-10), -1.0, 1.0)

        # توسّع المدى: هل هذه الشمعة أوسع من المعتاد؟ (Effort → Result)
        prior_rng = (high - low)[max(0, i - RVOL_LOOKBACK):i]
        avg_rng = prior_rng.mean() if len(prior_rng) > 0 else candle_range
        rng_exp = candle_range / (avg_rng + 1e-10)
        s_rngexp = np.clip((rng_exp - 0.8) / 1.2, -1.0, 1.0)

        score_activity = np.clip(
            (AC_W_RVOL * s_rvol) + (AC_W_DIR * s_voldir) + (AC_W_RANGE * s_rngexp), -1.0, 1.0
        )

        # ══════════════════════════════════════════════════════
        # 3. CONTEXT — 20-25%
        # ══════════════════════════════════════════════════════
        # أ) استمرارية: متوسط موقع الإغلاق لآخر CANDLE_CONTEXT_LEN شموع
        ctx0 = max(0, i - CANDLE_CONTEXT_LEN + 1)
        recent_cpos = float(np.nanmean(cpos_arr[ctx0:i+1]))
        s_sustain = np.clip((recent_cpos - CLOSE_POS_MID) / 0.20, -1.0, 1.0)

        # ب) موقع الإغلاق ضمن مدى آخر RANGE_POS_LOOKBACK شمعة (اختراق أم لا)
        rp0 = max(0, i - RANGE_POS_LOOKBACK + 1)
        lo_min = float(np.min(low[rp0:i+1]))
        hi_max = float(np.max(high[rp0:i+1]))
        span = hi_max - lo_min
        s_rangepos = np.clip(((price - lo_min) / (span + 1e-10) - 0.55) / 0.30, -1.0, 1.0) if span > 0 else 0.0

        # ج) مساحة الصعود حتى أقرب تجمع سيولة مقابل مساحة الهبوط
        room_up = bsl_pool - price
        room_down = price - ssl_pool
        s_room = np.clip((room_up - room_down) / (tr * 5.0 + 1e-10), -1.0, 1.0)

        score_context = np.clip(
            (CX_W_SUSTAIN * s_sustain) + (CX_W_RANGEPOS * s_rangepos) + (CX_W_ROOM * s_room), -1.0, 1.0
        )

        # ══════════════════════════════════════════════════════
        # ---- WEIGHTED EVIDENCE MODEL (Candle Radar 1.0) ----
        # فلتر الاتجاه وزن ناعم فقط: لا جلب شبكة إن كان أفضل سيناريو
        # نظرياً (اتجاه +1 كامل) لن يصل أصلاً لعتبة score_min.
        # ══════════════════════════════════════════════════════
        trend_enabled = trend_tf is not None
        w_vol = WEIGHT_VOLUME_WITH_TREND if trend_enabled else WEIGHT_VOLUME_NO_TREND
        w_ctx = WEIGHT_CONTEXT_WITH_TREND if trend_enabled else WEIGHT_CONTEXT_NO_TREND

        base_composite = float(
            (WEIGHT_CANDLE * score_anatomy) +
            (w_vol * score_activity) +
            (w_ctx * score_context)
        )

        score_trend = 0.0
        if trend_enabled:
            if base_composite + WEIGHT_TREND < score_min:
                return None
            score_trend = get_trend_score(ticker, trend_tf)  # محايد (0.0) عند الفشل، وليس رفضاً
            composite = float(np.clip(base_composite + (WEIGHT_TREND * score_trend), -1.0, 1.0))
        else:
            composite = base_composite

        # ═══ فقط إشارات الشراء (BUY) ═══
        if composite < score_min:
            return None
        sig = "BUY (CANDLE+)"

        # نمط الشمعة (للعرض فقط)
        pattern = classify_candle(open_[i], close[i], open_[i-1], close[i-1],
                                  body_pct, upper_wick, lower_wick, close_pos)

        # ═══ فلتر القيمة السوقية (فحص متأخر لتقليل طلبات الشبكة) ═══
        mcap = None
        if min_market_cap > 0:
            mcap = get_market_cap(ticker)
            if mcap is None or mcap < min_market_cap:
                return None

        # ══════════════════════════════════════════════════════════════
        # التسلسل: Entry → SL → أقرب Liquidity TP → R:R (كما في الرادار الأول)
        # ══════════════════════════════════════════════════════════════
        tp_price = (high + low + close) / 3.0
        tpv = tp_price * vol
        r_tpv = _rolling_sum(tpv, 14)
        r_vol = _rolling_sum(vol, 14)
        vwap = r_tpv[i] / (r_vol[i] + 1e-10)

        buffer = tr * 0.2

        entry = price
        sl = (min(ssl_pool, vwap) - buffer) if price > vwap else (ssl_pool - buffer)
        risk = entry - sl
        tp = bsl_pool
        if not ((bsl_pool - entry) > (risk * 1.0)):
            tp = entry + (risk * 2.0)

        if not all(np.isfinite(v) for v in (sl, tp, risk, entry)):
            return None
        if risk <= 0:
            return None
        reward = tp - entry
        if reward <= 0:
            return None

        rr = reward / risk
        if rr < MIN_RISK_REWARD:
            tp = entry + (risk * MIN_RISK_REWARD)
            reward = tp - entry
            rr = reward / risk

        return dict(
            Ticker=ticker, Signal=sig, Pattern=pattern, Price=round(entry, 2),
            AN=round(score_anatomy, 2), VLM=round(score_activity, 2), CTX=round(score_context, 2),
            TR=round(score_trend, 2) if trend_enabled else None,
            RVOL=round(rvol, 2),
            Body=round(body_pct*100), Upper=round(upper_wick*100),
            Lower=round(lower_wick*100), ClosePos=round(close_pos*100),
            TP=round(tp, 2), SL=round(sl, 2), RR=round(rr, 2),
            MarketCapB=round(mcap / 1e9, 2) if mcap else None,
            Score=round(composite, 3), _score=composite
        )
    except Exception:
        return None

# ═══════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════
st.set_page_config(page_title="Candle Radar 1.0 🕯️", page_icon="🕯️", layout="centered")

st.markdown("""
<style>
.stButton > button { height: 3rem !important; font-size: 1.1rem !important; font-weight: 700 !important; border-radius: 8px !important; }
.card-buy { background: linear-gradient(135deg,#0a2e12,#11471d); border-left: 5px solid #00e676; border-radius: 8px; padding: 12px; margin: 8px 0; color: #e0ffe0; }
.tag-buy  { background:#00e676; color:#000; border-radius:4px; padding:2px 8px; font-weight:bold; font-size:0.8rem; }
.metric-pill { background:#1e3a5f; color:#7dd3fc; border-radius:4px; padding:2px 6px; font-size:0.8rem; margin-right:5px;}
.candle-pill { background:#0f2a1a; color:#81c784; border-radius:4px; padding:2px 6px; font-size:0.8rem; margin-right:5px; }
.score-high { color: #00e676; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🕯️ Candle Radar 1.0")
st.caption("تشريح الشمعة الموحّد (40%) + الحجم والنشاط (30-35%) + السياق والموقع (20-25%) + فلتر اتجاه 1H/4H — إشارات الشراء فقط")
st.divider()

with st.expander("📖 دليل قراءة مقاييس الشمعة الموحّدة", expanded=False):
    st.markdown("""
    <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
      <tr style="background:#1e3a5f; color:#7dd3fc;"><th style="padding:6px;">المقياس</th><th style="padding:6px;">الحساب</th><th style="padding:6px;">المعنى</th></tr>
      <tr><td style="padding:6px;"><b>Range</b></td><td style="padding:6px;">High − Low</td><td style="padding:6px;">التقلب الفعلي (أساس التوحيد)</td></tr>
      <tr><td style="padding:6px;"><b>Body %</b></td><td style="padding:6px;">abs(C−O) / Range</td><td style="padding:6px;">حسم الحركة أم التردد؟</td></tr>
      <tr><td style="padding:6px;"><b>Upper %</b></td><td style="padding:6px;">(High − max(C,O)) / Range</td><td style="padding:6px;">الرفض البيعي بالأعلى</td></tr>
      <tr><td style="padding:6px;"><b>Lower %</b></td><td style="padding:6px;">(min(C,O) − Low) / Range</td><td style="padding:6px;">دعم الثيران بالأسفل</td></tr>
      <tr><td style="padding:6px;"><b>Close@ %</b></td><td style="padding:6px;">(Close − Low) / Range</td><td style="padding:6px;"><b>الأهم</b>: موقع الإغلاق (0=القاع، 100=القمة)</td></tr>
      <tr><td style="padding:6px;"><b>RVOL</b></td><td style="padding:6px;">Volume / SMA(Vol, 20)</td><td style="padding:6px;">سيولة طبيعية أم اهتمام مؤسسي؟</td></tr>
    </table>
    """, unsafe_allow_html=True)

with st.spinner("📡 جلب القائمة الأساسية..."):
    nasdaq_tuple, source, err = fetch_all_nasdaq_symbols()
    nasdaq_all = list(nasdaq_tuple)

with st.expander("⚙️ إعدادات رادار الشموع", expanded=True):
    tf = st.selectbox("⏱️ الإطار الزمني (Timeframe)", list(TF_CONFIG.keys()), index=2, format_func=lambda x: TF_CONFIG[x]["label"])

    c1, c2 = st.columns(2)
    with c1:
        score_min = st.slider("🎯 حساسية الإشارة (Score Limit)", 0.20, 0.90, 0.35, 0.05, help="خفض الرقم يتيح ظهور الفرص بشكل أسرع وأسهل. رفعه يشدد الفلترة على أقوى الشموع فقط.")
        max_stocks = st.selectbox("📊 عدد الأسهم للمسح", [50, 100, 200, 500, 1000, 2000, 3000, 4000], index=1)
    with c2:
        min_price = st.number_input("💵 الحد الأدنى للسعر ($)", value=5.0, step=1.0)
        min_vol_avg = st.number_input("💧 أدنى سيولة متوسطة", value=5000, step=1000)

    min_market_cap_b = st.number_input(
        "🏢 الحد الأدنى للقيمة السوقية (مليار $)",
        value=DEFAULT_MIN_MARKET_CAP_B, step=0.5, min_value=0.0,
        help="يتم تجاهل أي شركة أقل من هذه القيمة السوقية (ضع 0 لتعطيل الفلتر)."
    )
    min_market_cap = min_market_cap_b * 1_000_000_000

    trend_choice = st.selectbox(
        "🧭 فلتر الاتجاه العام (Trend Filter)",
        ["1h", "4h", "تعطيل"],
        index=0,
        format_func=lambda x: {"1h": "🕒 1 ساعة (1H)", "4h": "📈 4 ساعات (4H)", "تعطيل": "🚫 بدون فلتر اتجاه"}[x],
        help="وزن إضافي ناعم (10%) يعكس اتجاه السهم على فريم أعلى (EMA20 مقابل EMA50). لا يرفض أي إشارة بمفرده."
    )
    trend_tf = None if trend_choice == "تعطيل" else trend_choice

    search_input = st.text_input("🔑 إضافة أسهم مخصصة (مفصولة بفاصلة)", placeholder="مثال: AAPL, TSLA, NVDA")

custom_tickers = [x.strip().upper() for x in re.split(r'[,\s]+', search_input) if x.strip()]
scan_list = list(dict.fromkeys(custom_tickers + nasdaq_all[:max_stocks]))

mcap_note = f"وتجاهل الشركات أقل من **{min_market_cap_b:g} مليار $**" if min_market_cap > 0 else "بدون فلتر قيمة سوقية"
trend_note = f"مع فلتر اتجاه **{trend_tf.upper()}** (وزن ناعم)" if trend_tf else "بدون فلتر اتجاه"
st.info(f"سيتم فحص **{len(scan_list)}** سهم على فريم **{tf}** — رصد الشموع القوية (فرص **الشراء فقط**) {mcap_note}، {trend_note}.")
if max_stocks >= 1000:
    st.caption("⚠️ مسح عدد كبير من الأسهم قد يستغرق وقتاً أطول بسبب حدود طلبات Yahoo Finance.")

if st.button("🔍 SCAN MARKET NOW", type="primary", use_container_width=True):
    start_time = time.time()
    results = []

    bar = st.progress(0)
    status_text = st.empty()

    chunks = [scan_list[i:i + CHUNK_SIZE] for i in range(0, len(scan_list), CHUNK_SIZE)]

    for idx, chunk in enumerate(chunks):
        status_text.text(f"📡 فحص تشريح الشموع الموحّد... الحزمة {idx+1}/{len(chunks)}")

        cfg = TF_CONFIG[tf]
        raw_data = download_raw_batch(
            tuple(chunk), cfg["yf_interval"], cfg["yf_period"],
            status_cb=lambda msg: status_text.text(msg),
            resample_rule=cfg.get("resample")
        )

        for ticker, df in raw_data.items():
            sig = get_radar_signal(ticker, df, score_min, min_price, min_vol_avg, min_market_cap, trend_tf)
            if sig:
                results.append(sig)

        bar.progress((idx + 1) / len(chunks))

    status_text.empty()
    bar.empty()

    results = sorted(results, key=lambda x: x["_score"], reverse=True)

    st.success(f"✅ اكتمل المسح في {time.time()-start_time:.1f} ثانية. تم رصد {len(results)} فرصة شراء.")

    for r in results:
        mcap_pill = f'<span class="metric-pill">Market Cap: ${r["MarketCapB"]}B</span>' if r.get("MarketCapB") else ""
        trend_pill = f'<span class="metric-pill">Trend: {r["TR"]}</span>' if r.get("TR") is not None else ""
        html = f"""
        <div class="card-buy">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:1.3rem; font-weight:bold; letter-spacing:1px;">{r['Ticker']}</span>
                <span class="tag-buy">{r['Signal']}</span>
            </div>
            <div style="font-size:0.9rem; color:#a5d6a7; margin-bottom:8px;">🕯️ {r['Pattern']}</div>
            <div style="font-size:0.95rem; margin-bottom:8px;">
                💵 Price: <b>${r['Price']}</b> &nbsp;|&nbsp;
                🎯 TP: <b style="color:#00e676">${r['TP']}</b> &nbsp;|&nbsp;
                🛡️ SL: <b style="color:#ff5252">${r['SL']}</b> &nbsp;|&nbsp;
                ⚖️ R:R <b>{r['RR']}</b>
            </div>
            <div>
                <span class="metric-pill">Anatomy (AN): {r['AN']}</span>
                <span class="metric-pill">Volume (VLM): {r['VLM']}</span>
                <span class="metric-pill">Context (CTX): {r['CTX']}</span>
                <span class="candle-pill">Body: {r['Body']}%</span>
                <span class="candle-pill">Upper: {r['Upper']}%</span>
                <span class="candle-pill">Lower: {r['Lower']}%</span>
                <span class="candle-pill">Close@: {r['ClosePos']}%</span>
                <span class="candle-pill">RVOL: {r['RVOL']}x</span>
                {trend_pill}
                {mcap_pill}
                <span class="metric-pill">Score: <span class="score-high">{r['Score']}</span></span>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    if not results:
        st.warning("لم يتم رصد أي شمعة شراء قوية تطابق المعايير الحالية. جرّب خفض (Score Limit) أو تغيير الفريم.")
