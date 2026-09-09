"""
Cash Flow + Liquidity + Momentum Radar — Streamlit Web App
رادار التدفق النقدي والسيولة والزخم (متوافق مع الهواتف)
--- نسخة: فرص الشراء (BUY) فقط، تم تجاهل إشارات البيع (SELL) ---
--- تم إضافة فلتر القيمة السوقية: تجاهل الشركات الصغيرة (أقل من الحد المحدد) ---
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
MFI_PERIOD         = 14   # فترة مؤشر التدفق النقدي القياسي (Money Flow Index)
RVOL_LOOKBACK      = 20   # متوسط الحجم للمقارنة (لا يشمل الشمعة الحالية)
OBV_SLOPE_LEN      = 5    # قياس تسارع الزخم في آخر الشموع
VOL_NORM_LOOKBACK  = 20   # نافذة حساب التذبذب (volatility) لتطبيع OBV والزخم إحصائياً
SWING_LEFT         = 2    # عدد الشموع على يسار القمة/القاع لتأكيده كـ Swing Point
SWING_RIGHT        = 2    # عدد الشموع على يمين القمة/القاع لتأكيده (لا يمكن تأكيد قمة قبل مرور هذا العدد)
LIQUIDITY_LOOKBACK = 50   # مدى البحث للخلف عن آخر تجمع سيولة (Swing) لم يتم "امتصاصه" بعد
MIN_RISK_REWARD    = 1.2  # أدنى نسبة عائد/مخاطرة مقبولة (يتم تمديد TP للوصول لها، لا رفض الإشارة)

# ── Cash Flow Radar 3.0 — أوزان النموذج المرجّح ──
# عند تفعيل فلتر الاتجاه العام (Trend Filter): يُعاد توزيع الوزن بحيث
# يحصل الاتجاه على وزن إضافي (10%) دون رفض أي إشارة بسببه (إضافة/خصم ناعم فقط).
WEIGHT_CASH_FLOW            = 0.40   # ثابت دائماً: Cash Flow 40% (CMF + MFI)
WEIGHT_LIQUIDITY_WITH_TREND = 0.30   # Liquidity عند تفعيل فلتر الاتجاه (ضمن نطاق 30-35%)
WEIGHT_MOMENTUM_WITH_TREND  = 0.20   # Momentum عند تفعيل فلتر الاتجاه (ضمن نطاق 20-25%)
WEIGHT_TREND                = 0.10   # Trend Filter (1H/4H) كوزن ناعم إضافي
WEIGHT_LIQUIDITY_NO_TREND   = 0.35   # Liquidity عند تعطيل فلتر الاتجاه
WEIGHT_MOMENTUM_NO_TREND    = 0.25   # Momentum عند تعطيل فلتر الاتجاه

TREND_EMA_FAST   = 20     # المتوسط السريع لفلتر الاتجاه العام
TREND_EMA_SLOW   = 50     # المتوسط البطيء لفلتر الاتجاه العام
TREND_SPREAD_NORM = 0.03  # نسبة تباعد بين المتوسطين تعادل أقصى قوة اتجاه (±1.0)

CHUNK_SIZE = 50  
CACHE_TTL  = 60

DEFAULT_MIN_MARKET_CAP_B = 1.0   # الحد الأدنى الافتراضي للقيمة السوقية بالمليار دولار

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
MAX_RETRIES        = 4     # أقصى عدد محاولات إعادة عند فشل الشبكة
BACKOFF_BASE        = 1.5  # ثانية (أساس التصاعد الأسي)
BACKOFF_MAX_SLEEP   = 20.0 # سقف زمن الانتظار بين المحاولات

def _download_chunk_with_backoff(chunk, yf_interval, yf_period, status_cb=None):
    """
    يحاول تحميل حزمة أسهم مع تأخير تصاعدي (exponential backoff + jitter)
    عند حدوث أخطاء شبكة أو تقييد معدل الطلبات (rate limiting) من Yahoo Finance.
    """
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
                # استنفدنا المحاولات: نتخلى عن هذه الحزمة ونكمل الفحص بدلاً من إيقاف البرنامج بالكامل
                if status_cb:
                    status_cb(f"⚠️ تعذر تحميل حزمة بعد {MAX_RETRIES} محاولات، سيتم تجاوزها.")
                return None
            # تصاعد أسي: 1.5^attempt مع سقف أقصى + عشوائية بسيطة (jitter) لتفادي تصادم الطلبات
            sleep_time = min(BACKOFF_BASE ** attempt, BACKOFF_MAX_SLEEP)
            sleep_time += np.random.uniform(0, 0.5)
            if status_cb:
                status_cb(f"🌐 خطأ شبكة/تقييد معدل الطلبات، إعادة المحاولة {attempt}/{MAX_RETRIES} بعد {sleep_time:.1f}ث...")
            time.sleep(sleep_time)

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    يجمّع شموع أقصر (مثل 1H) إلى شموع أطول (مثل 4H) عبر pandas resample.
    يعتمد على أن الفهرس (index) هو DatetimeIndex قادم من yfinance.
    """
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna()
    return out

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
    """
    يجلب القيمة السوقية للسهم. يُستدعى فقط للأسهم التي اجتازت
    فلاتر السعر/السيولة/الإشارة لتقليل عدد طلبات الشبكة.
    يُرجع None إذا تعذر الجلب (وفي هذه الحالة يتم استبعاد السهم احتياطياً).
    """
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
    """
    يقيس اتجاه السهم على فريم أعلى (1H أو 4H) عبر تباعد متوسطين متحركين
    أسّيين (EMA20 مقابل EMA50). كلما زاد تباعد EMA السريع فوق البطيء
    زادت قوة الاتجاه الصاعد، والعكس صحيح.
    هذا "فلتر ناعم" (soft score) وليس شرط قبول/رفض: عند تعذر الجلب أو
    نقص البيانات يُرجع 0.0 (محايد) بدل استبعاد السهم، حتى لا تقل عدد
    الإشارات بسبب مشاكل شبكة عابرة.
    """
    try:
        cfg = TF_CONFIG[trend_tf]
        raw = yf.download(
            tickers=ticker, period=cfg["yf_period"], interval=cfg["yf_interval"],
            progress=False, auto_adjust=True
        )
        if raw is None or raw.empty:
            return 0.0
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):  # قد يعيد yfinance عمود متعدد المستويات لسهم واحد
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

def calculate_mfi(high, low, close, vol, period=14):
    """
    تطبيق قياسي لمؤشر Money Flow Index (MFI):
    1) Typical Price = (H+L+C)/3
    2) Raw Money Flow = Typical Price * Volume
    3) مقارنة TP الحالي بالسابق لتصنيف التدفق كموجب أو سالب
    4) Money Ratio = مجموع التدفق الموجب / مجموع التدفق السالب على مدى "period"
    5) MFI = 100 - 100/(1+Money Ratio)
    محسوبة بشكل متجه (vectorized) عبر نافذة متحركة حقيقية بدل عدد ثابت من الشموع.
    """
    n = len(close)
    tp = (high + low + close) / 3.0
    raw_mf = tp * vol

    tp_diff = tp[1:] - tp[:-1]
    pos_mf = np.where(tp_diff > 0, raw_mf[1:], 0.0)
    neg_mf = np.where(tp_diff < 0, raw_mf[1:], 0.0)

    # محاذاة الطول الأصلي (n) بإضافة صفر في البداية (لا يوجد تدفق قبل أول شمعة)
    pos_full = np.concatenate(([0.0], pos_mf))
    neg_full = np.concatenate(([0.0], neg_mf))

    pos_sum = _rolling_sum(pos_full, period)
    neg_sum = _rolling_sum(neg_full, period)

    mfi = np.full(n, 50.0)  # قيمة محايدة افتراضية عند نقص البيانات (أول period-1 شمعة)
    valid = ~np.isnan(pos_sum) & ~np.isnan(neg_sum)
    if valid.any():
        p, ng = pos_sum[valid], neg_sum[valid]
        mfi[valid] = np.where(ng <= 1e-10, 100.0, 100.0 - (100.0 / (1.0 + p / (ng + 1e-10))))
    return mfi

def find_swing_points(high, low, left=2, right=2):
    """
    يحدد Swing High / Swing Low حقيقيين (Fractals): قمة أعلى من "left" شموع
    قبلها و"right" شموع بعدها، وقاع أدنى من محيطه بنفس الطريقة.
    لا يمكن تأكيد أي Swing إلا بعد مرور "right" شموع عليه (كما يحدث فعلياً في التداول).
    """
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
    """
    يبحث للخلف عن أقرب Swing High/Low لم يتم "امتصاصه" بعد (Unmitigated):
    أي أن السعر لم يتجاوزه منذ تشكّله وحتى الشمعة الحالية.
    هذا هو التعريف الفعلي لـ "تجمّع السيولة" (Liquidity Pool) — مستوى ينتظر فيه
    أمر إيقاف/دخول كثيف لم يتم تفعيله بعد. إن لم يُعثر على أي منها ضمن النطاق،
    يتم الرجوع لأعلى/أدنى سعر في النطاق كحل احتياطي (fallback) فقط.
    """
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
    # fallback احتياطي فقط إن لم يُعثر على Swing غير ممسوح ضمن النطاق
    if bsl is None:
        bsl = np.max(high[start:i]) if i > start else high[i]
    if ssl is None:
        ssl = np.min(low[start:i]) if i > start else low[i]
    return bsl, ssl

def get_radar_signal(ticker, df, score_min, min_price, min_vol_avg, min_market_cap=0.0, trend_tf=None):
    """
    Cash Flow Radar 3.0 — ترصد هذه الدالة فرص الشراء (BUY) فقط.

    هيكل الأوزان:
      • Cash Flow  40%  (CMF 60% + MFI 40%)
      • Liquidity  30-35% (RVOL 60% + Liquidity/Swing Structure 40%)
      • Momentum   20-25% (Price momentum 70% + OBV 30%)
      • Trend Filter (1H/4H) 10% — وزن ناعم إضافي فقط عند تفعيله، وليس شرط رفض.

    التدفق: Entry → SL → أقرب Liquidity TP → R:R.
    لا يوجد أي رفض غير ضروري للإشارات: فلتر الاتجاه إضافة/خصم ناعم على
    الـ Score فقط، وفشل جلبه لا يستبعد السهم (يُعامل كمحايد = 0).
    """
    try:
        if df is None or len(df) < 35: return None
        
        close = df["Close"].values.astype(float)
        high  = df["High"].values.astype(float)
        low   = df["Low"].values.astype(float)
        vol   = np.maximum(df["Volume"].values.astype(float), 0.0)
        
        price = close[-1]
        i = len(close) - 1
        
        # ══════════════════════════════════════════════════════
        # فلتر السعر والسيولة — RVOL/متوسط الحجم يُحسب من الشموع
        # السابقة فقط، بدون تضمين الشمعة الحالية.
        # ══════════════════════════════════════════════════════
        if price < min_price: return None
        prior_window = vol[max(0, i - RVOL_LOOKBACK):i]  # يستبعد الشمعة i نفسها
        avg_vol = prior_window.mean() if len(prior_window) > 0 else vol[i]
        if avg_vol < min_vol_avg: return None

        # ══════════════════════════════════════════════════════
        # 1. CASH FLOW — 40% (CMF 60% + MFI 40%)
        # ══════════════════════════════════════════════════════
        cmf_arr = calculate_cmf(high, low, close, vol, CMF_PERIOD)
        cmf_val = cmf_arr[i]

        mfi_arr = calculate_mfi(high, low, close, vol, MFI_PERIOD)
        mfi_val = mfi_arr[i]

        s_cf_cmf = np.clip(cmf_val / 0.15, -1.0, 1.0)
        s_cf_mfi = np.clip((mfi_val - 50) / 30.0, -1.0, 1.0)
        score_cash_flow = (s_cf_cmf * 0.6) + (s_cf_mfi * 0.4)

        # ══════════════════════════════════════════════════════
        # تحديد Swing Points وتجمعات السيولة (Liquidity Pools) مبكراً،
        # لاستخدامها في: (أ) درجة السيولة الهيكلية أدناه، و(ب) SL/TP لاحقاً.
        # ══════════════════════════════════════════════════════
        swing_high, swing_low = find_swing_points(high, low, SWING_LEFT, SWING_RIGHT)
        bsl_pool, ssl_pool = find_liquidity_pools(high, low, swing_high, swing_low, i, LIQUIDITY_LOOKBACK)
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        tr = tr if np.isfinite(tr) and tr > 0 else price * 0.01

        # ══════════════════════════════════════════════════════
        # 2. LIQUIDITY — 30-35% (RVOL 60% + Liquidity/Swing Structure 40%)
        # ══════════════════════════════════════════════════════
        rvol = vol[i] / (avg_vol + 1e-10)
        score_liquidity_rvol = np.clip((rvol - 0.7) / 2.0, -0.5, 1.0)
        if rvol < 0.4: score_liquidity_rvol = -1.0

        # درجة هيكلية: هل المساحة نحو أقرب سيولة علوية (هدف محتمل) أكبر من
        # المسافة نحو أقرب سيولة سفلية (دعم/وقف قريب)؟ هذا انعكاس حقيقي
        # لبنية السيولة (Swing) بدل الاعتماد فقط على RVOL.
        room_up = bsl_pool - price
        room_down = price - ssl_pool
        score_liquidity_structure = np.clip((room_up - room_down) / (tr * 5.0 + 1e-10), -1.0, 1.0)

        score_liquidity = np.clip((score_liquidity_rvol * 0.6) + (score_liquidity_structure * 0.4), -1.0, 1.0)

        # ══════════════════════════════════════════════════════
        # 3. MOMENTUM — 20-25% (Price momentum 70% + OBV 30%)
        # تطبيع إحصائي (z-score) لكل من تغيّر السعر وتغيّر OBV بالنسبة
        # لتذبذبهما الطبيعي على مدى VOL_NORM_LOOKBACK شمعة.
        # ══════════════════════════════════════════════════════
        d = np.sign(np.diff(close))
        obv = np.empty(len(close))
        obv[0] = vol[0]
        for j in range(1, len(close)): obv[j] = obv[j - 1] + d[j - 1] * vol[j]
        obv_step = np.diff(obv)

        obv_change = obv[i] - obv[i - OBV_SLOPE_LEN]
        recent_obv_steps = obv_step[max(0, i - VOL_NORM_LOOKBACK):i]
        obv_std = recent_obv_steps.std() if len(recent_obv_steps) >= 5 else np.nan
        if not np.isfinite(obv_std) or obv_std < 1e-9:
            obv_z_score = 0.0
        else:
            expected_obv_dispersion = obv_std * np.sqrt(OBV_SLOPE_LEN)
            obv_z_score = obv_change / (expected_obv_dispersion + 1e-10)
        score_obv = np.clip(obv_z_score / 3.0, -1.0, 1.0)

        returns = np.diff(close) / close[:-1]
        prc_change = (close[i] - close[i - OBV_SLOPE_LEN]) / (close[i - OBV_SLOPE_LEN] + 1e-10)
        recent_returns = returns[max(0, i - VOL_NORM_LOOKBACK):i]
        ret_std = recent_returns.std() if len(recent_returns) >= 5 else np.nan
        if not np.isfinite(ret_std) or ret_std < 1e-9:
            prc_z_score = 0.0
        else:
            expected_price_dispersion = ret_std * np.sqrt(OBV_SLOPE_LEN)
            prc_z_score = prc_change / (expected_price_dispersion + 1e-10)
        score_price = np.clip(prc_z_score / 3.0, -1.0, 1.0)

        score_momentum = np.clip((score_price * 0.7) + (score_obv * 0.3), -1.0, 1.0)

        # ══════════════════════════════════════════════════════
        # ---- WEIGHTED EVIDENCE MODEL (Cash Flow Radar 3.0) ----
        # فلتر الاتجاه (Trend Filter) وزن ناعم إضافي فقط: نحسب الـ composite
        # الأساسي أولاً، وإن لم يكن هناك حتى نظرياً (بأفضل اتجاه ممكن) أمل
        # ببلوغ score_min نتجنب استدعاء الشبكة إطلاقاً لجلب الاتجاه.
        # ══════════════════════════════════════════════════════
        trend_enabled = trend_tf is not None
        w_liq = WEIGHT_LIQUIDITY_WITH_TREND if trend_enabled else WEIGHT_LIQUIDITY_NO_TREND
        w_mom = WEIGHT_MOMENTUM_WITH_TREND if trend_enabled else WEIGHT_MOMENTUM_NO_TREND

        base_composite = float(
            (WEIGHT_CASH_FLOW * score_cash_flow) +
            (w_liq * score_liquidity) +
            (w_mom * score_momentum)
        )

        score_trend = 0.0
        if trend_enabled:
            # لا داعي لجلب الاتجاه إن كان أفضل سيناريو ممكن (اتجاه صاعد كامل +1)
            # لن يوصل الـ composite أصلاً لعتبة score_min — توفير طلبات شبكة.
            if base_composite + WEIGHT_TREND < score_min:
                return None
            score_trend = get_trend_score(ticker, trend_tf)  # محايد (0.0) عند الفشل، وليس رفضاً
            composite = float(np.clip(base_composite + (WEIGHT_TREND * score_trend), -1.0, 1.0))
        else:
            composite = base_composite

        # ══════════════════════════════════════════════════════
        # فقط إشارات الشراء (BUY): يتم تجاهل أي شيء غير ذلك بالكامل
        # ══════════════════════════════════════════════════════
        if composite < score_min:
            return None
        sig = "BUY (CASH FLOW+)"

        # ══════════════════════════════════════════════════════
        # فلتر القيمة السوقية: تجاهل الشركات الصغيرة (Small Caps)
        # يتم الفحص هنا فقط (بعد اجتياز باقي الشروط) لتقليل عدد
        # الطلبات الإضافية للشبكة على yfinance.
        # ══════════════════════════════════════════════════════
        mcap = None
        if min_market_cap > 0:
            mcap = get_market_cap(ticker)
            if mcap is None or mcap < min_market_cap:
                return None

        # ══════════════════════════════════════════════════════════════
        # التسلسل المطلوب: Entry → SL → أقرب Liquidity TP → R:R
        # SL: أقرب تجمع سيولة سفلي (ssl_pool) أو VWAP أيهما أقرب، مع هامش أمان.
        # TP: أقرب تجمع سيولة علوي غير ممتص (bsl_pool)، ويُمدَّد فقط إذا كان
        #     قريباً جداً بما لا يحقق حداً أدنى معقولاً من العائد/المخاطرة
        #     (لا يتم رفض الإشارة أبداً بسبب ذلك — فقط تعديل الهدف).
        # ══════════════════════════════════════════════════════════════
        tp_price = (high + low + close) / 3.0
        tpv = tp_price * vol
        r_tpv = _rolling_sum(tpv, 14)
        r_vol = _rolling_sum(vol, 14)
        vwap = r_tpv[i] / (r_vol[i] + 1e-10)

        buffer = tr * 0.2

        # Entry
        entry = price
        # SL
        sl = (min(ssl_pool, vwap) - buffer) if price > vwap else (ssl_pool - buffer)
        risk = entry - sl
        # TP = أقرب Liquidity Pool علوي
        tp = bsl_pool
        if not ((bsl_pool - entry) > (risk * 1.0)):
            # التجمع الأقرب قريب جداً؛ نمدّ الهدف بمقياس حركة مكافئة بدل رفض الإشارة
            tp = entry + (risk * 2.0)

        # ══════════════════════════════════════════════════════════════
        # التحقق (Validation) من صلاحية TP/SL قبل قبول الإشارة
        # ══════════════════════════════════════════════════════════════
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
            Ticker=ticker, Signal=sig, Price=round(entry, 2),
            CF=round(score_cash_flow, 2), LQ=round(score_liquidity, 2), MO=round(score_momentum, 2),
            TR=round(score_trend, 2) if trend_enabled else None,
            RVOL=round(rvol, 2), CMF=round(cmf_val, 2), MFI=round(mfi_val, 1),
            TP=round(tp, 2), SL=round(sl, 2), RR=round(rr, 2),
            MarketCapB=round(mcap / 1e9, 2) if mcap else None,
            Score=round(composite, 3), _score=composite
        )
    except:
        return None

# ═══════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════
st.set_page_config(page_title="Cash Flow Radar 3.0 ⚡", page_icon="⚡", layout="centered")

st.markdown("""
<style>
.stButton > button { height: 3rem !important; font-size: 1.1rem !important; font-weight: 700 !important; border-radius: 8px !important; }
.card-buy { background: linear-gradient(135deg,#0a2e12,#11471d); border-left: 5px solid #00e676; border-radius: 8px; padding: 12px; margin: 8px 0; color: #e0ffe0; }
.tag-buy  { background:#00e676; color:#000; border-radius:4px; padding:2px 8px; font-weight:bold; font-size:0.8rem; }
.metric-pill { background:#1e3a5f; color:#7dd3fc; border-radius:4px; padding:2px 6px; font-size:0.8rem; margin-right:5px;}
.score-high { color: #00e676; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Cash Flow Radar 3.0")
st.caption("Cash Flow (40%) + Liquidity/Swing (30-35%) + Momentum (20-25%) + فلتر اتجاه 1H/4H — إشارات الشراء فقط")
st.divider()

with st.spinner("📡 جلب القائمة الأساسية..."):
    nasdaq_tuple, source, err = fetch_all_nasdaq_symbols()
    nasdaq_all = list(nasdaq_tuple)

with st.expander("⚙️ إعدادات الرادار الثلاثي", expanded=True):
    tf = st.selectbox("⏱️ الإطار الزمني (Timeframe)", list(TF_CONFIG.keys()), index=2, format_func=lambda x: TF_CONFIG[x]["label"])
    
    c1, c2 = st.columns(2)
    with c1:
        score_min = st.slider("🎯 حساسـية الإشارة (Score Limit)", 0.20, 0.90, 0.35, 0.05, help="خفض الرقم يتيح ظهور الفرص بشكل أسرع وأسهل. رفعه يشدد الفلترة على أقوى الإشارات فقط.")
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
        help="وزن إضافي ناعم (10%) يعكس اتجاه السهم على فريم أعلى (EMA20 مقابل EMA50). "
             "لا يرفض أي إشارة بمفرده — فقط يرفع أو يخفض الـ Score قليلاً، حفاظاً على تكرار الإشارات."
    )
    trend_tf = None if trend_choice == "تعطيل" else trend_choice

    search_input = st.text_input("🔑 إضافة أسهم مخصصة (مفصولة بفاصلة)", placeholder="مثال: AAPL, TSLA, NVDA")

custom_tickers = [x.strip().upper() for x in re.split(r'[,\s]+', search_input) if x.strip()]
scan_list = list(dict.fromkeys(custom_tickers + nasdaq_all[:max_stocks]))

mcap_note = f"وتجاهل الشركات أقل من **{min_market_cap_b:g} مليار $**" if min_market_cap > 0 else "بدون فلتر قيمة سوقية"
trend_note = f"مع فلتر اتجاه **{trend_tf.upper()}** (وزن ناعم)" if trend_tf else "بدون فلتر اتجاه"
st.info(f"سيتم فحص **{len(scan_list)}** سهم على فريم **{tf}** — عرض فرص **الشراء فقط** {mcap_note}، {trend_note}.")
if max_stocks >= 1000:
    st.caption("⚠️ مسح عدد كبير من الأسهم قد يستغرق وقتاً أطول بسبب حدود طلبات Yahoo Finance.")

if st.button("🔍 SCAN MARKET NOW", type="primary", use_container_width=True):
    start_time = time.time()
    results = []
    
    bar = st.progress(0)
    status_text = st.empty()
    
    chunks = [scan_list[i:i + CHUNK_SIZE] for i in range(0, len(scan_list), CHUNK_SIZE)]
    
    for idx, chunk in enumerate(chunks):
        status_text.text(f"📡 فحص تدفق السيولة والزخم... الحزمة {idx+1}/{len(chunks)}")
        
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
            <div style="font-size:0.95rem; margin-bottom:8px;">
                💵 Price: <b>${r['Price']}</b> &nbsp;|&nbsp; 
                🎯 TP: <b style="color:#00e676">${r['TP']}</b> &nbsp;|&nbsp; 
                🛡️ SL: <b style="color:#ff5252">${r['SL']}</b> &nbsp;|&nbsp;
                ⚖️ R:R <b>{r['RR']}</b>
            </div>
            <div>
                <span class="metric-pill">CashFlow (CF): {r['CF']}</span>
                <span class="metric-pill">Liquidity (LQ): {r['LQ']}</span>
                <span class="metric-pill">Momentum (MO): {r['MO']}</span>
                <span class="metric-pill">CMF: {r['CMF']}</span>
                <span class="metric-pill">MFI: {r['MFI']}</span>
                {trend_pill}
                {mcap_pill}
                <span class="metric-pill">Score: <span class="score-high">{r['Score']}</span></span>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        
    if not results:
        st.warning("لم يتم اكتشاف أي فرص شراء تتطابق مع معايير التدفق النقدي الحالية. جرّب خفض (Score Limit).")
