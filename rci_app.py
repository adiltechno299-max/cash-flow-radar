import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# ==========================================
# إعدادات الواجهة
# ==========================================
st.set_page_config(page_title="Candlestick Radar Scanner", layout="wide")

st.title("🎯 رادار الشموع الكمي (Candlestick Radar)")
st.markdown("يقوم هذا الرادار بمسح قائمة أصول السوق تلقائياً عبر بيانات **Yahoo Finance**، والبحث عن الشموع التي تطابق المعايير الكمية لرفض الأسعار والسيولة في آخر شمعة مغلقة.")

# ==========================================
# الشريط الجانبي للإعدادات والفلاتر
# ==========================================
st.sidebar.header("⚙️ إعدادات الرادار")

default_tickers = "AAPL, MSFT, TSLA, NVDA, GOOGL, AMZN, META, AMD, NFLX, BTC-USD, ETH-USD, EURUSD=X, GC=X"
tickers_input = st.sidebar.text_area("قائمة الرموز (مفصولة بفاصلة)", value=default_tickers, height=100)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

period = st.sidebar.selectbox("فترة البيانات التاريخية", ["1mo", "3mo", "6mo", "1y"], index=1)
interval = st.sidebar.selectbox("الإطار الزمني", ["1d", "1h"], index=0)

st.sidebar.divider()
st.sidebar.subheader("🛠️ الشروط الكمية للشمعة الأخيرة")
min_lower_wick = st.sidebar.slider("الحد الأدنى للذيل السفلي (%)", 0, 100, 50, help="رفض قوي من الأسفل (مثل البن بار)")
max_body = st.sidebar.slider("الحد الأقصى لحجم الجسم (%)", 0, 100, 30, help="جسم شمعة صغيراً نسبياً")
min_close_pos = st.sidebar.slider("الحد الأدنى لموقع الإغلاق (%)", 0, 100, 70, help="الإغلاق قرب أعلى الشمعة")
min_rvol = st.sidebar.slider("الحد الأدنى للسيولة النسبية (RVOL)", 0.0, 5.0, 1.2, 0.1, help="حجم تداول أعلى من المتوسط بـ N مرة")

# ==========================================
# إنشاء جلسة اتصال متجاوزة للحظر لـ yfinance
# ==========================================
def get_custom_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

# ==========================================
# دالة فحص وحساب الخصائص لكل أصل
# ==========================================
@st.cache_data(ttl=1800)
def scan_ticker(ticker, period, interval):
    try:
        session = get_custom_session()
        df = yf.download(ticker, period=period, interval=interval, session=session, progress=False, auto_adjust=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1)
            
        if df.empty or len(df) < 25:
            return None
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        # هندسة الميزات الكمية
        df['Range'] = df['High'] - df['Low']
        df['Range'] = df['Range'].replace(0, 1e-10)
        
        df['Body_%'] = abs(df['Close'] - df['Open']) / df['Range'] * 100
        df['Upper_Wick_%'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Range'] * 100
        df['Lower_Wick_%'] = (df['Open', 'Close'].min(axis=1) - df['Low']) / df['Range'] * 100
        df['Close_Position_%'] = (df['Close'] - df['Low']) / df['Range'] * 100
        
        df['SMA_Volume_20'] = df['Volume'].rolling(20).mean()
        df['Relative_Volume'] = df['Volume'] / (df['SMA_Volume_20'] + 1e-10)
        
        # أخذ الشمعة المكتملة الأخيرة (قبل الأخيرة في الإطار)
        prev = df.iloc[-2]
        
        return {
            'Ticker': ticker,
            'Time': df.index[-2],
            'Close': float(prev['Close']),
            'Body_%': float(prev['Body_%']),
            'Lower_Wick_%': float(prev['Lower_Wick_%']),
            'Close_Position_%': float(prev['Close_Position_%']),
            'Relative_Volume': float(prev['Relative_Volume'])
        }
    except Exception as e:
        return None

# ==========================================
# تشغيل الرادار عند الضغط على الزر
# ==========================================
if st.button("🚀 تشغيل رادار الفحص الشامل", type="primary"):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(tickers):
        status_text.text(f"جاري جلب وفحص: {t} ({i+1}/{len(tickers)})")
        res = scan_ticker(t, period, interval)
        if res:
            results.append(res)
        progress_bar.progress((i + 1) / len(tickers))
        
    status_text.text("اكتمل الفحص بنجاح!")
    progress_bar.empty()
    
    if results:
        scan_df = pd.DataFrame(results)
        
        # تصفية النتائج بناءً على الفلاتر الكمية
        matched_df = scan_df[
            (scan_df['Lower_Wick_%'] >= min_lower_wick) &
            (scan_df['Body_%'] <= max_body) &
            (scan_df['Close_Position_%'] >= min_close_pos) &
            (scan_df['Relative_Volume'] >= min_rvol)
        ].copy()
        
        st.divider()
        
        c1, c2 = st.columns(2)
        c1.metric("إجمالي الأصول التي تم فحصها بنجاح", len(scan_df))
        c2.metric("الأصول المطابقة للشروط الآن", len(matched_df))
        
        st.subheader("📡 الأصول المطابقة لإشارات الرادار:")
        if not matched_df.empty:
            st.dataframe(matched_df.style.format({
                'Close': "{:.2f}",
                'Body_%': "{:.1f}%",
                'Lower_Wick_%': "{:.1f}%",
                'Close_Position_%': "{:.1f}%",
                'Relative_Volume': "{:.2f}x"
            }), use_container_width=True)
        else:
            st.warning("لا توجد أصول تطابق هذه المعايير الصارمة في آخر شمعة. جرب خفض نسب الذيل أو السيولة من القائمة الجانبية.")
            
        with st.expander("عرض تفاصيل كافة الأصول المفحوصة"):
            st.dataframe(scan_df.style.format({
                'Close': "{:.2f}",
                'Body_%': "{:.1f}%",
                'Lower_Wick_%': "{:.1f}%",
                'Close_Position_%': "{:.1f}%",
                'Relative_Volume': "{:.2f}x"
            }), use_container_width=True)
    else:
        st.error("تعذر جلب البيانات. تأكد من اتصال الإنترنت وصحة رموز الأصول (بعض الشبكات أو المزودين يحظرون اتصالات واجهات برمجة التطبيقات المالية مباشرة).")
