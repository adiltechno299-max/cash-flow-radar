import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Candlestick Radar Scanner", layout="wide")

st.title("🎯 رادار الشموع الكمي (Candlestick Radar)")
st.markdown("مسح أوتوماتيكي للسوق لتحليل الشموع الأخيرة بناءً على النسب الرياضية للسيولة وبنية الشمعة.")

st.sidebar.header("⚙️ إعدادات الرادار")
default_tickers = "AAPL, MSFT, TSLA, NVDA, GOOGL, AMZN, META, AMD, NFLX, BTC-USD, ETH-USD, EURUSD=X, GC=X"
tickers_input = st.sidebar.text_area("قائمة الأصول (مفصولة بفاصلة)", value=default_tickers, height=100)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

period = st.sidebar.selectbox("فترة البيانات", ["1mo", "3mo", "6mo"], index=0)
interval = st.sidebar.selectbox("الإطار الزمني", ["1d", "1h"], index=1)

st.sidebar.divider()
st.sidebar.subheader("🛠️ الفلاتر الكمية للشمعة الأخيرة")
min_lower_wick = st.sidebar.slider("الحد الأدنى للذيل السفلي (%)", 0, 100, 50)
max_body = st.sidebar.slider("الحد الأقصى لحجم الجسم (%)", 0, 100, 30)
min_close_pos = st.sidebar.slider("الحد الأدنى لموقع الإغلاق (%)", 0, 100, 70)
min_rvol = st.sidebar.slider("الحد الأدنى للسيولة النسبية (RVOL)", 0.0, 5.0, 1.2, 0.1)

def fetch_and_analyze(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty or len(df) < 25:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.levels[1]:
                df = df.xs(ticker, axis=1, level=1)
            else:
                df.columns = df.columns.droplevel(1)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        if len(df) < 25:
            return None

        df['Range'] = df['High'] - df['Low']
        df['Range'] = df['Range'].replace(0, 1e-10)
        
        df['Body_%'] = abs(df['Close'] - df['Open']) / df['Range'] * 100
        df['Upper_Wick_%'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Range'] * 100
        df['Lower_Wick_%'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Range'] * 100
        df['Close_Position_%'] = (df['Close'] - df['Low']) / df['Range'] * 100
        
        df['SMA_Volume_20'] = df['Volume'].rolling(20).mean()
        df['Relative_Volume'] = df['Volume'] / (df['SMA_Volume_20'] + 1e-10)
        
        prev = df.iloc[-2]
        
        return {
            'Ticker': ticker,
            'Time': str(df.index[-2]),
            'Close': float(prev['Close']),
            'Body_%': float(prev['Body_%']),
            'Lower_Wick_%': float(prev['Lower_Wick_%']),
            'Close_Position_%': float(prev['Close_Position_%']),
            'Relative_Volume': float(prev['Relative_Volume'])
        }
    except Exception:
        return None

if st.button("🚀 تشغيل الرادار", type="primary"):
    results = []
    bar = st.progress(0)
    txt = st.empty()
    
    for i, t in enumerate(tickers):
        txt.text(f"جاري الفحص: {t} ({i+1}/{len(tickers)})")
        res = fetch_and_analyze(t, period, interval)
        if res:
            results.append(res)
        bar.progress((i + 1) / len(tickers))
        
    txt.text("تم الانتهاء من الفحص!")
    bar.empty()
    
    if results:
        res_df = pd.DataFrame(results)
        
        filtered = res_df[
            (res_df['Lower_Wick_%'] >= min_lower_wick) &
            (res_df['Body_%'] <= max_body) &
            (res_df['Close_Position_%'] >= min_close_pos) &
            (res_df['Relative_Volume'] >= min_rvol)
        ].copy()
        
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("الأصول المفحوصة بنجاح", len(res_df))
        c2.metric("المطابقة للشروط", len(filtered))
        
        st.subheader("📡 النتائج المطابقة:")
        if not filtered.empty:
            st.dataframe(filtered.style.format({
                'Close': "{:.2f}",
                'Body_%': "{:.1f}%",
                'Lower_Wick_%': "{:.1f}%",
                'Close_Position_%': "{:.1f}%",
                'Relative_Volume': "{:.2f}x"
            }), use_container_width=True)
        else:
            st.warning("لا توجد أصول تطابق الشروط حالياً. جرب تخفيف الفلاتر.")
            
        with st.expander("جميع الأصول المفحوصة"):
            st.dataframe(res_df.style.format({
                'Close': "{:.2f}",
                'Body_%': "{:.1f}%",
                'Lower_Wick_%': "{:.1f}%",
                'Close_Position_%': "{:.1f}%",
                'Relative_Volume': "{:.2f}x"
            }), use_container_width=True)
    else:
        st.error("لم يتم العثور على بيانات صالحة.")س
