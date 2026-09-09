import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 1. دوال التحليل الكمي (Feature & Target Engineering)
# ==========================================
@st.cache_data(ttl=3600)
def load_and_prep_data(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def engineer_features(df):
    df = df.copy()
    df['Range'] = df['High'] - df['Low']
    df['Range'] = df['Range'].replace(0, 1e-10) 
    
    # تشريح الشمعة
    df['Body_%'] = abs(df['Close'] - df['Open']) / df['Range'] * 100
    df['Upper_Wick_%'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Range'] * 100
    df['Lower_Wick_%'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Range'] * 100
    df['Close_Position_%'] = (df['Close'] - df['Low']) / df['Range'] * 100
    
    # السياق
    df['SMA_Volume_20'] = df['Volume'].rolling(20).mean()
    df['Relative_Volume'] = df['Volume'] / (df['SMA_Volume_20'] + 1e-10)
    
    return df

def engineer_targets(df, n_candles):
    df = df.copy()
    # العائد بعد N شموع
    df[f'Target_Return_{n_candles}C_%'] = (df['Close'].shift(-n_candles) - df['Close']) / df['Close'] * 100
    
    # أقصى انعكاس سلبي (Drawdown) خلال الـ N شموع
    future_lows = df['Low'].iloc[::-1].rolling(n_candles).min().iloc[::-1]
    df[f'Max_Drawdown_{n_candles}C_%'] = (future_lows.shift(-1) - df['Close']) / df['Close'] * 100
    
    return df

# ==========================================
# 2. واجهة Streamlit
# ==========================================
st.set_page_config(page_title="Quantitative Candlestick Engine", layout="wide")

st.title("📊 محرك التحليل الكمي للشموع اليابانية")
st.markdown("ابحث عن **الميزة الإحصائية (Edge)** من خلال اختبار أشكال الشموع والسيولة ضد البيانات التاريخية الحقيقية.")

# إعدادات البيانات
col1, col2, col3, col4 = st.columns(4)
with col1:
    ticker = st.text_input("رمز السهم (Ticker)", value="AAPL")
with col2:
    interval = st.selectbox("الإطار الزمني", ["1h", "1d", "1wk"], index=1)
with col3:
    period = st.selectbox("مدة البيانات", ["6mo", "1y", "2y", "5y"], index=2)
with col4:
    n_candles = st.number_input("قياس المستقبل (بعد كم شمعة؟)", min_value=1, max_value=20, value=5)

# جلب ومعالجة البيانات
if st.button("تحميل ومعالجة البيانات", type="primary"):
    with st.spinner("جاري جلب البيانات وحساب الميزات..."):
        raw_df = load_and_prep_data(ticker, period, interval)
        if raw_df.empty:
            st.error("لم يتم العثور على بيانات. تأكد من الرمز.")
        else:
            features_df = engineer_features(raw_df)
            final_df = engineer_targets(features_df, n_candles).dropna()
            st.session_state['data'] = final_df
            st.success(f"تمت معالجة {len(final_df)} شمعة بنجاح.")

st.divider()

# قسم الفلاتر التفاعلية (صناعة الاستراتيجية)
if 'data' in st.session_state:
    st.subheader("🛠️ بناء الاستراتيجية (تحديد شروط الشمعة)")
    
    df = st.session_state['data']
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_lower_wick = st.slider("الحد الأدنى للذيل السفلي (%)", 0, 100, 50)
    with c2:
        max_body = st.slider("الحد الأقصى لحجم الجسم (%)", 0, 100, 30)
    with c3:
        min_close_pos = st.slider("الحد الأدنى لموقع الإغلاق (%)", 0, 100, 70, help="100% يعني إغلاق عند أعلى نقطة (High)")
    with c4:
        min_rvol = st.slider("الحد الأدنى للسيولة النسبية (RVOL)", 0.0, 5.0, 1.2, 0.1)

    # تطبيق الفلاتر
    condition = (
        (df['Lower_Wick_%'] >= min_lower_wick) &
        (df['Body_%'] <= max_body) &
        (df['Close_Position_%'] >= min_close_pos) &
        (df['Relative_Volume'] >= min_rvol)
    )
    
    filtered_df = df[condition].copy()
    
    st.divider()
    
    # عرض النتائج الإحصائية
    st.subheader(f"📈 نتائج الاختبار الإحصائي (خلال الـ {n_candles} شموع التالية)")
    
    total_signals = len(filtered_df)
    
    if total_signals > 0:
        win_rate = (len(filtered_df[filtered_df[f'Target_Return_{n_candles}C_%'] > 0]) / total_signals) * 100
        avg_return = filtered_df[f'Target_Return_{n_candles}C_%'].mean()
        avg_drawdown = filtered_df[f'Max_Drawdown_{n_candles}C_%'].mean()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("عدد الإشارات المطابقة", f"{total_signals} شمعة")
        m2.metric("نسبة الصفقات الرابحة (Win Rate)", f"{win_rate:.1f}%")
        m3.metric("متوسط العائد المتوقع", f"{avg_return:.2f}%")
        m4.metric("متوسط الانعكاس (Drawdown)", f"{avg_drawdown:.2f}%", help="يساعدك في تحديد مكان الـ Stop Loss")
        
        st.write("### تفاصيل الشموع التي طابقت الشروط:")
        cols_to_display = ['Close', 'Relative_Volume', 'Lower_Wick_%', 'Close_Position_%', f'Target_Return_{n_candles}C_%', f'Max_Drawdown_{n_candles}C_%']
        st.dataframe(filtered_df[cols_to_display].style.format("{:.2f}"))
        
    else:
        st.warning("لا توجد شموع تطابق هذه الشروط القاسية. حاول تخفيف الشروط من المنزلقات أعلاه.")
