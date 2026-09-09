import numpy as np
import pandas as pd
import yfinance as yf

def engineer_candlestick_features(df):
    """
    تقوم هذه الدالة بتحويل بيانات OHLCV الخام إلى ميزات رياضية (Features)
    مستقلة عن السعر المطلق، لتسهيل التحليل الإحصائي وتعلم الآلة.
    """
    df = df.copy()
    
    # 1. المدى والسيولة الأساسية
    df['Range'] = df['High'] - df['Low']
    # لتجنب القسمة على صفر في الشموع المسطحة تماماً (Doji)
    df['Range'] = df['Range'].replace(0, 1e-10) 
    
    # 2. تشريح الشمعة (Candle Anatomy) كنسب مئوية من 0 إلى 1
    df['Body_%'] = abs(df['Close'] - df['Open']) / df['Range']
    df['Upper_Wick_%'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Range']
    df['Lower_Wick_%'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Range']
    
    # أهم مقياس: أين أغلق السعر؟ (1 = قمة الشمعة، 0 = قاع الشمعة)
    df['Close_Position_%'] = (df['Close'] - df['Low']) / df['Range']
    
    # 3. السياق (Context)
    # الحجم النسبي مقارنة بمتوسط 20 شمعة
    df['SMA_Volume_20'] = df['Volume'].rolling(20).mean()
    df['Relative_Volume'] = df['Volume'] / (df['SMA_Volume_20'] + 1e-10)
    
    # حجم الشمعة النسبي مقارنة بمتوسط المدى (ATR تقريبي) لـ 14 شمعة
    df['SMA_Range_14'] = df['Range'].rolling(14).mean()
    df['Relative_Size'] = df['Range'] / (df['SMA_Range_14'] + 1e-10)
    
    # الموقع من الاتجاه: بعد السعر عن متوسط 20 شمعة
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['Trend_Distance_%'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    
    # زخم السعر: نسبة التغير في آخر 3 شموع
    df['Past_3_Return_%'] = df['Close'].pct_change(3)
    
    return df

def engineer_future_targets(df, n_candles=5):
    """
    تقوم هذه الدالة بتحديد "الأهداف" (Targets).
    ماذا حدث للسعر خلال الـ N شموع التالية؟
    """
    df = df.copy()
    
    # 1. العائد الصافي بعد N شموع
    # shift(-n) تقوم بسحب بيانات المستقبل إلى السطر الحالي
    df[f'Target_Return_{n_candles}C_%'] = (df['Close'].shift(-n_candles) - df['Close']) / df['Close']
    
    # 2. أقصى صعود وأقصى هبوط خلال الشموع الـ N التالية (لتحديد احتمالات ضرب الوقف أو الهدف)
    # نقوم بعكس البيانات، ثم حساب الـ Rolling Max/Min، ثم إعادتها
    future_highs = df['High'].iloc[::-1].rolling(n_candles).max().iloc[::-1]
    future_lows = df['Low'].iloc[::-1].rolling(n_candles).min().iloc[::-1]
    
    # نعمل إزاحة بمقدار -1 حتى لا نحسب الشمعة الحالية ضمن المستقبل
    df[f'Max_Up_{n_candles}C_%'] = (future_highs.shift(-1) - df['Close']) / df['Close']
    df[f'Max_Down_{n_candles}C_%'] = (future_lows.shift(-1) - df['Close']) / df['Close']
    
    return df

# ==========================================
# تجربة الكود على بيانات حقيقية
# ==========================================
if __name__ == "__main__":
    print("جلب بيانات AAPL لاختبار النظام...")
    # جلب بيانات فريم 1 ساعة لآخر 60 يوم
    raw_data = yf.download("AAPL", period="60d", interval="1h", progress=False, auto_adjust=True)
    
    # إذا كان yfinance يعيد MultiIndex (كما في التحديثات الأخيرة)
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data = raw_data.xs('AAPL', axis=1, level=1)
        
    raw_data = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    # 1. تطبيق هندسة الميزات (تحويل الشموع لأرقام)
    features_df = engineer_candlestick_features(raw_data)
    
    # 2. تطبيق هندسة الأهداف (النظر للمستقبل لـ 5 شموع)
    final_df = engineer_future_targets(features_df, n_candles=5)
    
    # تنظيف البيانات من قيم NaN التي تنتج عن المتوسطات أو الشموع المستقبلية المجهولة
    final_df = final_df.dropna()
    
    # عرض الأعمدة التي تهمنا فقط لشمعة عشوائية
    cols_to_show = [
        'Close_Position_%', 'Body_%', 'Lower_Wick_%', 'Relative_Volume', 
        'Relative_Size', 'Target_Return_5C_%', 'Max_Up_5C_%', 'Max_Down_5C_%'
    ]
    
    print("\n--- عينة من قاعدة البيانات الجاهزة للتحليل الإحصائي ---")
    # نضرب في 100 لتسهيل قراءة النسب المئوية
    display_df = (final_df[cols_to_show].tail(5) * 100).round(2)
    print(display_df)
