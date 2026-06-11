import pandas as pd
import numpy as np

def calculate_vix_filter(df_vix):
    if df_vix is None or df_vix.empty:
        return pd.Series()
    
    vix_close = df_vix['Close']
    vix_multiplier = pd.Series(1.0, index=vix_close.index)
    vix_multiplier[vix_close > 20] = 0.5
    vix_multiplier[vix_close > 25] = 0.0
    
    return vix_multiplier

def calculate_sector_rs_filter(df_sector, df_nifty):
    if df_sector is None or df_sector.empty or df_nifty is None or df_nifty.empty:
        return pd.Series()
    
    idx = df_sector.index.intersection(df_nifty.index)
    sector_close = df_sector['Close'].loc[idx]
    nifty_close = df_nifty['Close'].loc[idx]
    
    sector_return_10d = sector_close.pct_change(10)
    nifty_return_10d = nifty_close.pct_change(10)
    
    rs_flag = sector_return_10d > nifty_return_10d
    return rs_flag

def calculate_market_regime_filter(df_nifty):
    """
    Filter 3 — Market Regime Filter (Replaces FII Proxy)
    Returns True if Nifty 50 is above its 50-day EMA (uptrend).
    """
    if df_nifty is None or df_nifty.empty:
        return pd.Series()
        
    df = df_nifty.copy()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['Regime_Allowed'] = df['Close'] > df['EMA_20']
    df['Macro_Crash'] = df['Close'] < df['EMA_200']
    
    return df[['Regime_Allowed', 'Macro_Crash']]

def calculate_technical_indicators(df):
    """
    Calculates ATR(14), RSI(14), and Bollinger Bands (20, 2)
    """
    if df is None or df.empty:
        return pd.DataFrame()
        
    df = df.copy()
    
    # 1. True Range & ATR(14)
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Prev_Close']),
            abs(df['Low'] - df['Prev_Close'])
        )
    )
    # Wilder's smoothing or simple rolling mean? Usually exponential for ATR
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # 2. RSI(14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. Bollinger Bands (20, 2)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    
    # 4. 50-day EMA for trend confirmation
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Uptrend'] = df['Close'] > df['EMA_50']
    
    return df
