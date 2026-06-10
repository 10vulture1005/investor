import pandas as pd
import numpy as np

def calculate_wilders_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
        
    prev_close = df['Close'].shift(1)
    
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = np.zeros(len(df))
    atr[:] = np.nan
    
    if len(df) >= period:
        atr[period-1] = tr.iloc[:period].mean()
        tr_values = tr.values
        for i in range(period, len(df)):
            atr[i] = (atr[i-1] * (period - 1) + tr_values[i]) / period
            
    return pd.Series(atr, index=df.index)

def calculate_rsi(series: pd.Series, period: int = 3) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100).where(loss != 0, 100)
    rsi = rsi.where((gain != 0) | (loss != 0), 50)
    
    return rsi

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
        
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # ATRs
    df['ATR_14'] = calculate_wilders_atr(df, period=14)
    df['ATR_20'] = calculate_wilders_atr(df, period=20)
    
    # Keltner Channels (20 EMA +/- 1.5 ATR(20))
    df['KC_Upper'] = df['EMA_20'] + (1.5 * df['ATR_20'])
    df['KC_Lower'] = df['EMA_20'] - (1.5 * df['ATR_20'])
    
    # RSI(3)
    df['RSI_3'] = calculate_rsi(df['Close'], period=3)
    
    # Momentum (90 days = ~4 months)
    df['ROC_90'] = df['Close'].pct_change(periods=90) * 100
    
    return df
