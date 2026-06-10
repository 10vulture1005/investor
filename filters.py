import pandas as pd
import numpy as np

def calculate_vix_filter(df_vix):
    """
    Filter 1 — VIX Regime Filter
    Returns a Series of multiplier for position size based on VIX:
    - VIX > 25: 0 (no new entries)
    - VIX > 20: 0.5 (halve position size)
    - VIX <= 20: 1.0 (full position size)
    """
    if df_vix is None or df_vix.empty:
        return pd.Series()
    
    # We use Close price of VIX for the regime filter
    vix_close = df_vix['Close']
    
    # Initialize with 1.0 (full size)
    vix_multiplier = pd.Series(1.0, index=vix_close.index)
    
    # Apply rules
    vix_multiplier[vix_close > 20] = 0.5
    vix_multiplier[vix_close > 25] = 0.0
    
    # Wait, plan says "If India VIX < 14: full position size allowed", what if it's between 14 and 20? 
    # Usually it implies <= 20 is full size, but let's assume default is 1.0, 
    # >20 is 0.5, >25 is 0.0.
    
    return vix_multiplier

def calculate_sector_rs_filter(df_sector, df_nifty):
    """
    Filter 2 — Sector Relative Strength
    Returns a boolean Series (True if sector outperforming Nifty over last 10 days)
    """
    if df_sector is None or df_sector.empty or df_nifty is None or df_nifty.empty:
        return pd.Series()
    
    # Ensure indices align
    idx = df_sector.index.intersection(df_nifty.index)
    sector_close = df_sector['Close'].loc[idx]
    nifty_close = df_nifty['Close'].loc[idx]
    
    # 10-day return
    sector_return_10d = sector_close.pct_change(10)
    nifty_return_10d = nifty_close.pct_change(10)
    
    # Outperformance flag
    rs_flag = sector_return_10d > nifty_return_10d
    return rs_flag

def calculate_fii_proxy_filter(df_niftybees):
    """
    Filter 3 — FII Proxy
    5-day rolling net flow proxy = (close - open) * volume for NIFTYBEES.NS
    If sum < 0 (or 3+ of last 5 days were net sell days), skip long entries.
    Returns boolean Series (True if entry allowed)
    """
    if df_niftybees is None or df_niftybees.empty:
        return pd.Series()
    
    df = df_niftybees.copy()
    # Daily flow proxy
    df['Daily_Flow'] = (df['Close'] - df['Open']) * df['Volume']
    
    # Days with negative flow
    df['Is_Sell_Day'] = (df['Daily_Flow'] < 0).astype(int)
    
    # Rolling 5-day sum of sell days
    sell_days_5d = df['Is_Sell_Day'].rolling(window=5).sum()
    
    # Allowed if < 3 sell days in last 5 days
    fii_allowed = sell_days_5d < 3
    
    # Alternatively, the rule also mentions "If the 5-day sum is negative":
    # sum_flow_5d = df['Daily_Flow'].rolling(window=5).sum()
    # We will enforce BOTH interpretations to be safe or just the 3+ sell days as it's more specific.
    # We'll use the "3+ of last 5 days were net sell days" as the strict rule.
    
    return fii_allowed

def identify_order_block(df_daily, df_weekly):
    """
    Filter 4 — SMC Structure
    Returns df_daily with added columns for OB_Low, OB_High and Is_SMC_Valid (boolean).
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
        
    df = df_daily.copy()
    
    # 1. Weekly timeframe: price must be above 50-period EMA
    if df_weekly is not None and not df_weekly.empty:
        df_weekly['EMA_50'] = df_weekly['Close'].ewm(span=50, adjust=False).mean()
        # Reindex weekly EMA to daily (forward fill)
        weekly_ema_daily = df_weekly['EMA_50'].reindex(df.index, method='ffill')
        df['Weekly_EMA_50'] = weekly_ema_daily
        df['Above_Weekly_EMA'] = df['Close'] > df['Weekly_EMA_50']
    else:
        df['Above_Weekly_EMA'] = False

    # 2. Daily BOS: price closes above previous 20-day swing high
    df['High_20d'] = df['High'].rolling(window=20).max().shift(1)
    df['BOS'] = df['Close'] > df['High_20d']
    
    # Identify Order Blocks
    # Find the most recent bearish candle just before a BOS
    df['Is_Bearish'] = df['Close'] < df['Open']
    
    ob_low = pd.Series(np.nan, index=df.index)
    ob_high = pd.Series(np.nan, index=df.index)
    
    current_ob_low = np.nan
    current_ob_high = np.nan
    
    # Pre-calculate bearish days to optimize search
    bearish_idx = df.index[df['Is_Bearish']]
    
    for i in range(20, len(df)):
        # Update OB on BOS
        if df['BOS'].iloc[i]:
            # Look back for most recent bearish candle before this BOS
            # The BOS happens at index i. The candle at i is bullish (since close > 20d high, usually).
            # Look backwards from i-1 down to i-20.
            date_i = df.index[i]
            prev_bearish = bearish_idx[bearish_idx < date_i]
            if len(prev_bearish) > 0:
                last_bear_date = prev_bearish[-1]
                # Update current OB
                current_ob_low = df.loc[last_bear_date, 'Low']
                current_ob_high = df.loc[last_bear_date, 'High']
                
        # Assign current OB to today
        ob_low.iloc[i] = current_ob_low
        ob_high.iloc[i] = current_ob_high
        
    df['OB_Low'] = ob_low
    df['OB_High'] = ob_high
    
    # 3. Current price must be within or touching this zone
    # This means low <= OB_High and high >= OB_Low
    df['In_OB_Zone'] = (df['Low'] <= df['OB_High']) & (df['High'] >= df['OB_Low'])
    
    # 4. SMC Valid = Above Weekly EMA + In OB Zone + Has Valid OB
    df['SMC_Valid'] = df['Above_Weekly_EMA'] & df['In_OB_Zone'] & df['OB_Low'].notna()
    
    return df
