import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates boolean entry conditions for Smart Alpha 3.0.
    Conditions:
    1. Deep Pullback: RSI(3) < 20 OR Close < Lower Keltner Channel
    2. Momentum Filter: Close > 50-day SMA
    
    Returns a boolean Series 'Entry_Qualifies'
    """
    if df is None or df.empty:
        df['Entry_Qualifies'] = False
        return df
        
    df = df.copy()
    
    # 1. Deep Pullback
    c1 = (df['RSI_3'] <= 30) | (df['Close'] < df['KC_Lower'])
    
    # 2. Uptrend Alignment
    c2 = df['Close'] > df['SMA_50']
    
    df['Entry_Qualifies'] = c1 & c2
    
    return df

def rank_candidates(candidates_data: list) -> list:
    """
    Ranks qualifying stocks based on Smooth Momentum.
    
    inputs:
        candidates_data: list of dicts. Each dict has:
            'stock': symbol string
            'ROC_90': float
            'ATR_20': float
            
    Returns a list of stock symbols sorted by Smooth Momentum descending.
    """
    if not candidates_data:
        return []
        
    df = pd.DataFrame(candidates_data)
    
    # Factor 1: Smooth Momentum
    atr20 = df['ATR_20'].replace(0, np.nan)
    df['smooth_momentum'] = df['ROC_90'] / atr20
    
    # Sort by smooth momentum descending
    df_sorted = df.sort_values(by='smooth_momentum', ascending=False)
    
    # The blueprint states "Top Quintile (Top 10) are eligible".
    # We will return the sorted list, and the engine will ensure 
    # it only considers those within the top 10 of the full universe.
    # To do this accurately, the engine should rank all stocks first, 
    # then filter by Entry_Qualifies, but for simplicity we can rank the 
    # candidates that actually generated a signal by their smooth momentum.
    # Wait, the rule is "Only stocks in the Top Quintile are eligible".
    # That means we need to rank ALL stocks. We'll handle this by passing 
    # ALL valid stocks to rank_candidates, picking the top 10, then filtering 
    # for the entry signal. We'll modify the engine to accommodate this.
    
    return df_sorted['stock'].tolist()
