import pandas as pd
import numpy as np

def generate_signals(df_stock):
    """
    Generate entry signals based on Technical Indicators.
    Requires df_stock to have ATR, RSI, BB_Lower, and Uptrend columns.
    """
    if df_stock is None or df_stock.empty:
        return pd.DataFrame()
        
    df = df_stock.copy()
    
    # 1. Candlestick Patterns (Bullish Engulfing or Pin Bar)
    df['Is_Bullish'] = df['Close'] > df['Open']
    df['Prev_Bearish'] = (df['Close'].shift(1) < df['Open'].shift(1))
    
    df['Candle_Signal'] = df['Is_Bullish']
    
    # Price must pullback to touch the 20 EMA (Low <= 20 EMA)
    # Price must close above the 20 EMA (Close > 20 EMA)
    df['Pullback'] = (df['Low'] <= df['EMA_20']) & (df['Close'] > df['EMA_20'])
    df['Tech_Valid'] = df['Uptrend'] & df['Pullback']
    
    # Final Entry Signal
    df['Entry_Signal'] = df['Tech_Valid'] & df['Candle_Signal']
    
    return df

def calculate_position_size(equity, risk_pct, entry_price, atr, vix_multiplier):
    """
    Calculate position size using dynamic ATR Stop Loss.
    SL = 1.5 * ATR
    If SL is > 15% of the entry price, skip the trade (return 0).
    """
    if pd.isna(atr) or atr <= 0 or vix_multiplier == 0:
        return 0, 0.0
        
    stop_loss = entry_price - (1.5 * atr)
    
    # Hard cap on Risk Distance: max 15%
    risk_distance_pct = (entry_price - stop_loss) / entry_price
    if risk_distance_pct > 0.15:
        return 0, stop_loss
        
    risk_amount = min(equity * risk_pct, 100000.0)
    risk_per_share = entry_price - stop_loss
    
    raw_size = risk_amount / risk_per_share
    adjusted_size = raw_size * vix_multiplier
    
    shares = int(np.floor(adjusted_size))
    return shares, stop_loss

class Trade:
    def __init__(self, stock, entry_date, entry_price, stop_loss, target_1, target_2, size, vix_val, sector, macro_crash_at_entry=False):
        self.stock = stock
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.initial_sl = stop_loss
        self.current_sl = stop_loss
        self.target_1 = target_1
        self.target_2 = target_2
        self.initial_size = size
        self.current_size = size
        self.vix_at_entry = vix_val
        self.sector = sector
        self.macro_crash_at_entry = macro_crash_at_entry
        
        self.status = 'OPEN'
        self.t1_hit = False
        self.hold_days = 0
        self.exit_events = []
        
    def update(self, date, open_p, high_p, low_p, close_p):
        self.hold_days += 1
        realized_pnl = 0.0
        
        if self.status == 'CLOSED':
            return 0.0
            
        # Time stop
        if self.hold_days >= 20:
            size_to_sell = self.current_size
            price = close_p
            pnl = (price - self.entry_price) * size_to_sell
            realized_pnl += pnl
            self.exit_events.append({'date': date, 'price': price, 'size': size_to_sell, 'reason': 'TIME_STOP', 'pnl': pnl})
            self.current_size = 0
            self.status = 'CLOSED'
            return realized_pnl
            
        # Stop Loss hit
        if low_p <= self.current_sl:
            price = self.current_sl
            size_to_sell = self.current_size
            pnl = (price - self.entry_price) * size_to_sell
            realized_pnl += pnl
            self.exit_events.append({'date': date, 'price': price, 'size': size_to_sell, 'reason': 'STOP_LOSS', 'pnl': pnl})
            self.current_size = 0
            self.status = 'CLOSED'
            return realized_pnl
            
        # Target 1 hit
        if not self.t1_hit and high_p >= self.target_1:
            price = max(open_p, self.target_1)
            size_to_sell = int(self.initial_size * 0.5)
            size_to_sell = min(size_to_sell, self.current_size)
            
            pnl = (price - self.entry_price) * size_to_sell
            realized_pnl += pnl
            self.exit_events.append({'date': date, 'price': price, 'size': size_to_sell, 'reason': 'TARGET_1', 'pnl': pnl})
            
            self.current_size -= size_to_sell
            self.t1_hit = True
            # Move SL to breakeven
            self.current_sl = max(self.entry_price, self.current_sl)
            
        # Target 2 hit
        if self.t1_hit and self.current_size > 0 and high_p >= self.target_2:
            price = max(open_p, self.target_2)
            size_to_sell = self.current_size
            
            pnl = (price - self.entry_price) * size_to_sell
            realized_pnl += pnl
            self.exit_events.append({'date': date, 'price': price, 'size': size_to_sell, 'reason': 'TARGET_2', 'pnl': pnl})
            
            self.current_size = 0
            self.status = 'CLOSED'
            
        return realized_pnl
