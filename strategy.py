import pandas as pd
import numpy as np

def generate_signals(df_stock):
    """
    Generate entry signals based on ENTRY TRIGGER rules.
    df_stock must already have 'SMC_Valid' from filters.py.
    """
    if df_stock is None or df_stock.empty:
        return pd.DataFrame()
        
    df = df_stock.copy()
    
    # Entry Trigger: Bullish Engulfing OR Pin Bar within Order Block
    # 1. Bullish Engulfing
    # Yesterday was bearish (Close < Open), Today is bullish (Close > Open)
    # Today's Body engulfs Yesterday's Body: Open <= Yesterday Close AND Close >= Yesterday Open
    df['Is_Bullish'] = df['Close'] > df['Open']
    df['Prev_Bearish'] = (df['Close'].shift(1) < df['Open'].shift(1))
    
    df['Bullish_Engulfing'] = (
        df['Is_Bullish'] & 
        df['Prev_Bearish'] & 
        (df['Open'] <= df['Close'].shift(1)) & 
        (df['Close'] >= df['Open'].shift(1))
    )
    
    # 2. Pin Bar
    # Lower wick >= 2 * body
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Pin_Bar'] = df['Lower_Wick'] >= (2 * df['Body_Size'])
    
    # Entry Signal
    # All 4 filters green is handled by the main loop checking other flags (VIX, FII, Sector).
    # Here we just check SMC + Candlestick Pattern
    df['Candle_Signal'] = df['Bullish_Engulfing'] | df['Pin_Bar']
    df['Entry_Signal'] = df['SMC_Valid'] & df['Candle_Signal']
    
    return df

def calculate_position_size(equity, risk_pct, entry_price, stop_loss, vix_multiplier):
    """
    Calculate position size.
    Risk 1.5% of current equity.
    Size = (equity * 0.015) / (entry_price - stop_loss_price)
    Apply VIX multiplier (halve size if VIX > 20, 0 if > 25).
    Returns integer number of shares.
    """
    if entry_price <= stop_loss or vix_multiplier == 0:
        return 0
        
    risk_amount = equity * risk_pct
    risk_per_share = entry_price - stop_loss
    
    raw_size = risk_amount / risk_per_share
    adjusted_size = raw_size * vix_multiplier
    
    # Number of shares must be integer
    shares = int(np.floor(adjusted_size))
    return shares

class Trade:
    def __init__(self, stock, entry_date, entry_price, stop_loss, target_1, target_2, size, vix_val, sector):
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
        
        self.status = 'OPEN'
        self.t1_hit = False
        self.hold_days = 0
        self.exit_events = [] # list of dicts: {'date', 'price', 'size_sold', 'reason', 'pnl'}
        
    def update(self, date, open_p, high_p, low_p, close_p):
        """
        Check for exits. Returns realized PnL for the day.
        Assumes we can hit target or SL intra-day using High/Low.
        """
        self.hold_days += 1
        realized_pnl = 0.0
        
        if self.status == 'CLOSED':
            return 0.0
            
        # Time stop
        if self.hold_days >= 20:
            size_to_sell = self.current_size
            price = close_p # exit at close of 20th day
            pnl = (price - self.entry_price) * size_to_sell
            realized_pnl += pnl
            self.exit_events.append({'date': date, 'price': price, 'size': size_to_sell, 'reason': 'TIME_STOP', 'pnl': pnl})
            self.current_size = 0
            self.status = 'CLOSED'
            return realized_pnl
            
        # Stop Loss hit (assume worst case: open below SL, else execution at SL)
        if low_p <= self.current_sl:
            price = min(open_p, self.current_sl) # Gap down scenario
            size_to_sell = self.current_size
            pnl = (price - self.entry_price) * size_to_sell
            realized_pnl += pnl
            self.exit_events.append({'date': date, 'price': price, 'size': size_to_sell, 'reason': 'STOP_LOSS', 'pnl': pnl})
            self.current_size = 0
            self.status = 'CLOSED'
            return realized_pnl
            
        # Target 1 hit
        if not self.t1_hit and high_p >= self.target_1:
            price = max(open_p, self.target_1) # Gap up scenario
            size_to_sell = int(self.initial_size * 0.5)
            # Ensure we don't sell more than we have
            size_to_sell = min(size_to_sell, self.current_size)
            
            pnl = (price - self.entry_price) * size_to_sell
            realized_pnl += pnl
            self.exit_events.append({'date': date, 'price': price, 'size': size_to_sell, 'reason': 'TARGET_1', 'pnl': pnl})
            
            self.current_size -= size_to_sell
            self.t1_hit = True
            # Move SL to breakeven
            self.current_sl = self.entry_price
            
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
