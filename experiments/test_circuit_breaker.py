import pandas as pd
import numpy as np
import os
import logging
from backtest_engine import BacktestEngine
from report import calculate_metrics
import strategy
from data_fetcher import DataFetcher
from filters import calculate_vix_filter, calculate_market_regime_filter

logging.getLogger().setLevel(logging.CRITICAL)

os.environ['INITIAL_CAPITAL'] = '1000000'
os.environ['RISK_PCT'] = '0.03'
os.environ['MAX_POSITIONS'] = '5'

fetcher = DataFetcher()
data_dict = fetcher.fetch_all()
vix_df = data_dict.get('VIX')
nifty_df = data_dict.get('Nifty50')

# Hardcode the tightened regime filter
nifty_df['EMA_20'] = nifty_df['Close'].ewm(span=20, adjust=False).mean()
nifty_df['Regime_Allowed'] = nifty_df['Close'] > nifty_df['EMA_20']

# Create engine to test
engine = BacktestEngine(data_dict, max_positions=5, risk_pct=0.03)

# Save original run method
original_run = engine.run

def custom_run(self):
    from datetime import timedelta
    # Basic setup (copied from run)
    all_dates = pd.DatetimeIndex([])
    for df in self.data_dict.values():
        if df is not None and not df.empty:
            all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()

    nifty_df = self.data_dict.get('Nifty50')
    vix_df = self.data_dict.get('VIX')

    for current_date in all_dates:
        if current_date not in nifty_df.index:
            continue
            
        regime_allowed = nifty_df.loc[current_date, 'Regime_Allowed']
        
        # --- NEW EMERGENCY EXIT LOGIC ---
        if not regime_allowed and len(self.open_trades) > 0:
            # Liquidate all open trades at today's close or tomorrow's open.
            # We'll approximate by closing them at today's close for testing.
            for trade in list(self.open_trades):
                if current_date in self.data_dict[trade.stock].index:
                    close_p = self.data_dict[trade.stock].loc[current_date, 'Close']
                    pnl = (close_p - trade.entry_price) * trade.current_size
                    trade.exit_events.append({
                        'date': current_date,
                        'price': close_p,
                        'size': trade.current_size,
                        'reason': 'EMERGENCY_MARKET_EXIT',
                        'pnl': pnl
                    })
                    self.cash += (trade.current_size * trade.entry_price) + pnl
                    trade.current_size = 0
                    trade.status = 'CLOSED'
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)
        # --------------------------------
        
        # Now update remaining trades (should be 0 if emergency exit fired)
        for trade in list(self.open_trades):
            df_stock = self.data_dict[trade.stock]
            if current_date in df_stock.index:
                row = df_stock.loc[current_date]
                realized_pnl = trade.update(
                    date=current_date,
                    open_p=row['Open'],
                    high_p=row['High'],
                    low_p=row['Low'],
                    close_p=row['Close']
                )
                if realized_pnl != 0:
                    exit_event = trade.exit_events[-1]
                    self.cash += (exit_event['size'] * trade.entry_price) + realized_pnl
                    
                if trade.status == 'CLOSED':
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)
                    
        # Record equity
        total_equity = self.cash
        for trade in self.open_trades:
            df_stock = self.data_dict[trade.stock]
            if current_date in df_stock.index:
                total_equity += trade.current_size * df_stock.loc[current_date, 'Close']
            else:
                total_equity += trade.current_size * trade.entry_price
                
        self.equity_curve.append({'date': current_date, 'equity': total_equity})
        
        # New Entries
        if regime_allowed:
            vix_multiplier = 1.0
            if vix_df is not None and current_date in vix_df.index:
                vix_multiplier = vix_df.loc[current_date, 'VIX_Multiplier']
                
            if vix_multiplier > 0 and len(self.open_trades) < self.max_positions:
                for ticker, df in self.data_dict.items():
                    if ticker in ['Nifty50', 'VIX'] or 'Sector_RS' in df.columns:
                        continue
                        
                    if current_date in df.index:
                        row = df.loc[current_date]
                        if row.get('Entry_Signal', False) and not self._has_open_position(ticker):
                            shares, sl = strategy.calculate_position_size(
                                equity=total_equity,
                                risk_pct=self.risk_pct,
                                entry_price=row['Close'],
                                atr=row['ATR'],
                                vix_multiplier=vix_multiplier
                            )
                            if shares > 0:
                                cost = shares * row['Close']
                                if self.cash >= cost:
                                    risk_per_share = row['Close'] - sl
                                    t1 = row['Close'] + (2.0 * risk_per_share)
                                    t2 = row['Close'] + (3.0 * risk_per_share)
                                    
                                    new_trade = strategy.Trade(
                                        stock=ticker,
                                        entry_date=current_date,
                                        entry_price=row['Close'],
                                        stop_loss=sl,
                                        target_1=t1,
                                        target_2=t2,
                                        size=shares,
                                        vix_val=vix_multiplier,
                                        sector='None'
                                    )
                                    self.open_trades.append(new_trade)
                                    self.cash -= cost
                                    
                                    if len(self.open_trades) >= self.max_positions:
                                        break
    return self.equity_curve, self.closed_trades

engine.run = custom_run.__get__(engine, BacktestEngine)
eq, closed = engine.run()
res = calculate_metrics(eq, closed)

print("\n--- RESULTS WITH EMERGENCY MARKET EXIT ---")
print(f"CAGR: {res['CAGR (%)']:.2f}%")
print(f"Max DD: {res['Max Drawdown (%)']:.2f}%")
print(f"Win Rate: {res['Win Rate (%)']:.2f}%")
print(f"Total Return: {res['Total Return (%)']:.2f}%")

