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

fetcher = DataFetcher(start_date='2017-01-01')
data_dict = fetcher.fetch_all()
vix_df = data_dict.get('VIX')
nifty_df = data_dict.get('Nifty50')
vix_df['VIX_Multiplier'] = calculate_vix_filter(vix_df)

nifty_df['EMA_20'] = nifty_df['Close'].ewm(span=20, adjust=False).mean()
nifty_df['EMA_100'] = nifty_df['Close'].ewm(span=100, adjust=False).mean()
nifty_df['Regime_Allowed'] = nifty_df['Close'] > nifty_df['EMA_20']
nifty_df['Macro_Crash'] = nifty_df['Close'] < nifty_df['EMA_100']

engine = BacktestEngine(data_dict, max_positions=5, risk_pct=0.03)

original_run = engine.run

def custom_run(self):
    all_dates = pd.DatetimeIndex([])
    for df in self.data_dict.values():
        if df is not None and not df.empty:
            all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()

    nifty_df = self.data_dict.get('Nifty50')
    vix_df = self.data_dict.get('VIX')
    stocks = self._get_active_stock_symbols()
    
    for i, date in enumerate(all_dates):
        if i == 0:
            continue
        prev_date = all_dates[i-1]
        
        vix_mult_prev = 1.0
        macro_crash = False
        if prev_date in vix_df.index:
            vix_mult_prev = vix_df.loc[prev_date].get('VIX_Multiplier', 1.0)
        if prev_date in nifty_df.index:
            macro_crash = nifty_df.loc[prev_date].get('Macro_Crash', False)
            
        if (macro_crash or vix_mult_prev == 0.0) and len(self.open_trades) > 0:
            for trade in self.open_trades:
                if trade.current_sl < trade.entry_price:
                    trade.current_sl = trade.entry_price
                    
        daily_realized_pnl = 0.0
        for trade in list(self.open_trades):
            df = self.data_dict[trade.stock]
            if date in df.index:
                row = df.loc[date]
                prev_size = trade.current_size
                pnl = trade.update(date, row['Open'], row['High'], row['Low'], row['Close'])
                if prev_size > trade.current_size:
                    size_exited = prev_size - trade.current_size
                    if trade.exit_events:
                        latest_event = trade.exit_events[-1]
                        if latest_event['date'] == date:
                            exit_price = latest_event['price']
                            value_sold = size_exited * exit_price
                            costs = value_sold * 0.0023
                            latest_event['costs'] = costs
                            latest_event['net_pnl'] = pnl - costs
                            self.cash += value_sold - costs
                            daily_realized_pnl += latest_event['net_pnl']
                if trade.status == 'CLOSED':
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)
                    
        current_portfolio_value = self.cash
        for trade in self.open_trades:
            df = self.data_dict[trade.stock]
            if date in df.index:
                current_portfolio_value += trade.current_size * df.loc[date, 'Close']
            else:
                current_portfolio_value += trade.current_size * trade.entry_price
                
        self.equity = current_portfolio_value
        self.equity_curve.append({'date': date, 'equity': self.equity, 'cash': self.cash})

        regime_allowed = True
        if prev_date in nifty_df.index:
            regime_allowed = nifty_df.loc[prev_date].get('Regime_Allowed', True)
            
        if not (vix_mult_prev > 0 and regime_allowed):
            continue
            
        if len(self.open_trades) >= self.max_positions:
            continue
            
        candidates = []
        for stock in stocks:
            df = self.data_dict[stock]
            if prev_date in df.index and date in df.index:
                prev_row = df.loc[prev_date]
                if prev_row.get('Entry_Signal', False) and not any(t.stock == stock for t in self.open_trades):
                    candidates.append((stock, df.loc[date]))
                    
        for stock, today_row in candidates:
            if len(self.open_trades) >= self.max_positions:
                break
            prev_row = self.data_dict[stock].loc[prev_date]
            shares, sl = strategy.calculate_position_size(
                equity=self.equity, risk_pct=self.risk_pct,
                entry_price=today_row['Open'], atr=prev_row.get('ATR', 0), vix_multiplier=vix_mult_prev
            )
            if shares > 0:
                cost = shares * today_row['Open']
                if self.cash >= cost:
                    rps = today_row['Open'] - sl
                    t1 = today_row['Open'] + (2.0 * rps)
                    t2 = today_row['Open'] + (3.0 * rps)
                    new_trade = strategy.Trade(
                        stock=stock, entry_date=date, entry_price=today_row['Open'],
                        stop_loss=sl, target_1=t1, target_2=t2, size=shares,
                        vix_val=vix_mult_prev, sector='None'
                    )
                    self.open_trades.append(new_trade)
                    self.cash -= cost

    return self.equity_curve, self.closed_trades

engine.run = custom_run.__get__(engine, BacktestEngine)
eq, closed = engine.run()
res = calculate_metrics(eq, closed)

print("\n--- RESULTS WITH EMA_100 CRASH PROTECTION (2017-2025) ---")
print(f"CAGR: {res.get('CAGR (%)', 0):.2f}%")
print(f"Max DD: {res.get('Max Drawdown (%)', 0):.2f}%")
print(f"Win Rate: {res.get('Win Rate (%)', 0):.2f}%")
print(f"Total Return: {res.get('Total Return (%)', 0):.2f}%")
