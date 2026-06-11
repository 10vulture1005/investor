import pandas as pd
import numpy as np
import logging
from core.strategy import Trade, calculate_position_size
from core.data_fetcher import SECTOR_MAP

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, data_dict, start_capital=1000000.0, max_positions=5, risk_pct=0.015):
        self.data_dict = data_dict
        self.initial_capital = start_capital
        self.equity = start_capital
        self.cash = start_capital
        self.max_positions = max_positions
        self.risk_pct = risk_pct
        
        self.open_trades = []
        self.closed_trades = []
        self.equity_curve = [] # list of dicts: {'date': date, 'equity': equity, 'cash': cash}
        
        # Extract global filters
        self.vix = self.data_dict.get('VIX')
        self.nifty = self.data_dict.get('Nifty50')
        
        # Prepare trading dates (intersection of all available dates, or just Nifty50 dates)
        nifty = self.data_dict.get('^NSEI')
        if nifty is not None:
            self.dates = nifty.index.sort_values()
        else:
            # Fallback if no Nifty50
            all_dates = pd.DatetimeIndex([])
            for k, df in self.data_dict.items():
                if not df.empty:
                    all_dates = all_dates.union(df.index)
            self.dates = all_dates.sort_values()

    def _get_active_stock_symbols(self):
        return [k for k in self.data_dict.keys() if k.endswith('.NS') and k != 'NIFTYBEES.NS']

    def run(self):
        logger.info(f"Starting backtest from {self.dates[0].date()} to {self.dates[-1].date()}")
        
        stocks = self._get_active_stock_symbols()
        
        for i, date in enumerate(self.dates):
            if i == 0:
                self.equity_curve.append({'date': date, 'equity': self.equity, 'cash': self.cash})
                continue
                
            prev_date = self.dates[i-1]
            
            # --- 0. Macro Crash Protection (Breakeven Trailing) ---
            vix_row_prev = self.vix.loc[prev_date] if prev_date in self.vix.index else {}
            nifty_row_prev = self.nifty.loc[prev_date] if prev_date in self.nifty.index else {}
            
            vix_mult_prev = vix_row_prev.get('VIX_Multiplier', 1.0)
            macro_crash = nifty_row_prev.get('Macro_Crash', False)
            
            if (macro_crash or vix_mult_prev == 0.0) and len(self.open_trades) > 0:
                for trade in self.open_trades:
                    if trade.current_sl < trade.entry_price:
                        trade.current_sl = trade.entry_price
            
            # --- 1. Manage existing trades (Exits) ---
            daily_realized_pnl = 0.0
            
            for trade in self.open_trades[:]: # iterate over copy
                df = self.data_dict[trade.stock]
                if date in df.index:
                    row = df.loc[date]
                    # Update trade state based on today's price action
                    # Apply slippage and costs on exit events within the Trade class or here.
                    # We'll apply costs here after the trade update returns pnl.
                    
                    # Capture state before update to know if an exit happened
                    prev_status = trade.status
                    prev_size = trade.current_size
                    
                    pnl = trade.update(date, row['Open'], row['High'], row['Low'], row['Close'])
                    
                    if prev_size > trade.current_size:
                        # An exit occurred, apply costs to the exited portion
                        size_exited = prev_size - trade.current_size
                        # Estimate exit price (avg)
                        # We reconstruct it roughly from the PnL or look at the latest event
                        if trade.exit_events:
                            latest_event = trade.exit_events[-1]
                            if latest_event['date'] == date:
                                exit_price = latest_event['price']
                                value_sold = size_exited * exit_price
                                
                                # Costs: Brokerage (0.03%), STT (0.1%), Slippage (0.1%)
                                costs = value_sold * (0.0003 + 0.001 + 0.001)
                                latest_event['costs'] = costs
                                latest_event['net_pnl'] = pnl - costs
                                
                                self.cash += value_sold - costs
                                daily_realized_pnl += latest_event['net_pnl']
                                
                    if trade.status == 'CLOSED':
                        self.closed_trades.append(trade)
                        self.open_trades.remove(trade)
            
            # --- 2. Calculate current equity for sizing ---
            # Equity = Cash + Value of open positions
            current_portfolio_value = self.cash
            for trade in self.open_trades:
                df = self.data_dict[trade.stock]
                if date in df.index:
                    current_price = df.loc[date, 'Close']
                    current_portfolio_value += trade.current_size * current_price
                else:
                    # Stale price if not traded today
                    current_portfolio_value += trade.current_size * trade.entry_price
                    
            self.equity = current_portfolio_value
            self.equity_curve.append({'date': date, 'equity': self.equity, 'cash': self.cash})

            # --- 3. Check for new entries ---
            if len(self.open_trades) >= self.max_positions:
                continue
                
            # Filter checks are based on yesterday's signal!
            if prev_date not in self.vix.index or prev_date not in self.nifty.index:
                continue
                
            vix_row = self.vix.loc[prev_date]
            nifty_row = self.nifty.loc[prev_date]
            
            vix_mult = vix_row.get('VIX_Multiplier', 1.0)
            regime_allowed = nifty_row.get('Regime_Allowed', False)
            
            if not (vix_mult > 0 and regime_allowed):
                continue
                
            # Evaluate stocks
            candidates = []
            for stock in stocks:
                df = self.data_dict[stock]
                if prev_date in df.index and date in df.index:
                    prev_row = df.loc[prev_date]
                    if prev_row.get('Entry_Signal', False):
                        # Sector RS is skipped for optimized backtest
                        candidates.append((stock, df.loc[date]))
                            
            # Sort candidates or pick first available
            for stock, today_row in candidates:
                if len(self.open_trades) >= self.max_positions:
                    break
                    
                prev_row = self.data_dict[stock].loc[prev_date]
                
                # Setup trade
                entry_price = today_row['Open']
                atr = prev_row.get('ATR', 0)
                
                shares, stop_loss = calculate_position_size(
                    equity=self.equity, 
                    risk_pct=self.risk_pct, 
                    entry_price=entry_price, 
                    atr=atr, 
                    vix_multiplier=vix_mult
                )
                
                if shares > 0:
                    cost = shares * entry_price
                    if self.cash >= cost:
                        risk_per_share = entry_price - stop_loss
                        target_1 = entry_price + (2.0 * risk_per_share)
                        target_2 = entry_price + (3.0 * risk_per_share)
                        
                        new_trade = Trade(
                            stock=stock, 
                            entry_date=date, 
                            entry_price=entry_price, 
                            stop_loss=stop_loss, 
                            target_1=target_1, 
                            target_2=target_2, 
                            size=shares, 
                            vix_val=vix_row['Close'], 
                            sector=SECTOR_MAP.get(stock, 'Other'),
                            atr_at_entry=atr,
                            macro_crash_at_entry=macro_crash
                        )
                        self.open_trades.append(new_trade)
                        self.cash -= cost

        logger.info(f"Backtest complete. Processed {len(self.dates)} days.")
        logger.info(f"Final Equity: ₹{self.equity:,.2f}")
        return self.equity_curve, self.closed_trades
