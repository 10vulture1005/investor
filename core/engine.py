import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import math
import os
import joblib

from core.signals import generate_signals, rank_candidates
from core.risk import calculate_position_size
from core.data_fetcher import SECTOR_MAP

logger = logging.getLogger(__name__)

class Trade:
    def __init__(self, stock: str, entry_date: pd.Timestamp, entry_price: float, stop_loss: float, shares: int, sector: str):
        self.stock = stock
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.shares = shares
        self.sector = sector
        
        self.hold_days = 0
        self.status = 'OPEN'
        self.exit_events = [] # list of dicts

class BacktestEngine:
    def __init__(self, data_dict: Dict[str, pd.DataFrame], start_capital: float = 10000000.0):
        self.data_dict = data_dict
        self.nifty = data_dict.get('Nifty50')
        self.vix = data_dict.get('VIX')
        
        self.ml_model = None
        if os.path.exists('models/xgb_filter.pkl'):
            self.ml_model = joblib.load('models/xgb_filter.pkl')
            logger.info("Loaded XGBoost ML Filter model.")
        
        self.equity = start_capital
        self.cash = start_capital
        
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve = []
        
        if self.nifty is not None:
            self.dates = self.nifty.index.sort_values()
        else:
            all_dates = pd.DatetimeIndex([])
            for k, df in self.data_dict.items():
                if not df.empty:
                    all_dates = all_dates.union(df.index)
            self.dates = all_dates.sort_values()
            
        self.stocks = [k for k in self.data_dict.keys() if k.endswith('.NS') and k != 'NIFTYBEES.NS']

    def run(self):
        logger.info(f"Starting Smart Alpha 3.0 Backtest...")
        
        signal_data = {}
        for stock in self.stocks:
            signal_data[stock] = generate_signals(self.data_dict[stock])
            
        for i, date in enumerate(self.dates):
            if i == 0:
                self.equity_curve.append({'date': date, 'equity': self.equity, 'cash': self.cash})
                continue
                
            prev_date = self.dates[i-1]
            
            # -----------------------------------------------------------
            # 1. PROCESS EXITS (Using day t data)
            # -----------------------------------------------------------
            daily_realized_pnl = 0.0
            
            for trade in self.open_trades[:]:
                df = self.data_dict[trade.stock]
                if date not in df.index:
                    continue
                    
                row = df.loc[date]
                trade.hold_days += 1
                
                exit_triggered = False
                exit_price = 0.0
                reason = ""
                
                # A. Hard Stop Loss (Intraday)
                if row['Low'] <= trade.stop_loss:
                    exit_triggered = True
                    reason = "STOP_LOSS"
                    exit_price = max(trade.stop_loss, row['Open']) if row['Open'] < trade.stop_loss else trade.stop_loss
                    
                # B. Primary Exit (Mean Reversion hit at close)
                elif row.get('RSI_3', 0) > 80 or row['Close'] > row.get('SMA_10', float('inf')):
                    exit_triggered = True
                    reason = "PRIMARY_EXIT"
                    exit_price = row['Close']
                    
                # C. Time Stop
                elif trade.hold_days >= 10:
                    exit_triggered = True
                    reason = "TIME_STOP"
                    exit_price = row['Close']
                        
                if exit_triggered:
                    gross_value = trade.shares * exit_price
                    net_value = gross_value * (1 - 0.0005) # 0.05% slippage/cost
                    cost_basis = trade.shares * trade.entry_price
                    net_pnl = net_value - cost_basis
                    
                    self.cash += net_value
                    daily_realized_pnl += net_pnl
                    
                    trade.exit_events.append({
                        'date': date,
                        'price': exit_price,
                        'shares': trade.shares,
                        'reason': reason,
                        'net_pnl': net_pnl
                    })
                    
                    trade.status = 'CLOSED'
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)

            # -----------------------------------------------------------
            # 2. UPDATE PORTFOLIO EQUITY
            # -----------------------------------------------------------
            current_portfolio_value = self.cash
            for trade in self.open_trades:
                df = self.data_dict[trade.stock]
                if date in df.index:
                    current_portfolio_value += trade.shares * df.loc[date, 'Close']
                else:
                    current_portfolio_value += trade.shares * trade.entry_price
                    
            self.equity = current_portfolio_value
            self.equity_curve.append({'date': date, 'equity': self.equity, 'cash': self.cash})
            
            # -----------------------------------------------------------
            # 3. EVALUATE REGIME & BREADTH (Using day t-1 data)
            # -----------------------------------------------------------
            vix_prev = self.vix.loc[prev_date] if (self.vix is not None and prev_date in self.vix.index) else None
            nifty_prev = self.nifty.loc[prev_date] if (self.nifty is not None and prev_date in self.nifty.index) else None
            
            if vix_prev is None or nifty_prev is None:
                continue
                
            vix_val = vix_prev['Close']
            nifty_close = nifty_prev['Close']
            nifty_sma100 = nifty_prev.get('SMA_100', np.nan)
            
            # Calculate Breadth
            stocks_above_50 = 0
            total_valid_stocks = 0
            
            for stock in self.stocks:
                df = self.data_dict[stock]
                if prev_date in df.index:
                    pr = df.loc[prev_date]
                    if not pd.isna(pr.get('SMA_50')):
                        total_valid_stocks += 1
                        if pr['Close'] > pr['SMA_50']:
                            stocks_above_50 += 1
                            
            breadth_pct = (stocks_above_50 / total_valid_stocks) if total_valid_stocks > 0 else 0
            
            # Check Regime Filters
            if pd.isna(nifty_sma100):
                continue
                
            if nifty_close <= nifty_sma100 or breadth_pct <= 0.30:
                continue # Skip all entries
                
            volatility_multiplier = 0.5 if breadth_pct < 0.40 else 1.0
            
            # -----------------------------------------------------------
            # 4. EVALUATE ENTRIES (Using day t-1 signals, executing at day t Open)
            # -----------------------------------------------------------
            if len(self.open_trades) >= 1:
                continue
                
            candidates_data = []
            current_stocks = {t.stock for t in self.open_trades}
            
            # Gather all valid stocks for ranking
            for stock in self.stocks:
                df_sig = signal_data[stock]
                if prev_date in df_sig.index:
                    row_prev = df_sig.loc[prev_date]
                    if not pd.isna(row_prev.get('ROC_90')) and not pd.isna(row_prev.get('ATR_20')):
                        candidates_data.append({
                            'stock': stock,
                            'ROC_90': row_prev['ROC_90'],
                            'ATR_20': row_prev['ATR_20'],
                            'Entry_Qualifies': row_prev.get('Entry_Qualifies', False)
                        })
                        
            if candidates_data:
                # Rank ALL valid stocks
                ranked_stocks = rank_candidates(candidates_data)
                
                # Keep only top 10 as eligible universe
                top_10 = ranked_stocks[:10]
                
                # Filter for Entry Qualifies AND not already open AND correlation control
                eligible_to_buy = []
                for stock in top_10:
                    if stock in current_stocks:
                        continue
                    
                    # Check if it actually triggered the pullback signal
                    stock_data = next((item for item in candidates_data if item["stock"] == stock), None)
                    if stock_data and stock_data['Entry_Qualifies']:
                        
                        # --- ML FILTER ---
                        prob_win = 1.0
                        if self.ml_model is not None:
                            df = self.data_dict[stock]
                            if prev_date in df.index:
                                row = df.loc[prev_date]
                                
                                nifty_row = self.nifty.loc[prev_date] if prev_date in self.nifty.index else None
                                vix_row = self.vix.loc[prev_date] if prev_date in self.vix.index else None
                                
                                nifty_sma100 = nifty_row['SMA_100'] if nifty_row is not None else np.nan
                                nifty_close = nifty_row['Close'] if nifty_row is not None else np.nan
                                vix_close = vix_row['Close'] if vix_row is not None else np.nan
                                
                                dist_sma50 = (row['Close'] - row['SMA_50']) / row['SMA_50']
                                smooth_mom = row['ROC_90'] / row['ATR_20'] if row['ATR_20'] > 0 else 0
                                dist_kc_lower = (row['Close'] - row['KC_Lower']) / row['KC_Lower']
                                
                                features = pd.DataFrame([{
                                    'RSI_3': row['RSI_3'],
                                    'ROC_90': row['ROC_90'],
                                    'ATR_pct': row['ATR_14'] / row['Close'],
                                    'dist_sma50': dist_sma50,
                                    'smooth_mom': smooth_mom,
                                    'dist_kc_lower': dist_kc_lower,
                                    'vix': vix_close,
                                    'nifty_trend': 1 if nifty_close > nifty_sma100 else 0
                                }])
                                
                                prob_win = self.ml_model.predict_proba(features)[0][1]
                                
                        if prob_win >= 0.50:
                            # Correlation Control: limit to max 2 from same sector
                            sector = SECTOR_MAP.get(stock, 'Other')
                            sector_count = sum(1 for t in self.open_trades if t.sector == sector)
                            sector_count += sum(1 for s in eligible_to_buy if SECTOR_MAP.get(s, 'Other') == sector)
                            
                            if sector_count < 2:
                                eligible_to_buy.append(stock)
                            
                slots_available = 1 - len(self.open_trades)
                stocks_to_buy = eligible_to_buy[:slots_available]
                
                for stock in stocks_to_buy:
                    df = self.data_dict[stock]
                    if date in df.index and prev_date in df.index:
                        entry_price = df.loc[date, 'Open']
                        atr_prev = df.loc[prev_date].get('ATR_14', 0)
                        
                        shares, stop_loss = calculate_position_size(
                            current_equity=self.equity,
                            entry_price=entry_price,
                            atr14=atr_prev,
                            volatility_multiplier=volatility_multiplier
                        )
                        
                        if shares > 0:
                            cost = shares * entry_price
                            if self.cash >= cost:
                                new_trade = Trade(
                                    stock=stock,
                                    entry_date=date,
                                    entry_price=entry_price,
                                    stop_loss=stop_loss,
                                    shares=shares,
                                    sector=SECTOR_MAP.get(stock, 'Other')
                                )
                                self.open_trades.append(new_trade)
                                self.cash -= cost

        logger.info(f"Backtest completed. Final Equity: ₹{self.equity:,.2f}")
        return self.equity_curve, self.closed_trades
