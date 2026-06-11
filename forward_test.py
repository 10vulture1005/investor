import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import joblib
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.data_fetcher import DataFetcher, NIFTY_50_TICKERS, SECTOR_INDICES, SECTOR_MAP
from core.filters import calculate_vix_filter, calculate_market_regime_filter
from core.indicators import calculate_technical_indicators
from core.signals import generate_signals
from core.risk import calculate_position_size
from core.broker import DhanBroker

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger('scanner')
console = Console()

def run_scanner():
    console.print(Panel.fit("[bold cyan]Smart Alpha 3.0 - Daily Forward Tester (ML Enhanced)[/bold cyan]"))
    
    # 1. Fetch recent data (150 days to warm up 50-EMA)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=150)).strftime('%Y-%m-%d')
    
    with console.status("[bold green]Fetching market data..."):
        fetcher = DataFetcher(start_date=start_date, end_date=end_date)
        data_dict = fetcher.fetch_all()
        
    if '^INDIAVIX' not in data_dict and 'VIX' not in data_dict:
        console.print("[bold red]VIX data missing. Cannot evaluate risk constraints. Aborting.[/bold red]")
        return
        
    vix_df = data_dict.get('VIX')
    nifty_df = data_dict.get('Nifty50')
    
    if nifty_df is None or nifty_df.empty:
        console.print("[bold red]Nifty 50 data missing. Aborting.[/bold red]")
        return
        
    # 2. Evaluate Market-wide Filters
    vix_df['VIX_Multiplier'] = calculate_vix_filter(vix_df)
    regime_df = calculate_market_regime_filter(nifty_df)
    nifty_df['Regime_Allowed'] = regime_df['Regime_Allowed']
    nifty_df['Macro_Crash'] = regime_df['Macro_Crash']
    
    # Load ML Model
    model = None
    if os.path.exists('models/xgb_filter.pkl'):
        model = joblib.load('models/xgb_filter.pkl')
        console.print("[bold green]✅ XGBoost ML Model Loaded.[/bold green]")
    else:
        console.print("[bold yellow]⚠️ ML Model not found. Running baseline rules.[/bold yellow]")
    
    # Get latest market status
    latest_date = nifty_df.index[-1]
    vix_mult = vix_df['VIX_Multiplier'].iloc[-1]
    regime_allowed = nifty_df['Regime_Allowed'].iloc[-1]
    date_str = latest_date.strftime('%Y-%m-%d')
    
    console.print(f"\n[bold]Market Status for {date_str}[/bold]")
    regime_color = "green" if regime_allowed else "red"
    vix_color = "green" if vix_mult == 1.0 else ("yellow" if vix_mult > 0 else "red")
    
    console.print(f"Market Regime (Nifty > 20 EMA): [{regime_color}]{'ALLOWED' if regime_allowed else 'BLOCKED'}[/{regime_color}]")
    console.print(f"VIX Status (Multiplier): [{vix_color}]{vix_mult}[/{vix_color}]")
    
    if not regime_allowed:
        console.print("\n[bold yellow]Market Regime is bearish. No new entries allowed.[/bold yellow]")
    if vix_mult == 0.0:
        console.print("\n[bold red]VIX is critically high. No new entries allowed.[/bold red]")
        
    # 3. Connect to Broker and Get Live Capital
    with console.status("[bold green]Connecting to Dhan API..."):
        broker = DhanBroker()
        live_equity = broker.get_live_capital()
        
        if live_equity is not None:
            equity = live_equity
            console.print(f"[bold green]Connected to Dhan HQ! Live Available Margin: ₹{equity:,.2f}[/bold green]")
        else:
            equity = 1000000.0 # Default assumption
            console.print(f"[bold yellow]Running in Offline Mode. Assumed Default Portfolio: ₹{equity:,.2f}[/bold yellow]")

    # 4. Generate Signals
    with console.status("[bold green]Scanning Nifty 50 universe..."):
        signals = []
        risk_pct = 0.03
        
        for ticker, df in data_dict.items():
            if ticker in NIFTY_50_TICKERS or ticker.endswith('.NS'):
                if df.empty or len(df) < 50:
                    continue
                    
                df = calculate_technical_indicators(df)
                df = generate_signals(df)
                
                # Check the last row
                last_row = df.iloc[-1]
                if last_row.get('Entry_Signal', False) or last_row.get('Entry_Qualifies', False):
                    # Features for ML
                    if model is not None:
                        # Extract features dynamically
                        atr_pct = last_row['ATR_14'] / last_row['Close'] if last_row.get('Close', 0) > 0 else 0
                        dist_sma50 = (last_row['Close'] - last_row['SMA_50']) / last_row['SMA_50'] if last_row.get('SMA_50', 0) > 0 else 0
                        smooth_mom = last_row['ROC_90'] / last_row['ATR_20'] if last_row.get('ATR_20', 0) > 0 else 0
                        dist_kc_lower = (last_row['Close'] - last_row['KC_Lower']) / last_row['KC_Lower'] if last_row.get('KC_Lower', 0) > 0 else 0
                        
                        features = {
                            'RSI_3': last_row.get('RSI_3', 0),
                            'ROC_90': last_row.get('ROC_90', 0),
                            'ATR_pct': atr_pct,
                            'dist_sma50': dist_sma50,
                            'smooth_mom': smooth_mom,
                            'dist_kc_lower': dist_kc_lower,
                            'vix': vix_df['Close'].iloc[-1],
                            'nifty_trend': 1 if regime_allowed else 0
                        }
                        
                        f_df = pd.DataFrame([features])
                        prob_win = model.predict_proba(f_df)[0][1]
                        
                        if prob_win < 0.50:
                            # ML Filter vetoes
                            continue
                    else:
                        prob_win = 0.0
                    
                    # Calculate position size
                    entry_price = last_row['Close']
                    atr = last_row['ATR']
                    
                    shares, stop_loss = calculate_position_size(
                        equity=equity, 
                        risk_pct=risk_pct, 
                        entry_price=entry_price, 
                        atr=atr, 
                        vix_multiplier=vix_mult
                    )
                    
                    if shares > 0:
                        risk_per_share = entry_price - stop_loss
                        target_1 = entry_price + (2.0 * risk_per_share)
                        
                        signals.append({
                            'Ticker': ticker.replace('.NS', ''),
                            'Close': entry_price,
                            'SL': stop_loss,
                            'T1 (2R)': target_1,
                            'Prob_Win': f"{prob_win*100:.1f}%" if model else "N/A",
                            'Shares': shares,
                            'Risk (INR)': shares * risk_per_share
                        })

    # 4. Display Signals
    if signals:
        console.print(f"\n[bold green]Found {len(signals)} Active Buy Signals![/bold green]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Ticker")
        table.add_column("Close (Entry)", justify="right")
        table.add_column("Stop Loss", justify="right", style="red")
        table.add_column("Target", justify="right", style="green")
        table.add_column("ML Prob", justify="right", style="cyan")
        table.add_column("Shares", justify="right")
        table.add_column("Risk", justify="right")
        
        for s in signals:
            table.add_row(
                s['Ticker'],
                f"₹{s['Close']:.2f}",
                f"₹{s['SL']:.2f}",
                f"₹{s['T1 (2R)']:.2f}",
                s['Prob_Win'],
                str(s['Shares']),
                f"₹{s['Risk (INR)']:.0f}"
            )
            
        console.print(table)
    else:
        console.print("\n[bold]No ML-approved buy signals found today.[/bold]")
        
    console.print(f"\n[italic grey]Evaluated against Equity: ₹{equity:,.2f} | Risk: {risk_pct*100}%[/italic grey]")

if __name__ == "__main__":
    run_scanner()
