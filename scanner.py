import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from data_fetcher import DataFetcher, NIFTY_50_TICKERS, SECTOR_INDICES, SECTOR_MAP
from filters import calculate_vix_filter, calculate_market_regime_filter, calculate_technical_indicators
from strategy import generate_signals, calculate_position_size
from broker import DhanBroker

# Setup logging to be less noisy for the scanner
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger('scanner')
console = Console()

def run_scanner():
    console.print(Panel.fit("[bold cyan]Quantitative Swing Trading - Daily Scanner[/bold cyan]"))
    
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
    nifty_df['Regime_Allowed'] = calculate_market_regime_filter(nifty_df)
    
    # Get latest market status
    latest_date = nifty_df.index[-1]
    vix_mult = vix_df['VIX_Multiplier'].iloc[-1]
    regime_allowed = nifty_df['Regime_Allowed'].iloc[-1]
    
    # Check if latest date matches today, else it's the last trading day
    date_str = latest_date.strftime('%Y-%m-%d')
    
    console.print(f"\n[bold]Market Status for {date_str}[/bold]")
    regime_color = "green" if regime_allowed else "red"
    vix_color = "green" if vix_mult == 1.0 else ("yellow" if vix_mult > 0 else "red")
    
    console.print(f"Market Regime (Nifty > 20 EMA): [{regime_color}]{'ALLOWED' if regime_allowed else 'BLOCKED'}[/{regime_color}]")
    console.print(f"VIX Status (Multiplier): [{vix_color}]{vix_mult}[/{vix_color}]")
    
    if not regime_allowed:
        console.print("\n[bold yellow]Market Regime is currently bearish. No new entries allowed today.[/bold yellow]")
        # We still scan to show what would trigger, but warn the user.
    if vix_mult == 0.0:
        console.print("\n[bold red]VIX is critically high. No new entries allowed today.[/bold red]")
        
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
                if last_row.get('Entry_Signal', False):
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
                        target_2 = entry_price + (3.0 * risk_per_share)
                        
                        signals.append({
                            'Ticker': ticker.replace('.NS', ''),
                            'Close': entry_price,
                            'SL': stop_loss,
                            'T1 (2R)': target_1,
                            'T2 (3R)': target_2,
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
        table.add_column("Target 1", justify="right", style="green")
        table.add_column("Target 2", justify="right", style="bold green")
        table.add_column("Shares", justify="right")
        table.add_column("Total Risk", justify="right")
        
        for s in signals:
            table.add_row(
                s['Ticker'],
                f"₹{s['Close']:.2f}",
                f"₹{s['SL']:.2f}",
                f"₹{s['T1 (2R)']:.2f}",
                f"₹{s['T2 (3R)']:.2f}",
                str(s['Shares']),
                f"₹{s['Risk (INR)']:.0f}"
            )
            
        console.print(table)
    else:
        console.print("\n[bold]No active buy signals found today.[/bold]")
        
    console.print(f"\n[italic grey]Scanner completed. Evaluated against Equity: ₹{equity:,.2f} | Risk: {risk_pct*100}%[/italic grey]")

if __name__ == "__main__":
    run_scanner()
