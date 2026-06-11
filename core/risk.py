import math
import pandas as pd

def calculate_position_size(
    current_equity: float, 
    entry_price: float, 
    atr14: float, 
    volatility_multiplier: float
) -> tuple[int, float]:
    """
    Calculates position size using Volatility Targeting.
    
    Target daily volatility: 2% (0.02).
    Shares = (Equity * 0.02 * Vol_Multiplier) / ATR14
    Max allocation per trade = 25% of equity.
    
    Returns:
        (shares, hard_stop_loss_price)
    """
    if pd.isna(atr14) or atr14 <= 0:
        return 0, 0.0
        
    target_volatility = 0.08
    daily_risk_budget = current_equity * target_volatility * volatility_multiplier
    
    shares = math.floor(daily_risk_budget / atr14)
    
    if shares <= 0:
        return 0, 0.0
        
    # Cap at 100% of equity
    position_cost = shares * entry_price
    max_allowed_cost = current_equity * 1.00
    
    if position_cost > max_allowed_cost:
        shares = math.floor(max_allowed_cost / entry_price)
        
    # Hard stop loss at Entry - (2.5 * ATR14) as per blueprint
    hard_stop_loss = entry_price - (2.5 * atr14)
        
    return shares, hard_stop_loss
