import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import logging
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_fetcher import DataFetcher
from core.indicators import calculate_technical_indicators
from core.signals import generate_signals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_dataset():
    logger.info("Fetching data for ML dataset...")
    fetcher = DataFetcher(start_date='2017-01-01', end_date='2025-12-31')
    raw_data_dict = fetcher.fetch_all()
    
    dataset_rows = []
    
    stocks = [k for k in raw_data_dict.keys() if k.endswith('.NS') and k != 'NIFTYBEES.NS']
    
    # Precompute for Nifty and VIX (Macro features)
    nifty = calculate_technical_indicators(raw_data_dict.get('Nifty50', pd.DataFrame()))
    vix = calculate_technical_indicators(raw_data_dict.get('VIX', pd.DataFrame()))
    
    for stock in stocks:
        df = raw_data_dict[stock]
        if df.empty:
            continue
            
        df = calculate_technical_indicators(df)
        df = generate_signals(df)
        
        # We need future prices to calculate the target
        df['Future_High_5d'] = df['High'].shift(-5).rolling(window=5, min_periods=1).max()
        df['Future_Close_5d'] = df['Close'].shift(-5)
        
        # Define target: Did it bounce? (e.g. 5-day future close is higher than next day's open)
        # Entry happens on next day's open
        df['Next_Open'] = df['Open'].shift(-1)
        
        # Target: 1 if Future_Close_5d > Next_Open * 1.01 (1% bounce at least) else 0
        df['Target'] = np.where(df['Future_Close_5d'] > df['Next_Open'] * 1.01, 1, 0)
        
        # Extract rows where Entry_Qualifies == True
        signals = df[df['Entry_Qualifies'] == True].copy()
        
        for date, row in signals.iterrows():
            if pd.isna(row['Target']) or pd.isna(row['ROC_90']) or pd.isna(row['ATR_14']):
                continue
                
            # Grab macro data for the same date
            nifty_row = nifty.loc[date] if date in nifty.index else None
            vix_row = vix.loc[date] if date in vix.index else None
            
            nifty_sma100 = nifty_row['SMA_100'] if nifty_row is not None else np.nan
            nifty_close = nifty_row['Close'] if nifty_row is not None else np.nan
            vix_close = vix_row['Close'] if vix_row is not None else np.nan
            
            # Distance to SMA 50
            dist_sma50 = (row['Close'] - row['SMA_50']) / row['SMA_50']
            
            # Smooth Momentum
            smooth_mom = row['ROC_90'] / row['ATR_20'] if row['ATR_20'] > 0 else 0
            
            # Distance from Keltner Lower
            dist_kc_lower = (row['Close'] - row['KC_Lower']) / row['KC_Lower']
            
            dataset_rows.append({
                'date': date,
                'stock': stock,
                'RSI_3': row['RSI_3'],
                'ROC_90': row['ROC_90'],
                'ATR_pct': row['ATR_14'] / row['Close'],
                'dist_sma50': dist_sma50,
                'smooth_mom': smooth_mom,
                'dist_kc_lower': dist_kc_lower,
                'vix': vix_close,
                'nifty_trend': 1 if nifty_close > nifty_sma100 else 0,
                'Target': row['Target']
            })

    df_ml = pd.DataFrame(dataset_rows)
    df_ml = df_ml.dropna()
    df_ml = df_ml.sort_values('date')
    logger.info(f"Dataset built with {len(df_ml)} samples.")
    return df_ml

def train_model(df_ml):
    logger.info("Training XGBoost Classifier...")
    
    features = ['RSI_3', 'ROC_90', 'ATR_pct', 'dist_sma50', 'smooth_mom', 'dist_kc_lower', 'vix', 'nifty_trend']
    
    X = df_ml[features]
    y = df_ml['Target']
    
    # Time-based split (Train on first 70%, Test on last 30%)
    split_idx = int(len(df_ml) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    logger.info(f"Out-of-Sample Accuracy: {acc:.4f}")
    logger.info(f"Out-of-Sample ROC AUC: {auc:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, preds))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgb_filter.pkl')
    logger.info("Model saved to models/xgb_filter.pkl")

if __name__ == "__main__":
    df = build_dataset()
    if not df.empty:
        train_model(df)
    else:
        logger.error("Empty dataset. Check data fetching.")
