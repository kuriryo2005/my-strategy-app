import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(r"c:\Users\kurir\Downloads\my_strategy")
sys.path.insert(0, str(PROJECT_ROOT))

from app import run_pca_sub_backtest
import src.config as config

def verify_daily_returns():
    target_date = pd.Timestamp("2026-04-08")
    
    # 商社あり設定に戻す
    config.JP_TICKERS = [
        "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T", 
        "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T", 
        "1631.T", "1632.T", "1633.T"
    ]
    
    # 2026年からのバックテスト実行
    daily_rets, df_pnl = run_pca_sub_backtest("2026-01-01")
    
    # 4/8以降のデータを抽出
    recent_rets = daily_rets[daily_rets.index >= target_date]
    recent_pnl = df_pnl[df_pnl['Date'] >= target_date]
    
    print(f"=== 4/8以降の日次リターン詳細 (商社あり) ===")
    cum_ret = 1.0
    for date, ret in recent_rets.items():
        cum_ret *= (1 + ret)
        # その日の商社の動きを確認
        shosha_day = recent_pnl[(recent_pnl['Date'] == date) & (recent_pnl['Ticker'] == "1629.T")]
        shosha_info = "---"
        if not shosha_day.empty:
            row = shosha_day.iloc[0]
            shosha_info = f"{row['Side']:5} | 損益 {row['PnL']:+.2f}%"
            
        print(f"日付: {date.date()} | 日次: {ret*100:+.2f}% | 累計: {(cum_ret-1)*100:+.2f}% | 商社寄与: {shosha_info}")

if __name__ == "__main__":
    verify_daily_returns()
