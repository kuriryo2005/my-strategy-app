import pandas as pd
import numpy as np
from pathlib import Path
import sys

# プロジェクトルート
PROJECT_ROOT = Path(r"c:\Users\kurir\Downloads\my_strategy")
sys.path.insert(0, str(PROJECT_ROOT))

from app import run_pca_sub_backtest
import src.config as config

def compare_shosha_effect():
    target_date = pd.Timestamp("2026-04-08")
    
    # 1. 商社あり (17銘柄) で計算
    # ※一時的に config を書き換える
    original_tickers = [
        "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T", 
        "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T", 
        "1631.T", "1632.T", "1633.T"
    ]
    config.JP_TICKERS = original_tickers
    daily_rets_with, _ = run_pca_sub_backtest("2026-01-01")
    
    # 2. 商社なし (16銘柄) で計算
    new_tickers = [t for t in original_tickers if t != "1629.T"]
    config.JP_TICKERS = new_tickers
    daily_rets_without, _ = run_pca_sub_backtest("2026-01-01")
    
    # 4/8以降を抽出
    rets_with = daily_rets_with[daily_rets_with.index >= target_date]
    rets_without = daily_rets_without[daily_rets_without.index >= target_date]
    
    # 累積リターン計算
    cum_with = (1 + rets_with).prod() - 1
    cum_without = (1 + rets_without).prod() - 1
    
    # 勝率
    win_with = (rets_with > 0).mean() * 100
    win_without = (rets_without > 0).mean() * 100
    
    print(f"=== 4/8以降のパフォーマンス比較 (レバレッジ 3.3倍換算) ===")
    print(f"期間: {target_date.date()} ～ {rets_with.index[-1].date()}")
    print(f"\n【商社あり (17銘柄)】")
    print(f"  累積リターン: {cum_with*100:+.2f}%")
    print(f"  勝率: {win_with:.1f}%")
    
    print(f"\n【商社なし (16銘柄)】")
    print(f"  累積リターン: {cum_without*100:+.2f}%")
    print(f"  勝率: {win_without:.1f}%")
    
    diff = (cum_without - cum_with) * 100
    print(f"\n差分 (改善幅): {diff:+.2f}%")

if __name__ == "__main__":
    compare_shosha_effect()
