import pandas as pd
import numpy as np
from pathlib import Path
import sys

# プロジェクトルート
PROJECT_ROOT = Path(r"c:\Users\kurir\Downloads\my_strategy")
sys.path.insert(0, str(PROJECT_ROOT))

# アプリ内の関数を使用して計算を再現
from app import run_pca_sub_backtest

def rigorous_verification():
    # 2026-04-08 以降のバックテストを実行 (商社なし 16銘柄設定中)
    target_date = pd.Timestamp("2026-04-08")
    daily_rets, df_pnl = run_pca_sub_backtest("2026-04-08")
    
    if daily_rets.empty:
        print("データが見つかりませんでした。")
        return

    print(f"{'日付':<12} | {'日次リターン(%)':<15} | {'累計リターン(%)':<15}")
    print("-" * 50)
    
    cum_ret = 1.0
    for date in sorted(daily_rets.index):
        ret = daily_rets[date]
        cum_ret *= (1 + ret)
        print(f"{date.date()} | {ret*100:>+14.2f}% | {(cum_ret-1)*100:>+14.2f}%")

    print("-" * 50)
    print(f"最終累計リターン (4/8 〜 現在): {(cum_ret-1)*100:+.2f}%")

if __name__ == "__main__":
    rigorous_verification()
