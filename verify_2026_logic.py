import pandas as pd
import numpy as np
from pathlib import Path
import sys

# プロジェクトルート
PROJECT_ROOT = Path(r"c:\Users\kurir\Downloads\my_strategy")
sys.path.insert(0, str(PROJECT_ROOT))

from app import run_pca_sub_backtest

def verify_calculations():
    # バックテスト実行 (2026/01/01から)
    daily_rets, df_pnl = run_pca_sub_backtest("2026-01-01")
    
    # 累積リターンの計算 (prod)
    cum_ret = (1 + daily_rets).cumprod()
    final_return = cum_ret.iloc[-1] - 1
    
    print(f"### 2026/01/01 〜 直近までの検証結果")
    print(f"データ件数: {len(daily_rets)} 日")
    print(f"最終累積リターン: {final_return*100:.2f}%")
    
    # 直近5日間の詳細ログ
    print("\n### 直近5日間の詳細損益ログ (3.3倍レバレッジ後)")
    recent = daily_rets.tail(5)
    for date, ret in recent.items():
        print(f"日付: {date.date()} | 日次損益: {ret*100:+.2f}% | その日の累積: {(cum_ret.loc[date]-1)*100:.2f}%")

    # 5月（順張り期間）の平均的な動きを確認
    may_rets = daily_rets[daily_rets.index >= pd.Timestamp("2026-05-01")]
    if not may_rets.empty:
        print(f"\n### 5月の統計 (順張りモード中)")
        print(f"5月平均日次リターン: {may_rets.mean()*100:+.2f}%")
        print(f"5月勝率: {(may_rets > 0).mean()*100:.1f}%")

if __name__ == "__main__":
    verify_calculations()
