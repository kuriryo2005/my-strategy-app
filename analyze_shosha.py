import pandas as pd
import numpy as np
from pathlib import Path
import sys

# プロジェクトルート
PROJECT_ROOT = Path(r"c:\Users\kurir\Downloads\my_strategy")
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import JP_TICKER_NAMES
from app import run_pca_sub_backtest

def analyze_shosha_failure():
    # 2026/01/01 からのバックテスト実行
    daily_rets, df_pnl = run_pca_sub_backtest("2026-01-01")
    
    ticker = "1629.T" # 商社・卸売
    shosha_pnl = df_pnl[df_pnl['Ticker'] == ticker].copy()
    
    if shosha_pnl.empty:
        print("商社・卸売（1629.T）の取引データが見つかりませんでした。")
        return

    # 勝敗分析
    shosha_pnl['IsWin'] = shosha_pnl['PnL'] > 0
    win_count = shosha_pnl['IsWin'].sum()
    total_count = len(shosha_pnl)
    win_rate = (win_count / total_count * 100) if total_count > 0 else 0
    total_pnl = shosha_pnl['PnL'].sum()
    
    print(f"=== 商社・卸売 (1629.T) 2026年詳細分析 ===")
    print(f"取引回数: {total_count} 回")
    print(f"勝利回数: {win_count} 回")
    print(f"勝率: {win_rate:.1f}%")
    print(f"累積損益貢献: {total_pnl:+.2f}%")
    
    # ショート/ロング別の勝率
    print("\n--- サイド別分析 ---")
    for side in ['Long', 'Short']:
        side_df = shosha_pnl[shosha_pnl['Side'] == side]
        if not side_df.empty:
            s_win = (side_df['PnL'] > 0).mean() * 100
            s_pnl = side_df['PnL'].sum()
            print(f"{side:5}: {len(side_df)}回 | 勝率 {s_win:5.1f}% | 合計損益 {s_pnl:+6.2f}%")

    # 4/8（構造変化）前後の比較
    print("\n--- 4/8構造変化 前後比較 ---")
    pre_408 = shosha_pnl[shosha_pnl.index < pd.Timestamp("2026-04-08")]
    post_408 = shosha_pnl[shosha_pnl.index >= pd.Timestamp("2026-04-08")]
    
    if not pre_408.empty:
        print(f"4/8以前: {len(pre_408)}回 | 勝率 {(pre_408['PnL']>0).mean()*100:.1f}% | 損益 {pre_408['PnL'].sum():+.2f}%")
    if not post_408.empty:
        print(f"4/8以後: {len(post_408)}回 | 勝率 {(post_408['PnL']>0).mean()*100:.1f}% | 損益 {post_408['PnL'].sum():+.2f}%")

    # 特徴的な負け日の抽出
    print("\n--- 直近の主な損失トレード ---")
    worst_trades = shosha_pnl.sort_values('PnL').head(5)
    for date, row in worst_trades.iterrows():
        print(f"日付: {date.date()} | サイド: {row['Side']:5} | 損益: {row['PnL']:+.2f}%")

if __name__ == "__main__":
    analyze_shosha_failure()
