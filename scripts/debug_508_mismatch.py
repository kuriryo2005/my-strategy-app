import sys
import pandas as pd
from pathlib import Path

# プロジェクトルートを追加
root = Path(r"c:\Users\kurir\Downloads\my_strategy")
sys.path.insert(0, str(root))

from app import run_pca_sub_backtest_v3, load_data

print("--- 5/8 Return Debug ---")
leverage = 3.0
daily_rets, df_pnl = run_pca_sub_backtest_v3("2026-05-01", leverage=leverage)

ret_508 = daily_rets.get(pd.Timestamp("2026-05-08"))
print(f"5/8 Total Portfolio Return (with {leverage}x leverage): {ret_508}")

if ret_508 is not None:
    print("\nIndividual Ticker Details for 5/8:")
    pnl_508 = df_pnl[df_pnl['Date'] == pd.Timestamp("2026-05-08")]
    print(pnl_508[['Ticker', 'Side', 'Weight', 'Return', 'PnL']])
    print(f"\nSum of Individual PnL: {pnl_508['PnL'].sum()}")
else:
    print("No data found for 5/8 in daily_rets.")
