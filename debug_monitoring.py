"""Debug script to identify why monitoring metrics fail."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import yfinance as yf
from src.config import (
    DATA_PROCESSED, JP_TICKERS, BASKET_MAPPING,
    CFULL_START, CFULL_END, ROLLING_WINDOW,
    REGULARIZATION_LAMBDA, NUM_FACTORS, TEST_START
)
from src.signal.prior_subspace import build_prior_subspace, compute_cfull
from src.signal.regularized_pca import regularized_pca, rolling_standardize
from src.portfolio.construction import construct_original_portfolio_with_crash_filter

print("=" * 60)
print("Step 1: Load base data")
print("=" * 60)

us_ret = pd.read_csv(DATA_PROCESSED / "us_returns.csv", index_col=0, parse_dates=True).sort_index()
jp_all = pd.read_csv(DATA_PROCESSED / "jp_returns.csv", header=[0, 1], index_col=0, parse_dates=True).sort_index()
jp_cc = jp_all.xs("cc", axis=1, level=1)
jp_oc = jp_all.xs("oc", axis=1, level=1)
date_map = pd.read_csv(DATA_PROCESSED / "us_jp_date_map.csv", parse_dates=["us_date", "jp_next_date"])

print(f"  us_ret: {us_ret.shape}, range: {us_ret.index[0]} ~ {us_ret.index[-1]}")
print(f"  jp_cc: {jp_cc.shape}, range: {jp_cc.index[0]} ~ {jp_cc.index[-1]}")
print(f"  jp_oc: {jp_oc.shape}, range: {jp_oc.index[0]} ~ {jp_oc.index[-1]}")
print(f"  date_map: {date_map.shape}")

print()
print("=" * 60)
print("Step 2: Load OHLCV data via yfinance")
print("=" * 60)

tickers = list(JP_TICKERS)
for basket in BASKET_MAPPING.values():
    tickers.extend(list(basket.keys()))
tickers = list(set(tickers))
print(f"  Total tickers to download: {len(tickers)}")

start_date = "2025-01-01"
try:
    ohlcv = yf.download(tickers, start=start_date, auto_adjust=False).sort_index()
    print(f"  ohlcv shape: {ohlcv.shape}")
    print(f"  ohlcv columns names: {ohlcv.columns.names}")
    print(f"  ohlcv columns levels (first 5): {ohlcv.columns[:5].tolist()}")
    print(f"  ohlcv index range: {ohlcv.index[0]} ~ {ohlcv.index[-1]}")
except Exception as e:
    print(f"  ERROR downloading OHLCV: {e}")
    sys.exit(1)

# Swap levels if needed
if ohlcv.columns.names == ['Price', 'Ticker'] or ohlcv.columns.names == ['Price', 'Symbols']:
    ohlcv.columns = ohlcv.columns.swaplevel(0, 1)
    print("  Swapped column levels")

print(f"  ohlcv columns names after swap: {ohlcv.columns.names}")
print(f"  ohlcv columns levels (first 5): {ohlcv.columns[:5].tolist()}")

print()
print("=" * 60)
print("Step 3: Extract Volume, Open, Close")
print("=" * 60)

try:
    # Check what level values exist
    level0_vals = ohlcv.columns.get_level_values(0).unique().tolist()
    level1_vals = ohlcv.columns.get_level_values(1).unique().tolist()
    print(f"  Level 0 unique values: {level0_vals[:10]}")
    print(f"  Level 1 unique values: {level1_vals[:10]}")
except Exception as e:
    print(f"  ERROR inspecting levels: {e}")

try:
    volume = ohlcv.xs('Volume', level=1, axis=1)
    print(f"  volume shape: {volume.shape}")
    print(f"  volume columns: {volume.columns.tolist()[:5]}")
except Exception as e:
    print(f"  ERROR extracting Volume: {e}")
    # Try other level
    try:
        volume = ohlcv.xs('Volume', level=0, axis=1)
        print(f"  volume shape (level=0): {volume.shape}")
    except Exception as e2:
        print(f"  ERROR extracting Volume (level=0): {e2}")

try:
    open_prices = ohlcv.xs('Open', level=1, axis=1)
    print(f"  open_prices shape: {open_prices.shape}")
except Exception as e:
    print(f"  ERROR extracting Open: {e}")

try:
    close_prices = ohlcv.xs('Close', level=1, axis=1)
    print(f"  close_prices shape: {close_prices.shape}")
except Exception as e:
    print(f"  ERROR extracting Close: {e}")

print()
print("=" * 60)
print("Step 4: Check date alignment")
print("=" * 60)

# Build us_to_jp map
us_to_jp = {}
for _, row in date_map.iterrows():
    us_str = row.iloc[0]
    jp_str = row.iloc[1]
    us_to_jp[pd.Timestamp(us_str)] = pd.Timestamp(jp_str)

test_start = pd.Timestamp(TEST_START)
print(f"  TEST_START: {test_start}")
print(f"  Total us_to_jp entries: {len(us_to_jp)}")

# Filter to test period
test_dates = {k: v for k, v in us_to_jp.items() if v >= test_start}
print(f"  Dates in test period: {len(test_dates)}")

if test_dates:
    jp_dates_test = sorted(test_dates.values())
    print(f"  JP test date range: {jp_dates_test[0]} ~ {jp_dates_test[-1]}")
    
    # Check OHLCV coverage
    ohlcv_dates = ohlcv.index.normalize()
    matched = 0
    unmatched = []
    for jd in jp_dates_test[:10]:
        jd_norm = jd.normalize()
        if jd_norm in ohlcv_dates:
            matched += 1
        else:
            unmatched.append(jd_norm)
    print(f"  First 10 JP test dates matched in OHLCV: {matched}/10")
    if unmatched:
        print(f"  Unmatched dates (first 5): {unmatched[:5]}")

    # Check timezone issue
    print(f"  OHLCV index timezone: {ohlcv.index.tz}")
    print(f"  OHLCV index dtype: {ohlcv.index.dtype}")
    print(f"  JP test date timezone: {jp_dates_test[0].tz if hasattr(jp_dates_test[0], 'tz') else 'None'}")

print()
print("=" * 60)
print("Step 5: Try running combined + z_scores computation")
print("=" * 60)

cfull_corr, us_tickers_cfull = compute_cfull(
    us_ret, jp_cc, cfull_start=CFULL_START, cfull_end=CFULL_END,
    us_tickers_available=None,
)
jp_tickers = list(JP_TICKERS)
V0, C0 = build_prior_subspace(us_tickers_cfull, jp_tickers, cfull_corr)

us = us_ret[us_tickers_cfull]
jp = jp_cc[jp_tickers]
combined = pd.concat([us, jp], axis=1).dropna()
z_scores = rolling_standardize(combined.values, window=ROLLING_WINDOW)

print(f"  combined shape: {combined.shape}")
print(f"  combined index range: {combined.index[0]} ~ {combined.index[-1]}")
print(f"  z_scores shape: {z_scores.shape}")

# Check which us_dates are in combined
in_combined = 0
not_in_combined = 0
for us_d in test_dates.keys():
    if us_d in combined.index:
        in_combined += 1
    else:
        not_in_combined += 1
print(f"  US test dates in combined: {in_combined}")
print(f"  US test dates NOT in combined: {not_in_combined}")

print()
print("=" * 60)
print("Step 6: Simulate first few iterations of monitoring loop")
print("=" * 60)

window = ROLLING_WINDOW
K = NUM_FACTORS
n_us = len(us_tickers_cfull)
metrics_count = 0
skip_reasons = {}

for i, us_date in enumerate(sorted(us_to_jp.keys())):
    jp_date = us_to_jp[us_date]
    
    if jp_date < test_start:
        continue
    
    if us_date not in combined.index:
        skip_reasons.setdefault("us_date not in combined", 0)
        skip_reasons["us_date not in combined"] += 1
        continue
    
    t = combined.index.get_loc(us_date)
    if t < window:
        skip_reasons.setdefault("t < window", 0)
        skip_reasons["t < window"] += 1
        continue

    z_win = z_scores[t - window : t]
    if np.isnan(z_win).any():
        skip_reasons.setdefault("NaN in z_win", 0)
        skip_reasons["NaN in z_win"] += 1
        continue

    V_K = regularized_pca(z_win, C0, lam=REGULARIZATION_LAMBDA, K=K)
    V_U = V_K[:n_us, :]
    V_J = V_K[n_us:, :]
    z_US_t = z_scores[t, :n_us]
    f_t = V_U.T @ z_US_t
    z_hat_J = V_J @ f_t

    signal = pd.Series(z_hat_J, index=jp_tickers)
    current_market_return = us.loc[us_date].mean()
    
    weights = construct_original_portfolio_with_crash_filter(
        signal=signal, 
        market_return=current_market_return, 
        crash_threshold=-0.015
    )
    
    long_tickers = weights[weights > 0].index.tolist()
    
    if jp_date not in jp_oc.index:
        skip_reasons.setdefault("jp_date not in jp_oc", 0)
        skip_reasons["jp_date not in jp_oc"] += 1
        continue
    
    # Check volume/open/close availability
    try:
        vol_available = jp_date in volume.index
        open_available = jp_date in open_prices.index
        close_available = jp_date in close_prices.index
    except:
        vol_available = False
        open_available = False
        close_available = False
    
    metrics_count += 1
    
    if metrics_count <= 3:
        print(f"  Date {metrics_count}: us={us_date.date()}, jp={jp_date.date()}")
        print(f"    Long tickers: {long_tickers}")
        print(f"    Volume available: {vol_available}, Open: {open_available}, Close: {close_available}")
        if vol_available:
            try:
                sma_volume = volume.rolling(20).mean()
                vol = pd.to_numeric(volume.loc[jp_date, long_tickers], errors='coerce')
                sma = pd.to_numeric(sma_volume.loc[jp_date, long_tickers], errors='coerce')
                vol_ratio = (vol / sma).mean()
                print(f"    Volume ratio: {vol_ratio}")
            except Exception as e:
                print(f"    Volume ratio ERROR: {e}")

print()
print(f"  Total metrics computed: {metrics_count}")
print(f"  Skip reasons: {skip_reasons}")
print()
print("DONE")
