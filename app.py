"""
Sruntreamlit Web App: 日米セクター リードラグ戦略ダッシュボード

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import os
import subprocess
import matplotlib.pyplot as plt

# --- 日本語フォント設定 ---
import matplotlib.font_manager as fm
_jp_font_candidates = ["IPAexGothic", "Hiragino Sans", "Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
_jp_font_set = False
for _fn in _jp_font_candidates:
    if any(_fn in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        _jp_font_set = True
        break
if not _jp_font_set:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy import stats
# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CFULL_END,
    CFULL_START,
    DATA_PROCESSED,
    DATA_RAW,
    JP_TICKERS,
    JP_TICKER_NAMES,
    NUM_FACTORS,
    QUANTILE_THRESHOLD,
    REGULARIZATION_LAMBDA,
    ROLLING_WINDOW,
    TEST_START,
)
from src.signal.prior_subspace import build_prior_subspace, compute_cfull
from src.signal.regularized_pca import regularized_pca, rolling_standardize
from src.portfolio.construction import construct_portfolio, construct_portfolio_with_crash_filter, construct_original_portfolio_with_crash_filter
from src.evaluation.metrics import compute_metrics

@st.cache_data(show_spinner="データを読み込み中...", ttl="12h")
def load_data():
    # DATA_PROCESSED（元々設定されているパス）を使って正しい場所を指定
    st.sidebar.info(f"Loading from: {os.path.abspath(DATA_PROCESSED)}")
    us_file = DATA_PROCESSED / "us_returns.csv"
    jp_file = DATA_PROCESSED / "jp_returns.csv"
    map_file = DATA_PROCESSED / "us_jp_date_map.csv"
    
    us_ret = pd.read_csv(us_file, index_col=0, parse_dates=True).sort_index()
    
    # jp_returns.csv はマルチインデックスヘッダー (Ticker, ReturnType)
    jp_all = pd.read_csv(jp_file, header=[0, 1], index_col=0, parse_dates=True).sort_index()
    
    # 🟢 JP_TICKERS に含まれる銘柄のみに制限 (除外対応)
    existing_tickers = [t for t in JP_TICKERS if t in jp_all.columns.get_level_values(0)]
    jp_all = jp_all[existing_tickers]
    
    jp_cc = jp_all.xs("cc", axis=1, level=1)
    jp_oc = jp_all.xs("oc", axis=1, level=1)
    
    date_map = pd.read_csv(map_file, parse_dates=["us_date", "jp_next_date"])
    
    return us_ret, jp_cc, jp_oc, date_map
from src.signal.regularized_pca import regularized_pca, rolling_standardize
from src.portfolio.construction import construct_portfolio, construct_portfolio_with_crash_filter, construct_original_portfolio_with_crash_filter
from src.evaluation.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="日米セクター リードラグ戦略",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

def _data_files_exist() -> bool:
    """Check whether the required processed data files exist."""
    us_path = DATA_PROCESSED / "us_returns.csv"
    jp_path = DATA_PROCESSED / "jp_returns.csv"
    map_path = DATA_PROCESSED / "us_jp_date_map.csv"
    return us_path.exists() and jp_path.exists() and map_path.exists()


@st.cache_data(show_spinner="日経平均をダウンロード中...", ttl="12h")
def load_nikkei_benchmark():
    """Load Nikkei 225 data for benchmark comparison."""
    nikkei = yf.download("^N225", start="2000-01-01", progress=False)
    if nikkei.empty:
        return pd.Series(dtype=float)
    if isinstance(nikkei.columns, pd.MultiIndex):
        close_col = nikkei["Close"]["^N225"]
    else:
        close_col = nikkei["Close"]
    rets = close_col.pct_change().dropna()
    rets.index = rets.index.tz_localize(None)
    rets = rets.sort_index()
    rets.name = "Nikkei_Return"
    return rets


@st.cache_data(show_spinner="JP始値データを読み込み中...", ttl="12h")
def load_jp_open_prices():
    """Load JP ETF open prices from raw OHLC for share count calculation."""
    ohlc_path = DATA_RAW / "jp_etf_ohlc.csv"
    if not ohlc_path.exists():
        return None
    ohlc = pd.read_csv(ohlc_path, header=[0, 1], index_col=0, parse_dates=True).sort_index()
    open_cols = {}
    for ticker in JP_TICKERS:
        if (ticker, "Open") in ohlc.columns:
            open_cols[ticker] = ohlc[(ticker, "Open")]
    if not open_cols:
        return None
    return pd.DataFrame(open_cols)


@st.cache_data(show_spinner="JP OHLCVデータを取得中...", ttl="12h")
def load_jp_ohlcv():
    """Load JP ETF and Proxy Basket OHLCV data using yfinance for monitoring indicators."""
    import yfinance as yf
    from src.config import JP_TICKERS, BASKET_MAPPING
    
    tickers = list(JP_TICKERS)
    for basket in BASKET_MAPPING.values():
        tickers.extend(list(basket.keys()))
    tickers = list(set(tickers))
    
    start_date = "2025-01-01"
    df = yf.download(tickers, start=start_date, auto_adjust=False, progress=False).sort_index()
    
    # yfinance returns MultiIndex columns. If 'Price' is level 0, swap it so 'Volume' etc is level 1 to match expected format
    if df.columns.names == ['Price', 'Ticker'] or df.columns.names == ['Price', 'Symbols']:
        df.columns = df.columns.swaplevel(0, 1)
        
    return df

@st.cache_data(show_spinner="Cfull/V0/C0/z-scoreを計算中...", ttl="12h")

def compute_all_artifacts():
    """Compute Cfull, V0, C0, combined returns, z-scores (all heavy work)."""
    us_ret = pd.read_csv(
        DATA_PROCESSED / "us_returns.csv", index_col=0, parse_dates=True,
    ).sort_index()
    
    # jp_returns.csv はマルチインデックスヘッダー (Ticker, ReturnType) を持つ
    jp_all = pd.read_csv(
        DATA_PROCESSED / "jp_returns.csv", header=[0, 1], index_col=0, parse_dates=True,
    ).sort_index()
    jp_cc = jp_all.xs("cc", axis=1, level=1)

    # Cfull (2010-2014, 9 US tickers automatically detected)
    cfull_corr, us_tickers_cfull = compute_cfull(
        us_ret, jp_cc,
        cfull_start=CFULL_START, cfull_end=CFULL_END,
        us_tickers_available=None,
    )
    jp_tickers = list(JP_TICKERS)
    V0, C0 = build_prior_subspace(us_tickers_cfull, jp_tickers, cfull_corr)

    # Combined returns matrix + rolling z-scores
    us = us_ret[us_tickers_cfull]
    jp = jp_cc[jp_tickers]
    # JP休場日（GW等）もUS側のデータを維持するため、
    # US営業日に合わせてJPをreindexし、休場日はリターン0（＝動かなかった）とする
    jp_aligned = jp.reindex(us.index).fillna(0.0)
    combined = pd.concat([us, jp_aligned], axis=1).dropna()
    z_scores = rolling_standardize(combined.values, window=ROLLING_WINDOW)

    return us_tickers_cfull, C0, combined, z_scores


@st.cache_data(show_spinner="バックテストを実行中...", ttl="12h")
def run_pca_sub_backtest(start_date_str: str = "2022-01-01"):
    """Run PCA_SUB backtest and return daily returns Series."""
    
    # 🟢 1. データ読み込みはこれ一行に集約します
    # 冒頭にあった個別の pd.read_csv は、エラーの原因（header=[0,1]など）になるので削除します
    us_ret, jp_cc, jp_oc, date_map = load_data()

    # 🟢 2. 事前情報の計算（ここはOKです）
    cfull_corr, us_tickers_cfull = compute_cfull(
        us_ret, jp_cc, cfull_start=CFULL_START, cfull_end=CFULL_END,
        us_tickers_available=None,
    )
    
    jp_tickers = list(JP_TICKERS)
    V0, C0 = build_prior_subspace(us_tickers_cfull, jp_tickers, cfull_corr)

    # ... この後に続くバックテストのループ処理へ ...

    # Combined + z-scores（JP休場日もUS側のデータを維持）
    us = us_ret[us_tickers_cfull]
    jp = jp_cc[jp_tickers]
    jp_aligned = jp.reindex(us.index).fillna(0.0)
    combined = pd.concat([us, jp_aligned], axis=1).dropna()
    z_scores = rolling_standardize(combined.values, window=ROLLING_WINDOW)

# Date map lookup
    us_to_jp = {}
    for _, row in date_map.iterrows():
        # CSVのカラム名がどうなっていても安全に取得できるように、
        # 列のインデックス（0番目と1番目）で指定します。
        us_str = row.iloc[0]  
        jp_str = row.iloc[1]
        
        # タイムスタンプに変換して辞書に格納
        us_to_jp[pd.Timestamp(us_str)] = pd.Timestamp(jp_str)

    test_start = pd.Timestamp(TEST_START)
    window = ROLLING_WINDOW
    K = NUM_FACTORS
    n_us = len(us_tickers_cfull)

    test_start = pd.Timestamp(start_date_str)
    daily_rets = []
    pnl_details = []
    for us_date in sorted(us_to_jp.keys()):
        jp_date = us_to_jp[us_date]
        if jp_date < test_start:
            continue
        if us_date not in combined.index:
            continue
        t = combined.index.get_loc(us_date)
        if t < window:
            continue

        z_win = z_scores[t - window : t]
        if np.isnan(z_win).any():
            continue

        V_K = regularized_pca(z_win, C0, lam=REGULARIZATION_LAMBDA, K=K)
        V_U = V_K[:n_us, :]
        V_J = V_K[n_us:, :]
        z_US_t = z_scores[t, :n_us]
        f_t = V_U.T @ z_US_t
        z_hat_J = V_J @ f_t

        signal = pd.Series(z_hat_J, index=jp_tickers)
        
        # --- 🟢 ハイブリッド戦略適用 (決算月は順張り) ---
        is_earnings = jp_date.month in [2, 5, 8, 11]
        current_market_return = us.loc[us_date].mean()
        
        # 順張りの場合はシグナルを反転させてから、ウェイト計算（除外設定を含む）に渡す
        if is_earnings:
            signal_input = -signal
        else:
            signal_input = signal

        weights = construct_portfolio_with_crash_filter(
            signal=signal_input, 
            market_return=current_market_return,
        )
        # --- 🟢 ここまで ---

        if jp_date not in jp_oc.index:
            continue
        oc_ret = jp_oc.loc[jp_date, jp_tickers]
        
        # 個別銘柄の損益を記録
        individual_pnl = weights * oc_ret
        for ticker, pnl in individual_pnl.items():
            if weights[ticker] != 0 and pd.notna(pnl):
                pnl_details.append({
                    'Date': jp_date,
                    'Ticker': ticker,
                    'Side': 'Long' if weights[ticker] > 0 else 'Short',
                    'Weight': weights[ticker],
                    'Return': oc_ret[ticker],
                    'PnL': pnl
                })
                
        port_ret = individual_pnl.sum()
        # 3.3倍レバレッジ換算 (現物1.0 + 信用2.3 を想定し、L/S各1.65倍)
        port_ret = port_ret * 1.65
        daily_rets.append((jp_date, port_ret))

    if not daily_rets:
        return pd.Series(dtype=float), pd.DataFrame()
    dates, rets = zip(*daily_rets)
    df_pnl = pd.DataFrame(pnl_details)
    return pd.Series(rets, index=pd.DatetimeIndex(dates), name="PCA_SUB").sort_index(), df_pnl


@st.cache_data(show_spinner="群集行動指標を計算中...", ttl="12h")
def run_monitoring_metrics():
    """Calculate original strategy metrics for crowd monitoring."""
    import traceback as _tb
    
    us_ret, jp_cc, jp_oc, date_map = load_data()
    ohlcv = load_jp_ohlcv()
    
    if ohlcv is None or ohlcv.empty:
        st.sidebar.warning("OHLCV データが取得できませんでした")
        return pd.DataFrame()

    # --- OHLCV から Volume / Open / Close を抽出 ---
    # yfinance の MultiIndex は (Ticker, Price) or (Price, Ticker) の場合がある
    try:
        col_names = ohlcv.columns.names
        # level 1 に 'Volume' etc. が入っているか確認
        level1_vals = ohlcv.columns.get_level_values(1).unique().tolist()
        if 'Volume' in level1_vals:
            price_level = 1
        else:
            price_level = 0
        
        volume = ohlcv.xs('Volume', level=price_level, axis=1)
        sma_volume = volume.rolling(20).mean()
        open_prices = ohlcv.xs('Open', level=price_level, axis=1)
        close_prices = ohlcv.xs('Close', level=price_level, axis=1)
    except Exception as e:
        st.sidebar.error(f"OHLCV データの解析に失敗: {e}")
        return pd.DataFrame()

    cfull_corr, us_tickers_cfull = compute_cfull(
        us_ret, jp_cc, cfull_start=CFULL_START, cfull_end=CFULL_END,
        us_tickers_available=None,
    )
    jp_tickers = list(JP_TICKERS)
    V0, C0 = build_prior_subspace(us_tickers_cfull, jp_tickers, cfull_corr)

    us = us_ret[us_tickers_cfull]
    jp = jp_cc[jp_tickers]
    # JP休場日（GW等）もUS側のデータを維持するため、
    # run_pca_sub_backtest() と同じアライメントを使用
    jp_aligned = jp.reindex(us.index).fillna(0.0)
    combined = pd.concat([us, jp_aligned], axis=1).dropna()
    z_scores = rolling_standardize(combined.values, window=ROLLING_WINDOW)

    us_to_jp = {}
    for _, row in date_map.iterrows():
        us_str = row.iloc[0]  
        jp_str = row.iloc[1]
        us_to_jp[pd.Timestamp(us_str)] = pd.Timestamp(jp_str)

    test_start = pd.Timestamp(TEST_START)
    window = ROLLING_WINDOW
    K = NUM_FACTORS
    n_us = len(us_tickers_cfull)

    metrics_list = []
    for us_date in sorted(us_to_jp.keys()):
        jp_date = us_to_jp[us_date]
        if jp_date < test_start:
            continue
        if us_date not in combined.index:
            continue
        t = combined.index.get_loc(us_date)
        if t < window:
            continue

        z_win = z_scores[t - window : t]
        if np.isnan(z_win).any():
            continue

        V_K = regularized_pca(z_win, C0, lam=REGULARIZATION_LAMBDA, K=K)
        V_U = V_K[:n_us, :]
        V_J = V_K[n_us:, :]
        z_US_t = z_scores[t, :n_us]
        f_t = V_U.T @ z_US_t
        z_hat_J = V_J @ f_t

        signal = pd.Series(z_hat_J, index=jp_tickers)
        current_market_return = us.loc[us_date].mean()
        
        # 元の戦略のウェイトを計算 (Long5, Short5, 順張り)
        weights = construct_original_portfolio_with_crash_filter(
            signal=signal, 
            market_return=current_market_return, 
            crash_threshold=-0.015
        )
        
        # ターゲット銘柄群（群集が買っている銘柄 = オリジナルのLong銘柄）
        long_tickers = weights[weights > 0].index.tolist()

        if jp_date not in jp_oc.index:
            continue
            
        oc_ret = jp_oc.loc[jp_date, jp_tickers]
        port_ret = (weights * oc_ret).sum()
        
        # 出来高スパイク (Volume Anomaly)
        vol_ratio = np.nan
        if jp_date in volume.index and jp_date in sma_volume.index:
            try:
                available_long = [t for t in long_tickers if t in volume.columns]
                if available_long:
                    vol = pd.to_numeric(volume.loc[jp_date, available_long], errors='coerce')
                    sma = pd.to_numeric(sma_volume.loc[jp_date, available_long], errors='coerce')
                    valid_mask = (sma > 0) & sma.notna() & vol.notna()
                    if valid_mask.any():
                        vol_ratio = (vol[valid_mask] / sma[valid_mask]).mean()
            except Exception:
                pass
            
        # イントラデイ・リバーサル (日中の押し戻し) と バスケット乖離率
        rev = np.nan
        div_list = []
        if jp_date in open_prices.index and jp_date in close_prices.index:
            try:
                available_long = [t for t in long_tickers if t in open_prices.columns]
                if available_long:
                    o = pd.to_numeric(open_prices.loc[jp_date, available_long], errors='coerce')
                    c = pd.to_numeric(close_prices.loc[jp_date, available_long], errors='coerce')
                    valid_mask = o.notna() & c.notna() & (o != 0)
                    if valid_mask.any():
                        rev = ((o[valid_mask] - c[valid_mask]) / o[valid_mask] * 100).mean()
            except Exception:
                pass
                
            # バスケット乖離率（プレミアム）
            from src.config import BASKET_MAPPING
            for etf_ticker in long_tickers:
                if etf_ticker in BASKET_MAPPING:
                    basket = BASKET_MAPPING[etf_ticker]
                    
                    try:
                        if etf_ticker not in open_prices.columns or etf_ticker not in close_prices.columns:
                            continue
                        etf_o = pd.to_numeric(open_prices.loc[jp_date, etf_ticker])
                        etf_c = pd.to_numeric(close_prices.loc[jp_date, etf_ticker])
                        if pd.isna(etf_o) or pd.isna(etf_c) or etf_o == 0:
                            continue
                        etf_ret = (etf_c - etf_o) / etf_o
                        
                        basket_ret = 0.0
                        valid_basket = False
                        for proxy_tick, weight in basket.items():
                            if proxy_tick not in open_prices.columns or proxy_tick not in close_prices.columns:
                                continue
                            p_o = pd.to_numeric(open_prices.loc[jp_date, proxy_tick])
                            p_c = pd.to_numeric(close_prices.loc[jp_date, proxy_tick])
                            if pd.notna(p_o) and pd.notna(p_c) and p_o != 0:
                                basket_ret += ((p_c - p_o) / p_o) * weight
                                valid_basket = True
                        
                        if valid_basket:
                            # 乖離率 (ETFリターン - バスケットリターン)
                            div = (etf_ret - basket_ret) * 100
                            div_list.append(div)
                    except Exception:
                        pass
            
        basket_div = np.mean(div_list) if div_list else np.nan
            
        metrics_list.append({
            'date': jp_date,
            'Original_Return': port_ret,
            'Volume_Anomaly': vol_ratio,
            'Intraday_Reversal': rev,
            'Basket_Divergence': basket_div
        })

    if not metrics_list:
        return pd.DataFrame()
        
    df_metrics = pd.DataFrame(metrics_list).set_index('date').sort_index()
    return df_metrics



# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------

def find_us_date_for_jp(jp_date: pd.Timestamp, date_map: pd.DataFrame):
    """Reverse-lookup: JP date -> US date."""
    match = date_map[date_map["jp_next_date"] == jp_date]
    if match.empty:
        return None
    return pd.Timestamp(match.iloc[-1]["us_date"])


def find_nearest_jp_date(target, date_map, direction="backward"):
    """Find the nearest JP business day in the date_map."""
    jp_dates = pd.DatetimeIndex(date_map["jp_next_date"].sort_values().unique())
    if direction == "backward":
        candidates = jp_dates[jp_dates <= target]
        return candidates[-1] if len(candidates) > 0 else None
    else:
        candidates = jp_dates[jp_dates >= target]
        return candidates[0] if len(candidates) > 0 else None


def generate_signal(us_date, combined, z_scores, us_tickers_cfull, C0):
    """Generate signal and weights for a single US date.
    
    一度生成したシグナルはファイルに保存し、同じ日は再計算しない。
    これにより場中にデータ更新してもシグナルが変わらない。
    """
    import json
    
    jp_tickers = list(JP_TICKERS)
    
    # --- シグナル保存ディレクトリ ---
    signal_dir = DATA_PROCESSED / "signals"
    signal_dir.mkdir(exist_ok=True)
    signal_file = signal_dir / f"{us_date.strftime('%Y-%m-%d')}.json"
    
    # --- 保存済みシグナルがあればそれを使う ---
    if signal_file.exists():
        with open(signal_file, "r") as f:
            saved = json.load(f)
        signal = pd.Series(saved["signal"])
        weights = pd.Series(saved["weights"])
        return signal, weights, None
    
    # --- なければ計算して保存 ---
    window = ROLLING_WINDOW
    K = NUM_FACTORS

    if us_date not in combined.index:
        return None, None, f"US日付 {us_date.date()} はデータに存在しません"

    t_loc = combined.index.get_loc(us_date)
    if t_loc < window * 2:
        return None, None, f"ローリングウィンドウに十分なデータがありません (位置={t_loc})"

    z_win = z_scores[t_loc - window : t_loc]
    if np.any(np.isnan(z_win)):
        return None, None, "PCAウィンドウにNaNが含まれています"

    V_K = regularized_pca(z_win, C0, lam=REGULARIZATION_LAMBDA, K=K)
    n_us = len(us_tickers_cfull)
    V_U = V_K[:n_us, :]
    V_J = V_K[n_us:, :]

    z_US_t = z_scores[t_loc, :n_us]
    f_t = V_U.T @ z_US_t
    z_hat_J = V_J @ f_t

    signal = pd.Series(z_hat_J, index=jp_tickers)
    
    current_market_return = combined.loc[us_date].iloc[:n_us].mean()
    weights = construct_portfolio_with_crash_filter(
        signal=signal, 
        market_return=current_market_return,
    )
    
    # --- ファイルに保存（次回以降は再計算しない） ---
    with open(signal_file, "w") as f:
        json.dump({
            "us_date": us_date.strftime("%Y-%m-%d"),
            "signal": signal.to_dict(),
            "weights": weights.to_dict(),
            "market_return": float(current_market_return),
        }, f, indent=2)
    
    return signal, weights, None


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

st.sidebar.title("日米セクター リードラグ戦略")

# --- 🗓️ バックテスト期間設定 ---
st.sidebar.subheader("📅 バックテスト設定")
bt_start_date = st.sidebar.date_input(
    "バックテスト開始日",
    value=pd.Timestamp("2022-01-01").date(),
    min_value=pd.Timestamp("2015-01-01").date(),
    max_value=pd.Timestamp.today().date(),
    help="この日付以降のデータを用いて実績およびシミュレーションを計算します。"
)
# config の TEST_START を上書き
test_start_str = bt_start_date.strftime("%Y-%m-%d")

# --- 💰 運用・シミュレーション設定 ---
st.sidebar.subheader("💰 運用・資金設定")
trading_days_year = st.sidebar.slider(
    "シミュレーション年間日数",
    min_value=100,
    max_value=250,
    value=245,
    help="日本株の年間取引日数（約245日）を設定します。"
)

page = st.sidebar.radio(
    "ページ選択",
    ["本日のシグナル", "バックテスト結果", "直近パフォーマンス", "群集行動モニタリング"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("データ管理")
if st.sidebar.button("データを最新に更新"):
    with st.spinner("最新データを取得中..."):
        try:
            script_path = PROJECT_ROOT / "manual_refresh.py"
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )
            st.cache_data.clear()
            st.sidebar.success("更新完了！（シグナルは固定済みのため変わりません）")
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.sidebar.error(f"更新失敗 (Code {e.returncode}):\n{e.stderr}")
        except Exception as e:
            st.sidebar.error(f"予期せぬエラー: {e}")

if st.sidebar.button("キャッシュをクリア"):
    st.cache_data.clear()
    # 🟢 保存済みシグナル（JSON）も削除
    signal_dir = DATA_PROCESSED / "signals"
    if signal_dir.exists():
        import shutil
        for f in signal_dir.glob("*.json"):
            try:
                f.unlink()
            except:
                pass
    st.sidebar.success("キャッシュと保存済みシグナルをクリアしました")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "**免責事項**\n\n"
    "本ツールは学術論文(中川ら, SIG-FIN-036)の"
    "再現実装をもとに栗田凌羽同時解析の上、研究・教育目的で公開しています。\n\n"
    "特定の金融商品の売買を推奨するものではありません。"
    "投資判断はご自身の責任で行ってください。\n\n"
    "データソース: Yahoo Finance, Kenneth French Data Library"
)

# ---------------------------------------------------------------------------
# Data availability check
# ---------------------------------------------------------------------------

if not _data_files_exist():
    st.error(
        "Required data files are missing. Run the scheduled HF update job "
        "to regenerate data for this Space."
    )
    st.caption("Setup instructions: docs/hf-jobs-setup.md")
    st.stop()

# Load base data (cheap, cached)
# Load base data (cheap, cached)
# xsでの抽出をやめ、直接4つのデータを受け取る
us_ret, jp_cc, jp_oc, date_map = load_data()


# ===================================================================
# Page 1: Today's Signal
# ===================================================================
if page == "本日のシグナル":
    st.header("本日の運用シグナル")
    
    # --- 💰 資金・元金の一括設定 ---
    st.markdown("### 💰 資金・ポジション設定")
    col_cap1, col_cap2 = st.columns([1, 2])
    with col_cap1:
        initial_capital = st.number_input(
            "運用元金（万円）",
            min_value=1,
            max_value=100000,
            value=100,
            step=10,
            key="main_capital_input",
            help="この元金をベースにポジションサイズを計算します。"
        )
    with col_cap2:
        st.info(f"現在の運用元金: **{initial_capital:,} 万円**")

    # Compute heavy artifacts (cached)
    us_tickers_cfull, C0, combined, z_scores = compute_all_artifacts()

    # Date picker
    jp_dates = pd.DatetimeIndex(date_map["jp_next_date"].sort_values().unique())
    today_jst = pd.Timestamp.today(tz="Asia/Tokyo").normalize().tz_localize(None)
    
    default_jp = today_jst
    if default_jp not in jp_dates:
        candidates = jp_dates[jp_dates <= today_jst]
        default_jp = candidates[-1] if not candidates.empty else jp_dates[-1]

    col_date, col_lev = st.columns([1.2, 1])
    with col_date:
        selected_date = st.date_input(
            "対象日（日本市場）",
            value=default_jp.date(),
            min_value=jp_dates[0].date(),
            max_value=max(jp_dates[-1].date(), today_jst.date()),
            help="日本市場での取引日を選択してください。"
        )
    with col_lev:
        leverage = st.slider("レバレッジ倍率", min_value=1.0, max_value=3.3, value=3.0, step=0.1)
        
    capital = initial_capital * 10000 * leverage
    target_ts = pd.Timestamp(selected_date)
    jp_date = target_ts
    us_date = find_us_date_for_jp(target_ts, date_map)
    
    if us_date is None:
        found_jp = find_nearest_jp_date(target_ts, date_map, direction="backward")
        if found_jp is not None:
            target_ts = found_jp
            us_date = find_us_date_for_jp(target_ts, date_map)
    
    if us_date is None or us_date not in combined.index:
        st.error("データが見つかりません。")
        st.stop()

    # --- 📅 戦略モードの自動判定 ---
    is_earnings_month = target_ts.month in [2, 5, 8, 11]
    
    # 推奨モードのデフォルト設定
    if is_earnings_month:
        default_mode = "順張り (Momentum)"
        mode_reason = "【決算期モード】決算発表による強いトレンドを利益に変える設定です。"
    else:
        default_mode = "逆張り (Contrarian)"
        mode_reason = "【平時モード】米国市場への過剰反応からの回帰を狙う標準設定です。"

    st.info(f"💡 **現在の推奨**: **{default_mode}** \n\n {mode_reason}")

    # モード切り替え
    strategy_mode = st.radio(
        "実行モードを選択（手動切り替え可能）",
        ["逆張り (Contrarian)", "順張り (Momentum)"],
        index=0 if default_mode == "逆張り (Contrarian)" else 1,
        horizontal=True
    )

    if target_ts >= pd.Timestamp("2026-04-08") and strategy_mode == "順張り (Momentum)" and not is_earnings_month:
        st.warning("⚠️ **4/8以降の警告**: この期間は逆張りが圧倒的に有効な相場構造になっています。あえて順張りを選択する場合は、強いトレンドがあることを確認してください。")

    # --- 📝 運用アクションガイド ---
    with st.expander("🛠️ 運用アクションガイド（レバレッジ3倍）", expanded=True):
        st.markdown(f"""
        1.  **寄付前 (08:50 - 08:59)**:
            - 推奨投資サイズ: **1.0x 〜 1.5x** を参考にロットを調整。
            - 現在のモード (**{strategy_mode}**) に基づいた銘柄リストが下に表示されます。
        2.  **寄付直後 (09:00 - 09:05)**:
            - **絶対厳守**: 買値/売値から **-0.5%** の位置に **「逆指値」** をすぐに入れてください。
        3.  **大引け前 (14:55 - 15:00)**:
            - 全ポジションを成行で決済。**持ち越しは破滅の元です。**
        """)

    st.markdown(f"**日本市場日付:** {selected_date} | **米国基準日:** {us_date.strftime('%Y-%m-%d')}")

    # 1. 生シグナルの生成
    raw_signal, _, err = generate_signal(us_date, combined, z_scores, us_tickers_cfull, C0)
    if err:
        st.error(err)
        st.stop()

    # 2. 🟢 戦略モードに応じてシグナルの極性を決定
    if strategy_mode == "順張り (Momentum)":
        signal_input = -raw_signal
    else:
        signal_input = raw_signal

    # 3. 🟢 反転後のシグナルに対して、正しい除外設定を適用してウェイトを再計算
    # market_return は combined から算出
    n_us = len(us_tickers_cfull)
    current_market_return = combined.loc[us_date].iloc[:n_us].mean()
    
    weights = construct_portfolio_with_crash_filter(
        signal=signal_input,
        market_return=current_market_return,
    )
    
    # 画面表示用のシグナルと変数名調整
    signal = signal_input

    # Fetch previous close prices
    ohlcv = load_jp_ohlcv()
    prev_close_prices = None
    if ohlcv is not None:
        close_df = ohlcv.xs('Close', level=1, axis=1)
        prev_dates = close_df.index[close_df.index < jp_date]
        if len(prev_dates) > 0:
            prev_close_prices = close_df.loc[prev_dates[-1]]

    # Build display table
    rows = []
    
    target_gross = initial_capital * 10000 * leverage
    gross_weight_sum = weights.abs().sum()
    norm_weights = weights / gross_weight_sum if gross_weight_sum > 0 else weights
    
    remaining_cash = initial_capital * 10000

    for ticker in signal.sort_values(ascending=False).index:
        w = norm_weights[ticker]
        if w > 0:
            side = "LONG"
        elif w < 0:
            side = "SHORT"
        else:
            side = "-"

        name = JP_TICKER_NAMES.get(ticker, "")
        sig_val = signal[ticker]
        target_jpy = abs(w) * target_gross

        shares_str = ""
        actual_jpy_str = ""
        order_type_str = ""
        prev_close_str = ""
        
        if w != 0 and prev_close_prices is not None and ticker in prev_close_prices.index:
            prev_close = prev_close_prices[ticker]
            if pd.notna(prev_close) and prev_close > 0:
                shares = max(1, int(target_jpy / prev_close))
                actual_jpy = shares * prev_close
                shares_str = f"{shares:,}"
                actual_jpy_str = f"{actual_jpy:,.0f}"
                prev_close_str = f"{prev_close:,.0f}"
                
                if side == "SHORT":
                    order_type_str = "信用売"
                elif side == "LONG":
                    shares_cash = min(shares, int(remaining_cash / prev_close))
                    shares_margin = shares - shares_cash
                    
                    remaining_cash -= shares_cash * prev_close
                    
                    if shares_cash == shares:
                        order_type_str = "現物買"
                    elif shares_cash == 0:
                        order_type_str = "信用買"
                    else:
                        order_type_str = f"現物買({shares_cash:,}口) + 信用買({shares_margin:,}口)"

        rows.append({
            "ティッカー": ticker,
            "セクター名": name,
            "サイド": side,
            "シグナル": round(sig_val, 4),
            "目標金額(円)": f"{target_jpy:,.0f}" if w != 0 else "",
            "前日終値(円)": prev_close_str,
            "推奨株数": shares_str,
            "注文区分": order_type_str,
            "実金額(円)": actual_jpy_str,
        })

    df_display = pd.DataFrame(rows)

    def style_side(val):
        if val == "LONG":
            return "color: #2ca02c; font-weight: bold"
        elif val == "SHORT":
            return "color: #d62728; font-weight: bold"
        return ""

    styled = df_display.style.map(style_side, subset=["サイド"])
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=35 * len(df_display) + 38,
    )
    st.caption("※ TOPIX-17業種ETFは1口単位で売買可能（個別株の100株単位とは異なります）")

    # Summary
    long_count = (weights > 0).sum()
    short_count = (weights < 0).sum()
    st.markdown(
        f"**ロング {long_count}銘柄** / **ショート {short_count}銘柄** / "
        f"計 {len(weights)}銘柄 |"
        f"グロスエクスポージャー: {weights.abs().sum():.1f} |"
        f"ネットエクスポージャー: {weights.sum():.4f}"
    )

    # Actual P&L if date is in the past
    if jp_date in jp_oc.index:
        oc_ret = jp_oc.loc[jp_date, list(JP_TICKERS)]
        # 生のリターン (2.0倍分)
        port_ret = (weights * oc_ret).sum()
        # レバレッジ係数
        leverage_factor = leverage / 2.0
        port_ret_leveraged = port_ret * leverage_factor

        if pd.Timestamp.today().normalize() > jp_date:
            st.markdown("---")
            st.subheader("実績（始値→終値）")

            pnl_jpy = port_ret_leveraged * (initial_capital * 10000)
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("ポートフォリオリターン", f"{port_ret_leveraged * 100:+.4f}%")
            with col_res2:
                st.metric("損益（円）", f"{pnl_jpy:+,.0f}")
                
            st.markdown("**セクター別 損益寄与 (Top 4)**")
            # 寄与度にもレバレッジを適用
            individual_pnl = (weights * oc_ret * leverage_factor).replace(0, np.nan).dropna().sort_values(ascending=False)
            
            if len(individual_pnl) > 0:
                top4_cols = st.columns(min(4, len(individual_pnl)))
                for i, (ticker, pnl) in enumerate(individual_pnl.head(4).items()):
                    with top4_cols[i]:
                        st.metric(
                            f"{i+1}位: {JP_TICKER_NAMES.get(ticker, ticker)}",
                            f"{pnl * 100:+.2f}%",
                            f"¥ {int(pnl * (initial_capital * 10000)):+,}"
                        )

    # ===================================================================
    # リアルタイム損益セクション (本日のポジションの含み損益)
    # ===================================================================
    # 本日の日付が選択されている場合にリアルタイム損益を表示
    today_norm = pd.Timestamp.today(tz="Asia/Tokyo").normalize().tz_localize(None)
    is_today = (pd.Timestamp(selected_date) == today_norm)
    
    if is_today and weights is not None:
        st.markdown("---")
        st.subheader("📊 本日のリアルタイム損益")
        
        # ポジションのある銘柄だけ抽出
        active_tickers = weights[weights != 0].index.tolist()
        
        if not active_tickers:
            st.info("本日はポジションがありません（暴落フィルターにより全銘柄キャッシュ）。")
        else:
            with st.spinner("最新の価格データを取得中..."):
                try:
                    import yfinance as yf
                    
                    # yfinanceで最新のデータ（本日分）を取得
                    live_data = yf.download(
                        active_tickers, 
                        period="1d", 
                        interval="1d",
                        auto_adjust=False, 
                        progress=False
                    )
                    
                    if live_data.empty:
                        st.warning("本日の価格データがまだ取得できません。市場が開くまでお待ちください。")
                    else:
                        # MultiIndexの場合とそうでない場合を処理
                        if isinstance(live_data.columns, pd.MultiIndex):
                            col_names = live_data.columns.names
                            level1_vals = live_data.columns.get_level_values(1).unique().tolist()
                            if 'Open' in level1_vals:
                                price_level = 1
                            else:
                                price_level = 0
                            open_today = live_data.xs('Open', level=price_level, axis=1).iloc[-1]
                            # 現在値 = 最終取引価格 (Close がリアルタイムでは最新値)
                            close_today = live_data.xs('Close', level=price_level, axis=1).iloc[-1]
                        else:
                            open_today = live_data['Open'].iloc[-1]
                            close_today = live_data['Close'].iloc[-1]
                        
                        # 各銘柄の損益を計算
                        pnl_rows = []
                        total_pnl_jpy = 0.0
                        total_invested = 0.0
                        
                        # ノーマライズされたウェイト
                        gross_w = weights.abs().sum()
                        nw = weights / gross_w if gross_w > 0 else weights
                        
                        # 🟢 レバレッジを明示的に定義 (3.3倍)
                        leverage = 3.3
                        st.sidebar.write(f"DEBUG: leverage={leverage}")
                        
                        for ticker in active_tickers:
                            w = nw[ticker]
                            side = "LONG" if w > 0 else "SHORT"
                            name = JP_TICKER_NAMES.get(ticker, ticker)
                            
                            if ticker not in open_today.index or ticker not in close_today.index:
                                continue
                            
                            o_price = pd.to_numeric(open_today[ticker], errors='coerce')
                            c_price = pd.to_numeric(close_today[ticker], errors='coerce')
                            
                            if pd.isna(o_price) or pd.isna(c_price) or o_price == 0:
                                continue
                            
                            # 始値→現在値のリターン
                            oc_return = (c_price - o_price) / o_price
                            
                            # ポジション金額 (元金 * レバレッジ を各銘柄のウェイトで配分)
                            position_jpy = abs(w) * (initial_capital * 10000 * leverage)
                            shares = max(1, int(position_jpy / o_price))
                            actual_position = shares * o_price
                            
                            # 損益計算 (ロングなら正のリターンが利益、ショートなら負のリターンが利益)
                            if side == "LONG":
                                ticker_pnl = (c_price - o_price) * shares
                            else:  # SHORT
                                ticker_pnl = (o_price - c_price) * shares
                            
                            ticker_return_pct = ticker_pnl / actual_position * 100 if actual_position > 0 else 0
                            
                            total_pnl_jpy += ticker_pnl
                            total_invested += actual_position
                            
                            pnl_rows.append({
                                "ティッカー": ticker,
                                "セクター名": name,
                                "サイド": side,
                                "始値": f"¥{o_price:,.0f}",
                                "現在値": f"¥{c_price:,.0f}",
                                "株数": f"{shares:,}",
                                "建玉金額": f"¥{actual_position:,.0f}",
                                "損益(円)": f"¥{ticker_pnl:+,.0f}",
                                "損益(%)": f"{ticker_return_pct:+.2f}%",
                            })
                        
                        if pnl_rows:
                            # サマリーメトリクス
                            total_return_pct = (total_pnl_jpy / total_invested * 100) if total_invested > 0 else 0
                            
                            col_pnl1, col_pnl2, col_pnl3 = st.columns(3)
                            with col_pnl1:
                                st.metric("合計損益（円）", f"¥{total_pnl_jpy:+,.0f}", delta=f"{total_return_pct:+.2f}%")
                            with col_pnl2:
                                st.metric("建玉合計", f"¥{total_invested:,.0f}")
                            with col_pnl3:
                                # 🟢 購入余力の計算 (レバレッジ枠 3.3倍の残り)
                                total_capacity = initial_capital * 10000 * leverage
                                buying_power = total_capacity - total_invested
                                st.metric("購入余力", f"¥{buying_power:,.0f}", delta=f"{(buying_power/total_capacity*100):.1f}%")
                            
                            # 銘柄別テーブル
                            df_pnl_live = pd.DataFrame(pnl_rows)
                            
                            def style_pnl(val):
                                if isinstance(val, str):
                                    if val.startswith("+") or val.startswith("¥+") or val.startswith("¥ +"):
                                        return "color: #2ca02c; font-weight: bold"
                                    elif val.startswith("-") or val.startswith("¥-") or val.startswith("¥ -"):
                                        return "color: #d62728; font-weight: bold"
                                return ""
                            
                            def style_side(val):
                                if val == "LONG":
                                    return "color: #2ca02c; font-weight: bold"
                                elif val == "SHORT":
                                    return "color: #d62728; font-weight: bold"
                                return ""
                            
                            styled_pnl = df_pnl_live.style.map(
                                style_pnl, subset=["損益(円)", "損益(%)"]
                            ).map(
                                style_side, subset=["サイド"]
                            )
                            st.dataframe(
                                styled_pnl, 
                                width="stretch", 
                                hide_index=True,
                                height=35 * len(df_pnl_live) + 38,
                            )
                            
                            st.caption("※ 価格データはyfinanceから取得しています。リアルタイムではなく最大20分程度の遅延がある場合があります。")
                            
                            # 再読み込みボタン
                            if st.button("🔄 価格を更新"):
                                st.rerun()
                        else:
                            st.warning("本日の価格データが取得できませんでした。")
                except Exception as e:
                    st.error(f"リアルタイム価格の取得に失敗しました: {e}")

    # ===================================================================
    # 1年後リターン予測シミュレーション
    # ===================================================================
    st.markdown("---")
    st.subheader("💰 1年後リターン予測シミュレーション")
    st.caption(f"現在の設定: 元金 {initial_capital}万円 / 年間 {trading_days_year}日 (設定はサイドバーから変更可能です)")

    # バックテスト結果からリターン分布を取得
    try:
        # 全期間のデータを取得
        daily_rets_full, _ = run_pca_sub_backtest(test_start_str)
        
        # 🟢 シミュレーションの根拠を 2026年以降に限定
        daily_rets_sim = daily_rets_full[daily_rets_full.index >= pd.Timestamp("2026-01-01")]
        
        if daily_rets_sim is not None and len(daily_rets_sim) > 0:
            # 🟢 取引が発生した日のみを対象にする
            ret_array = daily_rets_sim[daily_rets_sim != 0].values
            
            # 指標再計算 (2026年ベース)
            metrics_2026 = compute_metrics(daily_rets_sim)

            # --- 実績統計 ---
            avg_daily = ret_array.mean()
            std_daily = ret_array.std()
            win_rate = (ret_array > 0).mean()
            n_days = len(ret_array)

            # --- モンテカルロシミュレーション ---
            np.random.seed(42)
            n_sims = 10000
            trading_days_year = 252
            sim_results = np.zeros(n_sims)
            sim_paths = np.zeros((n_sims, trading_days_year + 1))
            sim_paths[:, 0] = initial_capital

            for i in range(n_sims):
                # 実績分布からリサンプリング
                sampled = np.random.choice(ret_array, size=trading_days_year, replace=True)
                cumulative = np.cumprod(1 + sampled)
                sim_results[i] = initial_capital * cumulative[-1]
                sim_paths[i, 1:] = initial_capital * cumulative

            # パーセンタイル
            p5 = np.percentile(sim_results, 5)
            p25 = np.percentile(sim_results, 25)
            p50 = np.percentile(sim_results, 50)
            p75 = np.percentile(sim_results, 75)
            p95 = np.percentile(sim_results, 95)
            mean_result = sim_results.mean()

            # 損失確率
            loss_prob = (sim_results < initial_capital).mean() * 100

            # --- 指標計算 (2026年ベース) ---
            metrics_real = metrics_2026

            # --- 表示 ---
            st.markdown(f"#### 📊 2026/01/01〜現在までのCAGR（複利年率）: **{metrics_real['CAGR']*100:.1f}%**")
            st.markdown("現在の相場における「稼ぐ力（2026年〜の実績）」に基づいた、今後1年間の資産予測です。")

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("悲観（下位5%）", f"{p5:.0f}万円", f"{(p5/initial_capital-1)*100:+.1f}%", delta_color="inverse")
            with col_s2:
                st.metric("中央値（50%）", f"{p50:.0f}万円", f"{(p50/initial_capital-1)*100:+.1f}%")
            with col_s3:
                st.metric("楽観（上位5%）", f"{p95:.0f}万円", f"{(p95/initial_capital-1)*100:+.1f}%")
            with col_s4:
                st.metric("元本割れ確率", f"{loss_prob:.1f}%")

            # (チャートとシナリオ表を削除)
            fig_sim, ax_sim = plt.subplots(figsize=(14, 6))

            days_x = np.arange(trading_days_year + 1)
            months_x = days_x / trading_days_year * 12

            # パーセンタイルバンド
            p5_path = np.percentile(sim_paths, 5, axis=0)
            p25_path = np.percentile(sim_paths, 25, axis=0)
            p50_path = np.percentile(sim_paths, 50, axis=0)
            p75_path = np.percentile(sim_paths, 75, axis=0)
            p95_path = np.percentile(sim_paths, 95, axis=0)

            ax_sim.fill_between(months_x, p5_path, p95_path, alpha=0.15, color='#4a90d9', label='5-95%')
            ax_sim.fill_between(months_x, p25_path, p75_path, alpha=0.25, color='#4a90d9', label='25-75%')
            ax_sim.plot(months_x, p50_path, color='#2ca02c', linewidth=2.5, label='中央値')
            ax_sim.axhline(y=initial_capital, color='grey', linestyle='--', linewidth=1, alpha=0.7, label=f'元金 ({initial_capital}万円)')

            ax_sim.set_xlabel("経過月数", fontsize=12)
            ax_sim.set_ylabel("資産額（万円）", fontsize=12)
            ax_sim.set_title("モンテカルロ・シミュレーション（10,000回）", fontsize=14)
            ax_sim.legend(loc='upper left', fontsize=10)
            ax_sim.spines["top"].set_visible(False)
            ax_sim.spines["right"].set_visible(False)
            ax_sim.set_xlim(0, 12)

            # Y軸フォーマット
            ax_sim.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
            fig_sim.tight_layout()
            st.pyplot(fig_sim)
            plt.close(fig_sim)

            # --- チャート: 最終資産の分布 ---
            fig_hist, ax_hist = plt.subplots(figsize=(14, 4))
            ax_hist.hist(sim_results, bins=100, color='#4a90d9', alpha=0.7, edgecolor='white', linewidth=0.3)
            ax_hist.axvline(x=initial_capital, color='red', linestyle='--', linewidth=1.5, label=f'元金 ({initial_capital}万円)')
            ax_hist.axvline(x=p50, color='#2ca02c', linestyle='-', linewidth=2, label=f'中央値 ({p50:.0f}万円)')
            ax_hist.set_xlabel("1年後の資産額（万円）", fontsize=12)
            ax_hist.set_ylabel("頻度", fontsize=12)
            ax_hist.set_title("1年後の資産額分布", fontsize=14)
            ax_hist.legend(fontsize=10)
            ax_hist.spines["top"].set_visible(False)
            ax_hist.spines["right"].set_visible(False)
            fig_hist.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)

            # --- 前提条件 ---
            st.markdown("#### ⚙️ シミュレーション前提")
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                st.metric("実績データ日数", f"{n_days}日")
            with col_p2:
                st.metric("日次平均リターン", f"{avg_daily*100:+.3f}%")
            with col_p3:
                st.metric("日次標準偏差", f"{std_daily*100:.3f}%")
            with col_p4:
                st.metric("実績勝率", f"{win_rate*100:.0f}%")

            st.caption(
                "※ モンテカルロ・シミュレーションは過去のバックテスト実績からリサンプリングして将来を予測しています。"
                "過去の実績が将来を保証するものではありません。手数料・税金・スリッページは考慮していません。"
                "ストップロス(-0.5%)は反映済みです。"
            )
        else:
            st.warning("バックテストデータがありません。先にバックテストを実行してください。")
    except Exception as e:
        st.error(f"シミュレーションに失敗しました: {e}")



# ===================================================================
# Page 2: Backtest Summary
# ===================================================================
elif page == "バックテスト結果":
    st.header("バックテスト結果 (PCA SUB)")

    # --- 📊 セクション1: 全期間ハイブリッド実績 ---
    st.header("1. 全期間ハイブリッド実績")
    
    # セクション固有の開始日選択
    col_bt_date, _ = st.columns([1, 2])
    with col_bt_date:
        bt_custom_start = st.date_input(
            "表示開始日を選択",
            value=bt_start_date,
            key="bt_custom_start",
            help="このセクションの分析対象期間を上書きします。"
        )
    test_start_str_custom = bt_custom_start.strftime("%Y-%m-%d")
    
    # カスタム期間でバックテストを実行
    daily_rets, df_pnl = run_pca_sub_backtest(test_start_str_custom)
    
    st.markdown(f"**分析対象期間:** {test_start_str_custom} 〜")
    st.markdown("決算月（2, 5, 8, 11月）は順張り、それ以外は逆張りで運用した実績です。")
    
    # 全期間の指標
    metrics_all = compute_metrics(daily_rets)
    # 取引が発生した日（リターン非ゼロ）のみで勝率を計算
    active_rets = daily_rets[daily_rets != 0]
    win_rate_val = (active_rets > 0).mean() if not active_rets.empty else 0.0

    col_a1, col_a2, col_a3, col_a4, col_a5 = st.columns(5)
    with col_a1: st.metric("年率リターン (単純)", f"{metrics_all['AR'] * 100:.1f}%")
    with col_a2: st.metric("累積リターン", f"{metrics_all['TOTAL_RETURN'] * 100:.1f}%")
    with col_a3: st.metric("最大ドローダウン", f"{metrics_all['MDD'] * 100:.2f}%")
    with col_a4: st.metric("リスクリターン比", f"{metrics_all['RR']:.2f}")
    with col_a5: st.metric("勝率 (取引日ベース)", f"{win_rate_val*100:.1f}%")

    st.subheader("累積リターン推移 (1/1 〜)")
    cum_all = (1.0 + daily_rets).cumprod()
    fig_all, ax_all = plt.subplots(figsize=(12, 4))
    ax_all.plot(cum_all.index, cum_all.values, color="#2ca02c", linewidth=2, label="Hybrid Strategy")
    ax_all.axhline(y=1.0, color="grey", linestyle="--", alpha=0.5)
    ax_all.set_ylabel("累積リターン")
    ax_all.legend(frameon=False)
    st.pyplot(fig_all)
    plt.close(fig_all)

    # 追加: 1/1〜のドローダウン
    st.subheader("ドローダウン推移 (1/1 〜)")
    running_max_all = cum_all.cummax()
    dd_all = (cum_all - running_max_all) / running_max_all
    fig_dd_all, ax_dd_all = plt.subplots(figsize=(12, 3))
    ax_dd_all.fill_between(dd_all.index, dd_all.values * 100, 0, alpha=0.3, color="#d62728")
    ax_dd_all.plot(dd_all.index, dd_all.values * 100, color="#d62728", linewidth=1)
    ax_dd_all.set_ylabel("DD (%)")
    st.pyplot(fig_dd_all)
    plt.close(fig_dd_all)

    st.markdown("---")

    # --- 📊 セクション2: 構造変化後の逆張り実績 (4/8 〜) ---
    st.header("2. 構造変化後の逆張り実績 (4/8 〜)")
    st.markdown("ユーザー様が特定した「4/8の構造変化」以降、純粋に**逆張り**として運用した場合のパフォーマンスです。")
    
    # 4/8以降を切り出し
    rets_post_408 = daily_rets[daily_rets.index >= pd.Timestamp("2026-04-08")]
    
    if not rets_post_408.empty:
        metrics_post = compute_metrics(rets_post_408)
        # 4/8以降も取引日ベースの勝率
        active_rets_post = rets_post_408[rets_post_408 != 0]
        win_rate_post = (active_rets_post > 0).mean() if not active_rets_post.empty else 0.0

        col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
        with col_p1: st.metric("期間累積リターン", f"{metrics_post['TOTAL_RETURN'] * 100:.1f}%")
        with col_p2: st.metric("平均日次 (取引日)", f"{active_rets_post.mean()*100:+.2f}%")
        with col_p3: st.metric("最大ドローダウン", f"{metrics_post['MDD'] * 100:.2f}%")
        with col_p4: st.metric("リスクリターン比", f"{metrics_post['RR']:.2f}")
        with col_p5: st.metric("勝率 (取引日ベース)", f"{win_rate_post*100:.1f}%")

        st.subheader("累積リターン推移 (4/8 〜)")
        # 4/8を1.0として再スタート
        cum_post = (1.0 + rets_post_408).cumprod()
        fig_post, ax_post = plt.subplots(figsize=(12, 4))
        ax_post.plot(cum_post.index, cum_post.values, color="#1f77b4", linewidth=2, label="Post-4/8 Contrarian")
        ax_post.axhline(y=1.0, color="grey", linestyle="--", alpha=0.5)
        ax_post.set_ylabel("累積リターン")
        ax_post.set_xlim(rets_post_408.index.min(), rets_post_408.index.max()) # X軸を限定
        ax_post.legend(frameon=False)
        st.pyplot(fig_post)
        plt.close(fig_post)

        # 追加: 4/8〜のドローダウン
        st.subheader("ドローダウン推移 (4/8 〜)")
        running_max_post = cum_post.cummax()
        dd_post = (cum_post - running_max_post) / running_max_post
        fig_dd_post, ax_dd_post = plt.subplots(figsize=(12, 3))
        ax_dd_post.fill_between(dd_post.index, dd_post.values * 100, 0, alpha=0.3, color="#d62728")
        ax_dd_post.plot(dd_post.index, dd_post.values * 100, color="#d62728", linewidth=1)
        ax_dd_post.set_ylabel("DD (%)")
        ax_dd_post.set_xlim(rets_post_408.index.min(), rets_post_408.index.max()) # X軸を限定
        st.pyplot(fig_dd_post)
        plt.close(fig_dd_post)
    else:
        st.warning("4/8以降のデータがまだありません。")

    st.markdown("---")

    # --- 📊 セクター貢献度分析の共通関数 ---
    def display_sector_contribution(pnl_df, title_suffix):
        if pnl_df.empty:
            return
        
        st.subheader(f"セクター別 損益貢献度 ({title_suffix})")
        st.markdown(f"期間中の各セクターのロング（買い）およびショート（空売り）によるトータルの損益寄与度（％）です。")
        
        # 集計
        pnl_summary = pnl_df.groupby(['Ticker', 'Side'])['PnL'].sum().unstack(fill_value=0) * 100
        for side in ['Long', 'Short']:
            if side not in pnl_summary.columns: pnl_summary[side] = 0.0
            
        pnl_summary['Total'] = pnl_summary['Long'] + pnl_summary['Short']
        pnl_summary = pnl_summary.sort_values('Total', ascending=True)
        
        # グラフ
        fig, ax = plt.subplots(figsize=(12, 6))
        y_pos = np.arange(len(pnl_summary))
        ax.barh(y_pos, pnl_summary['Long'], color='#2ca02c', label='Long Profit', alpha=0.8)
        ax.barh(y_pos, pnl_summary['Short'], color='#d62728', label='Short Profit', alpha=0.8)
        
        ticker_names = [JP_TICKER_NAMES.get(t, t) for t in pnl_summary.index]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ticker_names)
        ax.set_xlabel("累積損益貢献度 (%)")
        ax.legend(frameon=False)
        ax.axvline(x=0, color='grey', linestyle='-', linewidth=0.8)
        st.pyplot(fig)
        plt.close(fig)
        
        # テーブル
        with st.expander(f"セクター別詳細データ ({title_suffix})"):
            def win_rate(group):
                return (group > 0).mean() if len(group) > 0 else 0
            
            wr = pnl_df.groupby(['Ticker', 'Side'])['PnL'].apply(win_rate).unstack(fill_value=0)
            for side in ['Long', 'Short']:
                if side not in wr.columns: wr[side] = 0.0
                
            detail = []
            for ticker in pnl_summary.sort_values('Total', ascending=False).index:
                detail.append({
                    "業種": JP_TICKER_NAMES.get(ticker, ticker),
                    "合計貢献度": f"{pnl_summary.loc[ticker, 'Total']:+.2f}%",
                    "ショート利益/勝率": f"{pnl_summary.loc[ticker, 'Short']:+.2f}% / {wr.loc[ticker, 'Short']*100:.1f}%",
                    "ロング利益/勝率": f"{pnl_summary.loc[ticker, 'Long']:+.2f}% / {wr.loc[ticker, 'Long']*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)

    # 1. 全期間の貢献度を表示
    if not df_pnl.empty:
        st.markdown("---")
        display_sector_contribution(df_pnl, "1/1〜 全期間")
        
        # 2. 4/8以降の貢献度を表示
        st.markdown("---")
        df_pnl_post = df_pnl[df_pnl['Date'] >= pd.Timestamp("2026-04-08")]
        display_sector_contribution(df_pnl_post, "4/8〜 構造変化後")


# ===================================================================
# Page 3: Recent Performance
# ===================================================================
elif page == "直近パフォーマンス":
    st.header("直近パフォーマンス (2026/01/01 〜)")

    daily_rets_all, df_pnl_all = run_pca_sub_backtest(test_start_str)
    
    # 🟢 2026/01/01 以降にフィルタリング
    recent_start = pd.Timestamp("2026-01-01")
    recent = daily_rets_all[daily_rets_all.index >= recent_start]
    
    # df_pnl_all のフィルタリング修正
    df_pnl_all['Date'] = pd.to_datetime(df_pnl_all['Date'])
    df_pnl = df_pnl_all[df_pnl_all['Date'] >= recent_start]

    if recent.empty:
        st.warning("2026年1月1日以降のデータがありません。")
        st.stop()

    n_days = len(recent)

    # Recent daily P&L table
    st.subheader("2026年 日次損益ログ")

    cum_pnl = (1 + recent).cumprod() - 1
    recent_df = pd.DataFrame({
        "日付": [d.strftime("%Y-%m-%d") for d in recent.index],
        "日次リターン": [f"{r * 100:+.4f}%" for r in recent.values],
        "累積リターン": [f"{c * 100:+.4f}%" for c in cum_pnl.values],
    })

    def highlight_return(val):
        if isinstance(val, str):
            if val.startswith("+"):
                return "color: #2ca02c"
            elif val.startswith("-"):
                return "color: #d62728"
        return ""

    styled = recent_df.style.map(
        highlight_return, subset=["日次リターン", "累積リターン"],
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    # Summary metrics for the period
    col1, col2, col3 = st.columns(3)
    with col1:
        total = (1 + recent).prod() - 1
        st.metric("2026年 累計リターン", f"{total * 100:+.2f}%")
    with col2:
        wins = (recent > 0).sum()
        st.metric("2026年 勝率", f"{wins}/{n_days} ({wins / n_days * 100:.1f}%)")
    with col3:
        avg = recent.mean()
        st.metric("2026年 平均日次リターン", f"{avg * 100:+.4f}%")

    # Cumulative P&L chart for recent period
    st.subheader("累積損益推移")
    cum_recent = (1 + recent).cumprod()

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ca02c" if r >= 0 else "#d62728" for r in recent.values]
    ax.bar(
        range(len(recent)), recent.values * 100,
        color=colors, alpha=0.7, label="日次リターン(%)",
    )
    ax2 = ax.twinx()
    ax2.plot(
        range(len(recent)), cum_recent.values,
        color="#1f77b4", linewidth=2, marker="o", markersize=4, label="累積リターン",
    )
    ax2.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)

    ax.set_xticks(range(len(recent)))
    ax.set_xticklabels(
        [d.strftime("%m/%d") for d in recent.index], rotation=45, fontsize=9,
    )
    ax.set_ylabel("日次リターン (%)")
    ax2.set_ylabel("累積リターン")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- 📊 追加分析: 決算期の季節性・傾向比較 ---
    st.markdown("---")
    st.subheader("🔍 決算期（2, 5月）の傾向比較分析")
    st.markdown(
        "現在の5月の相場が、過去の決算期（2025年5月、2026年2月）とどのように異なるかを分析します。"
        "特に4/8の構造変化以降、決算期であっても「逆張り」が有利になる兆候があるかを検証します。"
    )

    # 各期間のデータを抽出
    daily_rets_all, _ = run_pca_sub_backtest(test_start_str)
    
    comp_periods = {
        "2025年5月 (前回春決算)": ("2025-05-01", "2025-05-31"),
        "2026年2月 (直近冬決算)": ("2026-02-01", "2026-02-28"),
        "2026年5月 (現在 - 4/8以降)": ("2026-05-01", "2026-05-31") # 5月全体
    }

    comp_results = []
    for name, (start, end) in comp_periods.items():
        mask = (daily_rets_all.index >= pd.Timestamp(start)) & (daily_rets_all.index <= pd.Timestamp(end))
        p_rets = daily_rets_all[mask]
        if p_rets.empty: continue
        
        # ハイブリッド実績 (決算月は順張り)
        hybrid_total = (1 + p_rets).prod() - 1
        # もし逆張りだったら (符号反転)
        contra_total = (1 + (-p_rets)).prod() - 1
        
        comp_results.append({
            "期間": name,
            "順張り実績": f"{hybrid_total*100:+.1f}%",
            "逆張り想定": f"{contra_total*100:+.1f}%",
            "ボラティリティ": f"{p_rets.std()*np.sqrt(252)*100:.1f}%",
            "優位性": "🟢 順張り" if hybrid_total > contra_total else "🔴 逆張り"
        })

    st.table(pd.DataFrame(comp_results))

    # ボラティリティの変化分析
    rets_pre_408 = daily_rets_all[(daily_rets_all.index >= pd.Timestamp("2026-01-01")) & (daily_rets_all.index < pd.Timestamp("2026-04-08"))]
    rets_post_408 = daily_rets_all[daily_rets_all.index >= pd.Timestamp("2026-04-08")]
    
    if not rets_pre_408.empty and not rets_post_408.empty:
        vol_pre = rets_pre_408.std() * np.sqrt(252)
        vol_post = rets_post_408.std() * np.sqrt(252)
        
        col_v1, col_v2 = st.columns(2)
        col_v1.metric("4/8以前のボラ (年率)", f"{vol_pre*100:.1f}%")
        col_v2.metric("4/8以後のボラ (年率)", f"{vol_post*100:.1f}%", delta=f"{(vol_post/vol_pre-1)*100:+.1f}%")
        
        if vol_post > vol_pre * 1.5:
            st.warning("⚠️ **分析結果**: 4/8以降、ボラティリティが1.5倍以上に急増しています。これは典型的な「オーバーシュート（行き過ぎ）」が発生しやすい環境であり、**決算期であっても逆張りの戻り圧力が順張りのトレンドを打ち消すリスク**が高まっています。")

# ===================================================================
# Page 4: Crowd Monitoring
# ===================================================================
elif page == "群集行動モニタリング":
    st.header("群集行動モニタリング (Crowd Monitoring)")
    st.markdown(
        "この画面では、**元の投資手法（順張り）の賞味期限** を推測するための3つの指標を監視します。"
        "これらの指標が平常時に戻ったり、ゼロに近づいたりした場合は、"
        "群集が去り、現在の**逆手法の優位性が失われつつあるサイン**となります。"
    )

    df_metrics = run_monitoring_metrics()
    if df_metrics.empty:
        st.warning("モニタリング指標が計算できませんでした。")
        st.stop()

    # 1. 出来高スパイク (Volume Anomaly)
    st.subheader("① 出来高異常値の減衰 (Volume Anomaly)")
    st.markdown("元の手法で「Long」となった銘柄の、過去20日平均に対する出来高倍率です。**1.0付近まで低下してきたら、群集が去ったサイン**です。")
    fig_vol, ax_vol = plt.subplots(figsize=(12, 3))
    ax_vol.plot(df_metrics.index, df_metrics['Volume_Anomaly'], color='purple', label='Volume Ratio (vs SMA20)')
    ax_vol.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax_vol.set_ylabel("倍率")
    ax_vol.legend(frameon=False)
    ax_vol.spines["top"].set_visible(False)
    ax_vol.spines["right"].set_visible(False)
    st.pyplot(fig_vol)
    plt.close(fig_vol)

    # 2. 元の手法のドローダウン (Original Strategy Drawdown)
    st.subheader("② 元の手法のドローダウン底打ち")
    st.markdown("元の手法の累積リターンのドローダウンです。損失の拡大が止まり、**横ばいや回復に転じたら、価格の行き過ぎ（オーバーシュート）が終わったサイン**です。")
    cum_orig = (1 + df_metrics['Original_Return']).cumprod()
    running_max_orig = cum_orig.cummax()
    dd_orig = (cum_orig - running_max_orig) / running_max_orig

    fig_dd, ax_dd = plt.subplots(figsize=(12, 3))
    ax_dd.fill_between(dd_orig.index, dd_orig.values * 100, 0, alpha=0.3, color="#d62728")
    ax_dd.plot(dd_orig.index, dd_orig.values * 100, color="#d62728", linewidth=1.0)
    ax_dd.set_ylabel("ドローダウン (%)")
    ax_dd.spines["top"].set_visible(False)
    ax_dd.spines["right"].set_visible(False)
    st.pyplot(fig_dd)
    plt.close(fig_dd)

    # 3. 日中の押し戻し (Intraday Reversal)
    st.subheader("③ 日中の価格の押し戻し (Intraday Reversal)")
    st.markdown("元の手法の対象銘柄の「(始値 - 終値) / 始値」の割合です。群集が朝方買って引けにかけて売る動きが強ければプラスに振れます。**0付近に落ち着いてきたら、群集の資金が枯渇したサイン**です。")
    
    fig_rev, ax_rev = plt.subplots(figsize=(12, 3))
    ax_rev.bar(df_metrics.index, df_metrics['Intraday_Reversal'], color='orange', alpha=0.7, label='Daily')
    ax_rev.axhline(y=0.0, color='gray', linestyle='--', alpha=0.7)
    ax_rev.set_ylabel("日中リバーサル (%)")
    ax_rev.legend(frameon=False)
    ax_rev.spines["top"].set_visible(False)
    ax_rev.spines["right"].set_visible(False)
    st.pyplot(fig_rev)
    plt.close(fig_rev)

    # 4. ETFと実態バスケットの乖離率 (Basket Divergence)
    st.subheader("④ ETFと実態バスケットの日中乖離率（プレミアム）")
    st.markdown("元の手法の対象ETFと、その中身（上位3〜5銘柄のバスケット）の「日中リターンの差分」です。プラスに大きく振れているほど、ETFだけが異常に買われている（プレミアムが発生している）ことを示します。**これが0%付近に落ち着いてきたら、異常価格が潰れ、戦略の優位性が完全に消滅した最強のサイン**です。")
    
    fig_div, ax_div = plt.subplots(figsize=(12, 3))
    ax_div.bar(df_metrics.index, df_metrics['Basket_Divergence'], color='green', alpha=0.7, label='Daily Divergence (%)')
    ax_div.axhline(y=0.0, color='gray', linestyle='--', alpha=0.7)
    ax_div.set_ylabel("乖離率 (%)")
    ax_div.legend(frameon=False)
    ax_div.spines["top"].set_visible(False)
    ax_div.spines["right"].set_visible(False)
    st.pyplot(fig_div)
    plt.close(fig_div)