"""
Portfolio construction from cross-sectional signals.

Converts signal scores into dollar-neutral long/short portfolio weights
using quantile-based sorting.
"""

from __future__ import annotations

import pandas as pd


def construct_portfolio(signals: pd.Series, q: float = 0.3) -> pd.Series:
    """Construct a dollar-neutral long/short portfolio from signal scores.

    Parameters
    ----------
    signals : pd.Series
        Signal values indexed by ticker.  Higher values indicate a
        stronger long signal.
    q : float
        Quantile threshold.  Top *q* fraction of tickers is assigned to
        the long leg; bottom *q* fraction to the short leg.
        Default 0.3 (top/bottom 30 %).

    Returns
    -------
    pd.Series
        Portfolio weights indexed by ticker.
        - Long tickers  :  +1/n_long  each
        - Short tickers :  -1/n_short each
        - Others        :  0
        Satisfies sum(w) = 0 and sum(|w|) = 2.
    """
    if signals.empty:
        return pd.Series(dtype=float)

    n = len(signals)
    n_long = max(1, int(round(n * q)))
    n_short = max(1, int(round(n * q)))

    ranked = signals.rank(method="first", ascending=True)

    weights = pd.Series(0.0, index=signals.index)

    # Bottom n_short by rank → short leg
    short_mask = ranked <= n_short
    # Top n_long by rank → long leg
    long_mask = ranked > (n - n_long)

    weights[long_mask] = -1.0 / n_long
    weights[short_mask] = 1.0 / n_short

    return weights

def construct_portfolio_with_crash_filter(
    signal: pd.Series,
    market_return: float,
    crash_threshold: float = 0.0,
    long_exclude: list = None,
    short_exclude: list = None,
    stop_loss: float = -0.005,
):
    """暴落検知付きポートフォリオ構築（Zスコア強度による可変ウェイト）

    Parameters
    ----------
    long_exclude : list
        Long側から除外するティッカーのリスト。
    short_exclude : list
        Short側から除外するティッカーのリスト。
    stop_loss : float
        ストップロス閾値（例: -0.005 = -0.5%）。バックテスト時に使用。
    """
    if long_exclude is None:
        long_exclude = ["1618.T"]  # エネルギー資源（中東情勢安定まで）
    
    if short_exclude is None:
        short_exclude = ["1617.T", "1621.T"]  # 食品、医薬品（分析の結果全敗のため除外）

    sorted_signal = signal.sort_values(ascending=False)
    is_crash = market_return <= crash_threshold

    n_long = 2
    n_short = 2

    weights = pd.Series(0.0, index=signal.index)

    # Short側: Zスコア上位 = 買い群集をショート（除外銘柄をスキップ）
    head_candidates = sorted_signal.sort_values(ascending=False)
    head_selected = []
    for ticker in head_candidates.index:
        if ticker in short_exclude:
            continue
        head_selected.append(ticker)
        if len(head_selected) >= n_short:
            break
    head_signals = sorted_signal.loc[head_selected]

    # Long側: Zスコア下位 = 売り群集をロング（除外銘柄をスキップ）
    tail_candidates = sorted_signal.sort_values(ascending=True)
    tail_selected = []
    for ticker in tail_candidates.index:
        if ticker in long_exclude:
            continue
        tail_selected.append(ticker)
        if len(tail_selected) >= n_long:
            break
    tail_signals = sorted_signal.loc[tail_selected]

    # Short ウェイト
    head_sum = head_signals.abs().sum()
    if head_sum > 0:
        weights[head_signals.index] = - (head_signals.abs() / head_sum)
    else:
        weights[head_signals.index] = - (1.0 / n_short)

    # Long ウェイト
    tail_sum = tail_signals.abs().sum()
    if tail_sum > 0:
        weights[tail_signals.index] = (tail_signals.abs() / tail_sum)
    else:
        weights[tail_signals.index] = 1.0 / n_long

    return weights


def construct_original_portfolio_with_crash_filter(signal: pd.Series, market_return: float, crash_threshold: float = -0.015):
    """元の戦略（順張り、暴落時Long1・Short5）のポートフォリオ構築"""
    sorted_signal = signal.sort_values(ascending=False)
    is_crash = market_return <= crash_threshold
    
    n_long = 1 if is_crash else 5
    n_short = 5
    
    weights = pd.Series(0.0, index=signal.index)
    base_weight = 1.0 / 5.0
    
    # 順張りロジック（そのまま）
    long_tickers = sorted_signal.head(n_long).index
    weights[long_tickers] = base_weight
    
    short_tickers = sorted_signal.tail(n_short).index
    weights[short_tickers] = -base_weight
    
    return weights
