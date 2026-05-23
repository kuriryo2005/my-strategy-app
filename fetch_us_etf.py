"""
米国 Select Sector SPDR ETF の OHLCV データ取得
yfinance 経由で取得し data/raw/us_etf_ohlc.csv に保存

XLC (2018/6/18〜) と XLRE (2015/10/7〜) は途中上場のため、
上場前の期間は NaN となる。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import DATA_RAW, DATA_START, DATA_END, US_TICKERS

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_WAIT_SEC = 5


def fetch_us_etf(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Select Sector SPDR ETF の OHLCV を yfinance で取得する（一括ダウンロード）.

    Parameters
    ----------
    tickers : list[str] | None
        取得対象ティッカー。None の場合 config.US_TICKERS を使用。
    start : str | None
        取得開始日 (YYYY-MM-DD)。None の場合 config.DATA_START。
    end : str | None
        取得終了日 (YYYY-MM-DD)。None の場合 config.DATA_END (= 直近)。

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (ticker, field)
        field ∈ {Open, High, Low, Close, Adj Close, Volume}
    """
    tickers = tickers or US_TICKERS
    start = start or DATA_START
    end = end or DATA_END

    df = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info("Downloading US ETFs batch (attempt %d/%d)...", attempt, _MAX_RETRIES)
            df = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
            if df is not None and not df.empty:
                # すべてのティッカーがダウンロードされたかチェック
                existing = df.columns.get_level_values(1).unique() if isinstance(df.columns, pd.MultiIndex) else df.columns.unique()
                missing = [t for t in tickers if t not in existing]
                if len(missing) == 0:
                    logger.info("Successfully downloaded all US ETFs.")
                    break
                else:
                    logger.warning("Missing US tickers: %s. Retrying...", missing)
            else:
                logger.warning("Downloaded empty US DataFrame. Retrying...")
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt, str(e))
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_WAIT_SEC)

    if df is None or df.empty:
        raise RuntimeError("US ETFのデータ取得に失敗しました（空のデータ）")

    # 列名の調整: (Price, Ticker) -> (Ticker, Field)
    col_names = list(df.columns.names)
    ticker_level = None
    field_level = None
    for i, name in enumerate(col_names):
        if name in ['Ticker', 'Symbols']:
            ticker_level = i
        elif name in ['Price', 'Field']:
            field_level = i
            
    if ticker_level is None or field_level is None:
        field_level = 0
        ticker_level = 1

    df_swapped = df.swaplevel(field_level, ticker_level, axis=1)
    df_swapped.columns.names = ["Ticker", "Field"]
    df_swapped = df_swapped.sort_index(axis=1)
    df_swapped.index.name = "Date"

    # 保存
    out_path = _ensure_dir(DATA_RAW) / "us_etf_ohlc.csv"
    df_swapped.to_csv(out_path)
    logger.info("Saved → %s  shape=%s", out_path, df_swapped.shape)

    return df_swapped


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    df = fetch_us_etf()
    print(f"\nShape: {df.shape}")
    print(df.head())
