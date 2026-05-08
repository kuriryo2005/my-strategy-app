import sys
from pathlib import Path
import logging

# Add project root to sys.path
PROJECT_ROOT = Path(r"c:\Users\kurir\Downloads\my_strategy")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)

from src.data.preprocess import preprocess_returns
from src.data.build_calendar import build_calendar
from src.data.fetch_jp_etf import fetch_jp_etf
from src.data.fetch_us_etf import fetch_us_etf

def main():
    print("Refreshing JP ETF data...")
    fetch_jp_etf()
    print("Refreshing US ETF data...")
    fetch_us_etf()
    print("Preprocessing returns...")
    us_returns, jp_returns = preprocess_returns()
    jp_cc = jp_returns.xs("cc", axis=1, level="ReturnType")
    print("Building calendar...")
    build_calendar(us_returns.index, jp_cc.index)
    print("Done!")

if __name__ == "__main__":
    main()
