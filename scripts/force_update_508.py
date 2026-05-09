import sys
from pathlib import Path

# プロジェクトルートをパスに追加
root = Path(r"c:\Users\kurir\Downloads\my_strategy")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data.fetch_jp_etf import fetch_jp_etf
from src.data.fetch_us_etf import fetch_us_etf
from src.data.preprocess import preprocess_returns
from src.data.build_calendar import build_calendar

print("Downloading Japanese ETF data...")
fetch_jp_etf()
print("Downloading US ETF data...")
fetch_us_etf()
print("Preprocessing returns...")
us_returns, jp_returns = preprocess_returns()
print("Building calendar map...")
jp_cc = jp_returns.xs("cc", axis=1, level="ReturnType")
build_calendar(us_returns.index, jp_cc.index)
print("SUCCESS: Data updated to May 8th!")
