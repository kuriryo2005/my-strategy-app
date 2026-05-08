"""
Project Babel — 日米業種リードラグ投資戦略
設定ファイル: 銘柄リスト・パラメータ・分類ラベル
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# --- 日本 TOPIX-17業種ETF ---
JP_TICKERS = [
    "1617.T",  # 食品
    "1618.T",  # エネルギー資源
    "1619.T",  # 建設・資材
    "1620.T",  # 素材・化学
    "1621.T",  # 医薬品
    "1622.T",  # 自動車・輸送機
    "1623.T",  # 鉄鋼・非鉄
    "1624.T",  # 機械
    "1625.T",  # 電機・精密
    "1626.T",  # 情報通信・サービスその他
    "1627.T",  # 電力・ガス
    "1628.T",  # 運輸・物流
    "1630.T",  # 小売
    "1631.T",  # 銀行
    "1632.T",  # 金融（除く銀行）
    "1633.T",  # 不動産
]

JP_TICKER_NAMES = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信・サービスその他",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融（除く銀行）",
    "1633.T": "不動産",
}

# --- 日本 TOPIX-17業種 実態バスケット（代替銘柄群） ---
# 各ETFの値動きの7割〜9割を説明する上位3〜5銘柄とそのおおよその構成比率
BASKET_MAPPING = {
    "1617.T": {"2914.T": 0.35, "2802.T": 0.20, "2502.T": 0.20, "2503.T": 0.15, "2269.T": 0.10}, # 食品
    "1618.T": {"1605.T": 0.70, "5020.T": 0.30}, # エネルギー資源
    "1619.T": {"1925.T": 0.35, "1928.T": 0.35, "1801.T": 0.15, "1802.T": 0.15}, # 建設・資材
    "1620.T": {"4063.T": 0.40, "4901.T": 0.25, "4911.T": 0.20, "4182.T": 0.15}, # 素材・化学
    "1621.T": {"4568.T": 0.40, "4502.T": 0.30, "4519.T": 0.20, "4523.T": 0.10}, # 医薬品
    "1622.T": {"7203.T": 0.50, "7267.T": 0.20, "6902.T": 0.15, "7269.T": 0.10, "7201.T": 0.05}, # 自動車・輸送機
    "1623.T": {"5401.T": 0.50, "5411.T": 0.25, "5713.T": 0.25}, # 鉄鋼・非鉄
    "1624.T": {"6506.T": 0.25, "6301.T": 0.35, "6326.T": 0.20, "6367.T": 0.20}, # 機械
    "1625.T": {"8035.T": 0.25, "6758.T": 0.25, "6861.T": 0.25, "6501.T": 0.25}, # 電機・精密
    "1626.T": {"9984.T": 0.30, "9432.T": 0.30, "9433.T": 0.25, "9434.T": 0.15}, # 情報通信
    "1627.T": {"9501.T": 0.35, "9502.T": 0.25, "9503.T": 0.25, "9531.T": 0.15}, # 電力・ガス
    "1628.T": {"9022.T": 0.30, "9020.T": 0.30, "9101.T": 0.25, "9104.T": 0.15}, # 運輸・物流
    "1629.T": {"8058.T": 0.25, "8031.T": 0.25, "8001.T": 0.20, "8053.T": 0.15, "8002.T": 0.15}, # 商社・卸売
    "1630.T": {"9983.T": 0.45, "3382.T": 0.25, "8267.T": 0.15, "9843.T": 0.15}, # 小売
    "1631.T": {"8306.T": 0.45, "8316.T": 0.30, "8411.T": 0.15, "8308.T": 0.10}, # 銀行
    "1632.T": {"8766.T": 0.35, "8750.T": 0.25, "8593.T": 0.25, "8604.T": 0.15}, # 金融
    "1633.T": {"8801.T": 0.40, "8802.T": 0.35, "8830.T": 0.15, "3289.T": 0.10}, # 不動産
}

# --- 米国 Select Sector SPDR ETF ---
US_TICKERS = [
    "XLB",   # 素材
    "XLC",   # コミュニケーション (2018/6/18〜)
    "XLE",   # エネルギー
    "XLF",   # 金融
    "XLI",   # 資本財
    "XLK",   # テクノロジー
    "XLP",   # 生活必需品
    "XLRE",  # 不動産 (2015/10/7〜)
    "XLU",   # 公益
    "XLV",   # ヘルスケア
    "XLY",   # 一般消費財
]

US_TICKER_NAMES = {
    "XLB": "素材",
    "XLC": "コミュニケーション",
    "XLE": "エネルギー",
    "XLF": "金融",
    "XLI": "資本財",
    "XLK": "テクノロジー",
    "XLP": "生活必需品",
    "XLRE": "不動産",
    "XLU": "公益",
    "XLV": "ヘルスケア",
    "XLY": "一般消費財",
}

# 途中上場銘柄
US_LATE_LISTING = {
    "XLC": "2018-06-18",
    "XLRE": "2015-10-07",
}

# --- シクリカル/ディフェンシブ分類 ---
US_CYCLICAL = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
US_NEUTRAL = [t for t in US_TICKERS if t not in US_CYCLICAL and t not in US_DEFENSIVE]

JP_CYCLICAL = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE = ["1617.T", "1621.T", "1627.T", "1630.T"]
JP_NEUTRAL = [t for t in JP_TICKERS if t not in JP_CYCLICAL and t not in JP_DEFENSIVE]

# --- ハイパーパラメータ ---
ROLLING_WINDOW = 120       # ローリングウィンドウ L=120日
REGULARIZATION_LAMBDA = 0.9  # 正則化パラメータ λ=0.9
NUM_FACTORS = 3            # 主成分数 K=3
QUANTILE_THRESHOLD = 0.3   # ロング/ショート選定閾値 q=0.3

# --- 期間設定 ---
DATA_START = "2010-01-01"
DATA_END = None  # None = 直近まで
CFULL_START = "2010-01-01"
CFULL_END = "2014-12-31"
TEST_START = "2022-01-01"
TEST_END = None  # None = 直近まで

# --- Fama-French Data Library ---
FF_FACTORS_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_3_Factors_Daily_CSV.zip"
FF_MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_Mom_Factor_Daily_CSV.zip"
# 日本市場用
FF_JAPAN_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_3_Factors_Daily_CSV.zip"
FF_JAPAN_MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_Mom_Factor_Daily_CSV.zip"
