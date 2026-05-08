import os
import pandas as pd
import yfinance as yf
from pathlib import Path

# 🟢 直接パスを定義します
data_dir = r"C:\Users\kurir\Downloads\my_strategy\data\processed"

# フォルダが存在しない場合に備えて作成
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# これで NameError が消えます
us_file = os.path.join(data_dir, "us_returns.csv")
jp_file = os.path.join(data_dir, "jp_returns.csv")
jp_oc_file = os.path.join(data_dir, "jp_oc.csv")
map_file = os.path.join(data_dir, "us_jp_date_map.csv")

us_df = pd.read_csv(us_file, index_col=0)
jp_df = pd.read_csv(jp_file, index_col=0)
us_tickers = us_df.columns.tolist()
jp_tickers = jp_df.columns.tolist()

map_df_old = pd.read_csv(map_file)
us_col_name = map_df_old.columns[0]
jp_col_name = map_df_old.columns[1]

print(f"🇺🇸 米国株({len(us_tickers)}銘柄)の最新データをダウンロード中...")
us_data = yf.download(us_tickers, start="2010-01-01")

print(f"🇯🇵 日本株({len(jp_tickers)}銘柄)の最新データをダウンロード中...")
jp_data = yf.download(jp_tickers, start="2010-01-01")

def get_price(df, col):
    if isinstance(df.columns, pd.MultiIndex):
        return df[col]
    return df[[col]]

# 欠損は前日の価格で引き継ぐ
us_close = get_price(us_data, "Close").ffill()
jp_close = get_price(jp_data, "Close").ffill()
jp_open = get_price(jp_data, "Open")

# 🌟 修正ポイント1: 始値の欠損補完
# Yahoo Financeの仕様で直近のOpenがNaNになる現象を防ぐため、前日のCloseで代用する
jp_open = jp_open.fillna(jp_close.shift(1))

print("🧮 リターンと日米日付マッピングを計算中...")
us_returns = us_close.pct_change().dropna(how="all")
jp_returns = jp_close.pct_change().dropna(how="all")

jp_oc = (jp_close - jp_open) / jp_open
jp_oc = jp_oc.dropna(how="all")

# タイムゾーンの消去
us_returns.index = pd.to_datetime(us_returns.index).tz_localize(None).normalize()
jp_returns.index = pd.to_datetime(jp_returns.index).tz_localize(None).normalize()
jp_oc.index = pd.to_datetime(jp_oc.index).tz_localize(None).normalize()

us_dates = us_returns.index
jp_dates = jp_oc.index

map_data = []
# 基本のマッピング（US -> 直後のJP）
for us_date in us_dates:
    future_jp = jp_dates[jp_dates > us_date]
    if len(future_jp) > 0:
        next_jp_date = future_jp[0]
        map_data.append({
            us_col_name: us_date.strftime("%Y-%m-%d"), 
            jp_col_name: next_jp_date.strftime("%Y-%m-%d")
        })

# 🌟 修正ポイント2: 孤立したJP日付の救済（安全装置）
# 上の処理でマッピングから漏れてしまったJP日付（例: 2026-05-01）がないかチェックし、
# 見つかった場合は「その直前にある最新のUS日付」と強制的にペアにする
existing_mapped_jp = {row[jp_col_name] for row in map_data}

for jp_date in jp_dates:
    jp_date_str = jp_date.strftime("%Y-%m-%d")
    if jp_date_str not in existing_mapped_jp:
        past_us = us_dates[us_dates < jp_date]
        if len(past_us) > 0:
            latest_us = past_us[-1].strftime("%Y-%m-%d")
            map_data.append({
                us_col_name: latest_us,
                jp_col_name: jp_date_str
            })

# 日付順に綺麗に並べ直す
new_map_df = pd.DataFrame(map_data).sort_values(by=us_col_name)

print("💾 CSVファイルを最新データで上書き保存中...")
us_returns.index = us_returns.index.strftime("%Y-%m-%d")
jp_returns.index = jp_returns.index.strftime("%Y-%m-%d")
jp_oc.index = jp_oc.index.strftime("%Y-%m-%d")

us_returns.to_csv(us_file)
jp_returns.to_csv(jp_file)
jp_oc.to_csv(jp_oc_file)
new_map_df.to_csv(map_file, index=False)

print("✨ データ更新がすべて完了しました！")