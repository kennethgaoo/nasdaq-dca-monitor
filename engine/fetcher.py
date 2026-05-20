"""数据获取模块：通过腾讯财经API获取纳斯达克100指数数据"""

import os
import pickle
from datetime import datetime, date

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
TENCENT_HEADERS = {"Referer": "https://gu.qq.com"}


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name):
    _ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{name}.pkl")


def _is_cached_today(name):
    path = _cache_path(name)
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return mtime.date() == date.today()


def _load_from_cache(name):
    with open(_cache_path(name), "rb") as f:
        return pickle.load(f)


def _save_to_cache(name, df):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(df, f)


def fetch_ndx_data():
    """从腾讯财经获取纳斯达克100指数(NDX)日线数据

    API: ifzq.gtimg.cn 美股历史K线

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume (兼容 yfinance 列名)
    """
    cache_name = "ndx"

    # 当天已缓存则直接读取
    if _is_cached_today(cache_name):
        return _load_from_cache(cache_name)

    url = "http://ifzq.gtimg.cn/appstock/app/usfqkline/get?param=usNDX,day,,,320,qfq"
    resp = requests.get(url, headers=TENCENT_HEADERS, timeout=15)
    data = resp.json()

    if data.get("code") != 0:
        # 如果API失败但有缓存，用缓存
        cached = _load_from_cache(cache_name) if os.path.exists(_cache_path(cache_name)) else None
        if cached is not None and not cached.empty:
            return cached
        raise RuntimeError(f"腾讯API返回错误: {data.get('msg', '未知错误')}")

    days = data["data"]["usNDX"]["day"]
    records = []
    for row in days:
        records.append({
            "Date": row[0],
            "Open": float(row[1]),
            "Close": float(row[2]),
            "High": float(row[3]),
            "Low": float(row[4]),
            "Volume": float(row[5]),
        })

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    _save_to_cache(cache_name, df)
    return df


def fetch_vix_data():
    """从 CBOE 官网获取 VIX 波动率指数日线数据（国内可直连）"""
    cache_name = "vix"

    if _is_cached_today(cache_name):
        return _load_from_cache(cache_name)

    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")
    # 格式: Date,Open,High,Low,Close  (MM/DD/YYYY)
    records = []
    for line in lines[1:]:  # 跳过表头
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue
        records.append({
            "Date": parts[0],
            "Open": float(parts[1]),
            "High": float(parts[2]),
            "Low": float(parts[3]),
            "Close": float(parts[4]),
        })

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()

    _save_to_cache(cache_name, df)
    return df


def fetch_all_data():
    """获取所有需要的数据，返回字典"""
    ndx = fetch_ndx_data()
    vix = fetch_vix_data()
    return {"ndx": ndx, "vix": vix}
