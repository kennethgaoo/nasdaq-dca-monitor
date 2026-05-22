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


def _fetch_vix_yahoo():
    """从 Yahoo Finance v8 API 获取 VIX（更新快，在 GitHub Actions 上可用）"""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?range=1mo&interval=1d"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0]
    # Yahoo 可能返回 adjclose
    adjclose = result["indicators"].get("adjclose")

    records = []
    for i in range(len(timestamps)):
        # 跳过 None 值
        if quotes["close"][i] is None:
            continue
        dt = __import__("datetime").datetime.fromtimestamp(timestamps[i])
        records.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "Open": quotes["open"][i],
            "High": quotes["high"][i],
            "Low": quotes["low"][i],
            "Close": quotes["close"][i],
            "Volume": quotes["volume"][i] or 0,
        })

    df = __import__("pandas").DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def _fetch_vix_cboe():
    """从 CBOE 官网获取 VIX CSV 数据（国内可直连兜底）"""
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")
    records = []
    for line in lines[1:]:
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
    return df


def fetch_vix_data():
    """获取 VIX 波动率指数：优先 Yahoo Finance（快），兜底 CBOE CSV（慢但国内可用）"""
    cache_name = "vix"

    if _is_cached_today(cache_name):
        return _load_from_cache(cache_name)

    df = None
    errors = []

    # 方案一：Yahoo Finance（更新最快）
    try:
        df = _fetch_vix_yahoo()
        print(f"[VIX] Yahoo Finance OK: {len(df)} rows, latest={df.index[-1].date()}")
    except Exception as e:
        errors.append(f"Yahoo: {e}")

    # 方案二：CBOE CSV（兜底）
    if df is None or df.empty:
        try:
            df = _fetch_vix_cboe()
            print(f"[VIX] CBOE CSV OK: {len(df)} rows, latest={df.index[-1].date()}")
        except Exception as e:
            errors.append(f"CBOE: {e}")

    if df is not None and not df.empty:
        _save_to_cache(cache_name, df)
        return df

    print(f"[VIX] 所有数据源失败: {'; '.join(errors)}")
    return pd.DataFrame()


def fetch_all_data():
    """获取所有需要的数据，返回字典"""
    ndx = fetch_ndx_data()
    vix = fetch_vix_data()
    return {"ndx": ndx, "vix": vix}
