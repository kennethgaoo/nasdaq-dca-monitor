"""技术指标计算模块"""

import numpy as np
import pandas as pd


def calculate_ma(df, windows=(20, 50, 200)):
    """计算移动平均线"""
    for w in windows:
        df[f"MA{w}"] = df["Close"].rolling(window=w).mean()
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    """计算 MACD"""
    ema_fast = df["Close"].ewm(span=fast).mean()
    ema_slow = df["Close"].ewm(span=slow).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def calculate_rsi(df, period=14):
    """计算 RSI（相对强弱指标）"""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def calculate_stochastic(df, k_period=14, d_period=3):
    """计算 Stochastic 随机指标"""
    low_k = df["Low"].rolling(window=k_period).min()
    high_k = df["High"].rolling(window=k_period).max()
    df["Stoch_%K"] = 100 * (df["Close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    df["Stoch_%D"] = df["Stoch_%K"].rolling(window=d_period).mean()
    return df


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """计算布林带"""
    df["BB_Middle"] = df["Close"].rolling(window=period).mean()
    bb_std = df["Close"].rolling(window=period).std()
    df["BB_Upper"] = df["BB_Middle"] + std_dev * bb_std
    df["BB_Lower"] = df["BB_Middle"] - std_dev * bb_std
    df["BB_%B"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"] * 100
    return df


def calculate_adx(df, period=14):
    """计算 ADX（平均趋向指数）"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * ((-minus_dm).rolling(window=period).mean() / atr.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["ADX"] = dx.rolling(window=period).mean()
    return df


def calculate_atr(df, period=14):
    """计算 ATR（平均真实波幅）"""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=period).mean()
    df["ATR_Pct"] = df["ATR"] / df["Close"] * 100
    return df


def calculate_roc(df, period=10):
    """计算 ROC（变化率指标）"""
    df["ROC"] = df["Close"].pct_change(periods=period) * 100
    return df


def calculate_deviation(df):
    """计算价格偏离度指标"""
    close = df["Close"]
    high_52w = close.rolling(window=252).max()
    low_52w = close.rolling(window=252).min()
    df["Pct_From_52W_High"] = (high_52w - close) / high_52w * 100
    df["Pct_From_52W_Low"] = (close - low_52w) / low_52w * 100
    df["Pct_From_MA200"] = (close - df["MA200"]) / df["MA200"] * 100
    df["Z_Score_20"] = (close - close.rolling(window=20).mean()) / close.rolling(window=20).std().replace(0, np.nan)
    return df


def calculate_volume_anomaly(df, period=20):
    """计算成交量异常指标"""
    df["Volume_MA20"] = df["Volume"].rolling(window=period).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)
    return df


def calculate_all_indicators(df_ndx, df_vix=None):
    """计算全部技术指标"""
    df = df_ndx.copy()

    df = calculate_ma(df)
    df = calculate_macd(df)
    df = calculate_rsi(df)
    df = calculate_stochastic(df)
    df = calculate_bollinger_bands(df)
    df = calculate_adx(df)
    df = calculate_atr(df)
    df = calculate_roc(df)
    df = calculate_deviation(df)
    df = calculate_volume_anomaly(df)

    # 整合 VIX 数据
    if df_vix is not None and not df_vix.empty:
        vix_close = df_vix["Close"].reindex(df.index, method="ffill")
        df["VIX"] = vix_close
    else:
        df["VIX"] = np.nan

    return df
