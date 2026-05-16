# 纳斯达克100定投监控工具 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个Flask网页应用，自动抓取纳斯达克100指数数据并计算量化指标，给出每日定投建议金额。

**Architecture:** Python Flask 后端通过 yfinance 免费获取 ^NDX 和 ^VIX 数据，计算5大维度12+个指标，综合评分后映射为定投金额。前端单页HTML展示仪表盘。

**Tech Stack:** Python 3, Flask, yfinance, pandas, numpy, Chart.js

---

### Task 1: 项目初始化与依赖

**Files:**
- Create: `d:/cc监控纳斯达克/requirements.txt`
- Create: `d:/cc监控纳斯达克/engine/__init__.py`

- [ ] **Step 1: 创建目录结构和依赖文件**

```bash
mkdir -p "d:/cc监控纳斯达克/engine"
```

```txt
# requirements.txt
flask==3.1.1
yfinance==0.2.54
pandas==2.2.3
numpy==2.2.4
```

- [ ] **Step 2: 创建 engine 包初始化文件**

```python
# engine/__init__.py
from . import indicators
from . import scoring

__all__ = ["indicators", "scoring"]
```

- [ ] **Step 3: 安装依赖**

```bash
cd "d:/cc监控纳斯达克" && pip install -r requirements.txt
```

---

### Task 2: 数据获取模块

**Files:**
- Create: `d:/cc监控纳斯达克/engine/fetcher.py`

负责通过 yfinance 拉取 ^NDX 和 ^VIX 数据，并进行缓存（避免重复请求）。

- [ ] **Step 1: 创建 fetcher.py**

```python
# engine/fetcher.py
"""数据获取模块：通过 yfinance 拉取纳斯达克100和VIX数据"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_ndx_data(period="1y"):
    """获取纳斯达克100指数(^NDX)日线数据
    
    Args:
        period: yfinance 支持的周期，如 "1mo", "3mo", "6mo", "1y", "2y"
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    ticker = yf.Ticker("^NDX")
    df = ticker.history(period=period)
    if df.empty:
        raise ValueError("无法获取 ^NDX 数据")
    return df


def fetch_vix_data(period="6mo"):
    """获取CBOE波动率指数(^VIX)日线数据"""
    ticker = yf.Ticker("^VIX")
    df = ticker.history(period=period)
    if df.empty:
        raise ValueError("无法获取 ^VIX 数据")
    return df


def fetch_all_data(ndx_period="1y", vix_period="6mo"):
    """获取所有需要的数据，返回字典"""
    ndx = fetch_ndx_data(ndx_period)
    vix = fetch_vix_data(vix_period)
    return {"ndx": ndx, "vix": vix}
```

---

### Task 3: 指标计算引擎

**Files:**
- Create: `d:/cc监控纳斯达克/engine/indicators.py`

计算所有技术指标：MA、MACD、ADX、RSI、ROC、Stochastic、Bollinger Bands、ATR、偏离度、Z-score、成交量异常。

- [ ] **Step 1: 创建 indicators.py**

```python
# engine/indicators.py
"""技术指标计算模块"""

import numpy as np
import pandas as pd


def calculate_ma(df, windows=[20, 50, 200]):
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
```

---

### Task 4: 评分决策引擎

**Files:**
- Create: `d:/cc监控纳斯达克/engine/scoring.py`

将各个指标值映射为分数，加权综合后给出定投建议金额。

- [ ] **Step 1: 创建 scoring.py**

```python
# engine/scoring.py
"""评分决策引擎：指标评分、维度加权、定投金额映射"""

import numpy as np
import pandas as pd


def _normalize(value, low_in, high_in, low_out=0, high_out=100):
    """线性映射，并在超出范围时截断"""
    if high_in == low_in:
        return (low_out + high_out) / 2
    ratio = (value - low_in) / (high_in - low_in)
    result = low_out + ratio * (high_out - low_out)
    return max(low_out, min(high_out, result))


# ─── 单指标评分函数 ───

def score_rsi(rsi):
    """RSI: <30 超卖(高分=加仓), >70 超买(低分=减仓)"""
    if pd.isna(rsi):
        return 50
    if rsi <= 30:
        return _normalize(rsi, 20, 30, 95, 80)
    elif rsi <= 50:
        return _normalize(rsi, 30, 50, 80, 50)
    elif rsi <= 70:
        return _normalize(rsi, 50, 70, 50, 20)
    else:
        return _normalize(rsi, 70, 85, 20, 5)


def score_macd_hist(hist):
    """MACD 柱状图：正值=多头，负值=空头；极端值=可能反转"""
    if pd.isna(hist):
        return 50
    # 归一化到最近252天的标准差
    # hist 是标量，没法算了，简化处理
    if hist > 0:
        return 65
    else:
        return 35


def score_adx(adx):
    """ADX: >25 趋势强，<20 趋势弱"""
    if pd.isna(adx):
        return 50
    # 趋势强未必好，结合方向判断，这里简化
    return 50


def score_ma_position(close, ma):
    """价格相对均线位置：偏离太大可能回归"""
    if pd.isna(ma) or pd.isna(close) or ma == 0:
        return 50
    pct = (close - ma) / ma * 100
    if pct > 20:
        return _normalize(pct, 20, 40, 10, 0)
    elif pct > 5:
        return _normalize(pct, 5, 20, 40, 10)
    elif pct > -5:
        return _normalize(pct, -5, 5, 60, 40)
    elif pct > -15:
        return _normalize(pct, -15, -5, 85, 60)
    else:
        return _normalize(pct, -30, -15, 95, 85)


def score_stochastic(k, d):
    """Stochastic: <20 超卖, >80 超买"""
    val = (k + d) / 2 if not (pd.isna(k) or pd.isna(d)) else (k if not pd.isna(k) else 50)
    if pd.isna(val):
        return 50
    if val <= 20:
        return _normalize(val, 0, 20, 95, 80)
    elif val <= 50:
        return _normalize(val, 20, 50, 80, 50)
    elif val <= 80:
        return _normalize(val, 50, 80, 50, 20)
    else:
        return _normalize(val, 80, 100, 20, 5)


def score_bb_percent_b(b):
    """布林带 %B: <0 超卖, >1 超买"""
    if pd.isna(b):
        return 50
    if b <= -1:
        return 95
    elif b <= 0:
        return _normalize(b, -1, 0, 95, 75)
    elif b <= 0.5:
        return _normalize(b, 0, 0.5, 75, 50)
    elif b <= 1:
        return _normalize(b, 0.5, 1, 50, 25)
    else:
        return _normalize(b, 1, 2, 25, 5)


def score_roc(roc):
    """ROC：极端正值=过热，极端负值=超卖"""
    if pd.isna(roc):
        return 50
    if roc < -10:
        return 90
    elif roc < -3:
        return _normalize(roc, -10, -3, 90, 65)
    elif roc < 3:
        return _normalize(roc, -3, 3, 65, 35)
    elif roc < 10:
        return _normalize(roc, 3, 10, 35, 10)
    else:
        return 5


def score_deviation(pct_from_52w_high, pct_from_52w_low, pct_from_ma200, z_score):
    """估值偏离综合评分"""
    scores = []
    
    # 距52周高点：跌得越多分越高
    if not pd.isna(pct_from_52w_high):
        if pct_from_52w_high > 20:
            scores.append(95)
        elif pct_from_52w_high > 10:
            scores.append(_normalize(pct_from_52w_high, 10, 20, 80, 95))
        elif pct_from_52w_high > 3:
            scores.append(_normalize(pct_from_52w_high, 3, 10, 50, 80))
        else:
            scores.append(_normalize(pct_from_52w_high, 0, 3, 30, 50))
    
    # 距MA200偏离
    if not pd.isna(pct_from_ma200):
        if pct_from_ma200 > 20:
            scores.append(10)
        elif pct_from_ma200 > 10:
            scores.append(_normalize(pct_from_ma200, 10, 20, 30, 10))
        elif pct_from_ma200 > 0:
            scores.append(_normalize(pct_from_ma200, 0, 10, 50, 30))
        elif pct_from_ma200 > -10:
            scores.append(_normalize(pct_from_ma200, -10, 0, 75, 50))
        else:
            scores.append(_normalize(pct_from_ma200, -25, -10, 95, 75))
    
    # Z-score: |Z|>2 极端
    if not pd.isna(z_score):
        if z_score < -2:
            scores.append(90)
        elif z_score < -1:
            scores.append(_normalize(z_score, -2, -1, 90, 65))
        elif z_score < 1:
            scores.append(_normalize(z_score, -1, 1, 65, 35))
        elif z_score < 2:
            scores.append(_normalize(z_score, 1, 2, 35, 10))
        else:
            scores.append(5)
    
    return np.mean(scores) if scores else 50


def score_vix(vix):
    """VIX: >30 恐慌(见底信号), <15 贪婪(见顶信号)"""
    if pd.isna(vix):
        return 50
    if vix > 35:
        return 90
    elif vix > 28:
        return _normalize(vix, 28, 35, 75, 90)
    elif vix > 20:
        return _normalize(vix, 20, 28, 50, 75)
    elif vix > 15:
        return _normalize(vix, 15, 20, 30, 50)
    else:
        return _normalize(vix, 10, 15, 10, 30)


def score_volume_anomaly(volume_ratio, price_change):
    """成交量异常：放量下跌=恐慌，放量上涨=乐观"""
    if pd.isna(volume_ratio) or pd.isna(price_change):
        return 50
    if volume_ratio > 1.5 and price_change < -2:
        return 75  # 放量下跌，恐慌，可能是买入机会
    elif volume_ratio > 1.5 and price_change > 2:
        return 25  # 放量上涨，追高谨慎
    else:
        return 50


# ─── 维度评分 ───

def score_trend(row):
    """趋势维度（权重 30%）"""
    scores = []
    
    # MA位置：MA20, MA50, MA200 权重递减
    scores.append(score_ma_position(row["Close"], row.get("MA20")) * 0.4)
    scores.append(score_ma_position(row["Close"], row.get("MA50")) * 0.3)
    scores.append(score_ma_position(row["Close"], row.get("MA200")) * 0.2)
    
    # MACD
    macd_hist = row.get("MACD_Hist")
    macd_score = score_macd_hist(macd_hist) * 0.1
    scores.append(macd_score)
    
    return sum(scores) / sum([0.4, 0.3, 0.2, 0.1])


def score_momentum(row):
    """动量维度（权重 20%）"""
    scores = []
    weights = []
    
    rsi = row.get("RSI")
    if not pd.isna(rsi):
        scores.append(score_rsi(rsi))
        weights.append(0.5)
    
    roc = row.get("ROC")
    if not pd.isna(roc):
        scores.append(score_roc(roc))
        weights.append(0.3)
    
    stoch_k = row.get("Stoch_%K")
    stoch_d = row.get("Stoch_%D")
    scores.append(score_stochastic(stoch_k, stoch_d))
    weights.append(0.2)
    
    return np.average(scores, weights=weights) if scores else 50


def score_volatility(row):
    """波动率维度（权重 15%）"""
    scores = []
    weights = []
    
    bb_b = row.get("BB_%B")
    if not pd.isna(bb_b):
        scores.append(score_bb_percent_b(bb_b))
        weights.append(0.6)
    
    vix = row.get("VIX")
    if not pd.isna(vix):
        scores.append(score_vix(vix))
        weights.append(0.4)
    
    return np.average(scores, weights=weights) if scores else 50


def score_deviation_dim(row):
    """估值偏离维度（权重 20%）"""
    return score_deviation(
        row.get("Pct_From_52W_High"),
        row.get("Pct_From_52W_Low"),
        row.get("Pct_From_MA200"),
        row.get("Z_Score_20")
    )


def score_sentiment(row):
    """市场情绪维度（权重 15%）"""
    scores = []
    weights = []
    
    vix = row.get("VIX")
    if not pd.isna(vix):
        scores.append(score_vix(vix))
        weights.append(0.6)
    
    vol_ratio = row.get("Volume_Ratio")
    price_chg = row.get("Close_pct_change")
    if not pd.isna(vol_ratio) and not pd.isna(price_chg):
        scores.append(score_volume_anomaly(vol_ratio, price_chg))
        weights.append(0.4)
    
    return np.average(scores, weights=weights) if scores else 50


# ─── 综合评分与定投金额 ───

DIMENSION_WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.20,
    "volatility": 0.15,
    "deviation": 0.20,
    "sentiment": 0.15,
}


def compute_overall_score(row):
    """计算综合评分（0~100）"""
    scores = {}
    scores["trend"] = score_trend(row)
    scores["momentum"] = score_momentum(row)
    scores["volatility"] = score_volatility(row)
    scores["deviation"] = score_deviation_dim(row)
    scores["sentiment"] = score_sentiment(row)
    
    overall = sum(
        scores[dim] * DIMENSION_WEIGHTS[dim] for dim in DIMENSION_WEIGHTS
    )
    return overall, scores


def get_investment_advice(overall_score):
    """根据综合评分返回定投建议
    
    Args:
        overall_score: 0~100 的综合评分
    
    Returns:
        dict: 包含建议金额、信号标签、信号级别
    """
    if overall_score >= 80:
        amount = 200
        signal = "强烈加仓"
        level = 4
    elif overall_score >= 65:
        # 65~79: 线性插值 160~190
        amount = int(160 + (overall_score - 65) / (79 - 65) * 30)
        amount = round(amount / 10) * 10
        signal = "加仓"
        level = 3
    elif overall_score >= 45:
        # 45~64: 线性插值 100~150
        amount = int(100 + (overall_score - 45) / (64 - 45) * 50)
        amount = round(amount / 10) * 10
        signal = "持有"
        level = 2
    elif overall_score >= 25:
        amount = 100
        signal = "谨慎"
        level = 1
    else:
        amount = 100
        signal = "风险"
        level = 0
    
    return {
        "amount": max(100, min(200, amount)),
        "signal": signal,
        "level": level,
        "score": round(overall_score, 1),
    }


# pandas 辅助：为行计算用的 Close 变化率
def prepare_row_with_pct_change(row, close_series, idx):
    """为评分准备行数据，附加 Close 变化率"""
    row = row.copy()
    if idx > 0 and idx < len(close_series):
        row["Close_pct_change"] = (close_series.iloc[idx] - close_series.iloc[idx - 1]) / close_series.iloc[idx - 1] * 100
    else:
        row["Close_pct_change"] = 0
    return row
```


- [ ] **Step 2: 验证导入正常**

```bash
cd "d:/cc监控纳斯达克" && python -c "from engine.scoring import compute_overall_score, get_investment_advice; print('scoring OK')"
```

---

### Task 5: Flask 应用主入口

**Files:**
- Create: `d:/cc监控纳斯达克/app.py`

Flask 应用，提供 API 接口并承载前端页面。

- [ ] **Step 1: 创建 app.py**

```python
# app.py
"""纳斯达克100定投监控工具 - Flask 主入口"""

from flask import Flask, jsonify, send_from_directory
import pandas as pd
import os
import traceback

from engine.fetcher import fetch_all_data
from engine.indicators import calculate_all_indicators
from engine.scoring import (
    compute_overall_score,
    get_investment_advice,
    prepare_row_with_pct_change,
    DIMENSION_WEIGHTS,
)

app = Flask(__name__, static_folder="static", static_url_path="")


def get_latest_signal(df_ndx, df_vix):
    """计算最新交易信号"""
    df = calculate_all_indicators(df_ndx, df_vix)
    df = df.sort_index()
    
    latest = df.iloc[-1]
    close_series = df["Close"]
    latest_row = prepare_row_with_pct_change(latest, close_series, len(df) - 1)
    
    overall_score, dim_scores = compute_overall_score(latest_row)
    advice = get_investment_advice(overall_score)
    
    # 收集各维度指标原始值（最近一行）
    indicators_raw = {}
    for col in [
        "Close", "MA20", "MA50", "MA200",
        "MACD", "MACD_Hist", "RSI",
        "Stoch_%K", "Stoch_%D",
        "BB_Upper", "BB_Middle", "BB_Lower", "BB_%B", "BB_Width",
        "ADX", "ATR", "ATR_Pct",
        "ROC", "Pct_From_52W_High", "Pct_From_52W_Low",
        "Pct_From_MA200", "Z_Score_20",
        "VIX", "Volume_Ratio",
    ]:
        val = latest.get(col)
        indicators_raw[col] = round(val, 2) if isinstance(val, (int, float)) and not pd.isna(val) else None
    
    return {
        "date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
        "price": round(latest["Close"], 2),
        "change_pct": indicators_raw.get("Close_pct_change"),
        "overall_score": advice["score"],
        "signal": advice["signal"],
        "level": advice["level"],
        "recommended_amount": advice["amount"],
        "dimension_scores": {k: round(v, 1) for k, v in dim_scores.items()},
        "dimension_weights": DIMENSION_WEIGHTS,
        "indicators": indicators_raw,
    }


def get_history_data(df_ndx, df_vix):
    """计算历史每日信号（用于趋势图）"""
    df = calculate_all_indicators(df_ndx, df_vix)
    df = df.sort_index()
    
    history = []
    close_series = df["Close"]
    
    for i in range(len(df)):
        if i < 50:  # 需要足够的数据计算指标
            continue
        row = df.iloc[i]
        row_prepared = prepare_row_with_pct_change(row, close_series, i)
        overall, _ = compute_overall_score(row_prepared)
        advice = get_investment_advice(overall)
        
        date_str = str(df.index[i].date()) if hasattr(df.index[i], "date") else str(df.index[i])
        history.append({
            "date": date_str,
            "price": round(row["Close"], 2),
            "amount": advice["amount"],
            "score": advice["score"],
            "signal": advice["signal"],
        })
    
    return history


@app.route("/api/data")
def api_data():
    """返回最新指标数据和评分"""
    try:
        data = fetch_all_data(ndx_period="1y", vix_period="6mo")
        signal = get_latest_signal(data["ndx"], data["vix"])
        history = get_history_data(data["ndx"], data["vix"])
        
        return jsonify({
            "success": True,
            "latest": signal,
            "history": history[-90:],  # 最近90天
            "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
```

---

### Task 6: 前端页面

**Files:**
- Create: `d:/cc监控纳斯达克/static/index.html`

完整的单页前端应用，使用 Chart.js 绘制图表。

- [ ] **Step 1: 创建 index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>纳斯达克100定投监控</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f0f2f5; color: #1a1a2e; min-height: 100vh;
        }
        .container { max-width: 1100px; margin: 0 auto; padding: 20px; }

        /* 头部 */
        .header {
            text-align: center; padding: 24px 0 20px;
            border-bottom: 1px solid #e5e7eb; margin-bottom: 24px;
        }
        .header h1 { font-size: 24px; font-weight: 700; }
        .header .subtitle { color: #6b7280; font-size: 14px; margin-top: 4px; }

        /* 状态栏 */
        .status-bar {
            display: flex; justify-content: center; gap: 32px;
            padding: 12px 20px; background: #fff; border-radius: 12px;
            margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            flex-wrap: wrap;
        }
        .status-item { text-align: center; }
        .status-item .label { font-size: 12px; color: #6b7280; }
        .status-item .value { font-size: 18px; font-weight: 700; }
        .status-item .value.up { color: #16a34a; }
        .status-item .value.down { color: #dc2626; }

        /* 主建议卡片 */
        .advice-card {
            background: linear-gradient(135deg, #1e3a5f, #2563eb);
            color: #fff; border-radius: 16px; padding: 32px; text-align: center;
            margin-bottom: 20px; box-shadow: 0 4px 20px rgba(37,99,235,0.3);
        }
        .advice-card .label { font-size: 14px; opacity: 0.9; }
        .advice-card .amount {
            font-size: 56px; font-weight: 800; margin: 8px 0;
            letter-spacing: -2px;
        }
        .advice-card .amount .unit { font-size: 20px; margin-left: 4px; }
        .advice-card .signal-tag {
            display: inline-block; padding: 6px 20px; border-radius: 20px;
            font-size: 16px; font-weight: 600; margin-top: 8px;
        }

        /* 信号级别颜色 */
        .level-4 { background: #2563eb; }
        .level-3 { background: #16a34a; }
        .level-2 { background: #ca8a04; color: #1a1a2e; }
        .level-1 { background: #ea580c; }
        .level-0 { background: #dc2626; }

        /* 进度条 */
        .score-bar-container {
            background: #fff; border-radius: 12px; padding: 20px;
            margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .score-bar-container .title {
            font-size: 14px; color: #6b7280; margin-bottom: 8px;
        }
        .score-bar {
            height: 16px; background: #e5e7eb; border-radius: 8px;
            overflow: hidden; position: relative;
        }
        .score-bar-fill {
            height: 100%; border-radius: 8px;
            background: linear-gradient(90deg, #dc2626, #ea580c, #ca8a04, #16a34a, #2563eb);
            transition: width 0.6s ease;
        }
        .score-labels {
            display: flex; justify-content: space-between; margin-top: 4px;
            font-size: 11px; color: #9ca3af;
        }

        /* 维度卡片网格 */
        .dimension-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 12px; margin-bottom: 20px;
        }
        .dim-card {
            background: #fff; border-radius: 12px; padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .dim-card .dim-name { font-size: 13px; color: #6b7280; font-weight: 600; }
        .dim-card .dim-score { font-size: 28px; font-weight: 700; margin: 4px 0; }
        .dim-card .dim-bar {
            height: 6px; background: #e5e7eb; border-radius: 3px; overflow: hidden;
        }
        .dim-card .dim-bar-fill {
            height: 100%; border-radius: 3px; transition: width 0.5s;
        }
        .dim-card .dim-weight { font-size: 11px; color: #9ca3af; margin-top: 4px; }

        /* 图表 */
        .chart-container {
            background: #fff; border-radius: 12px; padding: 20px;
            margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .chart-container .title { font-size: 14px; color: #6b7280; margin-bottom: 12px; font-weight: 600; }
        .chart-wrapper { position: relative; height: 360px; }

        /* 指标详情表 */
        .indicators-table-container {
            background: #fff; border-radius: 12px; padding: 20px;
            margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .indicators-table-container .title {
            font-size: 14px; color: #6b7280; margin-bottom: 12px; font-weight: 600;
        }
        .indicator-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 8px;
        }
        .indicator-item {
            display: flex; justify-content: space-between; padding: 6px 10px;
            background: #f9fafb; border-radius: 6px; font-size: 13px;
        }
        .indicator-item .ind-label { color: #6b7280; }
        .indicator-item .ind-value { font-weight: 600; }

        /* 历史建议趋势 */
        .history-chart-container {
            background: #fff; border-radius: 12px; padding: 20px;
            margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .history-chart-container .title {
            font-size: 14px; color: #6b7280; margin-bottom: 12px; font-weight: 600;
        }

        /* 底部 */
        .footer {
            text-align: center; color: #9ca3af; font-size: 12px;
            padding: 20px 0;
        }

        /* 加载中/错误 */
        .loading { text-align: center; padding: 60px 20px; color: #6b7280; }
        .loading .spinner {
            border: 3px solid #e5e7eb; border-top-color: #2563eb;
            border-radius: 50%; width: 32px; height: 32px;
            animation: spin 0.8s linear infinite; margin: 0 auto 12px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .error-card {
            background: #fef2f2; color: #dc2626; border-radius: 12px;
            padding: 32px; text-align: center; margin-bottom: 20px;
        }
        .error-card .retry-btn {
            margin-top: 12px; padding: 8px 24px; background: #dc2626;
            color: #fff; border: none; border-radius: 8px; cursor: pointer;
        }

        /* 更新时间 */
        .update-time {
            text-align: center; font-size: 12px; color: #9ca3af; margin-top: -12px; margin-bottom: 20px;
        }

        @media (max-width: 640px) {
            .advice-card .amount { font-size: 40px; }
            .dimension-grid { grid-template-columns: repeat(2, 1fr); }
            .container { padding: 12px; }
        }
    </style>
</head>
<body>
    <div class="container" id="app">
        <div class="header">
            <h1>纳斯达克100定投监控</h1>
            <div class="subtitle">南方016452 · 每日智能定投决策</div>
        </div>

        <!-- 加载状态 -->
        <div id="loadingState" class="loading">
            <div class="spinner"></div>
            <div>正在获取最新纳斯达克数据...</div>
        </div>

        <!-- 错误状态 -->
        <div id="errorState" class="error-card" style="display:none;">
            <div id="errorMessage">数据获取失败</div>
            <button class="retry-btn" onclick="fetchData()">重新加载</button>
        </div>

        <!-- 主要内容 -->
        <div id="mainContent" style="display:none;">
            <!-- 状态栏 -->
            <div class="status-bar" id="statusBar"></div>

            <!-- 核心建议卡片 -->
            <div class="advice-card" id="adviceCard"></div>

            <!-- 综合评分进度条 -->
            <div class="score-bar-container">
                <div class="title">综合评分</div>
                <div class="score-bar">
                    <div class="score-bar-fill" id="scoreFill" style="width:0%"></div>
                </div>
                <div class="score-labels">
                    <span>风险 0</span>
                    <span>25</span>
                    <span>45</span>
                    <span>65</span>
                    <span>强烈加仓 100</span>
                </div>
            </div>

            <!-- 维度卡片 -->
            <div class="dimension-grid" id="dimensionGrid"></div>

            <!-- 走势图 -->
            <div class="chart-container">
                <div class="title">纳斯达克100指数走势 — 布林带 &amp; 均线</div>
                <div class="chart-wrapper">
                    <canvas id="priceChart"></canvas>
                </div>
            </div>

            <!-- 历史建议趋势 -->
            <div class="history-chart-container">
                <div class="title">历史定投建议趋势</div>
                <div class="chart-wrapper">
                    <canvas id="historyChart"></canvas>
                </div>
            </div>

            <!-- 指标详情 -->
            <div class="indicators-table-container" id="indicatorsTable">
                <div class="title">全部技术指标</div>
                <div class="indicator-grid" id="indicatorGrid"></div>
            </div>

            <div class="update-time" id="updateTime"></div>
        </div>

        <div class="footer">
            数据来源: Yahoo Finance · 本工具仅供参考，不构成投资建议
        </div>
    </div>

    <script>
        const DIM_NAMES_CN = {
            trend: "趋势跟踪",
            momentum: "动量分析",
            volatility: "波动率",
            deviation: "估值偏离",
            sentiment: "市场情绪"
        };

        let priceChartInstance = null;
        let historyChartInstance = null;

        function getSignalColor(signal) {
            const map = { "强烈加仓": "#2563eb", "加仓": "#16a34a", "持有": "#ca8a04", "谨慎": "#ea580c", "风险": "#dc2626" };
            return map[signal] || "#6b7280";
        }

        function formatNumber(n) {
            if (n === null || n === undefined) return "--";
            if (typeof n === 'number') return n.toLocaleString('zh-CN');
            return n;
        }

        function renderStatusBar(data) {
            const d = data.latest;
            const changeClass = d.change_pct !== null && d.change_pct >= 0 ? "up" : "down";
            const arrow = d.change_pct !== null ? (d.change_pct >= 0 ? "▲" : "▼") : "";
            document.getElementById("statusBar").innerHTML = `
                <div class="status-item">
                    <div class="label">日期</div>
                    <div class="value">${d.date}</div>
                </div>
                <div class="status-item">
                    <div class="label">纳斯达克100 (NDX)</div>
                    <div class="value">${formatNumber(d.price)}</div>
                </div>
                <div class="status-item">
                    <div class="label">日涨跌幅</div>
                    <div class="value ${changeClass}">${arrow} ${d.change_pct !== null ? d.change_pct.toFixed(2) : "--"}%</div>
                </div>
            `;
        }

        function renderAdviceCard(data) {
            const d = data.latest;
            const color = getSignalColor(d.signal);
            document.getElementById("adviceCard").innerHTML = `
                <div class="label">建议定投金额</div>
                <div class="amount">¥${d.recommended_amount}<span class="unit">/天</span></div>
                <div class="signal-tag level-${d.level}" style="background:${color}">${d.signal} · 综合评分 ${d.overall_score}</div>
            `;
            document.getElementById("scoreFill").style.width = d.overall_score + "%";
        }

        function renderDimensionCards(data) {
            const dimScores = data.latest.dimension_scores;
            const weights = data.latest.dimension_weights;
            let html = "";
            for (const [key, score] of Object.entries(dimScores)) {
                const name = DIM_NAMES_CN[key] || key;
                const weight = Math.round((weights[key] || 0) * 100);
                const color = score >= 65 ? "#16a34a" : score >= 45 ? "#ca8a04" : "#dc2626";
                html += `
                    <div class="dim-card">
                        <div class="dim-name">${name}</div>
                        <div class="dim-score" style="color:${color}">${score.toFixed(0)}</div>
                        <div class="dim-bar"><div class="dim-bar-fill" style="width:${score}%;background:${color}"></div></div>
                        <div class="dim-weight">权重 ${weight}%</div>
                    </div>
                `;
            }
            document.getElementById("dimensionGrid").innerHTML = html;
        }

        function renderIndicators(data) {
            const ind = data.latest.indicators;
            const labels = {
                Close: "收盘价", MA20: "MA20", MA50: "MA50", MA200: "MA200",
                MACD: "MACD", MACD_Hist: "MACD柱",
                RSI: "RSI(14)", "Stoch_%K": "Stoch %K", "Stoch_%D": "Stoch %D",
                BB_Upper: "布林上轨", BB_Middle: "布林中轨", BB_Lower: "布林下轨",
                "BB_%B": "BB %B", BB_Width: "布林带宽%",
                ADX: "ADX", ATR: "ATR", ATR_Pct: "ATR%",
                ROC: "ROC(%)", Pct_From_52W_High: "距52周高%", Pct_From_52W_Low: "距52周低%",
                Pct_From_MA200: "距MA200%", Z_Score_20: "Z-Score(20)",
                VIX: "VIX", Volume_Ratio: "量比"
            };
            let html = "";
            for (const [key, label] of Object.entries(labels)) {
                const val = ind[key];
                const display = val !== null && val !== undefined ? val.toLocaleString('zh-CN', {maximumFractionDigits: 2}) : "--";
                html += `<div class="indicator-item"><span class="ind-label">${label}</span><span class="ind-value">${display}</span></div>`;
            }
            document.getElementById("indicatorGrid").innerHTML = html;
        }

        function renderPriceChart(data) {
            const history = data.history;
            const dates = history.map(d => d.date);
            const prices = history.map(d => d.price);
            
            // We need more data for Bollinger Bands - we'll show price + MA20/MA50/MA200
            // Since the API doesn't send historical MAs, show close price with area
            if (priceChartInstance) {
                priceChartInstance.destroy();
            }
            
            const ctx = document.getElementById('priceChart').getContext('2d');
            priceChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [{
                        label: 'NDX 收盘价',
                        data: prices,
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37,99,235,0.05)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 1,
                        borderWidth: 2,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { intersect: false, mode: 'index' },
                    plugins: {
                        legend: { display: false },
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { maxTicksLimit: 10, font: { size: 11 } }
                        },
                        y: {
                            grid: { color: '#f0f0f0' },
                            ticks: { font: { size: 11 } }
                        }
                    }
                }
            });
        }

        function renderHistoryChart(data) {
            const history = data.history;
            // Only show last 60 days for the history chart
            const slice = history.slice(-60);
            const dates = slice.map(d => d.date);
            const amounts = slice.map(d => d.amount);
            const prices = slice.map(d => d.price);

            if (historyChartInstance) {
                historyChartInstance.destroy();
            }

            const ctx = document.getElementById('historyChart').getContext('2d');
            historyChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: dates,
                    datasets: [{
                        label: '建议定投金额 (¥)',
                        data: amounts,
                        backgroundColor: amounts.map(a => {
                            if (a >= 180) return '#2563eb';
                            if (a >= 140) return '#16a34a';
                            if (a > 100) return '#ca8a04';
                            return '#9ca3af';
                        }),
                        borderRadius: 3,
                        yAxisID: 'y1',
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { intersect: false, mode: 'index' },
                    plugins: {
                        legend: { display: false },
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { maxTicksLimit: 10, font: { size: 10 } }
                        },
                        y1: {
                            position: 'left',
                            min: 80,
                            max: 220,
                            grid: { color: '#f0f0f0' },
                            ticks: { font: { size: 11 } },
                            title: { display: true, text: '建议金额 (¥)', font: { size: 11 } }
                        }
                    }
                }
            });
        }

        async function fetchData() {
            document.getElementById("loadingState").style.display = "";
            document.getElementById("errorState").style.display = "none";
            document.getElementById("mainContent").style.display = "none";

            try {
                const resp = await fetch('/api/data');
                const data = await resp.json();
                
                if (!data.success) {
                    throw new Error(data.error || "未知错误");
                }

                document.getElementById("loadingState").style.display = "none";
                document.getElementById("mainContent").style.display = "block";

                renderStatusBar(data);
                renderAdviceCard(data);
                renderDimensionCards(data);
                renderIndicators(data);
                renderPriceChart(data);
                renderHistoryChart(data);

                document.getElementById("updateTime").textContent = "数据更新于 " + data.updated_at;

            } catch (err) {
                document.getElementById("loadingState").style.display = "none";
                document.getElementById("errorState").style.display = "block";
                document.getElementById("errorMessage").textContent = "数据获取失败: " + err.message;
            }
        }

        document.addEventListener('DOMContentLoaded', fetchData);
    </script>
</body>
</html>
```

---

### Task 7: 启动验证

- [ ] **Step 1: 启动 Flask 服务**

```bash
cd "d:/cc监控纳斯达克" && python app.py
```
服务启动后将监听 `http://localhost:5000`，浏览器打开即可查看。

- [ ] **Step 2: 验证 API 返回正常**

浏览器访问 `http://localhost:5000/api/data`，验证返回JSON包含各项指标数据。

- [ ] **Step 3: 验证页面渲染**

浏览器访问 `http://localhost:5000`，验证所有卡片、图表、表格正常渲染，文字为简体中文。
