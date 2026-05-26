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
    """MACD 柱状图：正值=多头，负值=空头"""
    if pd.isna(hist):
        return 50
    if hist > 0:
        return 65
    else:
        return 35


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
    weights = []

    # MA位置：MA20, MA50, MA200 权重递减
    ma20_score = score_ma_position(row.get("Close"), row.get("MA20"))
    scores.append(ma20_score)
    weights.append(0.4)

    ma50_score = score_ma_position(row.get("Close"), row.get("MA50"))
    scores.append(ma50_score)
    weights.append(0.3)

    ma200_score = score_ma_position(row.get("Close"), row.get("MA200"))
    scores.append(ma200_score)
    weights.append(0.2)

    macd_score = score_macd_hist(row.get("MACD_Hist"))
    scores.append(macd_score)
    weights.append(0.1)

    return np.average(scores, weights=weights) if scores else 50


def score_momentum(row):
    """动量维度（权重 20%）"""
    scores = []
    weights = []

    rsi = row.get("RSI")
    if not pd.isna(rsi):
        scores.append(score_rsi(rsi))
        weights.append(0.5)

    roc_val = row.get("ROC")
    if not pd.isna(roc_val):
        scores.append(score_roc(roc_val))
        weights.append(0.3)

    stoch_k = row.get("Stoch_%K")
    stoch_d = row.get("Stoch_%D")
    if not pd.isna(stoch_k):
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
        row.get("Z_Score_20"),
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
    scores = {
        "trend": score_trend(row),
        "momentum": score_momentum(row),
        "volatility": score_volatility(row),
        "deviation": score_deviation_dim(row),
        "sentiment": score_sentiment(row),
    }

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
        amount = 300
        signal = "强烈加仓"
        level = 4
    elif overall_score >= 65:
        # 65~79: 线性插值 200~290
        amount = int(200 + (overall_score - 65) / (79 - 65) * 90)
        amount = round(amount / 10) * 10
        signal = "加仓"
        level = 3
    elif overall_score >= 45:
        # 45~64: 线性插值 100~190
        amount = int(100 + (overall_score - 45) / (64 - 45) * 90)
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
        "amount": max(100, min(300, amount)),
        "signal": signal,
        "level": level,
        "score": round(overall_score, 1),
    }


def prepare_row_with_pct_change(row, close_series, idx):
    """为评分准备行数据，附加 Close 变化率"""
    row = row.copy()
    if idx > 0 and idx < len(close_series):
        row["Close_pct_change"] = (
            (close_series.iloc[idx] - close_series.iloc[idx - 1])
            / close_series.iloc[idx - 1] * 100
        )
    else:
        row["Close_pct_change"] = 0
    return row
