"""纳斯达克100定投监控工具 - Flask 主入口"""

from flask import Flask, jsonify, send_from_directory
import pandas as pd
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
    cols = [
        "Close", "MA20", "MA50", "MA200",
        "MACD", "MACD_Hist", "RSI",
        "Stoch_%K", "Stoch_%D",
        "BB_Upper", "BB_Middle", "BB_Lower", "BB_%B", "BB_Width",
        "ADX", "ATR", "ATR_Pct",
        "ROC", "Pct_From_52W_High", "Pct_From_52W_Low",
        "Pct_From_MA200", "Z_Score_20",
        "VIX", "Volume_Ratio",
    ]
    for col in cols:
        val = latest.get(col)
        indicators_raw[col] = (
            round(val, 2)
            if isinstance(val, (int, float)) and not pd.isna(val)
            else None
        )

    # 提取涨跌幅（不在 indicators_raw 中）
    change_pct = latest_row.get("Close_pct_change")
    if change_pct is not None and not pd.isna(change_pct):
        change_pct = round(change_pct, 2)

    return {
        "date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
        "price": round(latest["Close"], 2),
        "change_pct": change_pct,
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
        entry = {
            "date": date_str,
            "price": round(row["Close"], 2),
            "amount": advice["amount"],
            "score": advice["score"],
            "signal": advice["signal"],
        }
        # 加入均线和布林带数据（用于前端绘图）
        for col in ["MA20", "MA50", "MA200", "BB_Upper", "BB_Lower"]:
            val = row.get(col)
            entry[col] = round(val, 2) if isinstance(val, (int, float)) and not pd.isna(val) else None
        history.append(entry)

    return history


@app.route("/api/data")
def api_data():
    """返回最新指标数据和评分"""
    try:
        data = fetch_all_data()
        signal = get_latest_signal(data["ndx"], data["vix"])
        history = get_history_data(data["ndx"], data["vix"])

        return jsonify({
            "success": True,
            "latest": signal,
            "history": history[-90:],  # 最近90天（用于图表）
            "week_scores": history[-8:],  # 最近8个交易日（用于评分回顾）
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
