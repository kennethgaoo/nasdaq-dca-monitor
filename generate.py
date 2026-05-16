"""每日静态页面生成器：抓取数据 → 计算指标 → 输出完整HTML"""

import os
import sys
import json
import shutil

# 确保能找到 engine 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from engine.fetcher import fetch_all_data
from engine.indicators import calculate_all_indicators
from engine.scoring import (
    compute_overall_score,
    get_investment_advice,
    prepare_row_with_pct_change,
    DIMENSION_WEIGHTS,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")


def get_latest_signal(df_ndx, df_vix):
    """计算最新交易信号（同 app.py 逻辑）"""
    df = calculate_all_indicators(df_ndx, df_vix)
    df = df.sort_index()
    latest = df.iloc[-1]
    close_series = df["Close"]
    latest_row = prepare_row_with_pct_change(latest, close_series, len(df) - 1)
    overall_score, dim_scores = compute_overall_score(latest_row)
    advice = get_investment_advice(overall_score)

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
            round(val, 2) if isinstance(val, (int, float)) and not pd.isna(val) else None
        )

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
    """计算历史每日信号（同 app.py 逻辑）"""
    df = calculate_all_indicators(df_ndx, df_vix)
    df = df.sort_index()
    history = []
    close_series = df["Close"]

    for i in range(len(df)):
        if i < 50:
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
        for col in ["MA20", "MA50", "MA200", "BB_Upper", "BB_Lower"]:
            val = row.get(col)
            entry[col] = round(val, 2) if isinstance(val, (int, float)) and not pd.isna(val) else None
        history.append(entry)

    return history


def generate_json_data():
    """生成完整的 JSON 数据"""
    print("Fetching data...")
    data = fetch_all_data()
    print("Computing indicators...")
    signal = get_latest_signal(data["ndx"], data["vix"])
    history = get_history_data(data["ndx"], data["vix"])

    return {
        "success": True,
        "latest": signal,
        "history": history[-90:],
        "week_scores": history[-8:],
        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_html(json_data):
    """读取模板，嵌入数据，输出完整 HTML"""
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 在 </head> 前注入数据 JSON
    data_script = (
        f'<script id="__DATA__" type="application/json">'
        f'{json.dumps(json_data, ensure_ascii=False)}'
        f'</script>\n'
    )
    html = html.replace("</head>", data_script + "</head>")
    return html


def main():
    print("=" * 40)
    print("[NDX Monitor] Page Generator")
    print("=" * 40)

    # 生成数据
    json_data = generate_json_data()

    # 打印摘要
    l = json_data["latest"]
    print(f"\nDate: {l['date']}")
    print(f"NDX: {l['price']}  ({l['change_pct']:+.2f}%)")
    print(f"Score: {l['overall_score']}")
    print(f"Signal: {l['signal']}")
    print(f"Amount: {l['recommended_amount']} CNY/day")
    print(f"Dimensions: {json.dumps(l['dimension_scores'])}")

    # 生成 HTML
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    html = build_html(json_data)

    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    file_size = os.path.getsize(output_path) / 1024
    print(f"\nOK - 页面已生成: {output_path} ({file_size:.0f} KB)")


if __name__ == "__main__":
    main()
