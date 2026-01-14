# weather_analysis.py
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from src.services.customerService import load_artifacts, get_customer_list

COL_WEATHER = "weather_type"   # 예: "비/눈/흐림/맑음" 들어있는 컬럼
COL_TEMP    = "average_temp_day"           # 기온 (°C)
COL_RAIN    = "average_rain_day"        # 강수량 (mm)
COL_PROB    = "no_show"   # 0~100

def infer_weather_type(df, rain_col=COL_RAIN, eps=0.1):
    out = df.copy()
    r = pd.to_numeric(out[rain_col], errors="coerce")
    out[COL_WEATHER] = np.where(r > eps, "비", "맑음")
    out.loc[r.isna(), COL_WEATHER] = "미상"
    return out

def add_bins(df, col_temp=COL_TEMP, col_rain=COL_RAIN):
    df = df.copy()

    temp_bins   = [-np.inf, 0, 10, 20, 30, np.inf]
    temp_labels = ["0°C 이하", "0–10°C", "10–20°C", "20–30°C", "30°C 이상"]
    df["temp_bin"] = pd.cut(df[col_temp], bins=temp_bins, labels=temp_labels, right=False)

    rain_bins   = [-0.01, 0.01, 5, 10, 20, np.inf]
    rain_labels = ["0mm", "1–5mm", "5–10mm", "10–20mm", "20mm 이상"]
    df["rain_bin"] = pd.cut(df[col_rain], bins=rain_bins, labels=rain_labels, right=False)

    return df

def mean_rate_by(df, group_col, prob_col=COL_PROB):
    out = (df.groupby(group_col, dropna=True)[prob_col]
             .mean()
             .reset_index()
             .rename(columns={prob_col: "rate"}))

    if out["rate"].max() <= 1.0:
        out["rate"] = out["rate"] * 100

    # ✅ NaN/inf 제거
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["rate"])

    out["rate"] = out["rate"].round(1)  
    return out

def build_insights(weather_tbl, temp_tbl, rain_tbl):
    w_top = weather_tbl.loc[weather_tbl["rate"].idxmax()]
    s1 = f"{w_top[COL_WEATHER]} 날 노쇼율이 {w_top['rate']}%로 가장 높음"

    r20 = rain_tbl.loc[rain_tbl["rain_bin"] == "20mm 이상", "rate"]
    s2 = f"강수량 20mm 이상 시 노쇼율 {int(r20.iloc[0])}%까지 급증" if len(r20) else "강수량 20mm 이상 구간 데이터 없음"

    t_min = temp_tbl.loc[temp_tbl["rate"].idxmin()]
    s3 = f"{t_min['temp_bin']} 구간에서 노쇼율 최저 ({t_min['rate']}%)"

    extreme = temp_tbl[temp_tbl["temp_bin"].isin(["0°C 이하", "30°C 이상"])]
    mid = temp_tbl[temp_tbl["temp_bin"] == "10–20°C"]
    s4 = "극한 기온(≤0°C, ≥30°C)에서 노쇼 위험 증가" if (len(extreme) and len(mid)) else "기온 구간 비교용 데이터가 일부 부족함"

    return [s1, s2, s3, s4]

def render_weather_dashboard():
    model, scaler, _ = load_artifacts()
    df = get_customer_list(model, scaler, limit = None)

    # 날씨 타입 추론 + 구간화
    df = infer_weather_type(df)
    df = add_bins(df)

    # 집계
    weather_tbl = mean_rate_by(df, COL_WEATHER)
    temp_tbl    = mean_rate_by(df, "temp_bin")
    rain_tbl    = mean_rate_by(df, "rain_bin")

    # 인사이트
    insights = build_insights(weather_tbl, temp_tbl, rain_tbl)

    st.markdown("### 📊 주요 인사이트 요약")
    st.markdown("\n".join([f"- **{x}**" for x in insights]))
    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 날씨 유형별 노쇼 예측 비율")
        fig1 = px.pie(weather_tbl, names=COL_WEATHER, values="rate", hole=0.55)
        fig1.update_traces(textinfo="percent")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("#### 기온별 노쇼 예측 비율")
        temp_order = ["0°C 이하", "0–10°C", "10–20°C", "20–30°C", "30°C 이상"]
        temp_tbl2 = temp_tbl.copy()
        temp_tbl2["temp_bin"] = pd.Categorical(temp_tbl2["temp_bin"], categories=temp_order, ordered=True)
        temp_tbl2 = temp_tbl2.sort_values("temp_bin")

        fig2 = px.bar(temp_tbl2, x="temp_bin", y="rate", text="rate")
        fig2.update_traces(texttemplate="%{text}", textposition="inside")
        fig2.update_layout(xaxis_title="기온 구간", yaxis_title="예측 노쇼율")
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        st.markdown("#### 강수량별 노쇼 예측 비율")
        rain_order = ["0mm", "1–5mm", "5–10mm", "10–20mm", "20mm 이상"]
        rain_tbl2 = rain_tbl.copy()
        rain_tbl2["rain_bin"] = pd.Categorical(rain_tbl2["rain_bin"], categories=rain_order, ordered=True)
        rain_tbl2 = rain_tbl2.sort_values("rain_bin")

        fig3 = px.bar(rain_tbl2, x="rain_bin", y="rate", text="rate")
        fig3.update_traces(texttemplate="%{text}", textposition="inside")
        fig3.update_layout(xaxis_title="강수량", yaxis_title="예측 노쇼율")
        st.plotly_chart(fig3, use_container_width=True)

