# weather_analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px

def render_weather_analysis():
    # 인사이트 요약
    with st.expander("주요 인사이트 요약", expanded=True, icon=":material/bar_chart:"):
        st.markdown("""
            - **비 오는 날** 노쇼율이 **32%**로 가장 높음  
            - **강수량 20mm 이상** 시 노쇼율 **42%까지 급증**  
            - **10–20°C** 구간에서 노쇼율 최저 (**18%**)  
            - **극한 기온(≤0°C, ≥30°C)** 에서 노쇼 위험 증가  
        """)

    col1, col2, col3 = st.columns(3, border=True)
    # 날씨 유형별 데이터
    with col1:
        weather_df = pd.DataFrame({
            "날씨": ["맑음", "흐림", "비", "눈"],
            "노쇼율": [18, 22, 32, 28],
            "건수": [450, 380, 280, 120]
        })

        st.subheader("날씨 유형별 노쇼 예측 비율")
        fig_weather = px.pie(
            weather_df,
            names="날씨",
            values="노쇼율",
            hover_data=["건수"],
            hole=0.45,
            color="날씨",
            color_discrete_map={
                "맑음": "#f59e0b",
                "흐림": "#6b7280",
                "비": "#3b82f6",
                "눈": "#8b5cf6",
            }
        )
        st.plotly_chart(fig_weather, use_container_width=True)

    # 기온별 분석
    with col2:
        temp_df = pd.DataFrame({
            "기온 구간": ["0°C 이하", "0–10°C", "10–20°C", "20–30°C", "30°C 이상"],
            "노쇼율": [25, 22, 18, 20, 28]
        })

        st.subheader("기온별 노쇼 예측 비율")
        fig_temp = px.bar(
            temp_df,
            x="기온 구간",
            y="노쇼율",
            text="노쇼율",
            color_discrete_sequence=["#ef4444"]
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    # 강수량별 분석
    with col3:
        rain_df = pd.DataFrame({
            "강수량": ["0mm", "1–5mm", "5–10mm", "10–20mm", "20mm 이상"],
            "노쇼율": [15, 22, 28, 35, 42]
        })

        st.subheader("강수량별 노쇼 예측 비율")
        fig_rain = px.bar(
            rain_df,
            x="강수량",
            y="노쇼율",
            text="노쇼율",
            color_discrete_sequence=["#3b82f6"]
        )
        st.plotly_chart(fig_rain, use_container_width=True)