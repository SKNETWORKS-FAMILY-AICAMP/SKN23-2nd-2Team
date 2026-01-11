import streamlit as st
import pandas as pd
import plotly.express as px

# 임시값
weekly_data = pd.DataFrame({
    "요일": ["월", "화", "수", "목", "금", "토", "일"],
    "노쇼율": [15, 22, 18, 25, 30, 12, 8]
})

age_data = pd.DataFrame({
    "연령대": ["10대", "20대", "30대", "40대", "50대", "60대+"],
    "노쇼율": [10, 28, 22, 18, 15, 12]
})

companion_data = pd.DataFrame({
    "구분": ["동행자 있음", "동행자 없음"],
    "비율": [15, 35]
})

heatmap_data = pd.DataFrame([
    ["월", "09:00", 12], ["월", "11:00", 18], ["월", "14:00", 15], ["월", "16:00", 20],
    ["화", "09:00", 15], ["화", "11:00", 25], ["화", "14:00", 22], ["화", "16:00", 28],
    ["수", "09:00", 14], ["수", "11:00", 20], ["수", "14:00", 18], ["수", "16:00", 22],
    ["목", "09:00", 18], ["목", "11:00", 26], ["목", "14:00", 24], ["목", "16:00", 30],
    ["금", "09:00", 22], ["금", "11:00", 32], ["금", "14:00", 28], ["금", "16:00", 35],
    ["토", "09:00", 8],  ["토", "11:00", 12], ["토", "14:00", 10], ["토", "16:00", 15],
], columns=["요일", "시간", "노쇼율"])


col_title, col_btn = st.columns([5, 1])

with col_title:
    st.subheader("📌 요일 / 시간대별 노쇼 예측 히트맵")

with col_btn:
    weather_clicked = st.button("🌦️ 날씨별 노쇼 예측")
        
heatmap_pivot = heatmap_data.pivot(
    index="시간", columns="요일", values="노쇼율"
)

fig_heatmap = px.imshow(
    heatmap_pivot,
    text_auto=True,
    color_continuous_scale="RdYlGn_r",
    aspect="auto"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 동행자 유무별 노쇼 비율")
    fig_pie = px.pie(
        companion_data,
        names="구분",
        values="비율",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:  
    st.subheader("📊 연령대별 노쇼 예측")
    fig_bar = px.bar(
        age_data,
        x="연령대",
        y="노쇼율",
        text="노쇼율"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

