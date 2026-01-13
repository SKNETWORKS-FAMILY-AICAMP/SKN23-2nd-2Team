import pandas as pd
import streamlit as st
import plotly.express as px

# 임시값
age_data = pd.DataFrame({
    "연령대": ["10대", "20대", "30대", "40대", "50대", "60대+"],
    "노쇼율": [10, 28, 22, 18, 15, 12]
})

companion_data = pd.DataFrame({
    "구분": ["동행자 있음", "동행자 없음"],
    "비율": [15, 35]
})

heatmap_data = {
    ("월", "09:00"): 12, ("화", "09:00"): 15, ("수", "09:00"): 14, ("목", "09:00"): 18, ("금", "09:00"): 22, ("토", "09:00"): 8,
    ("월", "11:00"): 18, ("화", "11:00"): 25, ("수", "11:00"): 20, ("목", "11:00"): 26, ("금", "11:00"): 32, ("토", "11:00"): 12,
    ("월", "14:00"): 15, ("화", "14:00"): 22, ("수", "14:00"): 18, ("목", "14:00"): 24, ("금", "14:00"): 28, ("토", "14:00"): 10,
    ("월", "16:00"): 20, ("화", "16:00"): 28, ("수", "16:00"): 22, ("목", "16:00"): 30, ("금", "16:00"): 35, ("토", "16:00"): 15,
}
days = ["월", "화", "수", "목", "금", "토"]
time_slots = ["09:00", "11:00", "14:00", "16:00"]

def rate_class(rate):
    if rate < 15:
        return "low"
    elif rate < 25:
        return "mid"
    return "high"


thead_str = "<th></th>"
tbody_str = ""

for day in days:
    thead_str += f"<th scope='col'>{day}요일</th>"

for time in time_slots:
    tbody_str += f"<tr><th scope='row' class='time'>{time}</th>"

    for day in days:
        rate = heatmap_data[(day, time)]
        cls = rate_class(rate)

        tbody_str += f"<td class='cell {cls}'><div class='cell-time'>{time}</div><div class='cell-rate'>{rate}%</div></td>"

    tbody_str += "</tr>"

# 카드 UI 시작
with st.container(key='datetime_container', width='stretch', border=True):
    with st.container(key='datetime_header_container', horizontal=True, horizontal_alignment="distribute"):
        st.subheader("요일/시간대별 노쇼 예측")
        if st.button("날씨별 노쇼 예측", type="primary", key='weather_modal_btn', icon=':material/clear_day:', width=170):
            st.session_state.weather_modal_open = True

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # 카드 콘텐츠: 테이블과 범례
    st.markdown(f"""
        <table>
            <thead>
                <tr>
                    {thead_str}
                </tr>
            </thead>
            <tbody>
                {tbody_str}
            </tbody>
        </table>
        <div class="legend">
            <span><div class="box low"></div> 낮음 ( &lt; 15% )</span>
            <span><div class="box mid"></div> 중간 ( 15 ~ 25% )</span>
            <span><div class="box high"></div> 높음 ( ≥ 25% )</span>
        </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2, border=True)

with col1:
    st.subheader("동행자 유무별 노쇼 비율")

    fig_pie = px.pie(
        companion_data,
        names="구분",
        values="비율",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:  
    st.subheader("연령대별 노쇼 예측")
    fig_bar = px.bar(
        age_data,
        x="연령대",
        y="노쇼율",
        text="노쇼율"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

