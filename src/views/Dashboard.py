import pandas as pd
import streamlit as st
import plotly.express as px
from src.modules.predict_noshow_proba_df import predict_noshow_proba_df
from src.modules.one_hot_module import build_df_onehot, fetch_df, rows_to_df_onehot
from src.services.customerService import load_artifacts, get_chart_data

# 페이지 스타일
st.markdown("""
    <style>
        [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            background-color: #FFFFFF !important;
            border-radius: 1rem !important;
        }
    </style>
    
""", unsafe_allow_html=True)

# weather_list = ["🌨️", "☀️", "🌤️", "🌨️", "☀️", "☀️"]

model, scaler, feature_cols = load_artifacts()
df = get_chart_data(model, scaler, limit = None)

df_pie = df.groupby("patient_needs_companion")["no_show"].mean().reset_index()
df_pie["patient_needs_companion"] = df_pie["patient_needs_companion"].apply(lambda x : "보호자 없음" if x == 0 else "보호자 있음")
df_hist = df.groupby("age")["no_show"].mean().reset_index()

def build_heatmap_data(df, days, time_slots, prob_col="no_show_prob"):
    df = df.copy()

    # datetime 파싱
    df["appointment_datetime"] = pd.to_datetime(df["appointment_datetime"])

    # 요일 한글명
    weekday_map = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
    df["day"] = df["appointment_datetime"].dt.dayofweek.map(weekday_map)

    # 시간 슬롯 (네 UI 시간대에 맞춰 커스텀)
    df["hour"] = df["appointment_datetime"].dt.hour
    bins = [0, 11, 14, 16, 24]
    labels = ["09:00", "11:00", "14:00", "16:00"]
    df["time_slot"] = pd.cut(df["hour"], bins=bins, labels=labels, right=False)

    # 요일×시간대 평균 노쇼확률
    mat = (df.groupby(["day", "time_slot"])[prob_col].mean().unstack("day"))

    # 순서 고정 (중요: 화면이 흔들리지 않음)
    mat = mat.reindex(index=time_slots, columns=days)

    # dict로 변환: heatmap_data[(day, time)] = int rate
    heatmap_data = {}
    for time in time_slots:
        for day in days:
            v = mat.loc[time, day]
            # 데이터 없는 칸 처리: 0으로 하거나 None으로 두기 (선택)
            if pd.isna(v):
                heatmap_data[(day, time)] = None   # or 0
            else:
                heatmap_data[(day, time)] = int(round(v))

    return heatmap_data, mat

days = ["월", "화", "수", "목", "금", "토"]  # 네가 보여준 화면 기준 (일요일 빼면)
time_slots = ["09:00", "11:00", "14:00", "16:00"]

heatmap_data, mat = build_heatmap_data(df, days, time_slots, prob_col="no_show_prob")


def rate_class(rate):
    if rate is None:
        return "na"   # CSS에서 회색 처리용
    if rate < 12:
        return "low"
    elif rate < 15:
        return "mid"
    return "high"


thead_str = "<th></th>"
tbody_str = ""

for idx, day in enumerate(days):
    thead_str += f"<th scope='col'>{day}요일" # {weather_list[idx]}</th>

for time in time_slots:
    tbody_str += f"<tr><th scope='row' class='time'>{time}</th>"

    for day in days:
        rate = heatmap_data.get((day, time))
        cls = rate_class(rate)
        rate_text = "-" if rate is None else f"{rate}%"

        tbody_str += (
            f"<td class='cell {cls}'>"
            f"<div class='cell-time'>{time}</div>"
            f"<div class='cell-rate'>{rate_text}</div>"
            f"</td>"
        )

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
            <span><div class="box low"></div> 낮음 ( &lt; 12% )</span>
            <span><div class="box mid"></div> 중간 ( 12 ~ 15% )</span>
            <span><div class="box high"></div> 높음 ( ≥ 15% )</span>
        </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2, border=True)

with col1:
    st.subheader("보호자 유무별 노쇼 비율")

    fig_pie = px.pie(
        df_pie,
        names="patient_needs_companion",
        values="no_show",
        hole=0.4,
        color_discrete_sequence=['#F59E0B', '#14B8A6']
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:  
    st.subheader("연령대별 노쇼율 예측")

    fig_hist = px.histogram(
        df_hist,
        x="age",
        y = "no_show",
        nbins=20,
        histfunc="avg",
        labels={
            "age": "연령",
            "no_show": "평균 노쇼율"
        },
        color_discrete_sequence=['#7C3AED']
    )
    fig_hist.update_yaxes(title_text="평균 노쇼율")
    fig_hist.update_layout(bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)

