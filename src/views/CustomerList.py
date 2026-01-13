import streamlit as st
import pandas as pd

# -----------------------------
# 기본 설정
# -----------------------------
st.subheader("👥 고객 목록")
st.caption("노쇼 예측 비율 및 예약 관리")
st.divider()

# -----------------------------
# Mock 데이터
# -----------------------------
customers = [
    {
        "id": 1,
        "name": "김민수",
        "age": 45,
        "gender": "남",
        "department": "내과",
        "companion": "없음",
        "appointment": "2026-01-15 14:00",
        "no_show": 65,
    },
    {
        "id": 2,
        "name": "이영희",
        "age": 32,
        "gender": "여",
        "department": "정형외과",
        "companion": "있음",
        "appointment": "2026-01-16 10:00",
        "no_show": 22,
    },
    {
        "id": 3,
        "name": "박철수",
        "age": 58,
        "gender": "남",
        "department": "이비인후과",
        "companion": "없음",
        "appointment": "2026-01-14 16:00",
        "no_show": 78,
    },
]
df = pd.DataFrame(customers)

# -----------------------------
# 필터 영역
# -----------------------------
with st.container():
    st.markdown("### 🔍 검색 필터")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age_filter = st.selectbox(
            "연령대",
            ["전체", "10대", "20대", "30대", "40대", "50대 이상"]
        )

    with col2:
        dept_filter = st.selectbox(
            "진료과",
            ["전체", "내과", "정형외과", "이비인후과"]
        )

    with col3:
        companion_filter = st.selectbox(
            "동반자 여부",
            ["전체", "있음", "없음"]
        )

    with col4:
        risk_filter = st.selectbox(
            "노쇼 위험군",
            ["전체", "고위험 (50% 이상)", "일반 (50% 미만)"]
        )

# -----------------------------
# 필터 로직
# -----------------------------
filtered_df = df.copy()

if age_filter != "전체":
    if age_filter == "50대 이상":
        filtered_df = filtered_df[filtered_df["나이"] >= 50]
    else:
        base = int(age_filter.replace("대", ""))
        filtered_df = filtered_df[
            (filtered_df["나이"] >= base) &
            (filtered_df["나이"] < base + 10)
        ]

if dept_filter != "전체":
    filtered_df = filtered_df[filtered_df["진료과"] == dept_filter]

if companion_filter != "전체":
    filtered_df = filtered_df[
        filtered_df["동반자"] == (companion_filter == "있음")
    ]

if risk_filter != "전체":
    if "고위험" in risk_filter:
        filtered_df = filtered_df[filtered_df["no_show"] >= 50]
    else:
        filtered_df = filtered_df[filtered_df["no_show"] < 50]

# -----------------------------
# 통계 요약
# -----------------------------
st.divider()
col1, col2, col3, col4 = st.columns(4)

col1.metric("총 고객 수", f"{len(filtered_df)}명")
col2.metric("고위험 고객", f"{len(filtered_df[filtered_df['no_show'] >= 50])}명")
col3.metric("중위험 고객", f"{len(filtered_df[(filtered_df['no_show'] >= 30) & (filtered_df['no_show'] < 50)])}명")
col4.metric("저위험 고객", f"{len(filtered_df[filtered_df['no_show'] < 30])}명")

# -----------------------------
# 테이블 출력
# -----------------------------

st.divider()
st.info(
    "노쇼 예측 비율이 **50% 이상인 고객**만 문자 전송 대상입니다.\n"
    "사전 알림을 통해 예약 이탈을 최소화할 수 있습니다."
)
st.divider()

for _, row in df.iterrows():
    cols = st.columns([2, 1, 1, 2, 2, 3, 2, 2])

    cols[0].write(row["name"])
    cols[1].write(f"{row['age']}세")
    cols[2].write(row["gender"])
    cols[3].write(row["department"])
    cols[4].write(row["companion"])
    cols[5].write(row["appointment"])

    # 노쇼율 뱃지
    if row["no_show"] >= 50:
        cols[6].markdown(
            f"<span style='background:#fee2e2;color:#991b1b;padding:4px 8px;border-radius:8px;'>고위험 {row['no_show']}%</span>",
            unsafe_allow_html=True
        )
    elif row["no_show"] >= 30:
        cols[6].markdown(
            f"<span style='background:#fef9c3;color:#92400e;padding:4px 8px;border-radius:8px;'>중위험 {row['no_show']}%</span>",
            unsafe_allow_html=True
        )
    else:
        cols[6].markdown(
            f"<span style='background:#dcfce7;color:#166534;padding:4px 8px;border-radius:8px;'>저위험 {row['no_show']}%</span>",
            unsafe_allow_html=True
        )

    # 문자 전송 버튼
    send_disabled = row["no_show"] < 50

    if cols[7].button(
        "📩 문자 전송",
        key=f"send_{row['id']}",
        disabled=send_disabled,
        type="primary" if not send_disabled else "secondary",
    ):
        st.session_state.selected_customer = row.to_dict()
        st.session_state.open_message_modal = True

    st.divider()