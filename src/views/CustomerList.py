import streamlit as st
import pandas as pd

st.markdown("""
    <style>
        hr {
            margin-top: 0 !important;
            margin-bottom: 1rem !important;
        }
    </style>
    
""", unsafe_allow_html=True)

# Mock 데이터
df = pd.DataFrame([
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
    {
        "id": 4,
        "name": "오지영",
        "age": 8,
        "gender": "여",
        "department": "소아과",
        "companion": "있음",
        "appointment": "2026-01-13 09:00",
        "no_show": 15,
    },
    {
        "id": 5,
        "name": "최강민",
        "age": 25,
        "gender": "남",
        "department": "피부과",
        "companion": "없음",
        "appointment": "2026-01-12 11:30",
        "no_show": 40,
    }
])

# 업데이트 로직
if 'updated_customer_info' in st.session_state and st.session_state.updated_customer_info:
    updated_info = st.session_state.updated_customer_info
    customer_id = updated_info['id']
    row_index = df.index[df['id'] == customer_id].tolist()
    if row_index:
        idx = row_index[0]
        # Update the relevant fields, excluding 'no_show'
        df.at[idx, 'name'] = updated_info['name']
        df.at[idx, 'age'] = updated_info['age']
        df.at[idx, 'gender'] = updated_info['gender']
        df.at[idx, 'department'] = updated_info['department']
        df.at[idx, 'companion'] = updated_info['companion']
        df.at[idx, 'appointment'] = updated_info['appointment']

    del st.session_state.updated_customer_info

# 검색바
with st.form("search_form"):
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        age_filter = st.selectbox(
            "연령대",
            ["전체", "10대 미만", "10대", "20대", "30대", "40대", "50대 이상"]
        )

    with col2:
        # 데이터프레임에서 'department'의 고유한 값으로 선택 상자 채우기
        dept_options = ["전체"] + list(df['department'].unique())
        dept_filter = st.selectbox("진료과", dept_options)

    with col3:
        risk_filter = st.selectbox("노쇼 위험군", ["전체", "고위험", "중위험", "저위험"])

    with col4:
        # 세로 정렬을 위해 div 추가
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(label="검 색", width="stretch", icon=":material/search:")

filtered_df = df.copy()

# 폼이 제출되었을 때만 필터링 수행
if submitted:
    if age_filter != "전체":
        if age_filter == "50대 이상":
            filtered_df = filtered_df[filtered_df["age"] >= 50]
        elif age_filter == "10대 미만":
            filtered_df = filtered_df[filtered_df["age"] < 10]
        else:
            base = int(age_filter.replace("대", ""))
            filtered_df = filtered_df[
                (filtered_df["age"] >= base) &
                (filtered_df["age"] < base + 10)
            ]

    if dept_filter != "전체":
        filtered_df = filtered_df[filtered_df["department"] == dept_filter]

    if risk_filter != "전체":
        if risk_filter == "고위험":
            filtered_df = filtered_df[filtered_df["no_show"] >= 50]
        elif risk_filter == "중위험":
            filtered_df = filtered_df[(filtered_df["no_show"] >= 30) & (filtered_df["no_show"] < 50)]
        elif risk_filter == "저위험":
            filtered_df = filtered_df[filtered_df["no_show"] < 30]

st.info("노쇼 예측 비율이 **50% 이상인 고객**만 문자 전송 대상입니다.\n 사전 알림을 통해 예약 이탈을 최소화할 수 있습니다.")

# 테이블 출력
with st.container(key='customer_container', width='stretch', border=True):
    if filtered_df.empty:
        st.warning("검색 조건에 맞는 고객이 없습니다.")
    else:
        cols_ratio = [1, 1, 1, 1, 2, 1.3, 2, 1]
        # 헤더 컬럼
        header_cols = st.columns(cols_ratio)
        column_names = ["이름", "나이", "성별", "진료과", "예약시간", "노쇼율", "문자 전송", "수정"]

        for col, name in zip(header_cols, column_names):
            col.markdown(f"<div style='display: flex; align-items: center; justify-content: center; height: 100%; font-weight: bold; padding: 8px; border-radius: 4px;'>{name}</div>", unsafe_allow_html=True)

        st.divider()

        # 셀 내용 (중앙 정렬)
        for _, row in filtered_df.iterrows():
            cols = st.columns(cols_ratio)
            cell_style = "display: flex; align-items: center; justify-content: center; height: 100%; padding: 0.25rem 0;"

            cols[0].markdown(f"<div style='{cell_style}'>{row['name']}</div>", unsafe_allow_html=True)
            cols[1].markdown(f"<div style='{cell_style}'>{row['age']}세</div>", unsafe_allow_html=True)
            cols[2].markdown(f"<div style='{cell_style}'>{row['gender']}</div>", unsafe_allow_html=True)
            cols[3].markdown(f"<div style='{cell_style}'>{row['department']}</div>", unsafe_allow_html=True)
            cols[4].markdown(f"<div style='{cell_style}'>{row['appointment']}</div>", unsafe_allow_html=True)

            # 노쇼율 뱃지
            badge_html = ""

            if row["no_show"] >= 50:
                badge_html = f"<span style='background:#fee2e2;color:#991b1b;padding:6px 10px;border-radius:8px;'>고위험 {row['no_show']}%</span>"
            elif row["no_show"] >= 30:
                badge_html = f"<span style='background:#fef9c3;color:#92400e;padding:6px 10px;border-radius:8px;'>중위험 {row['no_show']}%</span>"
            else:
                badge_html = f"<span style='background:#dcfce7;color:#166534;padding:6px 10px;border-radius:8px;'>저위험 {row['no_show']}%</span>"

            cols[5].markdown(f"<div style='{cell_style}'>{badge_html}</div>", unsafe_allow_html=True)

            # 문자 전송 버튼
            send_disabled = row["no_show"] < 50

            with cols[6]:
                if st.button(
                    "문자 전송",
                    key=f"send_{row['id']}",
                    disabled=send_disabled,
                    type="primary" if not send_disabled else "secondary",
                    width='stretch',
                    icon=":material/mail:"
                ):
                    st.session_state.selected_customer = row.to_dict()
                    st.session_state.open_message_modal = True
                    # st.rerun()

            with cols[7]:
                if st.button(
                    "수정",
                    key=f"edit_{row['id']}",
                    width='stretch',
                    icon=":material/edit:"
                ):
                    st.session_state.selected_customer_for_edit = row.to_dict()
                    st.session_state.open_edit_modal = True
                    # st.rerun()