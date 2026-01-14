import streamlit as st
import pandas as pd
from src.modules.one_hot_module import SPECIALTY_KO_MAP
from src.services.customerService import load_artifacts, get_customer_list
# 페이지 스타일
st.markdown("""
    <style>
        hr {
            margin-top: 0 !important;
            margin-bottom: 1rem !important;
        }
    </style>
    
""", unsafe_allow_html=True)

ITEMS_PER_PAGE = 5
column_names = ["이름", "나이", "성별", "전문의", "예약시간", "노쇼율", "문자 전송", "수정"]

# 데이터 호출
model, scaler, feature_cols = load_artifacts()
df = get_customer_list(model, scaler)

# 세션 작업
if 'df_data' not in st.session_state:
    st.session_state.df_data = df.copy()

if 'page_num' not in st.session_state:
    st.session_state.page_num = 1

# 업데이트 로직
if 'updated_customer_info' in st.session_state and st.session_state.updated_customer_info:
    updated_info = st.session_state.updated_customer_info
    customer_id = updated_info['id']
    row_index = df.index[df['id'] == customer_id].tolist()
    if row_index:
        idx = row_index[0]
        df.at[idx, 'name'] = updated_info['name']
        df.at[idx, 'age'] = updated_info['age']
        df.at[idx, 'gender'] = updated_info['gender']
        df.at[idx, 'specialty'] = updated_info['specialty']
        df.at[idx, 'companion'] = updated_info['companion']
        df.at[idx, 'appointment'] = updated_info['appointment']

    del st.session_state.updated_customer_info

# 검색바
with st.form("search_form"):
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        age_filter = st.selectbox("연령대", ["전체", "10대 미만", "10대", "20대", "30대", "40대", "50대 이상"])

    with col2:
        dept_filter = st.selectbox("전문의", ["전체"] + list(SPECIALTY_KO_MAP.values()))

    with col3:
        risk_filter = st.selectbox("노쇼 위험군", ["전체", "고위험", "중위험", "저위험"])

    with col4:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(label="검 색", width="stretch", icon=":material/search:")

filtered_df = st.session_state.df_data.copy()

# 폼이 제출되었을 때만 필터링 수행
if submitted:
    st.session_state.page_num = 1

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
        reverse_specialty_map = {v: k for k, v in SPECIALTY_KO_MAP.items()}
        selected_specialty_en = reverse_specialty_map.get(dept_filter)

        if selected_specialty_en:
            filtered_df = filtered_df[filtered_df["specialty"] == selected_specialty_en]

    if risk_filter != "전체":
        if risk_filter == "고위험":
            filtered_df = filtered_df[filtered_df["no_show_prob"] >= 30]
        elif risk_filter == "중위험":
            filtered_df = filtered_df[(filtered_df["no_show_prob"] >= 20) & (filtered_df["no_show_prob"] < 30)]
        elif risk_filter == "저위험":
            filtered_df = filtered_df[filtered_df["no_show_prob"] < 20]

st.info("노쇼 예측 비율이 **50% 이상인 고객**만 문자 전송 대상입니다.\n 사전 알림을 통해 예약 이탈을 최소화할 수 있습니다.")

# 테이블 출력
with st.container(key='customer_container', border=True):
    if filtered_df.empty:
        st.warning("검색 조건에 맞는 고객이 없습니다.")
    else:
        # Pagination 설정
        total_items = len(filtered_df)
        total_pages = (total_items - 1) // ITEMS_PER_PAGE + 1

        # 현재 페이지 번호가 전체 페이지 수를 초과하지 않도록 조정
        if st.session_state.page_num > total_pages:
            st.session_state.page_num = total_pages

        if total_pages == 0:
            total_pages = 1

        start_idx = (st.session_state.page_num - 1) * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        paginated_df = filtered_df.iloc[start_idx:end_idx]

        cols_ratio = [1, 1, 1, 1, 2, 1.3, 2, 1]
        # 헤더 컬럼
        header_cols = st.columns(cols_ratio)

        for col, name in zip(header_cols, column_names):
            col.markdown(f"<div style='display: flex; align-items: center; justify-content: center; height: 100%; font-weight: bold; padding: 8px; border-radius: 4px;'>{name}</div>", unsafe_allow_html=True)

        st.divider()

        # 셀 내용 (중앙 정렬)
        for _, row in paginated_df.iterrows():
            cols = st.columns(cols_ratio)
            cell_style = "display: flex; align-items: center; justify-content: center; height: 100%; padding: 0.25rem 0;"

            cols[0].markdown(f"<div style='{cell_style}'>{row['name']}</div>", unsafe_allow_html=True)
            cols[1].markdown(f"<div style='{cell_style}'>{row['age']}세</div>", unsafe_allow_html=True)
            cols[2].markdown(f"<div style='{cell_style}'>{'여' if row['gender'] else '남'}</div>", unsafe_allow_html=True)
            cols[3].markdown(f"<div style='{cell_style}'>{SPECIALTY_KO_MAP[row['specialty']]}</div>", unsafe_allow_html=True)
            cols[4].markdown(f"<div style='{cell_style}'>{row['appointment_datetime']}</div>", unsafe_allow_html=True)

            # 노쇼율 뱃지
            badge_html = ""

            if row["no_show_prob"] >= 30:
                badge_html = f"<span style='background:#fee2e2;color:#991b1b;padding:6px 10px;border-radius:8px;'>고위험 {row['no_show_prob']:.1f}%</span>"
            elif row["no_show_prob"] >= 20:
                badge_html = f"<span style='background:#fef9c3;color:#92400e;padding:6px 10px;border-radius:8px;'>중위험 {row['no_show_prob']:.1f}%</span>"
            else:
                badge_html = f"<span style='background:#dcfce7;color:#166534;padding:6px 10px;border-radius:8px;'>저위험 {row['no_show_prob']:.1f}%</span>"

            cols[5].markdown(f"<div style='{cell_style}'>{badge_html}</div>", unsafe_allow_html=True)

            # 문자 전송 버튼
            send_disabled = row["no_show_prob"] < 30

            with cols[6]:
                if st.button(
                    "문자 전송",
                    key=f"send_{row['appointment_id']}",
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
                    key=f"edit_{row['appointment_id']}",
                    width='stretch',
                    icon=":material/edit:"
                ):
                    st.session_state.selected_customer_for_edit = row.to_dict()
                    st.session_state.open_edit_modal = True
                    # st.rerun()

        st.divider()

        # 페이지네이션 컨트롤
        _, col1, _ = st.columns([4, 2, 4])

        with col1:
            prev, pages, next = st.columns([1, 3, 1])

            with prev:
                if st.button("", icon=":material/keyboard_double_arrow_left:", disabled=st.session_state.page_num <= 1):
                    st.session_state.page_num -= 1

            with pages:
                st.markdown(f"<div style='text-align: center; padding: 0.5rem 0;'>{st.session_state.page_num} / {total_pages}</div>", unsafe_allow_html=True)

            with next:
                if st.button("", icon=":material/keyboard_double_arrow_right:", disabled=st.session_state.page_num >= total_pages):
                    st.session_state.page_num += 1