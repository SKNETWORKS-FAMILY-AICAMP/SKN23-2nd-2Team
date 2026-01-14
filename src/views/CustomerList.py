import numpy as np
import pandas as pd
import streamlit as st
from src.modules.one_hot_module import SPECIALTY_KO_MAP, _SPECIALTY_CATS_KO, rows_to_df_onehot
from src.services.customerService import load_artifacts, get_chart_data, update_customer_info, search_filters

# 페이지 스타일
st.markdown("""
    <style>
        hr {
            margin-top: 0 !important;
            margin-bottom: 1rem !important;
        }

        [data-testid="stLayoutWrapper"] > [data-testid="stForm"],
        [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
            background-color: #FFFFFF !important;
        }

        [data-testid="stForm"] {
            padding: 0.8rem 1.7rem 1rem 1.7rem
        }

        [data-testid="stAlert"] > [data-testid="stAlertContainer"] {
            padding: 0.7rem 1rem;
        }

        [data-testid="stBaseButton-tertiary"] {
            color: #7C7C7C;
        }
        [data-testid="stBaseButton-tertiary"]:hover {
            color: #242424;
        }
    </style>
    
""", unsafe_allow_html=True)

ITEMS_PER_PAGE = 5
column_names = ["이름", "나이", "성별", "전문의", "예약시간", "노쇼율", "문자 전송", "수정"]

# 데이터 호출
model, scaler, feature_cols = load_artifacts()
df = get_chart_data(model, scaler)

# 세션 작업
if 'org_data' not in st.session_state:
    st.session_state.org_data = df.copy()

if 'df_data' not in st.session_state:
    st.session_state.df_data = df.copy()

if 'page_num' not in st.session_state:
    st.session_state.page_num = 1

filtered_df = st.session_state.df_data.copy()

# 업데이트 로직
def updated_customer_info_data(target_df, updated_info):
    row_idx = target_df.index[target_df['appointment_id'] == updated_info['appointment_id']].tolist()

    if row_idx:
        idx = row_idx[0]

        for t in ['name', 'age', 'gender', 'specialty', 'appointment_datetime']:
            target_df.at[idx, t] = updated_info[t]

        target_df = update_customer_info(model, scaler, target_df)

if 'updated_customer_info' in st.session_state and st.session_state.updated_customer_info:
    updated_info = st.session_state.updated_customer_info

    for target_df in [st.session_state.org_data, st.session_state.df_data, filtered_df]:
        updated_customer_info_data(target_df, updated_info)

    del st.session_state.updated_customer_info

# 검색 및 필터 초기화를 위한 세션 상태 초기화
if 'age_filter' not in st.session_state:
    st.session_state.age_filter = "전체"
if 'dept_filter' not in st.session_state:
    st.session_state.dept_filter = "전체"
if 'risk_filter' not in st.session_state:
    st.session_state.risk_filter = "전체"

# 검색 및 초기화 콜백 함수
def search_action():
    result_df = search_filters(
        st.session_state.age_filter,
        st.session_state.dept_filter,
        st.session_state.risk_filter
    )

    st.session_state.df_data = result_df
    st.session_state.page_num = 1

def reset_action():
    st.session_state.age_filter = "전체"
    st.session_state.dept_filter = "전체"
    st.session_state.risk_filter = "전체"

    search_action()

with st.form("search_form"):
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])

    with col1:
        st.selectbox("연령대", ["전체", "10대 미만", "10대", "20대", "30대", "40대", "50대 이상"], key="age_filter")

    with col2:
        st.selectbox("전문의", ["전체"] + _SPECIALTY_CATS_KO, key="dept_filter")

    with col3:
        st.selectbox("노쇼 위험군", ["전체", "고위험", "중위험", "저위험"], key="risk_filter")

    with col4:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        st.form_submit_button(label="검 색", on_click=search_action, use_container_width=True, icon=":material/search:")

    with col5:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        st.form_submit_button(label="초기화", on_click=reset_action, use_container_width=True, icon=":material/replay:")

st.info("노쇼 예측 비율이 **20% 이상인 고객**만 문자 전송 대상입니다.\n 사전 알림을 통해 예약 이탈을 최소화할 수 있습니다.")

# 테이블 출력
with st.container(key=f'customer_container_{len(st.session_state.df_data)}', border=True):
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
        print(f"idx: start-{start_idx}, end-{end_idx}")
        paginated_df = filtered_df.iloc[start_idx:end_idx]

        cols_ratio = [1, 1, 1, 1, 2, 1.3, 2, 1]
        # 헤더 컬럼
        header_cols = st.columns(cols_ratio)

        for col, name in zip(header_cols, column_names):
            col.markdown(f"<div style='display: flex; align-items: center; justify-content: center; height: 100%; font-weight: bold; padding: 8px; border-radius: 4px;'>{name}</div>", unsafe_allow_html=True)

        st.divider()

        # 셀 내용 (중앙 정렬)
        for _, row in paginated_df.iterrows():
            # 각 행을 고유한 key를 가진 container로 감싸 렌더링 문제를 해결합니다.
            with st.container(key=f"row_{row['appointment_id']}"):
                cols = st.columns(cols_ratio)
                cell_style = "display: flex; align-items: center; justify-content: center; height: 100%; padding: 0.25rem 0;"

                cols[0].markdown(f"<div style='{cell_style}'>{row['name']}</div>", unsafe_allow_html=True)
                cols[1].markdown(f"<div style='{cell_style}'>{row['age']}세</div>", unsafe_allow_html=True)
                cols[2].markdown(f"<div style='{cell_style}'>{'여' if row['gender'] else '남'}</div>", unsafe_allow_html=True)
                cols[3].markdown(f"<div style='{cell_style}'>{SPECIALTY_KO_MAP[row['specialty']]}</div>", unsafe_allow_html=True)
                cols[4].markdown(f"<div style='{cell_style}'>{row['appointment_datetime']}</div>", unsafe_allow_html=True)

                # 노쇼율 뱃지
                badge_html = ""

                if row["no_show_prob"] >= 20:
                    badge_html = f"<span style='background:#fee2e2;color:#991b1b;padding:6px 10px;border-radius:8px;'>고위험 {row['no_show_prob']:.1f}%</span>"
                elif row["no_show_prob"] >= 10:
                    badge_html = f"<span style='background:#fef9c3;color:#92400e;padding:6px 10px;border-radius:8px;'>중위험 {row['no_show_prob']:.1f}%</span>"
                else:
                    badge_html = f"<span style='background:#dcfce7;color:#166534;padding:6px 10px;border-radius:8px;'>저위험 {row['no_show_prob']:.1f}%</span>"

                cols[5].markdown(f"<div style='{cell_style}'>{badge_html}</div>", unsafe_allow_html=True)

                # 문자 전송 버튼
                send_disabled = row["no_show_prob"] < 20

                with cols[6]:
                    if st.button(
                        "문자 전송",
                        key=f"send_{row['appointment_id']}",
                        disabled=send_disabled,
                        # type="primary",
                        type="primary" if not send_disabled else "secondary",
                        width='stretch',
                        icon=":material/mail:"
                    ):
                        st.session_state.selected_customer = row.to_dict()
                        st.session_state.open_message_modal = True
                        st.rerun()

                with cols[7]:
                    if st.button(
                        "수정",
                        key=f"edit_{row['appointment_id']}",
                        width='stretch',
                        icon=":material/edit:"
                    ):
                        st.session_state.selected_customer_for_edit = row.to_dict()
                        st.session_state.open_edit_modal = True
                        st.rerun()

        st.divider()

        # 페이지네이션
        _, col1, _ = st.columns([4, 2, 4])

        with col1:
            prev, pages, next = st.columns([1, 3, 1])

            with prev:
                if st.button("", icon=":material/keyboard_double_arrow_left:", type="tertiary", disabled=st.session_state.page_num <= 1):
                    st.session_state.page_num -= 1
                    st.rerun()

            with pages:
                st.markdown(f"<div style='text-align: center; padding: 0.5rem 0;'>{st.session_state.page_num} / {total_pages}</div>", unsafe_allow_html=True)

            with next:
                if st.button("", icon=":material/keyboard_double_arrow_right:", type="tertiary", disabled=st.session_state.page_num >= total_pages):
                    st.session_state.page_num += 1
                    st.rerun()