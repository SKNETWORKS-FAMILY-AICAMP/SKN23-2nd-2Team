import streamlit as st
import pandas as pd
import numpy as np
import torch

from src.modules.one_hot_module import SPECIALTY_KO_MAP, _SPECIALTY_CATS_KO, disability_onehot
from src.services.customerService import load_artifacts, get_customer_list  # update_customer_info는 이제 안 씀
from src.modules.one_hot_module import rows_to_df_onehot  # <- 너희 원핫 함수 경로에 맞게 수정

def safe_disability_onehot(df, column_name="disability"):
    s = df[column_name].copy()

    # null 처리: NaN이면 "null"로 통일 (너희 데이터 규칙에 맞게)
    s = s.fillna("null").replace({"None": "null", "": "null"})

    dummies = pd.get_dummies(s, prefix=column_name)

    required = ["disability_intellectual", "disability_motor", "disability_null"]
    for col in required:
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[required]

    out = df.drop(columns=[column_name], errors="ignore").copy()
    out = pd.concat([out, dummies], axis=1)
    return out



# ----------------------------
# 0) Style
# ----------------------------
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


# ----------------------------
# 1) Artifacts (resource cache 권장)
# ----------------------------
@st.cache_resource
def get_dl_artifacts():
    return load_artifacts()

model, scaler, feature_cols = get_dl_artifacts()


# ----------------------------
# 2) Initial data
# ----------------------------
@st.cache_data
def load_initial_customer_df():
    # get_customer_list가 내부에서 전체 예측까지 포함한다면(초기 no_show_prob 포함),
    # 여기서 한 번만 전체 계산하고, 이후 수정은 "행 단위"로 처리
    df0 = get_customer_list(model, scaler)
    return df0

df = load_initial_customer_df()

if "df_data" not in st.session_state:
    st.session_state.df_data = df.copy()

if "page_num" not in st.session_state:
    st.session_state.page_num = 1


# ----------------------------
# 3) Row-only predict
# ----------------------------
def predict_one_row_prob_percent(row_df: pd.DataFrame) -> float:
    # ✅ 모델이 학습할 때의 "appointment 원본 스키마" 컬럼만 남기기
    keep_cols = [
        "appointment_id", "name", "age", "gender", "specialty",
        "appointment_date", "appointment_datetime", "entry_service_date",
        "disability", "icd",
    ]
    row_df = row_df.copy()
    row_df = row_df[[c for c in keep_cols if c in row_df.columns]]

    X_oh = rows_to_df_onehot(row_df).reindex(columns=feature_cols, fill_value=0)
    X = X_oh.values.astype(np.float32)

    # scaler transform
    Xs = scaler.transform(X)

    # model
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.tensor(Xs, dtype=torch.float32, device=device)
        logits = model(xt).view(-1)
        prob = torch.sigmoid(logits).detach().cpu().numpy()[0]

    return float(prob * 100.0)


def apply_update_one_row_inplace(base_df: pd.DataFrame, updated_info: dict) -> None:
    """
    base_df(세션 df_data)에서 appointment_id로 행을 찾고
    - 원본 컬럼 갱신
    - no_show_prob만 "그 행" 기준으로 재예측하여 갱신
    """
    # ✅ df가 아니라 base_df에서 찾는다 (중요)
    row_idx = base_df.index[base_df["appointment_id"] == updated_info["appointment_id"]]
    if len(row_idx) == 0:
        return

    idx = row_idx[0]

    # 1) 원본 값 갱신 (dtype도 가능하면 맞춰주기)
    base_df.at[idx, "name"] = updated_info["name"]
    base_df.at[idx, "age"] = int(updated_info["age"])
    base_df.at[idx, "gender"] = bool(updated_info["gender"])
    base_df.at[idx, "specialty"] = updated_info["specialty"]
    base_df.at[idx, "appointment_datetime"] = updated_info["appointment_datetime"]

    # 2) 해당 행만 예측
    one = base_df.loc[[idx]].copy()
    if "no_show_prob" in one.columns:
        one = one.drop(columns=["no_show_prob"])

    new_prob = predict_one_row_prob_percent(one)
    base_df.at[idx, "no_show_prob"] = new_prob


# ----------------------------
# 4) Update hook (from edit modal)
# ----------------------------
if "updated_customer_info" in st.session_state and st.session_state.updated_customer_info:
    updated_info = st.session_state.updated_customer_info

    # ✅ 세션의 df_data 자체를 "행 단위"로 갱신
    apply_update_one_row_inplace(st.session_state.df_data, updated_info)

    # cleanup
    del st.session_state.updated_customer_info
    st.rerun()


# ----------------------------
# 5) Filtering (submitted 기반 유지)
# ----------------------------
filtered_df = st.session_state.df_data  # ✅ copy() 지양. 필요 시 마지막에만 copy

with st.form("search_form"):
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        age_filter = st.selectbox("연령대", ["전체", "10대 미만", "10대", "20대", "30대", "40대", "50대 이상"])

    with col2:
        dept_filter = st.selectbox("전문의", ["전체"] + _SPECIALTY_CATS_KO)

    with col3:
        risk_filter = st.selectbox("노쇼 위험군", ["전체", "고위험", "중위험", "저위험"])

    with col4:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(label="검 색", use_container_width=True, icon=":material/search:")

if submitted:
    st.session_state.page_num = 1
    tmp = filtered_df

    # age
    if age_filter != "전체":
        if age_filter == "50대 이상":
            tmp = tmp[tmp["age"] >= 50]
        elif age_filter == "10대 미만":
            tmp = tmp[tmp["age"] < 10]
        else:
            base = int(age_filter.replace("대", ""))
            tmp = tmp[(tmp["age"] >= base) & (tmp["age"] < base + 10)]

    # dept
    if dept_filter != "전체":
        reverse_specialty_map = {v: k for k, v in SPECIALTY_KO_MAP.items()}
        selected_specialty_en = reverse_specialty_map.get(dept_filter)
        if selected_specialty_en:
            tmp = tmp[tmp["specialty"] == selected_specialty_en]

    # risk
    if risk_filter != "전체":
        if risk_filter == "고위험":
            tmp = tmp[tmp["no_show_prob"] >= 30]
        elif risk_filter == "중위험":
            tmp = tmp[(tmp["no_show_prob"] >= 20) & (tmp["no_show_prob"] < 30)]
        elif risk_filter == "저위험":
            tmp = tmp[tmp["no_show_prob"] < 20]

    filtered_df = tmp
else:
    # 폼 제출 안 했으면 "현재 세션 df" 그대로
    filtered_df = filtered_df


st.info("노쇼 예측 확률이 **20% 이상인 고객**만 문자 전송 대상입니다.\n 사전 알림을 통해 예약 이탈을 최소화할 수 있습니다.")


# ----------------------------
# 6) Table + Pagination (기존 레이아웃 유지)
# ----------------------------
with st.container(key="customer_container", border=True):
    if filtered_df.empty:
        st.warning("검색 조건에 맞는 고객이 없습니다.")
    else:
        total_items = len(filtered_df)
        total_pages = (total_items - 1) // ITEMS_PER_PAGE + 1
        total_pages = max(total_pages, 1)

        if st.session_state.page_num > total_pages:
            st.session_state.page_num = total_pages

        start_idx = (st.session_state.page_num - 1) * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        paginated_df = filtered_df.iloc[start_idx:end_idx]

        cols_ratio = [1, 1, 1, 1, 2, 1.3, 2, 1]

        header_cols = st.columns(cols_ratio)
        for col, name in zip(header_cols, column_names):
            col.markdown(
                f"<div style='display:flex;align-items:center;justify-content:center;height:100%;"
                f"font-weight:bold;padding:8px;border-radius:4px;'>{name}</div>",
                unsafe_allow_html=True
            )

        st.divider()

        for _, row in paginated_df.iterrows():
            cols = st.columns(cols_ratio)
            cell_style = "display:flex;align-items:center;justify-content:center;height:100%;padding:0.25rem 0;"

            cols[0].markdown(f"<div style='{cell_style}'>{row['name']}</div>", unsafe_allow_html=True)
            cols[1].markdown(f"<div style='{cell_style}'>{row['age']}세</div>", unsafe_allow_html=True)
            cols[2].markdown(f"<div style='{cell_style}'>{'여' if row['gender'] else '남'}</div>", unsafe_allow_html=True)
            cols[3].markdown(f"<div style='{cell_style}'>{SPECIALTY_KO_MAP[row['specialty']]}</div>", unsafe_allow_html=True)
            cols[4].markdown(f"<div style='{cell_style}'>{row['appointment_datetime']}</div>", unsafe_allow_html=True)

            if row["no_show_prob"] >= 30:
                badge_html = f"<span style='background:#fee2e2;color:#991b1b;padding:6px 10px;border-radius:8px;'>고위험 {row['no_show_prob']:.1f}%</span>"
            elif row["no_show_prob"] >= 20:
                badge_html = f"<span style='background:#fef9c3;color:#92400e;padding:6px 10px;border-radius:8px;'>중위험 {row['no_show_prob']:.1f}%</span>"
            else:
                badge_html = f"<span style='background:#dcfce7;color:#166534;padding:6px 10px;border-radius:8px;'>저위험 {row['no_show_prob']:.1f}%</span>"

            cols[5].markdown(f"<div style='{cell_style}'>{badge_html}</div>", unsafe_allow_html=True)

            send_disabled = row["no_show_prob"] < 20
            with cols[6]:
                if st.button(
                    "문자 전송",
                    key=f"send_{row['appointment_id']}",
                    disabled=send_disabled,
                    type="primary" if not send_disabled else "secondary",
                    use_container_width=True,
                    icon=":material/mail:"
                ):
                    st.session_state.selected_customer = row.to_dict()
                    st.session_state.open_message_modal = True

            with cols[7]:
                if st.button(
                    "수정",
                    key=f"edit_{row['appointment_id']}",
                    use_container_width=True,
                    icon=":material/edit:"
                ):
                    st.session_state.selected_customer_for_edit = row.to_dict()
                    st.session_state.open_edit_modal = True

        st.divider()

        _, col1, _ = st.columns([4, 2, 4])
        with col1:
            prev, pages, nxt = st.columns([1, 3, 1])

            with prev:
                if st.button("", icon=":material/keyboard_double_arrow_left:", disabled=st.session_state.page_num <= 1):
                    st.session_state.page_num -= 1
                    st.rerun()

            with pages:
                st.markdown(
                    f"<div style='text-align:center;padding:0.5rem 0;'>{st.session_state.page_num} / {total_pages}</div>",
                    unsafe_allow_html=True
                )

            with nxt:
                if st.button("", icon=":material/keyboard_double_arrow_right:", disabled=st.session_state.page_num >= total_pages):
                    st.session_state.page_num += 1
                    st.rerun()
