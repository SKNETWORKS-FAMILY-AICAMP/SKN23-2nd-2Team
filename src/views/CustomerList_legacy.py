# app.py
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import src.modules.one_hot_module as oh  # 업로드한 one_hot_module.py를 같은 폴더에 두기

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

# ✅ 모듈 함수를 안전 버전으로 교체
oh.disability_onehot = safe_disability_onehot


# -----------------------
# 1) 모델 클래스 (학습 때와 동일해야 함)
# -----------------------
class NoShowMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1)  # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# -----------------------
# 2) 로드
# -----------------------
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "src/artifacts"

@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    scaler = joblib.load("src/artifacts/scaler.joblib")

    model = NoShowMLP(input_dim=len(feature_cols))
    state = torch.load("src/artifacts/mlp_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, scaler, feature_cols

# -----------------------
# 3) 학습 때와 동일한 전처리(= 노트북 그대로)
# -----------------------
def build_df_onehot(df_appt: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    df = df_appt.copy()
    w = df_weather.copy()

    # 날짜 처리 + 파생변수 (노트북 셀 기준)
    df.replace("", pd.NA, inplace=True)
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    df["entry_service_date"] = pd.to_datetime(df["entry_service_date"])

    df["days_since_first_visit"] = (df["appointment_date"] - df["entry_service_date"]).dt.days
    df["entry_service_date_missing"] = df["entry_service_date"].isna().astype(int)

    mean_days = df["days_since_first_visit"].mean()
    df["days_since_first_visit"] = df["days_since_first_visit"].fillna(mean_days).round().astype(int)
    df = df.drop(columns=["entry_service_date"])

    # weather merge (노트북에서 drop 하던 컬럼은 있으면 drop)
    for c in ["heat_intensity", "rain_intensity", "weather_id"]:
        if c in w.columns:
            w = w.drop(columns=[c])

    w["weather_date"] = pd.to_datetime(w["weather_date"])
    df_merged = df.merge(w, how="left", left_on="appointment_date", right_on="weather_date")

    # one-hot / multi-hot (업로드 모듈 사용)
    df_onehot = oh.date_to_weekday_onehot(df_merged, column_name="appointment_date")
    df_onehot = oh.date_to_month_onehot(df_onehot, column_name="appointment_datetime")
    df_onehot = oh.disability_onehot(df_onehot, column_name="disability")
    df_onehot = oh.specialty_ko_onehot(df_onehot, column_name="specialty", keep_korean_column=False)
    df_onehot = oh.icd_multihot(df_onehot, column_name="icd")

    # 노트북에서 drop 하던 컬럼 (있으면 drop)
    drop_cols = ["appointment_id", "name", "entry_service_date_missing", "weather_date"]
    df_onehot = df_onehot.drop(columns=[c for c in drop_cols if c in df_onehot.columns], errors="ignore")

    return df_onehot

@torch.no_grad()
def predict_proba_df(model, scaler, feature_cols, df_onehot: pd.DataFrame) -> np.ndarray:
    X = df_onehot.drop(columns=["no_show"], errors="ignore")
    X = X.reindex(columns=feature_cols, fill_value=0).values.astype(np.float32)
    Xs = scaler.transform(X)
    xt = torch.tensor(Xs, dtype=torch.float32)
    logits = model(xt)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def risk_bucket(p: float) -> str:
    if p >= 0.50:
        return "고위험"
    if p >= 0.30:
        return "중위험"
    return "저위험"

# -----------------------
# 4) UI
# -----------------------
st.set_page_config(layout="wide")
st.title("고객 목록")
st.caption("데이터 수정 → 노쇼 확률 실시간 갱신")

model, scaler, feature_cols = load_artifacts()

df_appt = pd.read_csv("assets/noshow_appt.csv")
df_weather = pd.read_csv("assets/noshow_weather.csv")


CALC_COLS = ["노쇼확률", "위험군", "문자전송가능"]

def add_predictions(df_base):
    df_onehot = build_df_onehot(df_base, df_weather)
    probs = predict_proba_df(model, scaler, feature_cols, df_onehot)

    out = df_base.copy()
    out["노쇼확률"] = (probs * 100).round(1)
    out["위험군"] = np.select([probs >= 0.5, probs >= 0.3], ["고위험", "중위험"], default="저위험")
    out["문자전송가능"] = probs >= 0.5
    return out

# ✅ 1) 최초 1회만: 원본 데이터에 예측 컬럼 붙여서 세션에 저장
if "table" not in st.session_state:
    base = df_appt.sample(200, random_state=42).reset_index(drop=True)
    st.session_state.table = add_predictions(base)

# ===== 옵션 B: 행 선택 + 상세 편집 패널 =====

def add_predictions_one_row(row_df):
    df_onehot = build_df_onehot(row_df, df_weather)
    probs = predict_proba_df(model, scaler, feature_cols, df_onehot)

    out = row_df.copy()
    out["노쇼확률"] = (probs * 100).round(1)
    out["위험군"] = np.select([probs >= 0.5, probs >= 0.3], ["고위험", "중위험"], default="저위험")
    out["문자전송가능"] = probs >= 0.5
    return out


DISPLAY_COLS = [
    "name", "age", "gender", "specialty", "appointment_datetime",
    "노쇼확률", "위험군", "문자전송가능"
]
EDITABLE_COLS = ["name", "age", "gender", "specialty", "appointment_datetime"]
CALC_COLS = ["노쇼확률", "위험군", "문자전송가능"]

full = st.session_state.table.copy()
view = full.loc[:, [c for c in DISPLAY_COLS if c in full.columns]].copy()

left, right = st.columns([3, 2])

with left:
    st.subheader("고객 목록")
    st.caption("고객을 선택하면 오른쪽에서 상세 정보를 수정할 수 있습니다.")
    st.dataframe(view, use_container_width=True)

    sel_idx = st.number_input(
        "수정할 고객 행 인덱스",
        min_value=int(view.index.min()),
        max_value=int(view.index.max()),
        value=int(view.index.min()),
        step=1
    )

with right:
    st.subheader("선택 고객 상세 편집")

    row = full.loc[[sel_idx]].copy()

    if "age" in row.columns:
        row.loc[sel_idx, "age"] = st.number_input(
            "나이", value=int(row.loc[sel_idx, "age"]), step=1
        )

    if "gender" in row.columns:
        row.loc[sel_idx, "gender"] = st.selectbox(
            "성별", [0, 1], index=int(row.loc[sel_idx, "gender"])
        )

    if "specialty" in row.columns:
        opts = sorted(full["specialty"].dropna().unique().tolist())
        cur = row.loc[sel_idx, "specialty"]
        row.loc[sel_idx, "specialty"] = st.selectbox(
            "진료 과목", opts, index=opts.index(cur) if cur in opts else 0
        )

    if "appointment_datetime" in row.columns:
        row.loc[sel_idx, "appointment_datetime"] = st.text_input(
            "예약 일시", value=str(row.loc[sel_idx, "appointment_datetime"])
        )

    if st.button("이 고객만 예측 갱신", type="primary", use_container_width=True):
        # 1) 수정값 반영
        for c in EDITABLE_COLS:
            full.loc[sel_idx, c] = row.loc[sel_idx, c]

        # 2) 선택 행만 재예측
        updated = add_predictions_one_row(
            full.loc[[sel_idx]].drop(columns=CALC_COLS, errors="ignore")
        )

        for c in CALC_COLS:
            full.loc[sel_idx, c] = updated.loc[sel_idx, c]

        st.session_state.table = full
        st.success("선택 고객의 노쇼 확률이 갱신되었습니다.")
        st.rerun()

    if st.button("전체 다시 계산", use_container_width=True):
        base_edited = full.drop(columns=CALC_COLS, errors="ignore")
        st.session_state.table = add_predictions(base_edited)
        st.success("전체 고객 예측이 갱신되었습니다.")
        st.rerun()

