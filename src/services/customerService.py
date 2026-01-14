import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
import streamlit as st
from src.modules.predict_noshow_proba_df import predict_noshow_proba_df
from src.modules.one_hot_module import rows_to_df_onehot, fetch_df, SPECIALTY_KO_MAP
"""
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
        return self.net(x)
"""
class NoShowMLP_KDY(nn.Module):
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
        return self.net(x)

@st.cache_resource
def load_artifacts():
    with open("src/artifacts/feature_columns.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    scaler = joblib.load("src/artifacts/scaler.joblib")
    input_dim = len(feature_cols)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NoShowMLP_KDY(input_dim=input_dim)
    state = torch.load("src/artifacts/mlp_model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, scaler, feature_cols

@st.cache_data
def load_dl_eval_artifacts(eval_df: pd.DataFrame):
    """
    eval_df: y_true를 포함한 평가용 데이터프레임 (예: test set)
    columns 예시: [... feature cols ..., "no_show"]
    """
    pred_path = "src/artifacts/preds.parquet"
    hist_path = "src/artifacts/hist.json"

    # ✅ 있으면 로드
    if os.path.exists(pred_path):
        df_pred = pd.read_parquet(pred_path)
    else:
        # ✅ 없으면 생성
        model, scaler, feature_cols = load_artifacts()

        # y_true 준비 (너 타겟 컬럼명에 맞춰 수정)
        y_true = eval_df["no_show"].astype(int).to_numpy()

        # X 만들기 (너희 원핫 함수 활용)
        X = rows_to_df_onehot(eval_df).reindex(columns=feature_cols, fill_value=0)
        Xs = scaler.transform(X.values.astype(np.float32))

        # model inference (pytorch MLP 기준)
        device = next(model.parameters()).device
        with torch.no_grad():
            xt = torch.tensor(Xs, dtype=torch.float32, device=device)
            logit = model(xt).view(-1)
            y_proba = torch.sigmoid(logit).detach().cpu().numpy()

        df_pred = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
        df_pred.to_parquet(pred_path, index=False)

    # hist는 없을 수도 있으니 안전 처리
    hist = None
    if os.path.exists(hist_path):
        hist = pd.read_json(hist_path)

    return df_pred, hist

@st.cache_data
def get_chart_data(_model, _scaler, limit = 40):
    rows = fetch_df("appointment", limit=limit)
    weather = fetch_df("weather", limit = limit)
    df = rows_to_df_onehot(rows)

    rows["appointment_date"] = pd.to_datetime(rows["appointment_date"]).dt.date
    weather["weather_date"] = pd.to_datetime(weather["weather_date"]).dt.date

    no_show_prob = predict_noshow_proba_df(_model, _scaler, df)["no_show_prob"]
    rows["no_show_prob"] = no_show_prob * 100
    rows = rows.merge(
        weather,
        left_on="appointment_date",
        right_on="weather_date",
        how="left"
    )
    rows = rows.drop(columns=["weather_date"])
    return rows

# 회원 정보 수정 후 노쇼율 재계산
@st.cache_data
def update_customer_info(_model, _scaler, dataframe):
    df = rows_to_df_onehot(dataframe)
    no_show_prob = predict_noshow_proba_df(_model, _scaler, df)["no_show_prob"]
    dataframe["no_show_prob"] = no_show_prob * 100

    return dataframe

# 검색바(필터)를 이용한 검색
def search_filters(age_filter, dept_filter, risk_filter):
    print(age_filter, dept_filter, risk_filter)
    st.session_state.page_num = 1
    df = st.session_state.org_data.copy()

    if age_filter != "전체":
        if age_filter == "50대 이상":
            df = df[df["age"] >= 50]
        elif age_filter == "10대 미만":
            df = df[df["age"] < 10]
        else:
            base = int(age_filter.replace("대", ""))
            df = df[(df["age"] >= base) & (df["age"] < base + 10)]

    if dept_filter != "전체":
        reverse_specialty_map = {v: k for k, v in SPECIALTY_KO_MAP.items()}
        selected_specialty_en = reverse_specialty_map.get(dept_filter)

        if selected_specialty_en:
            df = df[df["specialty"] == selected_specialty_en]

    if risk_filter != "전체":
        if risk_filter == "고위험":
            df = df[df["no_show_prob"] >= 20]
        elif risk_filter == "중위험":
            df = df[(df["no_show_prob"] >= 10) & (df["no_show_prob"] < 20)]
        elif risk_filter == "저위험":
            df = df[df["no_show_prob"] < 10]

    return df
