import json
import torch
import joblib
import torch.nn as nn
import streamlit as st
from src.modules.predict_noshow_proba_df import predict_noshow_proba_df
from src.modules.one_hot_module import rows_to_df_onehot, fetch_df
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
def get_customer_list(_model, _scaler, limit = 40):
    rows = fetch_df("appointment", limit=limit)
    df = rows_to_df_onehot(rows)
    no_show_prob = predict_noshow_proba_df(_model, _scaler, df)["no_show_prob"]
    rows["no_show_prob"] = no_show_prob * 100

    return rows