from modules.one_hot_module import build_df_onehot
import modules.one_hot_module
import modules.connect_db_module
from modules.predict_noshow_proba_df import predict_noshow_proba_df
import pandas as pd
import joblib
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve
)

df_onehot = build_df_onehot()

# ============================
# No-show Prediction (Tabular DL) - One-shot Script
# Requirements: pandas, numpy, scikit-learn, torch
# Input: df_onehot (pandas DataFrame) with target column "no_show"
# ============================



# ----------------------------
# 1) Dataset
# ----------------------------
class TabularDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------
# 2) Model (MLP)
# ----------------------------
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


# ----------------------------
# 3) Train / Eval Utils
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_auc(model, loader, device):
    model.eval()
    preds, targets = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy()

        preds.append(prob)
        targets.append(yb.numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    return roc_auc_score(targets, preds)


# ----------------------------
# 4) Main Pipeline
# ----------------------------
def run_no_show_dl(
    df_onehot,
    target_col="no_show",
    test_size=0.2,
    random_state=42,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=20
):
    # ---- basic checks ----
    if target_col not in df_onehot.columns:
        raise ValueError(f"Target column '{target_col}' not found in df_onehot.")

    # ---- split X/y ----
    X_df = df_onehot.drop(columns=[target_col])
    y_sr = df_onehot[target_col]

    # ---- to numpy ----
    # NOTE: df_onehot이 object dtype 섞여있으면 astype에서 터질 수 있으니
    # 원-핫 완료되어 numeric/bool만 남아있다는 가정.
    feature_cols = X_df.columns.tolist()

    X = X_df.values.astype(np.float32)
    y = y_sr.values.astype(np.float32)

    # ---- train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ---- scaling ----
    # with_mean=False는 one-hot / sparse-like 행렬에서 안전
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---- to torch tensors ----
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # ---- loaders ----
    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TabularDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # ---- model / train setup ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NoShowMLP_KDY(input_dim=X_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- training ----
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        auc = eval_auc(model, test_loader, device)

        print(f"[Epoch {epoch:02d}/{epochs}] Train Loss: {train_loss:.4f} | Test AUC: {auc:.4f}")

    return model, scaler, feature_cols, train_loader, test_loader


# ----------------------------
# 5) Run (example)
# ----------------------------
model, scaler, feature_cols, train_loader, test_loader = run_no_show_dl(df_onehot, epochs=400)
@torch.no_grad()
def collect_probs_targets(model, loader, device):
    model.eval()
    probs, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy().ravel()
        probs.append(prob)
        targets.append(yb.numpy().ravel())
    return np.concatenate(probs), np.concatenate(targets)

def evaluate_binary(probs, y_true):
    roc = roc_auc_score(y_true, probs)
    pr  = average_precision_score(y_true, probs)

    # threshold sweep: F1 최대 지점 찾기
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx-1] if best_idx > 0 else 0.5  # 안전 처리

    y_pred = (probs >= best_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "best_threshold_f1": float(best_threshold),
        "confusion_matrix": cm,
        "classification_report": report
    }

# 사용 예시
device = "cuda" if torch.cuda.is_available() else "cpu"
probs, y_true = collect_probs_targets(model, test_loader, device)
metrics = evaluate_binary(probs, y_true)

@torch.no_grad()
def predict_noshow_proba_row(
    model,
    scaler,
    df_onehot,
    feature_cols,
    idx,
    target_col="no_show",
    device=None
):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # 1) 한 행 뽑기 (DataFrame 유지)
    row_df = df_onehot.loc[[idx]]

    # 2) 타겟 제거 + 컬럼 정렬/누락 보정
    if target_col in row_df.columns:
        row_df = row_df.drop(columns=[target_col])

    row_df = row_df.reindex(columns=feature_cols, fill_value=0)

    # 3) float32로 변환 → scaler transform
    x = row_df.values.astype(np.float32)
    x_scaled = scaler.transform(x)

    # 4) torch → model → sigmoid
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    logit = model(x_tensor).view(-1)          # (1,)
    proba = torch.sigmoid(logit)[0].item()    # 확률

    return proba

p = predict_noshow_proba_row(model, scaler, df_onehot, feature_cols, idx=344)


df_with_prob = predict_noshow_proba_df(model, scaler, df_onehot)
df_with_prob.iloc[0]
torch.save(model.state_dict(), "artifacts/mlp_model.pt")

os.makedirs("artifacts", exist_ok=True)

with open("artifacts/feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, ensure_ascii=False, indent=2)

os.makedirs("artifacts", exist_ok=True)

joblib.dump(scaler, "artifacts/scaler.joblib")

