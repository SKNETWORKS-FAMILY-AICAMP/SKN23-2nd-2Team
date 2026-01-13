import numpy as np
import torch
import pandas as pd

@torch.no_grad()
def predict_noshow_proba_df(
    model,
    scaler,
    df_onehot,
    feature_cols,
    target_col="no_show",
    device=None,
    batch_size=4096,   # 메모리/속도에 맞게 조절
):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    X_df = df_onehot.copy()

    # 타겟 제거
    if target_col in X_df.columns:
        X_df = X_df.drop(columns=[target_col])

    # 컬럼 정렬/누락 보정
    X_df = X_df.reindex(columns=feature_cols, fill_value=0)

    # float32 + scaler
    X = X_df.values.astype(np.float32)
    X_scaled = scaler.transform(X)

    # 배치 추론
    probs = np.empty((X_scaled.shape[0],), dtype=np.float32)

    for start in range(0, X_scaled.shape[0], batch_size):
        end = start + batch_size
        xb = torch.tensor(X_scaled[start:end], dtype=torch.float32, device=device)
        logits = model(xb).view(-1)
        probs[start:end] = torch.sigmoid(logits).detach().cpu().numpy()

    df_with_prob = df_onehot.copy()
    df_with_prob["no_show_prob"] = probs
    
    return df_with_prob