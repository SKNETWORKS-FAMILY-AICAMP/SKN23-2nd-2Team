import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from src.services.customerService import load_artifacts, load_dl_eval_artifacts
from src.modules.one_hot_module import rows_to_df_onehot
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_fscore_support, roc_curve, precision_recall_curve
)

BASE_DIR = Path(__file__).resolve().parents[2]  # deepTap.py 기준이면 보통 src/.. 조정 필요
ART_DIR = BASE_DIR / "src" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL = "no_show"   # 너희 타겟 컬럼명에 맞게 수정

def build_preds_from_eval_df(eval_df: pd.DataFrame):
    model, scaler, feature_cols = load_artifacts()

    if TARGET_COL not in eval_df.columns:
        raise ValueError(f"eval_df에 타겟 컬럼 '{TARGET_COL}'이 없습니다.")

    y_true = eval_df[TARGET_COL].astype(int).to_numpy()

    # ✅ 너희 프로젝트 함수 사용
    X_oh = rows_to_df_onehot(eval_df).reindex(columns=feature_cols, fill_value=0)
    Xs = scaler.transform(X_oh.values.astype(np.float32))

    with torch.no_grad():
        xt = torch.tensor(Xs, dtype=torch.float32, device="cpu")
        logit = model(xt).view(-1)
        y_proba = torch.sigmoid(logit).detach().cpu().numpy()

    df_pred = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    return df_pred

@st.cache_data
def load_dl_eval_artifacts():
    pred_path = ART_DIR / "preds.parquet"
    eval_path = ART_DIR / "eval_df.parquet"

    df_pred = pd.read_parquet(pred_path) if pred_path.exists() else None
    eval_df = pd.read_parquet(eval_path) if eval_path.exists() else None

    hist_path = ART_DIR / "hist.json"
    hist = pd.read_json(hist_path) if hist_path.exists() else None

    return eval_df, df_pred, hist

def compute_metrics(y_true, y_proba, thr):
    y_pred = (y_proba >= thr).astype(int)

    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return auc, ap, p, r, f1, cm

def render_confusion_matrix(cm):
    """
    cm: confusion_matrix 결과 (2x2)
    """

    st.space()

    # 보기 좋게 표도 같이
    st.dataframe(
        pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
        width="stretch"
    )

    st.space()

    with st.container(width="stretch", horizontal=True, horizontal_alignment="center"):
        with st.container(width=600):
            fig, ax = plt.subplots(figsize = (4, 4))
            im = ax.imshow(cm)  # 색 지정 안 함 (기본)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["0", "1"])
            ax.set_yticklabels(["0", "1"])

            # 숫자 표시
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center")

            st.pyplot(fig)


def render_roc_pr_curves(y_true, y_proba):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    
    st.markdown("**ROC Curve**")
    st.line_chart(roc_df.set_index("FPR"))  # y=TPR 자동

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_df = pd.DataFrame({"Recall": rec, "Precision": prec})

    st.divider()

    st.markdown("**Precision–Recall Curve**")
    st.line_chart(pr_df.set_index("Recall"))  # y=Precision 자동

def render_threshold_sweep(y_true, y_proba):
    thrs = np.linspace(0, 1, 101)
    ps, rs, f1s = [], [], []

    for t in thrs:
        y_pred = (y_proba >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        ps.append(p); rs.append(r); f1s.append(f1)

    df = pd.DataFrame({"thr": thrs, "precision": ps, "recall": rs, "f1": f1s})
    st.line_chart(df.set_index("thr"))

def render_loss_curve(hist):
    st.markdown("### Training Curves")

    if hist is None:
        st.info("Training history(hist)가 없습니다. (hist.json 미생성)")
        return

    if isinstance(hist, dict):
        df_hist = pd.DataFrame(hist)
    else:
        df_hist = hist.copy()

    if "epoch" not in df_hist.columns:
        df_hist["epoch"] = np.arange(1, len(df_hist) + 1)

    cols = [c for c in ["train_loss", "val_loss", "val_acc", "val_auc"] if c in df_hist.columns]
    if not cols:
        st.info("hist에 표시할 컬럼(train_loss/val_loss 등)이 없습니다.")
        return

    st.line_chart(df_hist.set_index("epoch")[cols])

def render_deep_learning_tab():
    eval_df, df_pred, hist = load_dl_eval_artifacts()

    st.subheader("1️⃣ Evaluation dataset")

    if eval_df is None:
        up = st.file_uploader("Upload evaluation dataset (csv/parquet)", type=["csv", "parquet"])

        if up is not None:
            if up.name.endswith(".csv"):
                eval_df = pd.read_csv(up)
            else:
                eval_df = pd.read_parquet(up)

            # 저장
            eval_df.to_parquet(ART_DIR / "eval_df.parquet", index=False)
            st.success("Saved eval_df.parquet")
            st.cache_data.clear()  # 새 파일 저장했으니 캐시 갱신
            st.rerun()
        else:
            st.info("평가용 데이터셋을 업로드하면 성능 지표를 계산할 수 있어요.")
            st.stop()

    _, col1, col2 = st.columns([0.3, 3, 7], gap="medium")

    with col1:
        st.space("large")
        col1.metric(label="eval_df shape", value=str(eval_df.shape))
    
    with col2:
        st.write("columns:", list(eval_df.columns))

    st.divider()

    st.subheader("2️⃣ Predictions (y_true / y_proba)")

    if df_pred is None:
        if st.button("Generate preds.parquet (run inference once)"):
            df_pred = build_preds_from_eval_df(eval_df)
            df_pred.to_parquet(ART_DIR / "preds.parquet", index=False)
            st.success("Saved preds.parquet")
            st.cache_data.clear()
            st.rerun()
        else:
            st.info("preds.parquet이 없어요. 버튼을 눌러 한 번만 생성하면 됩니다.")
            st.stop()

    y_true = df_pred["y_true"].to_numpy()
    y_proba = df_pred["y_proba"].to_numpy()

    with st.container(key="deep_slider_container", width="stretch", border=True, gap="large"):
        thr = st.slider("Decision threshold", 0.0, 1.0, 0.35, 0.01)
        auc, ap, p, r, f1, cm = compute_metrics(y_true, y_proba, thr)

        _, c1, c2, c3, c4, c5, c6 = st.columns([0.3, 1, 1, 1, 1, 1, 1])
        c1.metric("ROC-AUC", f"{auc:.3f}")
        c2.metric("PR-AUC", f"{ap:.3f}")
        c3.metric("F1 (pos)", f"{f1:.3f}")
        c4.metric("Recall (pos)", f"{r:.3f}")
        c5.metric("Precision (pos)", f"{p:.3f}")
        c6.metric("Threshold", f"{thr:.2f}")

    st.space("medium")

    with st.expander("Confusion Matrix", expanded=True):
        render_confusion_matrix(cm)

    with st.expander("Curves", expanded=True):
        render_roc_pr_curves(y_true, y_proba)

    with st.expander("Threshold Sweep (Precision / Recall / F1)", expanded=True):
        render_threshold_sweep(y_true, y_proba)
