# - 모든 차트 함수는 plt.show() 대신 matplotlib Figure를 반환
# - Streamlit에서는 st.pyplot(fig)로 출력하면 됨

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from imblearn.over_sampling import SMOTE


# 0) 컬럼명 sanitize
def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    seen = {}
    for c in df.columns:
        c2 = str(c).strip()
        c2 = re.sub(r"[^0-9a-zA-Z_]+", "_", c2)     # 영문/숫자/_ 외는 _
        c2 = re.sub(r"_{2,}", "_", c2).strip("_")   # __ -> _
        if c2 == "":
            c2 = "col"

        # 중복 처리
        if c2 in seen:
            seen[c2] += 1
            c2 = f"{c2}_{seen[c2]}"
        else:
            seen[c2] = 0

        new_cols.append(c2)

    df.columns = new_cols
    return df


# 1) feature 컬럼 결정 + datetime 제거
def resolve_feature_cols(df: pd.DataFrame, target_col: str, feature_col=None, drop_cols=None):
    drop_cols = drop_cols or []

    if feature_col is None:
        feature_col = [c for c in df.columns if c != target_col and c not in drop_cols]

    # datetime 컬럼 제거 (Timestamp -> float 에러 방지)
    dt_cols = df[feature_col].select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    feature_col = [c for c in feature_col if c not in dt_cols]

    missing = [c for c in feature_col if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature cols: {missing}")

    return feature_col


# 2) split/fit/predict_proba (시각화용)
def fit_for_viz(
    model,
    df: pd.DataFrame,
    target_col: str = "no_show",
    feature_col=None,
    drop_cols=None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    threshold: float = 0.5,
    fillna_zero: bool = False,
    smote: bool = False,
    smote_k_neighbors: int = 5,
):
    feature_col = resolve_feature_cols(df, target_col, feature_col, drop_cols)

    X = df[feature_col].copy()
    y = df[target_col].to_numpy()

    # 숫자화/결측 처리 (SMOTE 쓰면 NaN 있으면 안 되므로 0 처리 권장)
    if fillna_zero or smote:
        X = X.replace(r"^\s*$", np.nan, regex=True)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(0)

    stratify_y = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )

    # SMOTE는 train에만 적용
    if smote:
        # minority count가 너무 적으면 SMOTE 실패하므로 가드
        y_train_int = pd.Series(y_train).astype(int)
        pos = int((y_train_int == 1).sum())
        neg = int((y_train_int == 0).sum())
        minority = min(pos, neg)

        if minority >= 2:
            k = min(smote_k_neighbors, minority - 1)
            sm = SMOTE(random_state=random_state, k_neighbors=k)
            X_train, y_train = sm.fit_resample(X_train, y_train)

    m = clone(model)
    m.fit(X_train, y_train)

    p_test = m.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= threshold).astype(int)

    return {
        "model": m,
        "feature_col": feature_col,
        "y_test": y_test,
        "p_test": p_test,
        "cm": confusion_matrix(y_test, y_pred),
        "metrics": {
            "AUC": roc_auc_score(y_test, p_test),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
        }
    }


# 3) 차트 함수들 (각각 Figure 반환)
def plot_roc(y_test, p_test, title: str):
    fpr, tpr, thr = roc_curve(y_test, p_test)
    auc = roc_auc_score(y_test, p_test)

    # Youden J 최대 threshold
    j = tpr - fpr
    k = int(np.argmax(j))
    best_thr = float(thr[k])

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.scatter([fpr[k]], [tpr[k]])
    ax.text(fpr[k], tpr[k], f" thr={best_thr:.3f}", va="bottom")

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig  # 필요하면 best_thr/auc도 같이 리턴하도록 변경 가능


def plot_pr(y_test, p_test, title: str):
    prec, rec, thr = precision_recall_curve(y_test, p_test)
    base = float(np.mean(y_test))

    # thr 길이 = len(prec)-1
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    k = int(np.argmax(f1))
    best_thr = float(thr[k]) if len(thr) > 0 else 0.5
    best_p, best_r, best_f1 = float(prec[k]), float(rec[k]), float(f1[k])

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(rec, prec, label="PR curve")
    ax.axhline(base, linestyle="--", label=f"baseline={base:.3f}")

    ax.scatter([best_r], [best_p])
    ax.text(
        best_r, best_p,
        f" thr={best_thr:.3f}\nP={best_p:.3f}, R={best_r:.3f}, F1={best_f1:.3f}",
        va="bottom"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig  # 필요하면 best_thr/점수도 같이 리턴하도록 변경 가능



def plot_lgbm_importance(fitted_lgbm, feature_names, topk: int = 20, title: str = "LightGBM feature importance"):
    imp = fitted_lgbm.feature_importances_
    s = pd.Series(imp, index=feature_names).sort_values(ascending=False).head(topk)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(s.index, s.values)
    ax.set_title(title)

    for b in bars:
        w = b.get_width()
        ax.text(w, b.get_y() + b.get_height() / 2, f"{int(w)}", va="center")

    fig.tight_layout()
    return fig

# 4) 편의 함수: LR/LGBM 기본 모델 생성
def make_default_models():
    lr_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    lgbm_model = LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05)
    return lr_model, lgbm_model


# 5) 편의 함수: 두 모델 평가 결과 만들기 (차트는 별도 호출)
def run_lr_lgbm(
    df: pd.DataFrame,
    target_col: str = "no_show",
    drop_cols=None,
    threshold: float = 0.5,
    random_state: int = 42,
    smote: bool = True,
    smote_k_neighbors: int = 5,
):
    drop_cols = drop_cols or ["appointment_id", "name"]

    lr_model, lgbm_model = make_default_models()

    lr_out = fit_for_viz(
        model=lr_model,
        df=df,
        target_col=target_col,
        drop_cols=drop_cols,
        stratify=True,
        threshold=threshold,
        fillna_zero=True,
        smote=smote,
        smote_k_neighbors=smote_k_neighbors,
        random_state=random_state,
    )

    lgbm_out = fit_for_viz(
        model=lgbm_model,
        df=df,
        target_col=target_col,
        drop_cols=drop_cols,
        stratify=True,
        threshold=threshold,
        fillna_zero=False,
        smote=smote,
        smote_k_neighbors=smote_k_neighbors,
        random_state=random_state,
    )

    summary = pd.DataFrame([
        {"model": "LogisticRegression", **lr_out["metrics"]},
        {"model": "LightGBM", **lgbm_out["metrics"]},
    ])

    return summary, lr_out, lgbm_out
