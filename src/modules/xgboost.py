from modules.one_hot_module import build_df_onehot
from modules.xgboost import XGB_module
from modules.randomForest import RF_module
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# ML_module
def ML_module(
    model,
    df,
    target_col: str,
    feature_col: list[str] | None = None,
    drop_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify=None,
    threshold: float = 0.5,
    imbalance: bool = False,
    smote: bool = False,
    smote_k_neighbors: int = 5,
):
    drop_cols = drop_cols or []

    # feature 선택
    if feature_col is None:
        X = df.drop(columns=[target_col] + drop_cols)
    else:
        X = df[feature_col]
    y = df[target_col]

    # train/test split
    stratify_data = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_data
    )

    # SMOTE 적용
    if smote:
        sm = SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # 모델 학습
    model.fit(X_train, y_train)

    # 예측
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # 학습용 예측
    train_pred = model.predict(X_train)

    # 평가 지표
    train_score = accuracy_score(y_train, train_pred)
    test_score = accuracy_score(y_test, y_pred)
    train_f1 = f1_score(y_train, train_pred)
    test_f1 = f1_score(y_test, y_pred)
    train_precision = precision_score(y_train, train_pred)
    test_precision = precision_score(y_test, y_pred)
    train_recall = recall_score(y_train, train_pred)
    test_recall = recall_score(y_test, y_pred)

    # 결과 반환
    return {
        "model": model,
        "feature_col": feature_col,
        "target_col": target_col,
        "train_score": train_score,
        "test_score": test_score,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_pred_proba,
    }


# XGB_module
def XGB_module(
    df,
    target_col: str,
    feature_col: list[str] | None = None,
    drop_cols: list[str] | None = None,
    task: str = "classifier",
    xgb_params: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify=None,
    threshold: float = 0.5,
    imbalance: bool = False,
    smote: bool = False,
    smote_k_neighbors: int = 5,
):
    xgb_params = xgb_params or {}
    drop_cols = drop_cols or []

    # 불균형 처리 (scale_pos_weight)
    if task == "classifier" and imbalance:
        y = df[target_col]
        n_pos = sum(y == 1)
        n_neg = sum(y == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        xgb_params["scale_pos_weight"] = scale_pos_weight
        print(f"[INFO] scale_pos_weight 적용: {scale_pos_weight:.2f}")

    # 모델 정의
    if task == "classifier":
        model = XGBClassifier(
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            **xgb_params
        )
    elif task == "regressor":
        model = XGBRegressor(
            random_state=random_state,
            **xgb_params
        )
    else:
        raise ValueError('task는 "classifier" 또는 "regressor"')

    # ML_module 호출
    ml_result = ML_module(
        model=model,
        df=df,
        target_col=target_col,
        feature_col=feature_col,
        drop_cols=drop_cols,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        threshold=threshold,
        imbalance=imbalance,
        smote=smote,
        smote_k_neighbors=smote_k_neighbors,
    )

    return ml_result
