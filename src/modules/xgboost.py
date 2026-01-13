from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from modules.machine_module import ML_module
import numpy as np

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
    imbalance: bool = False,   # 불균형 처리
    smote: bool = False,
    smote_k_neighbors: int = 5,
):
    xgb_params = xgb_params or {}
    drop_cols = drop_cols or []

    # 불균형 처리
    if task == "classifier" and imbalance:
        # target 컬럼에서 양성/음성 비율 계산
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

    # 이미 train/test 성능 지표가 있으면 그대로 사용
    metrics_keys = ['train_score', 'test_score', 'train_precision', 'train_recall',
                    'train_f1', 'test_precision', 'test_recall', 'test_f1']
    metrics = {k: ml_result[k] for k in metrics_keys if k in ml_result}

    # 최종 리턴
    return {**ml_result, "metrics": metrics}
    