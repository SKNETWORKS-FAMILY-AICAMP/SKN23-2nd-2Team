from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, r2_score
)
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.over_sampling import SMOTE


def ML_module(
        model,                          # sklearn 모델
        df,                             # 사용할 DataFrame
        target_col: str,                # 타겟 컬럼명
        feature_col: list[str] | None = None,  # 사용할 feature 컬럼명
        drop_cols: list[str] | None = None,
        test_size: float = 0.2,         # 테스트사이즈 크기
        random_state: int = 42,         # 재현성 구현
        stratify=None,                  # stratify에 쓸 y를 직접 넘기고 싶으면 y 또는 None
        threshold: float = 0.5,         # 분류에서 임계값 조정용 (0.5 기본)
        imbalance: bool = False,        # 불균형 처리 사용 여부
        smote: bool = False,            # SMOTE 적용 여부
        smote_k_neighbors: int = 5,     # SMOTE k_neighbors (기본 5)
        ):

    # 타켓 컬럼 없을 시 오류 메세지
    if target_col not in df.columns:
        raise ValueError(f'{target_col}이 없음')

    drop_cols = drop_cols or []

    # feature 컬럼 미입력시, drop 컬럼을 제외한 컬럼을 feature컬럼으로
    if feature_col is None:
        feature_col = [c for c in df.columns if c != target_col and c not in drop_cols]

    # 존재하지 않는 feature 컬럼 입력시 오류 메세지
    missing = [c for c in feature_col if c not in df.columns]
    if missing:
        raise ValueError(f"{missing}은 존재하지 않는 컬럼")

    X_df = df[feature_col].copy()
    y = df[target_col].to_numpy()

    # 학습데이터, 테스트데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    result = {
        "model": model,
        "feature_col": feature_col,
        "target_col": target_col,
    }

    # 이진분류(0/1) 여부를 먼저 확인
    uniq = set(np.unique(y_test[~np.isnan(y_test)])) if np.issubdtype(y_test.dtype, np.number) else set(np.unique(y_test))
    is_binary = uniq.issubset({0, 1, True, False}) and len(uniq) == 2

    # (SMOTE용) 학습에 실제로 넣을 X/y를 분리해서 둠
    X_fit, y_fit = X_train, y_train

    # SMOTE 적용(이진분류일 때, train 데이터에만 적용)
    if smote and is_binary:
        # SMOTE는 숫자 피처만 처리 가능
        sm = SMOTE(random_state=random_state, k_neighbors=smote_k_neighbors)
        X_fit, y_fit = sm.fit_resample(X_train, y_train)

        result["smote"] = True
        result["smote_k_neighbors"] = smote_k_neighbors

    # 불균형 처리(이진분류일 때만)
    fit_kwargs = {}

    if imbalance and is_binary:
        y_fit_int = y_fit.astype(int) if np.issubdtype(np.asarray(y_fit).dtype, np.number) else y_fit
        pos = int(np.sum(y_fit_int == 1))
        neg = int(np.sum(y_fit_int == 0))

        if pos > 0 and neg > 0:
            pos_weight = neg / pos  # 음성/양성 비율
            result["pos_weight"] = pos_weight  # 참고용 기록

            # class_weight 지원 모델이면 class_weight로 처리
            if hasattr(model, "get_params") and hasattr(model, "set_params") and ("class_weight" in model.get_params()):
                model.set_params(class_weight={0: 1.0, 1: pos_weight})

            # scale_pos_weight 지원 모델이면 scale_pos_weight로 처리 (XGBoost/LightGBM sklearn API)
            elif hasattr(model, "get_params") and hasattr(model, "set_params") and ("scale_pos_weight" in model.get_params()):
                model.set_params(scale_pos_weight=pos_weight)

            # fit에 sample_weight를 넣을 수 있으면 sample_weight로 처리
            else:
                sample_weight = np.where(y_fit_int == 1, pos_weight, 1.0)
                fit_kwargs["sample_weight"] = sample_weight

    # 모델 학습은 (SMOTE 적용 시) X_fit, y_fit로 수행
    model.fit(X_fit, y_fit, **fit_kwargs)

    # 이진분류면 threshold로 예측값 생성 (predict 대신 predict_proba 사용)
    if is_binary and hasattr(model, "predict_proba"):  # predict_proba 지원 시
        p_train = model.predict_proba(X_train)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
        y_pred_train = (p_train >= threshold).astype(int)
        y_pred_test = (p_test >= threshold).astype(int)
        result["threshold"] = threshold                    # 사용한 임계값 기록
    else:
        # (회귀/멀티클래스/확률 미지원) 기존처럼 predict 사용
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

    # train_score / test_score는 model.score 대신 "현재 예측값 기준"으로 계산
    # (threshold를 바꾸면 model.score(기본 0.5 predict)와 값이 달라질 수 있어서)
    if is_binary:
        result["train_score"] = accuracy_score(y_train, y_pred_train)
        result["test_score"] = accuracy_score(y_test, y_pred_test)
    else:
        # 회귀/기타는 기존 score 유지
        result["train_score"] = model.score(X_train, y_train)
        result["test_score"] = model.score(X_test, y_test)

    # 이진분류(0/1)일 때만 precision/recall/f1 계산
    if is_binary:
        result["train_precision"] = precision_score(y_train, y_pred_train, zero_division=0)
        result["train_recall"] = recall_score(y_train, y_pred_train, zero_division=0)
        result["train_f1"] = f1_score(y_train, y_pred_train, zero_division=0)

        result["test_precision"] = precision_score(y_test, y_pred_test, zero_division=0)
        result["test_recall"] = recall_score(y_test, y_pred_test, zero_division=0)
        result["test_f1"] = f1_score(y_test, y_pred_test, zero_division=0)

    # 회귀일 때만 R2 계산(타깃이 0/1이면 의미 없어서 계산 제외)
    if not is_binary:
        if np.issubdtype(y_test.dtype, np.number):
            result["train_r2"] = r2_score(y_train, y_pred_train)
            result["test_r2"] = r2_score(y_test, y_pred_test)

    return result


# LightGBM 모델
def LGBM_module(
    df,
    target_col: str,
    feature_col: list[str] | None = None,
    drop_cols: list[str] | None = None,
    task: str = "classifier",        # "classifier" 또는 "regressor"
    lgbm_params: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify=None,
    threshold: float = 0.5,          # 분류에서 임계값 조정용
    imbalance: bool = False,         # 불균형 처리 사용 여부
    smote: bool = False,             # SMOTE 적용 여부
    smote_k_neighbors: int = 5,      # SMOTE k_neighbors
):
    lgbm_params = lgbm_params or {}

    # 타켓 컬럼 없을 시 오류 메세지
    drop_cols = drop_cols or []
    if task == "classifier":
        model = LGBMClassifier(random_state=random_state, **lgbm_params)
    elif task == "regressor":
        model = LGBMRegressor(random_state=random_state, **lgbm_params)
    else:
        raise ValueError('task는 "classifier" 또는 "regressor"')

    # 동일한 학습/평가 흐름 재사용
    result = ML_module(
        model=model,
        df=df,
        target_col=target_col,
        feature_col=feature_col,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        drop_cols=drop_cols,
        threshold=threshold,
        imbalance=imbalance,
        smote=smote,
        smote_k_neighbors=smote_k_neighbors,
    )

    return result

# XGBoost 모델
def XGB_module(
    df,
    target_col: str,
    feature_col=None,
    drop_cols=None,
    task="classifier",
    xgb_params=None,
    test_size=0.2,
    random_state=42,
    stratify=None,
    threshold=0.5,
    imbalance=True,
    smote=True,
    smote_k_neighbors=5,
):
    xgb_params = xgb_params or {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    if task == "classifier":
        model = XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            **xgb_params
        )
    else:
        model = XGBRegressor(random_state=random_state, **xgb_params)

    return ML_module(
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
    
# 랜덤포레스트 모델
def RF_module(
    df,
    target_col: str,
    feature_col: list[str] | None = None,
    drop_cols: list[str] | None = None,
    task: str = "classifier",
    rf_params: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify=None,
    threshold: float = 0.5,
    imbalance: bool = False,
    smote: bool = False,
    smote_k_neighbors: int = 5,
):
    rf_params = rf_params or {}
    drop_cols = drop_cols or []

    if task == "classifier":
        model = RandomForestClassifier(
            random_state=random_state,
            **rf_params
        )
    elif task == "regressor":
        model = RandomForestRegressor(
            random_state=random_state,
            **rf_params
        )
    else:
        raise ValueError('task는 "classifier" 또는 "regressor"')

    return ML_module(
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
