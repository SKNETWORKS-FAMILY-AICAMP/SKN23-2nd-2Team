from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

def ML_module(
    model,
    df,
    target_col,
    feature_col=None,
    drop_cols=None,
    test_size=0.2,
    random_state=42,
    stratify=None,
    threshold=0.5,
    imbalance=False,
    smote=False,
    smote_k_neighbors=5,
):
    rop_cols = drop_cols or []

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
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # 평가 지표
    train_pred = model.predict(X_train)
    train_score = accuracy_score(y_train, train_pred)
    test_score = accuracy_score(y_test, y_pred)
    train_f1 = f1_score(y_train, train_pred)
    test_f1 = f1_score(y_test, y_pred)
    train_precision = precision_score(y_train, train_pred)
    test_precision = precision_score(y_test, y_pred)
    train_recall = recall_score(y_train, train_pred)
    test_recall = recall_score(y_test, y_pred)

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
        "y_test": y_test,      # 추가
        "y_pred": y_pred,      # 추가
        "y_proba": y_pred_proba  # 추가
    }


# RF_module
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
        model = RandomForestClassifier(random_state=random_state, **rf_params)
    elif task == "regressor":
        model = RandomForestRegressor(random_state=random_state, **rf_params)
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