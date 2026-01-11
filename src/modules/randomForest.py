from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from modules.machine_module import ML_module

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
