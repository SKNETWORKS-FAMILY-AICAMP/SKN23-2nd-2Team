from sklearn.model_selection import train_test_split


def ML_module(
        model,                  # sklearn 모델
        df,                     # 사용할 DataFrame
        target_col: str,                                       # 타겟 컬럼명
        feature_col: list[str] | None = None,            # 사용할 feature 컬럼명
        drop_cols: list[str] | None = None,
        test_size: float = 0.2,                          # 테스트사이즈 크기
        random_state: int = 42,         # 재현성 구현
        stratify=None,                  # stratify에 쓸 y를 직접 넘기고 싶으면 y 또는 None
        ):

    if target_col not in df.columns:
        raise ValueError(f'{target_col}이 없음')

    drop_cols = drop_cols or []

    if feature_col is None:
        feature_col = [c for c in df.columns if c != target_col and c not in drop_cols]

    missing = [c for c in feature_col if c not in df.columns]

    if missing:
        raise ValueError(f"존재하지 않는 컬럼")

    X_df = df[feature_col].copy()
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    model.fit(X_train, y_train)

    return {
        "model": model,
        "feature_col": feature_col,
        "target_col": target_col,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_score": model.score(X_train, y_train),
        "test_score": model.score(X_test, y_test),
    }
