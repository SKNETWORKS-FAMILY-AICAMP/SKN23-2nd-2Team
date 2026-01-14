import json, joblib, re
import pandas as pd


def sanitize_one(name: str) -> str:
    s = str(name)
    s = re.sub(r"[\x00-\x1f\x7f]", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"""["'\\{}\[\]:,]""", "_", s)
    s = re.sub(r"_+", "_", s).strip("_") or "col"
    return s


MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _set_onehot_row(X: pd.DataFrame, row_i: int, base: str, value: str,
                    unknown_col: str | None = None):
    v = "" if value is None else str(value).strip()

    if v == "":
        if unknown_col and unknown_col in X.columns:
            X.iat[row_i, X.columns.get_loc(unknown_col)] = 1
        return

    candidates = [
        f"{base}_{v}",
        sanitize_one(f"{base}_{v}")
    ]
    for c in candidates:
        if c in X.columns:
            X.iat[row_i, X.columns.get_loc(c)] = 1
            return

    if unknown_col and unknown_col in X.columns:
        X.iat[row_i, X.columns.get_loc(unknown_col)] = 1


def predict_no_show_prob_lgbm(
    new_raw: pd.DataFrame,
    model_path="artifacts/lgbm_model.joblib",
    feature_cols_path="artifacts/lgbm_feature_columns.json"
):

    model = joblib.load(model_path)
    with open(feature_cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    X = pd.DataFrame(0, index=new_raw.index, columns=feature_cols, dtype=float)

    # 숫자형/이진형: 학습 피처에 있는 건 다 채움
    numeric_cols = [
        "gender", "age", "patient_needs_companion",
        "under_12_years_old", "over_60_years_old",
        "days_since_first_visit",
        "average_temp_day", "max_temp_day",
        "average_rain_day", "max_rain_day",
        "storm_day_before",
    ]
    for col in numeric_cols:
        if col in new_raw.columns and col in X.columns:
            X[col] = pd.to_numeric(new_raw[col], errors="coerce").fillna(0)

    # 요일 원핫
    if "appointment_date" in new_raw.columns:
        d = pd.to_datetime(new_raw["appointment_date"], errors="coerce")
        w = d.dt.strftime("%a")
        for i, wd in enumerate(w):
            if pd.isna(wd):
                continue
            c = f"appointment_date_{wd}"
            if c in X.columns:
                X.iat[i, X.columns.get_loc(c)] = 1

    # 월 원핫
    if "appointment_datetime" in new_raw.columns:
        dt = pd.to_datetime(new_raw["appointment_datetime"], errors="coerce")
        m = dt.dt.month
        for i, mm in enumerate(m):
            if pd.isna(mm):
                continue
            abbr = MONTH_ABBR[int(mm) - 1]
            c = f"appointment_datetime_{abbr}"
            if c in X.columns:
                X.iat[i, X.columns.get_loc(c)] = 1

    # specialty 원핫 (원본/ sanitize 둘 다 대응)
    if "specialty" in new_raw.columns:
        for i, v in enumerate(new_raw["specialty"].fillna("").astype(str)):
            _set_onehot_row(X, i, "specialty", v, unknown_col="specialty_unknown")

    # disability 원핫 (빈값이면 disability_null)
    if "disability" in new_raw.columns:
        for i, v in enumerate(new_raw["disability"].fillna("").astype(str)):
            v = v.strip()
            if v == "":
                if "disability_null" in X.columns:
                    X.iat[i, X.columns.get_loc("disability_null")] = 1
                continue
            parts = [p.strip() for p in re.split(r"[,\|;/]+", v) if p.strip()]
            for p in parts:
                _set_onehot_row(X, i, "disability", p, unknown_col="disability_null")

    # icd 원핫 (빈값이면 icd_null) - 대/소문자 정규화도 같이
    if "icd" in new_raw.columns:
        for i, v in enumerate(new_raw["icd"].fillna("").astype(str)):
            v = v.strip()
            if v == "":
                if "icd_null" in X.columns:
                    X.iat[i, X.columns.get_loc("icd_null")] = 1
                continue
            v = v.lower()  # 학습 피처가 icd_f71 처럼 소문자라면 필요
            # 원본/ sanitize 둘 다
            candidates = [f"icd_{v}", sanitize_one(f"icd_{v}")]
            placed = False
            for c in candidates:
                if c in X.columns:
                    X.iat[i, X.columns.get_loc(c)] = 1
                    placed = True
                    break
            if not placed and "icd_null" in X.columns:
                X.iat[i, X.columns.get_loc("icd_null")] = 1

    proba = model.predict_proba(X)[:, 1]
    out = new_raw.copy()
    out["no_show_prob"] = proba
    return out
