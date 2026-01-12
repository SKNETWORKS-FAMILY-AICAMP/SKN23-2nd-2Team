import re
import numpy as np
import pandas as pd
from faker import Faker

def to_snake(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[’'“”\"]", "", s)
    s = re.sub(r"[^0-9a-zA-Z\s_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


def standardize_str_series(sr: pd.Series) -> pd.Series:
    return (
        sr.astype("string")
          .str.strip()
          .str.replace(r"\s+", " ", regex=True)
          .str.lower()
          .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
    )


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        c = to_snake(c)
        if c in df.columns:
            return c
    return None


def clean_for_db_no_new_cols(df: pd.DataFrame, drop_leak_cols: bool = False) -> pd.DataFrame:
    """
    DB 적재 전 정제(컬럼 수 증가 없음)
    - 컬럼명 snake_case
    - 문자열 값 표준화
    - 날짜/시간 표준 포맷 문자열로 변환(기존 컬럼 값만 변경)
    - 의미 있는 결측(ICD/Disability/No-show Reason 등)은 유지
    - 나머지 결측은 채움
    - 컬럼 추가 없음
    - (선택) leakage 컬럼 삭제 가능
    """
    df = df.copy()
    df.columns = [to_snake(c) for c in df.columns]

    # 주요 컬럼 탐색
    col_no_show = first_existing(df, ["No-show", "no_show", "noshow"])
    col_no_show_reason = first_existing(df, ["No-show Reason", "no_show_reason", "noshow_reason"])
    col_icd = first_existing(df, ["ICD", "icd"])
    col_disability = first_existing(df, ["Disability", "disability"])
    col_gender = first_existing(df, ["Gender", "gender"])

    col_appt_date = first_existing(df, ["Appointment Date", "appointment_date"])
    col_appt_time = first_existing(df, ["Appointment Time", "appointment_time"])
    col_dob = first_existing(df, ["Date of Birth", "date_of_birth", "dob"])
    col_entry = first_existing(df, ["Date of Entry into the Service", "date_of_entry_into_the_service", "entry_service_date"])

    # (선택) leakage 컬럼 삭제
    if drop_leak_cols and col_no_show_reason is not None:
        df = df.drop(columns=[col_no_show_reason])

    # 문자열 컬럼 값 표준화
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        df[c] = standardize_str_series(df[c])

    # 타깃(no_show) 표준화(yes/no)
    if col_no_show is not None:
        df[col_no_show] = (
            df[col_no_show]
            .astype("string").str.strip().str.lower()
            .replace({"y": "yes", "n": "no", "true": "yes", "false": "no", "1": "yes", "0": "no"})
        )

    # 날짜/시간 표준 포맷으로 변환 (컬럼 추가 없이 값만 바꿈)
    def to_date_str(series: pd.Series) -> pd.Series:
        dt = pd.to_datetime(
            series,
            format="%d/%m/%Y",
            errors="coerce"
        )
        return dt.dt.strftime("%Y-%m-%d")

    def to_time_str(series: pd.Series) -> pd.Series:
        dt = pd.to_datetime(
            series,
            errors="coerce"
        )
        return dt.dt.strftime("%H:%M:%S")

    for c in [col_appt_date, col_dob, col_entry]:
        if c is not None and c in df.columns:
            df[c] = to_date_str(df[c])

    if col_appt_time is not None and col_appt_time in df.columns:
        df[col_appt_time] = to_time_str(df[col_appt_time])

    # appointment_month: 'oct' -> 10 (컬럼 추가 없이 값만 정규화)
    month_col = first_existing(df, ["Appointment Month", "appointment_month"])
    if month_col is not None and month_col in df.columns:
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        m = df[month_col]
        m_num = pd.to_numeric(m, errors="coerce")
        m_txt = m.astype("string").str.strip().str.lower().map(month_map)
        df[month_col] = m_num.fillna(m_txt)

    # 성별 값 표준화(있으면)
    if col_gender is not None and col_gender in df.columns:
        df[col_gender] = df[col_gender].replace({"m": "male", "f": "female", "man": "male", "woman": "female"})

    # --- 결측치 처리 정책 ---
    # 결측 자체가 의미 있을 수 있는 컬럼: 채우지 않음
    meaningful_missing = set([c for c in [col_icd, col_disability] if c is not None])
    # no_show_reason은 drop_leak_cols=False면 남아있을 수 있는데, 이 또한 보통 "의미 있는 결측"이라 유지 권장
    if not drop_leak_cols and col_no_show_reason is not None:
        meaningful_missing.add(col_no_show_reason)

    # 결측 채우기: 의미 있는 결측 제외 + 타깃 제외
    skip_fill = meaningful_missing | set([c for c in [col_no_show] if c is not None])

    # 숫자형 컬럼: 중앙값
    # 문자열/범주형: "unknown"
    for c in df.columns:
        if c in skip_fill:
            continue

        # 날짜/시간 문자열 컬럼은 결측이면 그대로 두는 편이 안전(필요하면 아래에서 "unknown"으로 바꿀 수도 있음)
        if c in [col_appt_date, col_appt_time, col_dob, col_entry]:
            continue

        if df[c].dtype.name in ("string", "object"):
            # 숫자처럼 생긴 컬럼이면 숫자로 간주해 중앙값 채움
            as_num = pd.to_numeric(df[c], errors="coerce")
            numeric_ratio = as_num.notna().mean()
            if numeric_ratio >= 0.9 and df[c].notna().any():
                med = as_num.median()
                df[c] = as_num.fillna(med)
            else:
                df[c] = df[c].fillna("unknown")
        else:
            if pd.api.types.is_numeric_dtype(df[c]):
                med = df[c].median()
                df[c] = df[c].fillna(med)

    # 나이 기본 검증(있으면) - 컬럼 추가 없이 값만 정리
    age_col = first_existing(df, ["Age", "age"])
    if age_col is not None and age_col in df.columns:
        age_num = pd.to_numeric(df[age_col], errors="coerce")
        df[age_col] = age_num.where(age_num.between(0, 120), pd.NA)

    return df


# -----------------------------
# 사용 예시
# -----------------------------
CSV_PATH = "medical-appointments-no-show-en.csv"
df_raw = pd.read_csv(CSV_PATH)
df_raw = df_raw.drop(columns = "no_show_reason")
# drop_leak_cols=True로 하면 no_show_reason 컬럼 삭제(원하면)
df_clean = clean_for_db_no_new_cols(df_raw, drop_leak_cols=False)

# 나이, 월 >> 정수형으로 바꾸기 
int_col = ["age", "appointment_month"]
for c in int_col:
    df_clean[c] = df_clean[c].astype("int")
# 성별 i 제거하고 m, f >> 0, 1로 맵핑  
# appointment_shift, no_show 컬럼도 마찬가지
df_clean = df_clean[df_clean["gender"] != "i"]

df_clean["gender"] = df_clean["gender"].map({
    "male" : 0,
    "female" : 1
})

df_clean["appointment_shift"] = df_clean["appointment_shift"].map({
    "morning" : 0,
    "afternoon" : 1
})

df_clean["no_show"] = df_clean["no_show"].map({
    "no" : 0,
    "yes" : 1
})

# appointment_datetime 이라는 새 컬럼 만들기  
df_clean["appointment_datetime"] = (
    pd.to_datetime(df_clean["appointment_date"], format="%Y-%m-%d", errors="coerce")
    + pd.to_timedelta(df_clean["appointment_time"], errors="coerce")
)

df = pd.read_csv("medical-appointments-no-show-en.csv")
lst1 = list(df_clean.columns)

for col in lst1:
    if df_clean[col].isna().sum() != 0:
        print(f"전) {col}의 null값 : {df[col].isna().sum()}, 후) {col}의 null값 : {df_clean[col].isna().sum()}" )

# 데이터에 가상의 이름 부여
fake = Faker("ko_KR")  
df_clean["name"] = [fake.name() for _ in range(len(df_clean))]
cols = ["name"] + [c for c in df_clean.columns if c != "name"]
df_clean = df_clean[cols]

csv1_col = [
    "name",
    "appointment_datetime",
    "appointment_date",
    "specialty",
    "gender",
    "age",
    "under_12_years_old",
    "over_60_years_old",
    "patient_needs_companion",
    "disability",
    "no_show",
    "icd",
    "entry_service_date"
]

csv2_col = [
    "weather_date",
    "storm_day_before",
    "rain_intensity",
    "max_temp_day",
    "max_rain_day",
    "heat_intensity",
    "average_temp_day",
    "average_rain_day"
]

df_clean[csv1_col].to_csv("noshow_appt.csv", index=False, encoding="utf-8-sig")

# weather 테이블 전용 데이터프레임 생성 & csv 파일 작성
df_weather = df_clean.copy()
df_weather = df_weather.drop_duplicates(subset = ["appointment_date"])
df_weather["weather_date"] = df_weather["appointment_date"]
df_weather[csv2_col].to_csv("noshow_weather.csv", index=False, encoding="utf-8-sig")
