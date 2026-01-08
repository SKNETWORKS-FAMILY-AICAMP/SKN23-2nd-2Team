import pandas as pd
from modules.connect_db_module import fetch_table_data


# 데이터 조회
def fetch_df(table_name: str, env_file: str = ".env", limit = None, offset=0):
    """
    connect_db_module의 fetch_table_data로 이용해
    특정 테이블 전체를 DataFrame으로 가져옴
    """

    rows = fetch_table_data(table_name=table_name, env_file=env_file, limit=limit, offset=offset)

    if rows:
        return pd.DataFrame(rows)
    else:
        print("DB import 에러")

# 날짜->요일 원핫인코딩
def date_to_weekday_onehot(df, column_name):
    """
    column_name(datetime)을 요일로 변환 후 원핫 인코딩해서 컬럼 확장
    - 월~일 문자열 기준
    """
    df_copy = df.copy()

    # datetime 변환
    dt = pd.to_datetime(df_copy[column_name], errors="coerce")

    # 요일 추출 (월~일)
    weekday = dt.dt.day_name().str[:3]  # 'Mon', 'Tue ... 'Sun' 3글자 파싱

    # 원핫 인코딩
    dummies = pd.get_dummies(weekday, prefix=f"{column_name}", dummy_na=False)

    # 기존 날짜 컬럼 제거 + 원핫 컬럼 붙이기
    df_copy = df_copy.drop(columns=[column_name])
    df_copy = pd.concat([df_copy, dummies], axis=1)

    return df_copy

# 날짜->월별 원핫인코딩
def date_to_month_onehot(df, column_name):
    """
    column_name(datetime)에서 월만 추출해서 원핫 인코딩 컬럼 생성
    예: appointment_date_Jan ... appointment_date_Dec
    """
    df_copy = df.copy()

    dt = pd.to_datetime(df_copy[column_name], errors="coerce")
    month_abbr = dt.dt.strftime("%b") 

    dummies = pd.get_dummies(month_abbr, prefix=column_name, dummy_na=False)

    # Jan~Dec 맵핑
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in months:
        col = f"{column_name}_{m}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"{column_name}_{m}" for m in months]]

    df_copy = df_copy.drop(columns=[column_name])
    df_copy = pd.concat([df_copy, dummies], axis=1)

    return df_copy

# 장애 원핫인코딩
def disability_onehot(df, column_name="disability"):
    """
    disability 컬럼("", motor, intellectual/intellecture)을 원핫 인코딩
    - ""(빈값) -> disability_null
    - intellectual -> disability_intellectual
    - motor -> disability_motor
    """
    df_copy = df.copy()

    # 문자열 정리
    s = df_copy[column_name].astype("string").fillna("").str.strip().str.lower()

    # 값 표준화: 빈값 -> null, 오타 통일
    s = s.replace({
        "": "null",
        "intellecture": "intellectual",
    })

    dummies = pd.get_dummies(s, prefix="disability", dtype=int)

    # 원하는 컬럼 순서/이름만 남기기
    dummies = dummies[["disability_intellectual", "disability_motor", "disability_null"]]
    df_copy = df_copy.drop(columns=[column_name])

    return pd.concat([df_copy, dummies], axis=1)


# specialty 영문 -> 한글 매핑
SPECIALTY_KO_MAP = {
    "physiotherapy": "물리치료",
    "psychotherapy": "심리치료",
    "speech therapy": "언어치료",
    "occupational therapy": "작업치료",
    "unknown": "알수없음",
    "enf": "간호",
    "assist": "보조",
    "pedagogo": "교육전문가",
    "sem especialidade": "전문의없음",
}

_SPECIALTY_CATS_EN = list(SPECIALTY_KO_MAP.keys())


def specialty_ko_onehot(df, column_name="specialty", keep_korean_column=True):
    """
    specialty 컬럼을 한글화(+선택)하고 원핫 인코딩 컬럼 확장
    """
    df_copy = df.copy()

    # 문자열 정리
    s = df_copy[column_name].astype("string").fillna("unknown").str.strip().str.lower()

    # 한글 컬럼 추가
    if keep_korean_column:
        df_copy["specialty_ko"] = s.map(SPECIALTY_KO_MAP)

    # 카테고리 고정 후 원핫
    s_cat = pd.Categorical(s, categories=_SPECIALTY_CATS_EN)
    dummies = pd.get_dummies(s_cat, prefix="specialty", dtype=int)
    dummies = dummies[[f"specialty_{c}" for c in _SPECIALTY_CATS_EN]]

    # 기존 specialty 컬럼 제거(항상)
    if column_name in df_copy.columns:
        df_copy = df_copy.drop(columns=[column_name])

    return pd.concat([df_copy, dummies], axis=1)