import pandas as pd
from modules.connect_db_module import fetch_table_data


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

    # 카테고리 고정(항상 3개 컬럼 나오게)
    s = pd.Categorical(s, categories=["null", "motor", "intellectual"])

    dummies = pd.get_dummies(s, prefix="disability", dtype=int)

    # 원하는 컬럼 순서/이름만 남기기
    dummies = dummies[["disability_intellectual", "disability_motor", "disability_null"]]
    df_copy = df_copy.drop(columns=[column_name])

    return pd.concat([df_copy, dummies], axis=1)