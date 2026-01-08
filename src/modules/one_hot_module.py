import pandas as pd
from connect_db_module import fetch_table_data
from sklearn.preprocessing import OneHotEncoder


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
