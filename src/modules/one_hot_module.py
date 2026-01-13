import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
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


def icd_multihot(
    df: pd.DataFrame,
    column_name: str = "icd",
    *,
    prefix: str = "icd",
    use_parent_code: bool = False,   # True면 f84.5 -> f84
    min_freq: int | None = None,     # 예: 50이면 50회 미만 ICD는 제거
    add_null_col: bool = True,       # 빈값이면 icd_null=1 컬럼 추가
    drop_original: bool = True       # 원본 icd 컬럼 제거 여부
) -> pd.DataFrame:
    """
    icd 컬럼(빈문자/단일/멀티라벨)을 multi-hot 인코딩하여 df에 붙여 반환.

    - 빈값(공백 포함) -> [] 로 처리
    - 멀티라벨은 sep 기준으로 split
    - use_parent_code=True면 '.' 앞까지만 사용 (예: f84.5 -> f84)
    - min_freq 지정 시, 빈도 낮은 ICD는 제거하여 차원 축소
    - add_null_col=True면, 원본이 빈값인 행은 {prefix}_null = 1로 표시
    """
    df_copy = df.copy()

    # 1) 문자열 정리: NaN이 아니라도 공백/빈문자 처리 필요
    s = df_copy[column_name].astype("string").fillna("").str.strip().str.lower()

    # 빈값 마스크(나중에 null 컬럼 만들 때 사용)
    is_null = s.eq("")

    # 2) 멀티라벨 파싱: 문자열 -> list[str]
    def _parse_codes(x: str) -> list[str]:
        if x == "":
            return []
        codes = [c.strip() for c in x.split("/")]
        codes = [c for c in codes if c]  # 빈 토큰 제거
        if use_parent_code:
            codes = [c.split(".")[0] for c in codes if c]
        # 중복 제거(순서 유지)
        seen = set()
        out = []
        for c in codes:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    icd_list = s.apply(_parse_codes)

    # 3) (옵션) 희귀 ICD 제거
    if min_freq is not None and min_freq > 1:
        flat = pd.Series([c for codes in icd_list for c in codes], dtype="string")
        valid = set(flat.value_counts()[lambda vc: vc >= min_freq].index.tolist())
        icd_list = icd_list.apply(lambda codes: [c for c in codes if c in valid])

    # 4) Multi-hot 인코딩
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(icd_list)

    icd_df = pd.DataFrame(
        encoded,
        columns=[f"{prefix}_{c}" for c in mlb.classes_],
        index=df_copy.index,
        dtype=int
    )

    # 5) null 컬럼(옵션)
    if add_null_col:
        icd_df[f"{prefix}_null"] = is_null.astype(int)

    # 6) 원본 컬럼 제거 + 병합
    if drop_original and column_name in df_copy.columns:
        df_copy = df_copy.drop(columns=[column_name])

    return pd.concat([df_copy, icd_df], axis=1)
