# src/services/weather_api.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


BASE = "https://api.openweathermap.org/data/2.5"
load_dotenv()


def _api_key():
    key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENWEATHER_API_KEY가 없습니다. (.env 또는 환경변수에 설정)")
    return key


def _get(path, params):
    url = f"{BASE}/{path}"
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()


def get_current_weather(city, units= "metric", lang="kr") :
    return _get("weather", {"q": city, "appid": _api_key(), "units": units, "lang": lang})


def get_forecast_5d3h(city, units="metric", lang="kr"):
    """
    5일/3시간 예보 DataFrame
    columns:
      dt, temp, feels_like, humidity,
      weather_main, weather_desc, ow_icon
    """
    data = _get("forecast", {"q": city, "appid": _api_key(), "units": units, "lang": lang})

    rows = []
    for item in data.get("list", []):
        w0 = (item.get("weather") or [{}])[0]
        main = item.get("main", {}) or {}

        rows.append(
            {
                "dt": item.get("dt_txt"),
                "temp": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "humidity": main.get("humidity"),
                "weather_main": w0.get("main"),          # <- 핵심
                "weather_desc": w0.get("description"),
                "ow_icon": w0.get("icon"),              # <- OpenWeather 아이콘 코드(예: 04d)
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["dt"] = pd.to_datetime(df["dt"])
    return df


def get_weather_data(city, units="metric", lang="kr"):
    """
    오늘부터 7일치 요일을 key로 만들고,
    forecast에서 하루(3시간*8) weather_main 최빈값을 value로 채움.
    마지막(또는 비어있는) 요일은 '마지막으로 확보된 값'으로 채움.
    예: Tue~Sun만 있으면 Mon은 Sun 값으로 채움.
    """
    df = get_forecast_5d3h(city, units=units, lang=lang)
    if df.empty:
        return {}

    df = df.sort_values("dt").copy()
    df["date"] = df["dt"].dt.date

    # 날짜별 최빈값 (하루 8개 중 가장 많이 나온 weather_main)
    daily = (
        df.groupby("date", sort=True)["weather_main"]
        .agg(lambda s: s.value_counts().idxmax())
    )

    if daily.empty:
        return {}

    today = datetime.now(ZoneInfo("Asia/Seoul")).date()

    out = {}

    # 시작값: 오늘 데이터가 없을 수도 있으니, 일단 첫 예보값으로 초기화
    last = daily.iloc[0]

    for i in range(7):
        d = today + timedelta(days=i)
        if d in daily.index:
            last = daily.loc[d]
        out[d.strftime("%a")] = last  # Tue, Wed, Thu ...

    return out