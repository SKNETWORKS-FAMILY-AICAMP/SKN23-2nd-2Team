import os
import requests
import pandas as pd
import pymysql
from sshtunnel import SSHTunnelForwarder
from datetime import datetime, timedelta
from dotenv import load_dotenv
from modules.connect_db_module import _open_conn, _close

# ----------------------------
# 환경 변수 로드
# ----------------------------
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Seoul"
BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

# ----------------------------
# 예약 날짜 가져오기
# ----------------------------
def get_appointment_dates(days_ahead=7):
    start_date = datetime.today().date()
    end_date = start_date + timedelta(days=days_ahead)

    query = f"""
    SELECT DISTINCT appointment_date 
    FROM appointmentList 
    WHERE appointment_date BETWEEN '{start_date}' AND '{end_date}' 
    ORDER BY appointment_date
    """
    tunnel, conn = _open_conn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        df = pd.DataFrame(rows)
    finally:
        _close(tunnel, conn)

    df['appointment_date'] = pd.to_datetime(df['appointment_date'], errors='coerce').dt.normalize()
    return df['appointment_date'].dropna().tolist()

# ----------------------------
# 날씨 예보 가져오기
# ----------------------------
def get_forecast(city=CITY, units="metric", lang="kr"):
    params = {"q": city, "appid": API_KEY, "units": units, "lang": lang}
    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# ----------------------------
# 하루 단위 요약
# ----------------------------
def parse_daily_weather(forecast_json):
    rows = []
    for item in forecast_json.get("list", []):
        dt_txt = item.get("dt_txt")
        main = item.get("main", {})
        weather = (item.get("weather") or [{}])[0]
        rain = item.get("rain", {}).get("3h", 0)

        rain_intensity = "no_rain" if weather.get("main") in ["Clear", "Clouds"] else "rain"

        temp = main.get("temp", 0)
        if temp < 0:
            heat_intensity = "cold"
        elif temp <= 20:
            heat_intensity = "mild"
        else:
            heat_intensity = "hot"

        rows.append({
            "weather_date": pd.to_datetime(dt_txt).date(),
            "average_temp_day": temp,
            "max_temp_day": main.get("temp_max"),
            "average_rain_day": item.get("pop", 0),
            "max_rain_day": rain,
            "rain_intensity": rain_intensity,
            "storm_day_before": 0,
            "heat_intensity": heat_intensity
        })

    df = pd.DataFrame(rows)
    daily = df.groupby("weather_date").agg({
        "average_temp_day": "mean",
        "max_temp_day": "max",
        "average_rain_day": "mean",
        "max_rain_day": "max",
        "rain_intensity": lambda x: x.value_counts().idxmax(),
        "storm_day_before": "max",
        "heat_intensity": lambda x: x.value_counts().idxmax()
    }).reset_index()
    return daily

# ----------------------------
# 예약 날짜 기준 필터링
# ----------------------------
def filter_for_appointments(daily_weather, appointment_dates):
    df_dates = pd.DataFrame({"weather_date": appointment_dates})
    df_dates['weather_date'] = pd.to_datetime(df_dates['weather_date'])
    daily_weather['weather_date'] = pd.to_datetime(daily_weather['weather_date'])
    
    merged = pd.merge(df_dates, daily_weather, on="weather_date", how="left")
    merged.fillna(method="ffill", inplace=True)
    return merged

# ----------------------------
# MySQL 업서트
# ----------------------------
def upsert_to_mysql(df):
    tunnel, conn = _open_conn()
    cursor = conn.cursor()

    sql = """
    INSERT INTO weatherDay 
    (weather_date, average_temp_day, max_temp_day, average_rain_day, max_rain_day, rain_intensity, storm_day_before, heat_intensity)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        average_temp_day=VALUES(average_temp_day),
        max_temp_day=VALUES(max_temp_day),
        average_rain_day=VALUES(average_rain_day),
        max_rain_day=VALUES(max_rain_day),
        rain_intensity=VALUES(rain_intensity),
        storm_day_before=VALUES(storm_day_before),
        heat_intensity=VALUES(heat_intensity)
    """
    data_tuples = [tuple(row) for row in df[[
        "weather_date", "average_temp_day", "max_temp_day", 
        "average_rain_day", "max_rain_day", "rain_intensity", 
        "storm_day_before", "heat_intensity"
    ]].to_numpy()]

    cursor.executemany(sql, data_tuples)
    conn.commit()
    cursor.close()
    _close(tunnel, conn)

# ----------------------------
# 전체 실행 함수
# ----------------------------
def run_weather_upsert(days_ahead=7, city=CITY):
    """
    1. 예약 날짜 가져오기
    2. 날씨 예보 가져오기
    3. 예약 날짜 기준 필터링
    4. MySQL 업서트
    Returns:
        filtered_df: 예약 날짜 기준 날씨 DataFrame
    """
    appointment_dates = get_appointment_dates(days_ahead)
    forecast_json = get_forecast(city)
    daily_weather = parse_daily_weather(forecast_json)
    filtered_df = filter_for_appointments(daily_weather, appointment_dates)
    upsert_to_mysql(filtered_df)
    return filtered_df
