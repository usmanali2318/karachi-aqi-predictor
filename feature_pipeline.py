import os, requests, pandas as pd
from datetime import datetime, timezone
import hopsworks

LAT, LON = 24.8608, 67.0104
OWM_KEY = os.environ["OPENWEATHER_API_KEY"]

PM25_BP = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
PM10_BP = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]

def us_aqi(pm25, pm10):
    def sub_index(c, bp):
        for lo, hi, ilo, ihi in bp:
            if lo <= c <= hi:
                return (ihi - ilo) / (hi - lo) * (c - lo) + ilo
        return bp[-1][3]
    return round(max(sub_index(pm25, PM25_BP), sub_index(pm10, PM10_BP)))

def fetch_features() -> pd.DataFrame:
    air_resp = requests.get("https://api.openweathermap.org/data/2.5/air_pollution",
                             params={"lat": LAT, "lon": LON, "appid": OWM_KEY})
    if "list" not in air_resp.json():
        raise RuntimeError(f"OpenWeather API error (check OPENWEATHER_API_KEY): {air_resp.status_code} {air_resp.text}")
    air = air_resp.json()["list"][0]
    wx = requests.get("https://api.openweathermap.org/data/2.5/weather",
                       params={"lat": LAT, "lon": LON, "appid": OWM_KEY, "units": "metric"}).json()

    ts = datetime.now(timezone.utc)
    row = {
        "timestamp": int(ts.timestamp()),
        "aqi": us_aqi(air["components"]["pm2_5"], air["components"]["pm10"]),
        **{k: v for k, v in air["components"].items()},  # pm2_5, pm10, no2, so2, o3, co, etc.
        "temp": wx["main"]["temp"],
        "humidity": wx["main"]["humidity"],
        "pressure": wx["main"]["pressure"],
        "wind_speed": wx["wind"]["speed"],
        "hour": ts.hour,
        "day": ts.day,
        "month": ts.month,
        "day_of_week": ts.weekday(),
    }
    df = pd.DataFrame([row])
    float_cols = ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3", "temp", "humidity", "pressure", "wind_speed"]
    df[float_cols] = df[float_cols].astype("float64")
    df[["timestamp", "hour", "day", "month", "day_of_week"]] = df[["timestamp", "hour", "day", "month", "day_of_week"]].astype("int64")
    return df

def push_to_hopsworks(df: pd.DataFrame):
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT"],
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="karachi_aqi_features",
        version=1,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly Karachi AQI + weather features",
    )
    fg.insert(df)

if __name__ == "__main__":
    df = fetch_features()
    print(df.T)
    push_to_hopsworks(df)
    print("Row inserted into Hopsworks feature store.")