import os, time, requests, pandas as pd
from datetime import datetime, timedelta, timezone
import hopsworks

LAT, LON = 24.8608, 67.0104
OWM_KEY = os.environ["OPENWEATHER_API_KEY"]
DAYS_BACK = 1095  # 3 years

# Extra points spread across Karachi, used only to compute city-wide spatial spread features
POINTS = {
    "site": (24.8944, 66.9874),      # industrial (west)
    "clifton": (24.8170, 67.0330),   # coastal/affluent (south)
    "landhi": (24.8504, 67.1999),    # industrial/residential (east)
    "malir": (24.8929, 67.1953),     # outer periphery (northeast)
}

PM25_BP = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
PM10_BP = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]

def us_aqi(pm25, pm10):
    def sub_index(c, bp):
        for lo, hi, ilo, ihi in bp:
            if lo <= c <= hi:
                return (ihi - ilo) / (hi - lo) * (c - lo) + ilo
        return bp[-1][3]
    return round(max(sub_index(pm25, PM25_BP), sub_index(pm10, PM10_BP)))

def fetch_pollution_at(lat, lon, start, end):
    rows, cur, step = [], start, timedelta(days=30)
    while cur < end:
        nxt = min(cur + step, end)
        r = requests.get("https://api.openweathermap.org/data/2.5/air_pollution/history",
                          params={"lat": lat, "lon": lon, "start": int(cur.timestamp()),
                                  "end": int(nxt.timestamp()), "appid": OWM_KEY}).json()
        rows += r.get("list", [])
        cur = nxt
        time.sleep(1)
    return rows

def fetch_pollution_history(start, end):
    rows = fetch_pollution_at(LAT, LON, start, end)
    return pd.DataFrame([{"timestamp": p["dt"], "aqi": us_aqi(p["components"]["pm2_5"], p["components"]["pm10"]),
                           **p["components"]} for p in rows])

def fetch_spatial_aggregates(start, end):
    dfs = []
    for name, (lat, lon) in POINTS.items():
        rows = fetch_pollution_at(lat, lon, start, end)
        d = pd.DataFrame([{"timestamp": p["dt"], name: us_aqi(p["components"]["pm2_5"], p["components"]["pm10"])} for p in rows])
        dfs.append(d)
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge_asof(merged.sort_values("timestamp"), d.sort_values("timestamp"), on="timestamp", direction="nearest")
    cols = list(POINTS.keys())
    merged["city_aqi_mean"] = merged[cols].mean(axis=1)
    merged["city_aqi_max"] = merged[cols].max(axis=1)
    merged["city_aqi_min"] = merged[cols].min(axis=1)
    merged["city_aqi_spread"] = merged["city_aqi_max"] - merged["city_aqi_min"]
    return merged[["timestamp", "city_aqi_mean", "city_aqi_max", "city_aqi_min", "city_aqi_spread"]]

def fetch_weather_history(start, end):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LAT, "longitude": LON, "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "UTC"}).json()["hourly"]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(r["time"]).astype("int64") // 10**9,
        "temp": r["temperature_2m"], "humidity": r["relative_humidity_2m"],
        "pressure": r["surface_pressure"], "wind_speed": r["wind_speed_10m"]})

def build_dataset():
    end, start = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
    poll = fetch_pollution_history(start, end)
    wx = fetch_weather_history(start, end)
    spatial = fetch_spatial_aggregates(start, end)

    df = pd.merge_asof(poll.sort_values("timestamp"), wx.sort_values("timestamp"), on="timestamp", direction="nearest")
    df = pd.merge_asof(df.sort_values("timestamp"), spatial.sort_values("timestamp"), on="timestamp", direction="nearest")

    ts = pd.to_datetime(df["timestamp"], unit="s")
    df["hour"], df["day"], df["month"], df["day_of_week"] = ts.dt.hour, ts.dt.day, ts.dt.month, ts.dt.dayofweek
    df = df.dropna()

    float_cols = ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3", "temp", "humidity", "pressure",
                  "wind_speed", "city_aqi_mean", "city_aqi_max", "city_aqi_min", "city_aqi_spread"]
    df[float_cols] = df[float_cols].astype("float64")
    df[["timestamp", "hour", "day", "month", "day_of_week"]] = df[["timestamp", "hour", "day", "month", "day_of_week"]].astype("int64")
    return df

def push_to_hopsworks(df):
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"], project=os.environ["HOPSWORKS_PROJECT"])
    fg = project.get_feature_store().get_or_create_feature_group(
        name="karachi_aqi_features", version=1, primary_key=["timestamp"], event_time="timestamp")
    fg.insert(df)

if __name__ == "__main__":
    df = build_dataset()
    print(f"Backfilled {len(df)} rows: {pd.to_datetime(df['timestamp'].min(), unit='s')} -> {pd.to_datetime(df['timestamp'].max(), unit='s')}")
    push_to_hopsworks(df)
    print("Backfill inserted into Hopsworks.")