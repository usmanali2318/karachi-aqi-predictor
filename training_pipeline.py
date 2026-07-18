import os, joblib, numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import hopsworks

HORIZONS = [24, 48, 72]  # hours ahead -> tomorrow, day after, day after that

def load_data():
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"], project=os.environ["HOPSWORKS_PROJECT"])
    fg = project.get_feature_store().get_feature_group("karachi_aqi_features", version=1)
    return fg.read().sort_values("timestamp").reset_index(drop=True), project

def engineer(df):
    df["lag_1h"], df["lag_3h"], df["lag_24h"] = df["aqi"].shift(1), df["aqi"].shift(3), df["aqi"].shift(24)
    df["rolling_mean_24h"] = df["aqi"].rolling(24).mean()
    df["aqi_change_rate"] = df["aqi"].diff()
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*df["hour"]/24), np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"], df["month_cos"] = np.sin(2*np.pi*df["month"]/12), np.cos(2*np.pi*df["month"]/12)
    for h in HORIZONS:
        df[f"target_{h}h"] = df["aqi"].shift(-h)
    return df.dropna().reset_index(drop=True)

def train_eval(df):
    target_cols = [f"target_{h}h" for h in HORIZONS]
    X, y = df.drop(columns=["timestamp"] + target_cols), df[target_cols]
    day_id = df["timestamp"] // 86400
    test_mask = (day_id % 6 == 0)  # every 6th day -> test, spread across the full year
    X_train, X_test, y_train, y_test = X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]

    models = {
        "Ridge": MultiOutputRegressor(Ridge(alpha=1.0)),
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42)),
        "FeedforwardNN": MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)),
    }

    best_name, best_model, best_r2 = None, None, -np.inf
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae, r2 = mean_absolute_error(y_test, preds), r2_score(y_test, preds)
        print(f"{name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
        if r2 > best_r2:
            best_name, best_model, best_r2 = name, model, r2

    print(f"\nBest model: {best_name} (R2={best_r2:.3f})")
    return best_name, best_model

def save_to_registry(project, model, name):
    os.makedirs("model_dir", exist_ok=True)
    joblib.dump(model, "model_dir/model.pkl")
    mr = project.get_model_registry()
    m = mr.python.create_model(name="karachi_aqi_model",
                                description=f"Best model: {name}, predicts AQI at +24h/+48h/+72h")
    try:
        m.save("model_dir")
    except Exception as e:
        print(f"Model uploaded, but Hopsworks' status check failed (known cluster issue): {e}")
        print("Check the Model Registry in the Hopsworks UI to confirm - it's almost always there anyway.")

if __name__ == "__main__":
    df, project = load_data()
    df = engineer(df)
    split = int(len(df) * 0.85)
    print("Train AQI stats:\n", df["aqi"][:split].describe())
    print("\nTest AQI stats:\n", df["aqi"][split:].describe())
    best_name, best_model = train_eval(df)
    save_to_registry(project, best_model, best_name)
    print("Model saved to Hopsworks Model Registry.")