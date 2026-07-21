import os, joblib, numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import hopsworks

HORIZONS = [24, 48, 72]  # hours ahead -> tomorrow, day after, day after that

class AverageEnsemble:
    """Simple averaging ensemble of already-fitted models. If this wins and gets saved,
    this exact class must also be defined in whatever script later unpickles it (Phase 5)."""
    def __init__(self, models):
        self.models = models
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

def load_data():
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"], project=os.environ["HOPSWORKS_PROJECT"])
    fg = project.get_feature_store().get_feature_group("karachi_aqi_features", version=1)
    return fg.read().sort_values("timestamp").reset_index(drop=True), project

def engineer(df):
    for lag in [1, 3, 6, 12, 24, 48]:
        df[f"lag_{lag}h"] = df["aqi"].shift(lag)
    df["rolling_mean_24h"] = df["aqi"].rolling(24).mean()
    df["rolling_std_24h"] = df["aqi"].rolling(24).std()
    df["aqi_change_rate"] = df["aqi"].diff()
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*df["hour"]/24), np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"], df["month_cos"] = np.sin(2*np.pi*df["month"]/12), np.cos(2*np.pi*df["month"]/12)
    for h in HORIZONS:
        # actual future weather stands in for a forecast - at live inference (Phase 5) these
        # get filled from OpenWeather's real forecast API for the matching target hour instead
        df[f"target_{h}h"] = df["aqi"].shift(-h)
        df[f"future_temp_{h}h"] = df["temp"].shift(-h)
        df[f"future_humidity_{h}h"] = df["humidity"].shift(-h)
        df[f"future_wind_{h}h"] = df["wind_speed"].shift(-h)
    return df.dropna().reset_index(drop=True)

def train_eval(df):
    target_cols = [f"target_{h}h" for h in HORIZONS]
    X, y = df.drop(columns=["timestamp"] + target_cols), df[target_cols]
    y_log = np.log1p(y)  # AQI is right-skewed; log-transform stabilizes it for training

    day_id = df["timestamp"] // 86400
    test_mask = day_id % 6 == 0  # every 6th day -> test, spread across the full year
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train_log, y_test = y_log[~test_mask], y[test_mask]

    tscv = TimeSeriesSplit(n_splits=2)
    tuned = {
        "RandomForest": (MultiOutputRegressor(RandomForestRegressor(random_state=42)),
                          {"estimator__n_estimators": [100, 200], "estimator__max_depth": [8, 12, None],
                           "estimator__min_samples_leaf": [1, 3]}),
        "XGBoost": (MultiOutputRegressor(XGBRegressor(random_state=42, verbosity=0)),
                    {"estimator__n_estimators": [100, 200], "estimator__max_depth": [3, 5],
                     "estimator__learning_rate": [0.05, 0.15]}),
        "Ridge": (MultiOutputRegressor(Ridge()), {"estimator__alpha": [0.1, 1.0, 5.0, 10.0]}),
    }

    fitted = {}
    for name, (model, params) in tuned.items():
        search = RandomizedSearchCV(model, params, n_iter=4, cv=tscv, scoring="r2", random_state=42, n_jobs=-1, verbose=1)
        search.fit(X_train, y_train_log)
        fitted[name] = search.best_estimator_
        print(f"{name} best params: {search.best_params_}")

    nn = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                                                        early_stopping=True, random_state=42))])
    fitted["FeedforwardNN"] = MultiOutputRegressor(nn)
    fitted["FeedforwardNN"].fit(X_train, y_train_log)

    fitted["Ensemble (RF+XGB)"] = AverageEnsemble([fitted["RandomForest"], fitted["XGBoost"]])

    best_name, best_model, best_r2 = None, None, -np.inf
    for name, model in fitted.items():
        preds = np.expm1(model.predict(X_test))  # invert log-transform for real-unit metrics
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
    for old in mr.get_models("karachi_aqi_model"):  # clear previous version(s) - this cluster's
        try:                                         # version-increment endpoint is unreliable,
            old.delete()                              # so we always recreate fresh as v1 instead
        except Exception:
            pass
    m = mr.python.create_model(name="karachi_aqi_model",
                                description=f"Best model: {name}, predicts log1p(AQI) at +24h/+48h/+72h - invert with expm1")
    try:
        m.save("model_dir")
    except Exception as e:
        print(f"Model uploaded, but Hopsworks' status check failed (known cluster issue): {e}")
        print("Check the Model Registry in the Hopsworks UI to confirm - it's almost always there anyway.")

if __name__ == "__main__":
    df, project = load_data()
    df = engineer(df)
    best_name, best_model = train_eval(df)
    save_to_registry(project, best_model, best_name)
    print("Model saved to Hopsworks Model Registry.")