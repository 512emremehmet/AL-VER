# backend/main_ultimate_fixed.py - NASA Parade Ultra Enhanced Edition (Fully Fixed & Stable)
import asyncio
import aiohttp
from fastapi import FastAPI, Query, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import io
import datetime
from typing import List, Dict, Any, Tuple, Optional
import hashlib
import json
import time
import os
import sys
from fpdf import FPDF
from collections import defaultdict
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Application ---
app = FastAPI(
    title="NASA Parade - Ultimate Backend (Fixed)",
    description="Enterprise-grade weather prediction system with ML ensemble, user personalization, and real-time updates",
    version="3.1.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Constants (DÜZELTİLDİ) ---
NASA_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"  # Boşluk kaldırıldı
START_YEAR = 2000
CURRENT_YEAR = datetime.datetime.utcnow().year
SEM = asyncio.Semaphore(20)
CACHE_TTL_SECONDS = 6 * 60 * 60
MAX_REQUESTS_PER_HOUR = 100

# VALID_NASA_PARAMS: COMMUNITY="AG" için tam destek (NASA POWER v3.0)
VALID_NASA_PARAMS = {
    "RE": ["T2M","T2M_MIN","T2M_MAX","PRECTOT","WS10M","WD10M","RH2M"],
    "AG": ["T2M","T2M_MIN","T2M_MAX","PRECTOT","WS10M","WD10M","RH2M"],
    "SB": ["T2M","T2M_MIN","T2M_MAX","PRECTOT","WS10M","WD10M","RH2M"]
}



COMMUNITY = "AG"  # En zengin veri → "AG" seçildi

# --- Activity Profiles (EKSİK EKLENDİ) ---
ACTIVITY_PROFILES = {
    "picnic": {
        "description": "Outdoor meal or gathering in a park or natural setting.",
        "optimal_temp": (15, 28),
        "max_rain": 2.0,
        "max_wind": 8.0,
        "sensitivities": ["rain", "wind"]
    },
    "hiking": {
        "description": "Walking in natural environments, often on trails.",
        "optimal_temp": (10, 25),
        "max_rain": 1.0,
        "max_wind": 10.0,
        "sensitivities": ["rain", "wind"]
    },
    "beach": {
        "description": "Relaxing or swimming at the seaside.",
        "optimal_temp": (22, 35),
        "max_rain": 1.0,
        "max_wind": 7.0,
        "sensitivities": ["rain", "wind"]
    },
    "skiing": {
        "description": "Winter sport on snow-covered slopes.",
        "optimal_temp": (-10, 5),
        "max_rain": 0.0,
        "max_wind": 15.0,
        "sensitivities": ["rain", "wind"]
    },
    "farming": {
        "description": "Agricultural activities requiring dry and mild conditions.",
        "optimal_temp": (10, 30),
        "max_rain": 5.0,
        "max_wind": 12.0,
        "sensitivities": ["rain", "wind"]
    }
}

# --- Smart Cache ---
class SmartCache:
    def __init__(self):
        self._cache: Dict[str, Tuple[float, Any, int]] = {}
        self._access_times: Dict[str, List[float]] = defaultdict(list)
    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if not entry:
            return None
        ts, value, hits = entry
        if time.time() - ts > CACHE_TTL_SECONDS:
            del self._cache[key]
            return None
        self._cache[key] = (ts, value, hits + 1)
        self._access_times[key].append(time.time())
        return value
    def set(self, key: str, value: Any):
        self._cache[key] = (time.time(), value, 0)
    def get_stats(self) -> Dict:
        total_keys = len(self._cache)
        total_hits = sum(hits for _, _, hits in self._cache.values())
        cache_size_mb = sum(sys.getsizeof(str(v)) for _, v, _ in self._cache.values()) / (1024*1024)
        hot_keys = sorted(
            [(k, hits) for k, (_, _, hits) in self._cache.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        return {
            "total_cached_keys": total_keys,
            "total_cache_hits": total_hits,
            "cache_size_mb": round(cache_size_mb, 2),
            "hot_keys": [{"key": k[:32], "hits": h} for k, h in hot_keys]
        }

smart_cache = SmartCache()

def make_cache_key(*args, **kwargs) -> str:
    key_raw = {"args": args, "kwargs": kwargs}
    s = json.dumps(key_raw, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()

# --- Rate Limiting ---
RATE_LIMIT_FILE = "rate_limits.json"
rate_limit_store: Dict[str, List[float]] = defaultdict(list)

def load_rate_limits():
    global rate_limit_store
    try:
        if os.path.exists(RATE_LIMIT_FILE):
            with open(RATE_LIMIT_FILE, 'r') as f:
                data = json.load(f)
                rate_limit_store = defaultdict(list, data)
            logger.info("Rate limits loaded from file")
    except Exception as e:
        logger.warning(f"Failed to load rate limits: {e}")

def save_rate_limits():
    try:
        with open(RATE_LIMIT_FILE, 'w') as f:
            json.dump(dict(rate_limit_store), f)
    except Exception as e:
        logger.warning(f"Failed to save rate limits: {e}")

load_rate_limits()

def check_rate_limit(client_id: str) -> bool:
    now = time.time()
    rate_limit_store[client_id] = [t for t in rate_limit_store[client_id] if now - t < 3600]
    if len(rate_limit_store[client_id]) >= MAX_REQUESTS_PER_HOUR:
        return False
    rate_limit_store[client_id].append(now)
    save_rate_limits()
    return True

# --- Coordinate Validation ---
def validate_coords(lat: float, lon: float):
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90")
    if not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180")

# --- Enhanced Parameters ---
TARGET_PARAMS = {
    'T2M': {'unit':'°C','thresholds':{'extreme_hot':35,'hot':30,'cold':5,'extreme_cold':0},'importance':10},
    'PRECTOT': {'unit':'mm','thresholds':{'extreme':20,'heavy':10,'moderate':5,'light':1},'importance':9},
    'WS10M': {'unit':'m/s','thresholds':{'extreme':15,'strong':10,'moderate':5},'importance':7},
    'SNOWFALL': {'unit':'mm','thresholds':{'heavy':10,'moderate':5,'light':1},'importance':6},
    'RH2M': {'unit':'%','thresholds':{'very_humid':80,'humid':60,'dry':30},'importance':5}
}

DAYS_7 = 7
DAYS_30 = 30
DAYS_180 = 180

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)
    async def heartbeat(self):
        while True:
            await asyncio.sleep(30)
            await self.broadcast({"type":"heartbeat","timestamp":time.time()})

manager = ConnectionManager()

# --- NASA Data Fetching (FULLY FIXED) ---
async def fetch_power(session: aiohttp.ClientSession, lat: float, lon: float, 
                      start: str, end: str, 
                      params: str = "T2M,T2M_MIN,T2M_MAX,PRECTOT,WS10M,WD10M,RH2M,SNOWFALL,CLRSKY"):
    # 1. Parametreleri COMMUNITY'ye göre filtrele
    params_list = [p for p in params.split(",") if p in VALID_NASA_PARAMS[COMMUNITY]]
    if not params_list:
        raise ValueError(f"No valid parameters for community '{COMMUNITY}'")
    params_str = ",".join(params_list)

    cache_key = make_cache_key("fetch_power", lat, lon, start, end, params_str)
    cached = smart_cache.get(cache_key)
    if cached is not None:
        return cached

    # 2. Tarih formatı kontrolü: YYYYMMDD olmalı
    try:
        datetime.datetime.strptime(start, "%Y%m%d")
        datetime.datetime.strptime(end, "%Y%m%d")
    except ValueError:
        raise ValueError("Start and end dates must be in YYYYMMDD format")

    url = (
        f"{NASA_BASE}?parameters={params_str}"
        f"&community={COMMUNITY}&start={start}&end={end}"
        f"&latitude={lat}&longitude={lon}&format=JSON"
    )

    async with SEM:
        try:
            # 3. Timeout 60 saniye yapıldı
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"NASA API error {resp.status}: {text[:300]}")
                    raise RuntimeError(f"NASA API returned status {resp.status}")
                data = await resp.json()
                result = data.get("properties", {}).get("parameter", {})
                smart_cache.set(cache_key, result)
                return result
        except Exception as e:
            logger.error(f"[fetch_power error] {e}")
            return None

# --- Custom Model Wrapper ---
class ModelWrapper:
    def __init__(self, target_key: str, model):
        self.target_key = target_key
        self.model = model
    def predict(self, X):
        # 4. Sklearn feature name uyarısını önlemek için .values kullan
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict(X)
    def __getattr__(self, name):
        return getattr(self.model, name)

# --- Ensemble Model Training (DÜZELTİLDİ: .values ile uyum) ---
def prepare_ensemble_model(full_data: Dict[str, Dict[str, float]], target_key: str) -> Tuple:
    temps_dict = full_data.get(target_key, {})
    df = pd.DataFrame(list(temps_dict.items()), columns=['date', target_key])
    if df.empty:
        return None, [], [], [], {}, [], []
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date'])
    df = df[(df[target_key] > -900) & (df[target_key].notna())].copy()
    if df.shape[0] < 20 or df[target_key].nunique() < 3:
        return None, [], [], [], {}, [], []

    # Feature engineering
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['day_sin'] = np.sin(2*np.pi*df['day_of_year']/365)
    df['day_cos'] = np.cos(2*np.pi*df['day_of_year']/365)

    features = ['day_of_year','month','day_of_week','quarter','is_weekend',
                'month_sin','month_cos','day_sin','day_cos']

    if target_key == 'T2M':
        tmax_dict = full_data.get('T2M_MAX', {})
        tmin_dict = full_data.get('T2M_MIN', {})
        df['ymd'] = df['date'].dt.strftime('%Y%m%d')
        df['tmax'] = df['ymd'].map(tmax_dict).astype(float)
        df['tmin'] = df['ymd'].map(tmin_dict).astype(float)
        df['temp_range'] = df['tmax'] - df['tmin']
        df = df.dropna(subset=['tmax','tmin','temp_range'])
        features.extend(['temp_range','tmax','tmin'])

    df = df.sort_values('date')
    for lag in [7,14,30]:
        df[f'{target_key}_lag_{lag}'] = df[target_key].shift(lag)
    for window in [7,30]:
        df[f'{target_key}_roll_mean_{window}'] = df[target_key].rolling(window=window).mean()
        df[f'{target_key}_roll_std_{window}'] = df[target_key].rolling(window=window).std()
    df = df.dropna()

    lag_features = [f'{target_key}_lag_{lag}' for lag in [7,14,30]]
    roll_features = [f'{target_key}_roll_mean_{w}' for w in [7,30]] + [f'{target_key}_roll_std_{w}' for w in [7,30]]
    all_features = features + lag_features + roll_features

    X = df[all_features].values  # ← .values ile numpy array → feature name uyarısı yok
    y = df[target_key].astype(float).values

    if X.size == 0 or y.size == 0:
        return None, [], [], [], {}, [], []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    ensemble = VotingRegressor(estimators=[('rf', rf), ('gb', gb)], weights=[0.6,0.4])
    ensemble.fit(X_train, y_train)

    y_pred_train = ensemble.predict(X_train)
    y_pred_test = ensemble.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    importances = {}
    try:
        rf_model = ensemble.named_estimators_['rf']
        fi = rf_model.feature_importances_
        importances = {all_features[i]: round(float(fi[i]),4) for i in range(len(all_features))}
        importances = dict(sorted(importances.items(), key=lambda x:x[1], reverse=True)[:10])
    except Exception as e:
        logger.warning(f"Feature importance error for {target_key}: {e}")

    comparison = {}
    try:
        last30 = df.tail(30).copy()
        if not last30.empty:
            last30_X = last30[all_features].values
            last30['pred'] = ensemble.predict(last30_X)
            comparison = {
                "dates": last30['date'].dt.strftime("%Y-%m-%d").tolist(),
                "actual": [round(float(v), 2) for v in last30[target_key].tolist()],
                "predicted": [round(float(v), 2) for v in last30['pred'].tolist()],
                "error": [round(float(a - p), 2) for a, p in zip(last30[target_key], last30['pred'])]
            }
    except Exception as e:
        logger.error(f"Comparison generation error for {target_key}: {e}")

    overfitting_score = abs(train_r2 - test_r2)
    overfitting_warning = "High overfitting detected" if overfitting_score > 0.1 else "Good generalization"

    metrics = {
        f"Train_R2_{target_key}": round(train_r2, 4),
        f"Test_R2_{target_key}": round(test_r2, 4),
        f"Train_MSE_{target_key}": round(train_mse, 2),
        f"Test_MSE_{target_key}": round(test_mse, 2),
        f"Train_MAE_{target_key}": round(train_mae, 2),
        f"Test_MAE_{target_key}": round(test_mae, 2),
        f"Overfitting_Score_{target_key}": round(overfitting_score, 4),
        f"Overfitting_Warning_{target_key}": overfitting_warning,
        f"Feature_Importances_{target_key}": importances,
        f"Comparison_{target_key}": comparison
    }

    dates = df['date'].dt.strftime("%Y-%m-%d").tolist()
    values = df[target_key].astype(float).tolist()
    logger.info(f"Model trained for {target_key}: R2={test_r2:.4f}, rows={len(dates)}")

    wrapped_model = ModelWrapper(target_key, ensemble)
    return wrapped_model, dates, values, all_features, metrics, X, y

# --- Confidence Intervals ---
def forecast_with_confidence_ensemble(model, X_future, lower_pct=5, upper_pct=95):
    try:
        rf_model = model.named_estimators_['rf']
        all_tree_preds = np.array([est.predict(X_future) for est in rf_model.estimators_])
        mean_preds = np.mean(all_tree_preds, axis=0)
        lower = np.percentile(all_tree_preds, lower_pct, axis=0)
        upper = np.percentile(all_tree_preds, upper_pct, axis=0)
        return [round(float(x), 2) for x in mean_preds], \
               [round(float(x), 2) for x in lower], \
               [round(float(x), 2) for x in upper]
    except Exception as e:
        logger.warning(f"Confidence interval error: {e}, falling back to mean")
        mean = model.predict(X_future)
        return [round(float(x), 2) for x in mean], \
               [round(float(x), 2) for x in mean], \
               [round(float(x), 2) for x in mean]

# --- Forecasting (Fixed: .values uyumu) ---
def forecast_future_enhanced(model, features, start_date, days, full_data=None, historical_df=None):
    future_dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    future_data = {
        'day_of_year': [d.timetuple().tm_yday for d in future_dates],
        'month': [d.month for d in future_dates],
        'day_of_week': [d.weekday() for d in future_dates],
        'quarter': [(d.month - 1) // 3 + 1 for d in future_dates],
        'is_weekend': [1 if d.weekday() >= 5 else 0 for d in future_dates],
    }
    future_data['month_sin'] = [np.sin(2 * np.pi * m / 12) for m in future_data['month']]
    future_data['month_cos'] = [np.cos(2 * np.pi * m / 12) for m in future_data['month']]
    future_data['day_sin'] = [np.sin(2 * np.pi * d / 365) for d in future_data['day_of_year']]
    future_data['day_cos'] = [np.cos(2 * np.pi * d / 365) for d in future_data['day_of_year']]

    X_future = pd.DataFrame(future_data)

    target_key = model.target_key

    if 'tmax' in features and full_data:
        tmax_values = [float(v) for v in full_data.get('T2M_MAX', {}).values() if v and float(v) > -900]
        tmin_values = [float(v) for v in full_data.get('T2M_MIN', {}).values() if v and float(v) > -900]
        avg_tmax = float(np.mean(tmax_values[-30:])) if tmax_values else 25.0
        avg_tmin = float(np.mean(tmin_values[-30:])) if tmin_values else 15.0
        X_future['tmax'] = avg_tmax
        X_future['tmin'] = avg_tmin
        X_future['temp_range'] = avg_tmax - avg_tmin

    if historical_df is not None and not historical_df.empty:
        last_values = historical_df['value'].values
        for lag in [7, 14, 30]:
            if len(last_values) >= lag:
                X_future[f'{target_key}_lag_{lag}'] = last_values[-lag]
            else:
                X_future[f'{target_key}_lag_{lag}'] = np.mean(last_values) if len(last_values) > 0 else 0.0
        X_future[f'{target_key}_roll_mean_7'] = np.mean(last_values[-7:]) if len(last_values) >= 7 else np.mean(last_values) if len(last_values) > 0 else 0.0
        X_future[f'{target_key}_roll_mean_30'] = np.mean(last_values) if len(last_values) > 0 else 0.0
        X_future[f'{target_key}_roll_std_7'] = np.std(last_values[-7:]) if len(last_values) >= 7 else np.std(last_values) if len(last_values) > 1 else 0.0
        X_future[f'{target_key}_roll_std_30'] = np.std(last_values) if len(last_values) > 1 else 0.0
    else:
        thresholds = TARGET_PARAMS.get(target_key, {}).get('thresholds', {})
        default_mean = list(thresholds.values())[0] if thresholds else 20.0
        default_std = 5.0
        for lag in [7, 14, 30]:
            X_future[f'{target_key}_lag_{lag}'] = default_mean
        X_future[f'{target_key}_roll_mean_7'] = default_mean
        X_future[f'{target_key}_roll_mean_30'] = default_mean
        X_future[f'{target_key}_roll_std_7'] = default_std
        X_future[f'{target_key}_roll_std_30'] = default_std

    X_future = X_future.reindex(columns=features, fill_value=0.0).values  # ← .values eklendi

    mean_preds, lower, upper = forecast_with_confidence_ensemble(model, X_future)
    logger.info(f"Forecast generated for {target_key}: {days} days, mean range {min(mean_preds):.1f}-{max(mean_preds):.1f}")
    return {"mean": mean_preds, "lower": lower, "upper": upper}

# --- Advanced Risk Analysis (unchanged) ---
def advanced_risk_analysis(activity: str, predictions: Dict, lat: float, lon: float) -> Dict:
    if activity not in ACTIVITY_PROFILES:
        activity = 'picnic'
        logger.warning(f"Unknown activity '{activity}', defaulting to 'picnic'")
    profile = ACTIVITY_PROFILES[activity]
    risk_score = 0
    risk_messages = []
    recommendations = []

    if 'T2M' in predictions:
        temps = predictions['T2M'].get('mean', [])
        opt_min, opt_max = profile['optimal_temp']
        for temp in temps:
            if temp < opt_min - 10:
                risk_score += 3
            elif temp < opt_min:
                risk_score += 1
            elif temp > opt_max + 10:
                risk_score += 3
            elif temp > opt_max:
                risk_score += 1
        avg_temp = np.mean(temps) if temps else 20
        if avg_temp < opt_min:
            risk_messages.append(f"🥶 Sıcaklık optimal aralığın altında ({avg_temp:.1f}°C)")
            recommendations.append("Kalın kıyafetler giyin, sıcak içecekler hazırlayın")
        elif avg_temp > opt_max:
            risk_messages.append(f"🥵 Sıcaklık optimal aralığın üstünde ({avg_temp:.1f}°C)")
            recommendations.append("Bol su için, güneş kremi kullanın, gölgede kalın")

    if 'PRECTOT' in predictions and 'rain' in profile['sensitivities']:
        rain = predictions['PRECTOT'].get('mean', [])
        rainy_days = sum(1 for r in rain if r > profile['max_rain'])
        if rainy_days > 0:
            risk_score += rainy_days * 2
            risk_messages.append(f"🌧️ {rainy_days} gün yağmur bekleniyor")
            recommendations.append("Yağmurluk ve şemsiye almayı unutmayın")

    if 'WS10M' in predictions and 'wind' in profile['sensitivities']:
        wind = predictions['WS10M'].get('mean', [])
        windy_days = sum(1 for w in wind if w > profile['max_wind'])
        if windy_days > 0:
            risk_score += windy_days
            risk_messages.append(f"💨 {windy_days} gün rüzgarlı hava bekleniyor")
            recommendations.append("Hafif eşyalarınızı sabitlemek için önlem alın")

    if 'SNOWFALL' in predictions:
        snow = predictions['SNOWFALL'].get('mean', [])
        snowy_days = sum(1 for s in snow if s > 1)
        if snowy_days > 0:
            risk_score += snowy_days * 2
            risk_messages.append(f"❄️ {snowy_days} gün kar yağışı bekleniyor")
            recommendations.append("Kış lastiği ve zincir kontrol edin")

    if risk_score >= 15:
        risk_level = "Çok Yüksek"
        risk_color = "red"
    elif risk_score >= 10:
        risk_level = "Yüksek"
        risk_color = "orange"
    elif risk_score >= 5:
        risk_level = "Orta"
        risk_color = "yellow"
    else:
        risk_level = "Düşük"
        risk_color = "green"

    best_days = []
    if 'T2M' in predictions:
        temps = predictions['T2M'].get('mean', [])
        rain_means = predictions.get('PRECTOT', {}).get('mean', [0.0] * len(temps))
        wind_means = predictions.get('WS10M', {}).get('mean', [0.0] * len(temps))
        for i, temp in enumerate(temps):
            day_risk = 0
            if temp < profile['optimal_temp'][0] or temp > profile['optimal_temp'][1]:
                day_risk += 2
            if i < len(rain_means) and rain_means[i] > profile['max_rain']:
                day_risk += 3
            if i < len(wind_means) and wind_means[i] > profile['max_wind']:
                day_risk += 1
            if day_risk <= 2:
                best_days.append(i)

    logger.info(f"Risk analysis for {activity} at {lat},{lon}: score={risk_score}, best_days={len(best_days)}")
    return {
        "activity": activity,
        "activity_description": profile['description'],
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_messages": risk_messages,
        "recommendations": recommendations,
        "best_days": best_days[:3],
        "optimal_temp_range": profile['optimal_temp']
    }

# --- PDF Report Generation (unchanged) ---
def generate_pdf_report(prediction_data: Dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "NASA PARADE - Weather Intelligence Report", 0, 1, "C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Location Information", 0, 1)
    pdf.set_font("Arial", "", 11)
    coords = prediction_data.get('coords', {})
    pdf.cell(0, 7, f"Latitude: {coords.get('lat', 'N/A')}, Longitude: {coords.get('lon', 'N/A')}", 0, 1)
    pdf.ln(3)

    if 'user_risk' in prediction_data:
        risk = prediction_data['user_risk']
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, f"Activity: {risk.get('activity', 'N/A')}", 0, 1)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, risk.get('activity_description', ''))
        pdf.ln(2)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, f"Risk Level: {risk.get('risk_level', 'N/A')} (Score: {risk.get('risk_score', 0)})", 0, 1)
        pdf.set_font("Arial", "", 10)
        for msg in risk.get('risk_messages', []):
            clean_msg = msg.replace('🥶', '[Cold]').replace('🥵', '[Hot]').replace('🌧️', '[Rain]').replace('💨', '[Wind]').replace('❄️', '[Snow]')
            pdf.cell(0, 6, f"  - {clean_msg}", 0, 1)
        pdf.ln(2)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, "Recommendations:", 0, 1)
        pdf.set_font("Arial", "", 10)
        for rec in risk.get('recommendations', []):
            pdf.multi_cell(0, 5, f"  * {rec}")
        best_days = risk.get('best_days', [])
        if best_days:
            pdf.ln(2)
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 7, f"Best Days for Activity: Day {', '.join(map(str, [d+1 for d in best_days]))}", 0, 1)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "7-Day Forecast Summary", 0, 1)
    pdf.set_font("Arial", "", 10)
    summaries = prediction_data.get('summaries', {})
    for key, value in summaries.items():
        if not key.startswith('special'):
            clean_value = value.replace('🌡️', '[Temp]').replace('🌧️', '[Rain]').replace('💨', '[Wind]').replace('❄️', '[Snow]').replace('💧', '[Humidity]')
            pdf.multi_cell(0, 5, f"  - {clean_value}")

    special_warnings = [v for k, v in summaries.items() if k.startswith('special')]
    if special_warnings:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 7, "SPECIAL WARNINGS:", 0, 1)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 10)
        for warning in special_warnings:
            clean_warning = warning.replace('🔥', '[Heat]').replace('🥶', '[Cold]').replace('⚠️', '[Alert]')
            pdf.multi_cell(0, 5, f"  ! {clean_warning}")

    pdf.ln(3)
    metrics = prediction_data.get('metrics', {})
    if metrics:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Model Performance Metrics", 0, 1)
        pdf.set_font("Arial", "", 9)
        for key, value in metrics.items():
            if not isinstance(value, dict) and not key.startswith('Feature') and not key.startswith('Comparison') and not key.startswith('Overfitting_Warning'):
                pdf.cell(0, 5, f"  {key}: {value}", 0, 1)

    pdf.ln(2)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 5, "Powered by NASA POWER API & Advanced ML Ensemble Models", 0, 1, "C")
    pdf.cell(0, 5, "This report is for informational purposes only.", 0, 1, "C")
    pdf_bytes = pdf.output(dest='S').encode('latin-1', errors='ignore')
    logger.info("PDF report generated successfully")
    return pdf_bytes

# --- ENDPOINTS (forecast endpoint düzeltildi) ---
@app.get("/health")
async def health():
    cache_stats = smart_cache.get_stats()
    return {
        "status": "healthy",
        "service": "NASA Parade Ultimate Backend (Fixed)",
        "version": "3.1.1",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "cache_stats": cache_stats,
        "active_websockets": len(manager.active_connections)
    }

@app.get("/predict")
async def predict(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    activity: str = Query("picnic", description="Activity type"),
    client_id: str = Query("anonymous", description="Client identifier for rate limiting")
):
    try:
        validate_coords(lat, lon)
    except HTTPException:
        raise
    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 100 requests per hour.")
    start_history = f"{START_YEAR}0101"
    end_history = f"{CURRENT_YEAR-1}1231"
    today = datetime.date.today()
    cache_key = make_cache_key("predict_v3", lat, lon, start_history, end_history, activity)
    cached = smart_cache.get(cache_key)
    if cached is not None:
        resp = cached.copy()
        resp["data_source"] = "cache"
        resp["cache_hit"] = True
        logger.info(f"Cache hit for prediction: {lat},{lon}")
        return resp

    async with aiohttp.ClientSession() as session:
        history_params = await fetch_power(session, lat, lon, start_history, end_history)
    if not history_params:
        raise HTTPException(status_code=500, detail="Failed to fetch NASA historical data. Check coordinates or try again later.")

    all_metrics = {}
    all_predictions = {}
    all_summaries = {}
    history_dates = []
    history_temps = []

    for target_key in TARGET_PARAMS.keys():
        try:
            result = prepare_ensemble_model(history_params, target_key)
            if len(result) == 7 and result[0] is None:
                all_predictions[target_key] = {"7d": {"mean": [], "lower": [], "upper": []}, "30d": {"mean": [], "lower": [], "upper": []}, "180d": {"mean": [], "lower": [], "upper": []}}
                all_summaries[f"summary_{target_key}"] = f"⚠️ {target_key}: Insufficient data"
                continue
            model, all_dates, all_values, features, metrics, X, y = result
            all_metrics.update(metrics)
            hist_df = pd.DataFrame({'date': all_dates, 'value': all_values})
            preds_7 = forecast_future_enhanced(model, features, today, DAYS_7, history_params, hist_df)
            preds_30 = forecast_future_enhanced(model, features, today, DAYS_30, history_params, hist_df)
            preds_180 = forecast_future_enhanced(model, features, today, DAYS_180, history_params, hist_df)
            all_predictions[target_key] = {
                "7d": preds_7,
                "30d": preds_30,
                "180d": preds_180
            }
            mean_7d = preds_7['mean']
            if target_key == 'T2M':
                avg_temp = np.mean(mean_7d)
                all_summaries[f"summary_{target_key}"] = f"🌡️ Avg temperature: {avg_temp:.1f}°C (range: {min(mean_7d):.1f}-{max(mean_7d):.1f}°C)"
                hot_streak = sum(1 for t in mean_7d if t > 35)
                cold_streak = sum(1 for t in mean_7d if t < 0)
                if hot_streak >= 3:
                    all_summaries["special_heatwave"] = f"🔥 HEATWAVE ALERT: {hot_streak} days above 35°C"
                if cold_streak >= 3:
                    all_summaries["special_coldwave"] = f"🥶 COLD WAVE ALERT: {cold_streak} days below 0°C"
                history_dates = all_dates[-30:] if len(all_dates) >= 30 else all_dates
                history_temps = all_values[-30:] if len(all_values) >= 30 else all_values
            elif target_key == 'PRECTOT':
                total_rain = sum(mean_7d)
                rainy_days = sum(1 for r in mean_7d if r > 1)
                all_summaries[f"summary_{target_key}"] = f"🌧️ Total rain: {total_rain:.1f}mm over {rainy_days} days"
                extreme_rain = sum(1 for r in mean_7d if r > 20)
                if extreme_rain > 0:
                    all_summaries["special_rain"] = f"⚠️ EXTREME RAINFALL: {extreme_rain} days with >20mm"
            elif target_key == 'WS10M':
                avg_wind = np.mean(mean_7d)
                windy_days = sum(1 for w in mean_7d if w > 10)
                all_summaries[f"summary_{target_key}"] = f"💨 Avg wind: {avg_wind:.1f}m/s, {windy_days} strong wind days"
                if windy_days >= 4:
                    all_summaries["special_wind"] = f"⚠️ WINDY PERIOD: {windy_days} days with strong winds"
            elif target_key == 'SNOWFALL':
                total_snow = sum(mean_7d)
                snowy_days = sum(1 for s in mean_7d if s > 1)
                if total_snow > 0:
                    all_summaries[f"summary_{target_key}"] = f"❄️ Snow expected: {total_snow:.1f}mm over {snowy_days} days"
            elif target_key == 'RH2M':
                avg_humidity = np.mean(mean_7d)
                all_summaries[f"summary_{target_key}"] = f"💧 Avg humidity: {avg_humidity:.1f}%"
        except Exception as e:
            logger.error(f"Error processing {target_key}: {e}")
            all_predictions[target_key] = {"7d": {"mean": [], "lower": [], "upper": []}, "30d": {"mean": [], "lower": [], "upper": []}, "180d": {"mean": [], "lower": [], "upper": []}}
            all_summaries[f"summary_{target_key}"] = f"❌ {target_key}: Processing error"

    preds_for_risk = {k: v.get('7d', {}) for k, v in all_predictions.items() if '7d' in v}
    user_risk = advanced_risk_analysis(activity, preds_for_risk, lat, lon)

    tomorrow = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow_temp = all_predictions.get('T2M', {}).get('7d', {}).get('mean', [None])[0] if all_predictions.get('T2M', {}).get('7d', {}).get('mean') else "N/A"
    tomorrow_rain = all_predictions.get('PRECTOT', {}).get('7d', {}).get('mean', [0])[0] if all_predictions.get('PRECTOT', {}).get('7d', {}).get('mean') else 0
    tomorrow_summary = f"📅 {tomorrow}: {tomorrow_temp}°C"
    if tomorrow_rain > 1:
        tomorrow_summary += f", {tomorrow_rain:.1f}mm rain expected"

    response = {
        "coords": {"lat": lat, "lon": lon},
        "tomorrow_summary": tomorrow_summary,
        "predictions": all_predictions,
        "summaries": all_summaries,
        "metrics": all_metrics,
        "user_risk": user_risk,
        "history": {
            "dates": history_dates,
            "temperatures": history_temps
        },
        "model": "Ensemble (RandomForest + GradientBoosting) v3.1",
        "data_source": "nasa_power_api",
        "cache_hit": False
    }

    smart_cache.set(cache_key, response)
    try:
        await manager.broadcast({
            "type": "new_prediction",
            "coords": {"lat": lat, "lon": lon},
            "summary": tomorrow_summary,
            "risk_level": user_risk['risk_level']
        })
    except Exception as e:
        logger.error(f"WebSocket broadcast error: {e}")

    logger.info(f"Prediction completed for {lat},{lon}: {len(all_predictions)} parameters")
    return response

@app.get("/forecast")
async def forecast_single_day(
    lat: float = Query(...),
    lon: float = Query(...),
    target_date: str = Query(..., description="YYYY-MM-DD format")
):
    try:
        validate_coords(lat, lon)
    except HTTPException:
        raise
    try:
        target_dt = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
        today = datetime.date.today()
        if target_dt < today:
            raise HTTPException(status_code=400, detail="Target date must be today or future")
        start_history = f"{START_YEAR}0101"
        end_history = f"{CURRENT_YEAR-1}1231"
        async with aiohttp.ClientSession() as session:
            history_params = await fetch_power(session, lat, lon, start_history, end_history)
        if not history_params:
            raise HTTPException(status_code=500, detail="Failed to fetch NASA historical data. Check coordinates or try again later.")
        result = prepare_ensemble_model(history_params, 'T2M')
        if result[0] is None:
            raise HTTPException(status_code=500, detail="Cannot train model: insufficient data")
        model, dates, values, features, metrics, X, y = result
        target_mmdd = f"{target_dt.month:02d}{target_dt.day:02d}"
        def calc_historical_avg(param_dict, mmdd_key):
            vals = [float(v) for k, v in param_dict.items() if k[4:] == mmdd_key and v and float(v) > -900]
            return round(np.mean(vals), 2) if vals else None
        hist_df = pd.DataFrame({'date': dates, 'value': values})
        days_ahead = (target_dt - today).days
        preds = forecast_future_enhanced(model, features, today, days_ahead + 1, history_params, hist_df)
        pred_temp = preds['mean'][-1]
        pred_lower = preds['lower'][-1]
        pred_upper = preds['upper'][-1]
        historical_data = {
            "T2M_historical_avg": calc_historical_avg(history_params.get('T2M', {}), target_mmdd),
            "T2M_MAX_historical_avg": calc_historical_avg(history_params.get('T2M_MAX', {}), target_mmdd),
            "T2M_MIN_historical_avg": calc_historical_avg(history_params.get('T2M_MIN', {}), target_mmdd),
            "PRECTOT_historical_avg": calc_historical_avg(history_params.get('PRECTOT', {}), target_mmdd),
            "WS10M_historical_avg": calc_historical_avg(history_params.get('WS10M', {}), target_mmdd),
        }
        return {
            "coords": {"lat": lat, "lon": lon},
            "target_date": target_date,
            "days_ahead": days_ahead,
            "prediction": {
                "T2M_mean": pred_temp,
                "T2M_lower_bound": pred_lower,
                "T2M_upper_bound": pred_upper,
                "confidence_interval": "90%"
            },
            "historical_context": historical_data,
            "model_quality": {
                "R2_score": metrics.get("Test_R2_T2M", "N/A"),
                "MAE": metrics.get("Test_MAE_T2M", "N/A")
            }
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Single day forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Diğer endpoint'ler (pdf_report, heatmap, download/csv, download/json, trends, activities, ws, cache/stats, cache/clear) aynı kalır.
# Uzunluk sınırı nedeniyle burada tekrar yazmadım, ancak gerekirse tamamını da verebilirim.

# --- Startup/Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(manager.heartbeat())
    logger.info("Application started with WebSocket heartbeat")

@app.on_event("shutdown")
async def shutdown_event():
    save_rate_limits()
    logger.info("Application shutdown: Rate limits saved")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")