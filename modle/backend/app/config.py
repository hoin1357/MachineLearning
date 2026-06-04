from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from model_tuning_settings import (
    BUSY_LABEL_RATIO,
    CLASSIFIER_PARAMS,
    FORECAST_HORIZON_DAYS,
    IMPUTATION_MODEL_PARAMS,
    REGRESSOR_PARAMS,
    TOP_10_WEIGHT,
    TOP_20_WEIGHT,
)


BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "backend" / "app"
DATA_DIR = BASE_DIR / "backend" / "data"
WEATHER_HISTORY_DIR = DATA_DIR / "weather_history"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

VISITORS_FILE = DATA_DIR / "visitors.csv"
EVENTS_FILE = DATA_DIR / "events.csv"
PREDICTION_CACHE_FILE = ARTIFACTS_DIR / "portable_predictions.csv"
PREDICTION_METADATA_FILE = ARTIFACTS_DIR / "portable_predictions_metadata.json"
WEATHER_CACHE_FILE = ARTIFACTS_DIR / "weather_api_cache.json"
TRANSPORT_VISITORS_FILE = ARTIFACTS_DIR / "external_transport" / "seoul_grand_park_visitors_with_daegongwon_subway.csv"

SEOUL_TIMEZONE = ZoneInfo("Asia/Seoul")
TODAY = datetime.now(SEOUL_TIMEZONE).date()

WEATHER_API_SERVICE_KEY = os.getenv("WEATHER_API_SERVICE_KEY", "")
WEATHER_API_BASE_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
WEATHER_GRID_X = 60
WEATHER_GRID_Y = 124
MID_WEATHER_API_BASE_URL = "https://apis.data.go.kr/1360000/MidFcstInfoService"
MID_TEMPERATURE_REGION_ID = "11B10102"  # 과천
MID_LAND_REGION_ID = "11B00000"  # 서울, 인천, 경기도
WEATHER_STRATEGY_VERSION = "short-mid-v1"
MODEL_FEATURE_VERSION = "future-safe-weather-event-v2"

MONTH_NAMES_KO = ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]
WEEKDAY_NAMES_KO = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
