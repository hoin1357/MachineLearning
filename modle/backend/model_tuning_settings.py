from __future__ import annotations

# 모델 튜닝은 이 파일에서만 수정하면 됩니다.
#
# 값을 바꾼 뒤에는 기존 예측 캐시를 지우고 다시 생성해야 새 설정이 반영됩니다.
# 지울 파일:
# - backend/data/artifacts/portable_predictions.csv
# - backend/data/artifacts/portable_predictions_metadata.json
# 다시 생성:
# - py scripts/prepare_artifacts.py


# 예측할 미래 날짜 범위입니다. 365면 오늘부터 1년 뒤까지 예측합니다.
FORECAST_HORIZON_DAYS = 365


# 1단계 혼잡일 분류 모델 설정입니다.
# CatBoostClassifier가 "혼잡일일 가능성"을 먼저 예측합니다.
CLASSIFIER_PARAMS = {
    "loss_function": "Logloss",
    "depth": 6,
    "learning_rate": 0.06,
    "iterations": 800,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}


# 2단계 방문객 수 예측 모델 설정입니다.
# CatBoostRegressor가 최종 방문객 수를 예측합니다.
REGRESSOR_PARAMS = {
    "loss_function": "RMSE",
    "depth": 10,
    "learning_rate": 0.04,
    "iterations": 1200,
    "l2_leaf_reg": 8.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}


# 방문객 데이터의 긴 누락 구간을 채우는 보정 모델 설정입니다.
# 보통은 위 CLASSIFIER_PARAMS / REGRESSOR_PARAMS부터 먼저 튜닝하세요.
IMPUTATION_MODEL_PARAMS = {
    "loss_function": "RMSE",
    "depth": 6,
    "learning_rate": 0.05,
    "iterations": 700,
    "l2_leaf_reg": 5.0,
    "min_data_in_leaf": 5,
    "random_strength": 1.0,
    "verbose": False,
    "random_seed": 42,
    "allow_writing_files": False,
}


# 방문객 수 상위 몇 퍼센트를 "혼잡일"로 볼지 정합니다.
# 0.15는 상위 15% 날짜를 혼잡일로 본다는 뜻입니다.
BUSY_LABEL_RATIO = 0.15


# 방문객 수가 많은 날에 더 큰 학습 가중치를 줍니다.
# 피크 날짜를 더 중요하게 맞추고 싶으면 이 값을 올려볼 수 있습니다.
TOP_10_WEIGHT = 4.0
TOP_20_WEIGHT = 2.0
