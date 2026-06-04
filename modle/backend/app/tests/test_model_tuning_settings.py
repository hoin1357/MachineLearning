from app import config
from app.services import model_runtime
from model_tuning_settings import (
    BUSY_LABEL_RATIO,
    CLASSIFIER_PARAMS,
    FORECAST_HORIZON_DAYS,
    IMPUTATION_MODEL_PARAMS,
    REGRESSOR_PARAMS,
    TOP_10_WEIGHT,
    TOP_20_WEIGHT,
)


def test_app_config_uses_top_level_tuning_settings() -> None:
    assert config.FORECAST_HORIZON_DAYS == FORECAST_HORIZON_DAYS
    assert config.BUSY_LABEL_RATIO == BUSY_LABEL_RATIO
    assert config.TOP_10_WEIGHT == TOP_10_WEIGHT
    assert config.TOP_20_WEIGHT == TOP_20_WEIGHT
    assert config.MODEL_FEATURE_VERSION == model_runtime.MODEL_FEATURE_VERSION


def test_model_runtime_uses_top_level_model_params() -> None:
    assert model_runtime.CLASSIFIER_PARAMS == CLASSIFIER_PARAMS
    assert model_runtime.REGRESSOR_PARAMS == REGRESSOR_PARAMS
    assert model_runtime.IMPUTATION_MODEL_PARAMS == IMPUTATION_MODEL_PARAMS
