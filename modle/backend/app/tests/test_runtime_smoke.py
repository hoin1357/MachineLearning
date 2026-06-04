import pandas as pd

from app.config import MODEL_FEATURE_VERSION, TODAY
from app.services.model_runtime import (
    FEATURE_COLUMNS,
    TRANSPORT_TRAINING_FEATURE_COLUMNS,
    PortablePredictionRuntime,
)


def test_prediction_cache_reuses_recent_cache_when_today_is_covered(tmp_path, monkeypatch) -> None:
    cache_file = tmp_path / "portable_predictions.csv"
    metadata_file = tmp_path / "portable_predictions_metadata.json"
    cache_file.write_text(f"date,predicted_visitors\n{TODAY.isoformat()},1000\n", encoding="utf-8")
    metadata_file.write_text(
        '{"today":"2000-01-01","known_data_end":"2025-07-31","horizon_days":365,'
        f'"weather_strategy_version":"short-mid-v1","model_feature_version":"{MODEL_FEATURE_VERSION}"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr("app.services.model_runtime.PREDICTION_CACHE_FILE", cache_file)
    monkeypatch.setattr("app.services.model_runtime.PREDICTION_METADATA_FILE", metadata_file)

    runtime = PortablePredictionRuntime()

    assert runtime._load_prediction_cache() is True


def test_prediction_cache_rejects_cache_that_ends_before_today(tmp_path, monkeypatch) -> None:
    cache_file = tmp_path / "portable_predictions.csv"
    metadata_file = tmp_path / "portable_predictions_metadata.json"
    cache_file.write_text("date,predicted_visitors\n2000-01-01,1000\n", encoding="utf-8")
    metadata_file.write_text(
        '{"today":"2000-01-01","known_data_end":"2025-07-31","horizon_days":365,'
        '"weather_strategy_version":"short-mid-v1"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("app.services.model_runtime.PREDICTION_CACHE_FILE", cache_file)
    monkeypatch.setattr("app.services.model_runtime.PREDICTION_METADATA_FILE", metadata_file)

    runtime = PortablePredictionRuntime()

    assert runtime._load_prediction_cache() is False


def test_runtime_can_predict_today_forward() -> None:
    runtime = PortablePredictionRuntime()
    runtime.initialize()
    prediction = runtime.predict_for_date(TODAY)
    assert prediction["date"] == TODAY.isoformat()
    assert prediction["predictedVisitors"] >= 0
    assert prediction["congestionLevel"] in {"한산함", "보통", "붐빔", "매우 붐빔"}


def test_final_prediction_features_are_future_safe() -> None:
    forbidden_fragments = ["지연값", "이동평균", "이동표준편차", "대공원역"]

    assert all(
        fragment not in feature
        for feature in FEATURE_COLUMNS
        for fragment in forbidden_fragments
    )
    assert "대공원역하차" in TRANSPORT_TRAINING_FEATURE_COLUMNS


def test_calendar_features_include_long_holiday_and_peak_season() -> None:
    runtime = PortablePredictionRuntime.__new__(PortablePredictionRuntime)
    frame = pd.DataFrame({"일자": pd.to_datetime(["2025-05-05", "2025-08-01"])})

    featured = runtime._add_calendar_features(frame)

    assert int(featured.loc[0, "연휴여부"]) == 1
    assert int(featured.loc[0, "봄성수기여부"]) == 1
    assert int(featured.loc[1, "여름방학시즌여부"]) == 1
    assert int(featured.loc[1, "방학성수기여부"]) == 1


def test_weather_features_include_too_cold_and_too_hot_flags() -> None:
    runtime = PortablePredictionRuntime.__new__(PortablePredictionRuntime)
    frame = pd.DataFrame({"평균기온": [-1.0, 31.0], "일강수량": [0.0, 12.0]})

    featured = runtime._add_weather_derived_features(frame)

    assert int(featured.loc[0, "너무추움여부"]) == 1
    assert int(featured.loc[1, "너무더움여부"]) == 1
    assert int(featured.loc[1, "강수여부"]) == 1
