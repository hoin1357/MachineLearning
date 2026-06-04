from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from app.config import (
    ARTIFACTS_DIR,
    BUSY_LABEL_RATIO,
    CLASSIFIER_PARAMS,
    FORECAST_HORIZON_DAYS,
    IMPUTATION_MODEL_PARAMS,
    MODEL_FEATURE_VERSION,
    PREDICTION_CACHE_FILE,
    PREDICTION_METADATA_FILE,
    REGRESSOR_PARAMS,
    TODAY,
    TOP_10_WEIGHT,
    TOP_20_WEIGHT,
    TRANSPORT_VISITORS_FILE,
    VISITORS_FILE,
    WEATHER_HISTORY_DIR,
    EVENTS_FILE,
    WEATHER_STRATEGY_VERSION,
)
from app.domain.commentary import (
    build_date_comment,
    color_for_prediction,
    congestion_level_from_visitors,
    random_comment_for_level,
)
from app.services.event_season_service import EventSeasonService, SEASON_COLUMNS
from app.services.holiday_service import is_holiday
from app.services.weather_service import WeatherService


FEATURE_COLUMNS = [
    "년",
    "월",
    "일",
    "요일번호",
    "주말여부",
    "분기",
    "연중일수",
    "연중주차",
    "시작후경과일",
    "월_sin",
    "월_cos",
    "연중일수_sin",
    "연중일수_cos",
    "공휴일여부",
    "주말또는공휴일여부",
    "연휴여부",
    "평균기온",
    "일강수량",
    "강수여부",
    "폭우여부",
    "한파일수",
    "무더위일수",
    "너무추움여부",
    "너무더움여부",
    *SEASON_COLUMNS,
    "season_active_count",
    "행사여부",
    "봄성수기여부",
    "여름방학시즌여부",
    "겨울방학시즌여부",
    "방학성수기여부",
]

TRANSPORT_TRAINING_FEATURE_COLUMNS = [
    "대공원역승차",
    "대공원역하차",
    "대공원역승하차",
]

IMPUTATION_FEATURES = [
    "년",
    "월",
    "일",
    "요일번호",
    "주말여부",
    "분기",
    "연중일수",
    "연중주차",
    "시작후경과일",
    "월_sin",
    "월_cos",
    "연중일수_sin",
    "연중일수_cos",
    "공휴일여부",
    "주말또는공휴일여부",
    "연휴여부",
    "평균기온",
    "일강수량",
    "강수여부",
    "폭우여부",
    "한파일수",
    "무더위일수",
    "너무추움여부",
    "너무더움여부",
    *SEASON_COLUMNS,
    "season_active_count",
    "행사여부",
    "봄성수기여부",
    "여름방학시즌여부",
    "겨울방학시즌여부",
    "방학성수기여부",
]

CALENDAR_ORIGIN = pd.Timestamp("2015-01-01")

@dataclass
class RuntimeState:
    initialized: bool = False
    source: str = "cold-start"
    error: str | None = None


class PortablePredictionRuntime:
    def __init__(self) -> None:
        self.state = RuntimeState()
        self.weather_service = WeatherService(WEATHER_HISTORY_DIR)
        self.event_service = EventSeasonService.from_csv(EVENTS_FILE)
        self.known_data_end: date | None = None
        self.prediction_frame: pd.DataFrame | None = None

    def initialize(self) -> None:
        if self.state.initialized:
            return
        try:
            if self._load_prediction_cache():
                self.state = RuntimeState(initialized=True, source="disk-cache")
                return
            self.prediction_frame = self._build_prediction_frame()
            self._save_prediction_cache()
            self.state = RuntimeState(initialized=True, source="fresh-train")
        except Exception as exc:  # noqa: BLE001
            self.state = RuntimeState(initialized=False, source="failed", error=str(exc))
            raise

    def health(self) -> dict[str, Any]:
        return {
            "initialized": self.state.initialized,
            "source": self.state.source,
            "error": self.state.error,
            "today": TODAY.isoformat(),
            "knownDataEnd": self.known_data_end.isoformat() if self.known_data_end else None,
            "predictionRangeEnd": self.max_supported_date.isoformat() if self.state.initialized else None,
            "modelFeatureVersion": MODEL_FEATURE_VERSION,
        }

    @property
    def min_supported_date(self) -> date:
        return TODAY

    @property
    def max_supported_date(self) -> date:
        if self.prediction_frame is not None and not self.prediction_frame.empty:
            return self.prediction_frame["date"].max().date()
        return TODAY + timedelta(days=FORECAST_HORIZON_DAYS)

    def _load_prediction_cache(self) -> bool:
        if not PREDICTION_CACHE_FILE.exists() or not PREDICTION_METADATA_FILE.exists():
            return False
        metadata = json.loads(PREDICTION_METADATA_FILE.read_text(encoding="utf-8"))
        if metadata.get("weather_strategy_version") != WEATHER_STRATEGY_VERSION:
            return False
        if metadata.get("model_feature_version") != MODEL_FEATURE_VERSION:
            return False
        prediction_frame = pd.read_csv(PREDICTION_CACHE_FILE, parse_dates=["date"])
        if prediction_frame.empty or prediction_frame["date"].max().date() < TODAY:
            return False
        prediction_frame["congestion_level"] = prediction_frame["predicted_visitors"].map(congestion_level_from_visitors)
        self.known_data_end = pd.to_datetime(metadata["known_data_end"]).date()
        self.prediction_frame = prediction_frame
        return True

    def _save_prediction_cache(self) -> None:
        assert self.prediction_frame is not None
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.prediction_frame.to_csv(PREDICTION_CACHE_FILE, index=False, encoding="utf-8-sig")
        PREDICTION_METADATA_FILE.write_text(
            json.dumps(
                {
                    "today": TODAY.isoformat(),
                    "known_data_end": self.known_data_end.isoformat(),
                    "horizon_days": FORECAST_HORIZON_DAYS,
                    "weather_strategy_version": WEATHER_STRATEGY_VERSION,
                    "model_feature_version": MODEL_FEATURE_VERSION,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_visitors(self) -> pd.DataFrame:
        visitors = pd.read_csv(VISITORS_FILE)
        visitors["일자"] = pd.to_datetime(visitors["일자"].astype(str))
        visitors = visitors.sort_values("일자").reset_index(drop=True)
        self.known_data_end = visitors.loc[visitors["방문인원수"].notna(), "일자"].max().date()
        return visitors

    def _add_calendar_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()
        dataframe["년"] = dataframe["일자"].dt.year
        dataframe["월"] = dataframe["일자"].dt.month
        dataframe["일"] = dataframe["일자"].dt.day
        dataframe["요일번호"] = dataframe["일자"].dt.dayofweek
        dataframe["주말여부"] = (dataframe["요일번호"] >= 5).astype(int)
        dataframe["분기"] = dataframe["일자"].dt.quarter
        dataframe["연중일수"] = dataframe["일자"].dt.dayofyear
        dataframe["연중주차"] = dataframe["일자"].dt.isocalendar().week.astype(int)
        dataframe["시작후경과일"] = (dataframe["일자"] - CALENDAR_ORIGIN).dt.days
        dataframe["월_sin"] = np.sin(2 * np.pi * dataframe["월"] / 12)
        dataframe["월_cos"] = np.cos(2 * np.pi * dataframe["월"] / 12)
        dataframe["연중일수_sin"] = np.sin(2 * np.pi * dataframe["연중일수"] / 365.25)
        dataframe["연중일수_cos"] = np.cos(2 * np.pi * dataframe["연중일수"] / 365.25)
        dataframe["공휴일여부"] = dataframe["일자"].dt.strftime("%Y-%m-%d").map(lambda text: int(is_holiday(text)))
        dataframe["주말또는공휴일여부"] = ((dataframe["주말여부"] == 1) | (dataframe["공휴일여부"] == 1)).astype(int)
        dataframe["연휴여부"] = dataframe["일자"].map(lambda value: int(self._is_long_holiday(pd.Timestamp(value).date())))
        dataframe = self._add_peak_season_features(dataframe)
        return dataframe

    def _add_weather_derived_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()
        dataframe["강수여부"] = (dataframe["일강수량"] > 0).astype(int)
        dataframe["폭우여부"] = (dataframe["일강수량"] >= 10).astype(int)
        dataframe["한파일수"] = (dataframe["평균기온"] <= 0).astype(int)
        dataframe["무더위일수"] = (dataframe["평균기온"] >= 30).astype(int)
        dataframe["너무추움여부"] = (dataframe["평균기온"] < 5).astype(int)
        dataframe["너무더움여부"] = (dataframe["평균기온"] > 28).astype(int)
        return dataframe

    def _is_rest_day(self, target_date: date) -> bool:
        return target_date.weekday() >= 5 or is_holiday(target_date)

    def _is_long_holiday(self, target_date: date) -> bool:
        if not self._is_rest_day(target_date):
            return False

        block_length = 1
        cursor = target_date - timedelta(days=1)
        while self._is_rest_day(cursor):
            block_length += 1
            cursor -= timedelta(days=1)

        cursor = target_date + timedelta(days=1)
        while self._is_rest_day(cursor):
            block_length += 1
            cursor += timedelta(days=1)

        return block_length >= 3

    def _add_peak_season_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()
        month_day = dataframe["일자"].dt.month * 100 + dataframe["일자"].dt.day
        dataframe["봄성수기여부"] = month_day.between(320, 515).astype(int)
        dataframe["여름방학시즌여부"] = month_day.between(720, 831).astype(int)
        dataframe["겨울방학시즌여부"] = ((month_day >= 1220) | (month_day <= 228)).astype(int)
        dataframe["방학성수기여부"] = (
            (dataframe["봄성수기여부"] == 1)
            | (dataframe["여름방학시즌여부"] == 1)
            | (dataframe["겨울방학시즌여부"] == 1)
        ).astype(int)
        return dataframe

    def _add_event_derived_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()
        dataframe["행사여부"] = (dataframe["season_active_count"] > 0).astype(int)
        return dataframe

    def _fill_missing_visitors(self, visitors: pd.DataFrame) -> pd.DataFrame:
        frame = visitors.copy()
        historical_weather = self.weather_service.weather_history.rename(
            columns={"평균기온": "평균기온", "일강수량": "일강수량"}
        )
        frame = frame.merge(historical_weather[["일자", "평균기온", "일강수량"]], on="일자", how="left")
        frame = self.event_service.merge_historical_flags(frame)
        frame = self._add_event_derived_features(frame)
        frame = self._add_calendar_features(frame)
        frame = self._add_weather_derived_features(frame)

        single_missing_date = pd.Timestamp("2019-09-07")
        yearly_mask = (frame["년"] == 2019) & frame["방문인원수"].notna()
        weekday_code = int(single_missing_date.dayofweek)
        yearly_average = frame.loc[yearly_mask, "방문인원수"].mean()
        weekday_average = frame.loc[yearly_mask & (frame["요일번호"] == weekday_code), "방문인원수"].mean()
        frame.loc[frame["일자"] == single_missing_date, "방문인원수"] = round((yearly_average + weekday_average) / 2)

        long_gap_mask = frame["일자"].between(pd.Timestamp("2016-12-18"), pd.Timestamp("2017-03-29"))
        train_mask = frame["방문인원수"].notna() & (~long_gap_mask)
        imputation_model = CatBoostRegressor(**IMPUTATION_MODEL_PARAMS)
        imputation_model.fit(frame.loc[train_mask, IMPUTATION_FEATURES], frame.loc[train_mask, "방문인원수"])
        predicted_gap_values = imputation_model.predict(frame.loc[long_gap_mask, IMPUTATION_FEATURES])
        frame.loc[long_gap_mask, "방문인원수"] = np.maximum(np.round(predicted_gap_values), 0).astype(int)
        frame["방문인원수"] = frame["방문인원수"].round().astype(int)
        return frame

    def _load_transport_training_data(self) -> pd.DataFrame:
        columns = ["일자", *TRANSPORT_TRAINING_FEATURE_COLUMNS]
        if not TRANSPORT_VISITORS_FILE.exists():
            return pd.DataFrame(columns=columns)

        transport = pd.read_csv(TRANSPORT_VISITORS_FILE, parse_dates=["date"])
        transport = transport.rename(
            columns={
                "date": "일자",
                "station_ride": "대공원역승차",
                "station_alight": "대공원역하차",
                "station_total": "대공원역승하차",
            }
        )
        return transport[columns].sort_values("일자").reset_index(drop=True)

    def _add_transport_training_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        transport = self._load_transport_training_data()
        if transport.empty:
            for column in TRANSPORT_TRAINING_FEATURE_COLUMNS:
                dataframe[column] = np.nan
            return dataframe
        return dataframe.merge(transport, on="일자", how="left")

    def _make_busy_sample_weights(self, target_values: pd.Series) -> np.ndarray:
        target_values = target_values.reset_index(drop=True)
        top_10_count = max(1, int(np.ceil(len(target_values) * 0.10)))
        top_20_count = max(top_10_count, int(np.ceil(len(target_values) * 0.20)))
        weights = pd.Series(1.0, index=target_values.index, dtype=float)
        weights.loc[target_values.nlargest(top_20_count).index] = TOP_20_WEIGHT
        weights.loc[target_values.nlargest(top_10_count).index] = TOP_10_WEIGHT
        return weights.to_numpy()

    def _make_stage1_busy_labels(self, target_values: pd.Series) -> pd.Series:
        target_values = target_values.reset_index(drop=True)
        busy_count = max(1, int(np.ceil(len(target_values) * BUSY_LABEL_RATIO)))
        labels = pd.Series(0, index=target_values.index, dtype=int)
        labels.loc[target_values.nlargest(busy_count).index] = 1
        return labels

    def _append_stage1_outputs(self, classifier: CatBoostClassifier, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = classifier.predict_proba(frame)[:, 1]
        augmented = frame.copy()
        augmented["혼잡일확률"] = probabilities
        augmented["혼잡일예측라벨"] = (probabilities >= 0.5).astype(int)
        return augmented

    def _train_models(self, frame: pd.DataFrame) -> tuple[CatBoostClassifier, CatBoostRegressor, pd.Series]:
        features = frame[FEATURE_COLUMNS]
        target = frame["방문인원수"]
        train_medians = features.median(numeric_only=True)
        filled_features = features.fillna(train_medians)
        classifier = CatBoostClassifier(**CLASSIFIER_PARAMS)
        classifier.fit(filled_features, self._make_stage1_busy_labels(target))
        augmented = self._append_stage1_outputs(classifier, filled_features)
        regressor = CatBoostRegressor(**REGRESSOR_PARAMS)
        regressor.fit(augmented, target, sample_weight=self._make_busy_sample_weights(target))
        return classifier, regressor, train_medians

    def _build_base_training_frame(self, filled_visitors: pd.DataFrame) -> pd.DataFrame:
        history = filled_visitors[["일자", "방문인원수"]].copy()
        history = self.event_service.merge_historical_flags(history)
        history = self._add_event_derived_features(history)
        history = history.merge(
            self.weather_service.weather_history[["일자", "평균기온", "일강수량"]],
            on="일자",
            how="left",
        )
        history = self._add_calendar_features(history)
        history = self._add_weather_derived_features(history)
        history["지연값_1"] = history["방문인원수"].shift(1)
        history["지연값_7"] = history["방문인원수"].shift(7)
        history["지연값_30"] = history["방문인원수"].shift(30)
        history["지연값_365"] = history["방문인원수"].shift(365)
        previous = history["방문인원수"].shift(1)
        history["이동평균_7"] = previous.rolling(window=7).mean()
        history["이동평균_30"] = previous.rolling(window=30).mean()
        history["이동표준편차_7"] = previous.rolling(window=7).std()
        history["이동표준편차_30"] = previous.rolling(window=30).std()
        history = self._add_transport_training_features(history)
        return history

    def _build_prediction_frame(self) -> pd.DataFrame:
        visitors = self._load_visitors()
        filled_visitors = self._fill_missing_visitors(visitors)
        history = self._build_base_training_frame(filled_visitors)
        classifier, regressor, train_medians = self._train_models(history)

        future_rows: list[dict[str, Any]] = []
        current_date = pd.Timestamp(self.known_data_end) + timedelta(days=1)
        end_date = pd.Timestamp(self.max_supported_date)

        while current_date <= end_date:
            weather = self.weather_service.feature_for_date(current_date.date())
            recurring_flags = self.event_service.recurring_flags_for_date(current_date.date())
            row = {
                "일자": current_date,
                "방문인원수": np.nan,
                "평균기온": weather.average_temperature,
                "일강수량": weather.average_precipitation,
                **recurring_flags,
            }
            row_frame = pd.DataFrame([row])
            row_frame = self._add_event_derived_features(row_frame)
            row_frame = self._add_calendar_features(row_frame)
            row_frame = self._add_weather_derived_features(row_frame)

            filled_features = row_frame[FEATURE_COLUMNS].fillna(train_medians)
            stage1_augmented = self._append_stage1_outputs(classifier, filled_features)
            predicted_visitors = int(max(round(regressor.predict(stage1_augmented)[0]), 0))

            future_rows.append(
                {
                    "date": current_date,
                    "predicted_visitors": predicted_visitors,
                    "busy_probability": float(stage1_augmented["혼잡일확률"].iloc[0]),
                    "weather_source": weather.source,
                    "average_temperature": weather.average_temperature,
                    "average_precipitation": weather.average_precipitation,
                    **recurring_flags,
                }
            )
            current_date += timedelta(days=1)

        prediction_frame = pd.DataFrame(future_rows)
        prediction_frame["congestion_level"] = prediction_frame["predicted_visitors"].map(congestion_level_from_visitors)
        prediction_frame["is_weekend"] = prediction_frame["date"].dt.dayofweek >= 5
        prediction_frame["is_holiday"] = prediction_frame["date"].dt.strftime("%Y-%m-%d").map(is_holiday)
        return prediction_frame

    def _build_prediction_frame_for_period(self, history_end_date: date, prediction_end_date: date) -> pd.DataFrame:
        visitors = pd.read_csv(VISITORS_FILE)
        visitors["일자"] = pd.to_datetime(visitors["일자"].astype(str))
        visitors = visitors.sort_values("일자").reset_index(drop=True)
        training_visitors = visitors.loc[visitors["일자"] <= pd.Timestamp(history_end_date)].copy()
        self.known_data_end = training_visitors.loc[training_visitors["방문인원수"].notna(), "일자"].max().date()

        filled_visitors = self._fill_missing_visitors(training_visitors)
        history = self._build_base_training_frame(filled_visitors)
        classifier, regressor, train_medians = self._train_models(history)

        future_rows: list[dict[str, Any]] = []
        current_date = pd.Timestamp(history_end_date) + timedelta(days=1)
        end_date = pd.Timestamp(prediction_end_date)

        while current_date <= end_date:
            weather = self.weather_service.feature_for_date(current_date.date())
            recurring_flags = self.event_service.recurring_flags_for_date(current_date.date())
            row = {
                "일자": current_date,
                "방문인원수": np.nan,
                "평균기온": weather.average_temperature,
                "일강수량": weather.average_precipitation,
                **recurring_flags,
            }
            row_frame = pd.DataFrame([row])
            row_frame = self._add_event_derived_features(row_frame)
            row_frame = self._add_calendar_features(row_frame)
            row_frame = self._add_weather_derived_features(row_frame)

            filled_features = row_frame[FEATURE_COLUMNS].fillna(train_medians)
            stage1_augmented = self._append_stage1_outputs(classifier, filled_features)
            predicted_visitors = int(max(round(regressor.predict(stage1_augmented)[0]), 0))

            future_rows.append(
                {
                    "date": current_date,
                    "predicted_visitors": predicted_visitors,
                    "busy_probability": float(stage1_augmented["혼잡일확률"].iloc[0]),
                    "weather_source": weather.source,
                    "average_temperature": weather.average_temperature,
                    "average_precipitation": weather.average_precipitation,
                    **recurring_flags,
                }
            )
            current_date += timedelta(days=1)

        prediction_frame = pd.DataFrame(future_rows)
        prediction_frame["congestion_level"] = prediction_frame["predicted_visitors"].map(congestion_level_from_visitors)
        prediction_frame["is_weekend"] = prediction_frame["date"].dt.dayofweek >= 5
        prediction_frame["is_holiday"] = prediction_frame["date"].dt.strftime("%Y-%m-%d").map(is_holiday)
        return prediction_frame

    def evaluate_ranking_metrics(
        self,
        validation_start: date | str = "2025-05-01",
        validation_end: date | str = "2025-07-31",
        k_values: tuple[int, ...] = (5, 10, 20),
        busy_ratio: float = BUSY_LABEL_RATIO,
    ) -> dict[str, Any]:
        if isinstance(validation_start, str):
            validation_start = date.fromisoformat(validation_start)
        if isinstance(validation_end, str):
            validation_end = date.fromisoformat(validation_end)

        history_end_date = validation_start - timedelta(days=1)
        prediction_frame = self._build_prediction_frame_for_period(history_end_date, validation_end)

        visitors = pd.read_csv(VISITORS_FILE)
        visitors["일자"] = pd.to_datetime(visitors["일자"].astype(str))
        actual = visitors.loc[
            visitors["일자"].between(pd.Timestamp(validation_start), pd.Timestamp(validation_end))
            & visitors["방문인원수"].notna(),
            ["일자", "방문인원수"],
        ].copy()
        actual = actual.rename(columns={"일자": "date", "방문인원수": "actual_visitors"})

        evaluation = prediction_frame.merge(actual, on="date", how="inner")
        if evaluation.empty:
            raise ValueError("검증 구간에 실제 방문객 데이터가 없습니다.")

        actual_busy_count = max(1, int(np.ceil(len(evaluation) * busy_ratio)))
        busy_threshold = evaluation["actual_visitors"].nlargest(actual_busy_count).min()
        evaluation["actual_busy"] = evaluation["actual_visitors"] >= busy_threshold
        evaluation["predicted_rank"] = evaluation["predicted_visitors"].rank(method="first", ascending=False).astype(int)
        evaluation["actual_rank"] = evaluation["actual_visitors"].rank(method="first", ascending=False).astype(int)
        ranked = evaluation.sort_values("predicted_visitors", ascending=False).reset_index(drop=True)
        ideal = evaluation.sort_values("actual_visitors", ascending=False).reset_index(drop=True)

        metrics = []
        total_actual_busy = int(evaluation["actual_busy"].sum())
        for k in k_values:
            top_k = ranked.head(k)
            hits = int(top_k["actual_busy"].sum())
            dcg = sum(float(value) / np.log2(index + 2) for index, value in enumerate(top_k["actual_visitors"]))
            idcg = sum(float(value) / np.log2(index + 2) for index, value in enumerate(ideal.head(k)["actual_visitors"]))
            metrics.append(
                {
                    "K": k,
                    "Precision@K": hits / min(k, len(ranked)),
                    "Recall@K": hits / total_actual_busy if total_actual_busy else 0.0,
                    "NDCG@K": dcg / idcg if idcg else 0.0,
                    "Hits": hits,
                    "ActualBusyDays": total_actual_busy,
                    "BusyThreshold": int(busy_threshold),
                }
            )

        ranking_columns = [
            "date",
            "predicted_visitors",
            "actual_visitors",
            "actual_busy",
            "predicted_rank",
            "actual_rank",
        ]
        ranking = ranked[ranking_columns].copy()
        ranking["date"] = ranking["date"].dt.date.astype(str)
        return {
            "validationStart": validation_start.isoformat(),
            "validationEnd": validation_end.isoformat(),
            "historyEndDate": history_end_date.isoformat(),
            "busyRatio": busy_ratio,
            "metrics": metrics,
            "ranking": ranking.to_dict(orient="records"),
        }

    def ensure_ready(self) -> None:
        if not self.state.initialized:
            self.initialize()

    def is_selectable(self, target_date: date) -> tuple[bool, str | None]:
        if target_date < TODAY:
            return False, "오늘 이전 날짜는 선택할 수 없습니다."
        if self.known_data_end and target_date <= self.known_data_end:
            return False, "학습/평가 데이터에 포함된 날짜는 예측할 수 없습니다."
        if target_date > self.max_supported_date:
            return False, f"현재는 {self.max_supported_date:%Y-%m-%d}까지 예측을 지원합니다."
        return True, None

    def list_predictions_for_window(self, center_date: date, days: int = 5) -> list[dict[str, Any]]:
        self.ensure_ready()
        assert self.prediction_frame is not None
        start = pd.Timestamp(center_date - timedelta(days=days))
        end = pd.Timestamp(center_date + timedelta(days=days))
        window = self.prediction_frame.loc[self.prediction_frame["date"].between(start, end)].copy()
        results = []
        for row in window.itertuples(index=False):
            results.append(
                {
                    "date": row.date.date().isoformat(),
                    "predictedVisitors": int(row.predicted_visitors),
                    "congestionLevel": row.congestion_level,
                    "riskColor": color_for_prediction(int(row.predicted_visitors)),
                }
            )
        return results

    def predict_for_date(self, target_date: date) -> dict[str, Any]:
        self.ensure_ready()
        selectable, reason = self.is_selectable(target_date)
        if not selectable:
            raise ValueError(reason)
        assert self.prediction_frame is not None
        matched = self.prediction_frame.loc[self.prediction_frame["date"] == pd.Timestamp(target_date)]
        if matched.empty:
            raise ValueError("예측 결과를 찾을 수 없습니다.")
        row = matched.iloc[0]
        season_flags = [season for season in SEASON_COLUMNS if int(row[season]) == 1]
        level = row["congestion_level"]
        return {
            "date": target_date.isoformat(),
            "predictedVisitors": int(row["predicted_visitors"]),
            "busyProbability": float(row["busy_probability"]),
            "congestionLevel": level,
            "weatherSource": row["weather_source"],
            "averageTemperature": round(float(row["average_temperature"]), 2),
            "averagePrecipitation": round(float(row["average_precipitation"]), 2),
            "isWeekend": bool(row["is_weekend"]),
            "isHoliday": bool(row["is_holiday"]),
            "seasonFlags": season_flags,
            "dateComment": build_date_comment(
                target_date=target_date,
                visitors=int(row["predicted_visitors"]),
                is_weekend=bool(row["is_weekend"]),
                is_holiday=bool(row["is_holiday"]),
                weather_source=str(row["weather_source"]),
                active_seasons=season_flags,
            ),
            "randomComment": random_comment_for_level(level, target_date.isoformat()),
            "textColor": color_for_prediction(int(row["predicted_visitors"])),
        }
