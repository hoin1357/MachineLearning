from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import ARTIFACTS_DIR, VISITORS_FILE  # noqa: E402
from app.services import model_runtime as mr  # noqa: E402


COVID_START = pd.Timestamp("2020-01-01")
COVID_END = pd.Timestamp("2022-12-31")
VALIDATION_START = date(2025, 5, 1)
VALIDATION_END = date(2025, 7, 31)
HISTORY_END = VALIDATION_START - timedelta(days=1)
COVID_SYNTHETIC_WEIGHT = 0.25

COMPARISON_FILE = ARTIFACTS_DIR / "covid_preprocessing_model_comparison.csv"
PREDICTIONS_FILE = ARTIFACTS_DIR / "covid_preprocessing_validation_predictions.csv"
COUNTERFACTUAL_SERIES_FILE = ARTIFACTS_DIR / "covid_counterfactual_training_series.csv"
METRICS_GRAPH_FILE = ARTIFACTS_DIR / "covid_preprocessing_metrics_bar.png"
ACTUAL_PREDICTED_GRAPH_FILE = ARTIFACTS_DIR / "covid_preprocessing_actual_vs_predicted.png"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    covid_mode: str = "baseline"
    covid_weight: float = 1.0
    classifier_params: dict[str, Any] | None = None
    regressor_params: dict[str, Any] | None = None
    busy_ratio: float | None = None
    top10_weight: float | None = None
    top20_weight: float | None = None


class CovidPreprocessingRuntime(mr.PortablePredictionRuntime):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.experiment_config = config
        self.classifier_params = {**mr.CLASSIFIER_PARAMS, **(config.classifier_params or {})}
        self.regressor_params = {**mr.REGRESSOR_PARAMS, **(config.regressor_params or {})}
        self.busy_ratio = mr.BUSY_LABEL_RATIO if config.busy_ratio is None else config.busy_ratio
        self.top10_weight = mr.TOP_10_WEIGHT if config.top10_weight is None else config.top10_weight
        self.top20_weight = mr.TOP_20_WEIGHT if config.top20_weight is None else config.top20_weight
        self.counterfactual_series: pd.DataFrame | None = None

    def _make_busy_sample_weights(self, target_values: pd.Series) -> np.ndarray:
        target_values = target_values.reset_index(drop=True)
        top_10_count = max(1, int(np.ceil(len(target_values) * 0.10)))
        top_20_count = max(top_10_count, int(np.ceil(len(target_values) * 0.20)))
        weights = pd.Series(1.0, index=target_values.index, dtype=float)
        weights.loc[target_values.nlargest(top_20_count).index] = self.top20_weight
        weights.loc[target_values.nlargest(top_10_count).index] = self.top10_weight
        return weights.to_numpy().copy()

    def _make_stage1_busy_labels(self, target_values: pd.Series) -> pd.Series:
        target_values = target_values.reset_index(drop=True)
        busy_count = max(1, int(np.ceil(len(target_values) * self.busy_ratio)))
        labels = pd.Series(0, index=target_values.index, dtype=int)
        labels.loc[target_values.nlargest(busy_count).index] = 1
        return labels

    def _fill_missing_visitors(self, visitors: pd.DataFrame) -> pd.DataFrame:
        frame = super()._fill_missing_visitors(visitors)
        if self.experiment_config.covid_mode != "counterfactual_downweight":
            return frame

        non_covid_mask = ~frame["일자"].between(COVID_START, COVID_END)
        covid_mask = frame["일자"].between(COVID_START, COVID_END)
        counterfactual_model = CatBoostRegressor(**mr.IMPUTATION_MODEL_PARAMS)
        counterfactual_model.fit(
            frame.loc[non_covid_mask, mr.IMPUTATION_FEATURES],
            frame.loc[non_covid_mask, "방문인원수"],
        )
        predicted_baseline = np.maximum(
            np.round(counterfactual_model.predict(frame.loc[covid_mask, mr.IMPUTATION_FEATURES])),
            0,
        ).astype(int)

        self.counterfactual_series = frame.loc[covid_mask, ["일자", "방문인원수"]].copy()
        self.counterfactual_series["counterfactual_visitors"] = predicted_baseline
        self.counterfactual_series["difference"] = (
            self.counterfactual_series["counterfactual_visitors"] - self.counterfactual_series["방문인원수"]
        )
        frame.loc[covid_mask, "방문인원수"] = predicted_baseline
        return frame

    def _train_models(self, frame: pd.DataFrame) -> tuple[CatBoostClassifier, CatBoostRegressor, pd.Series]:
        features = frame[mr.FEATURE_COLUMNS]
        target = frame["방문인원수"]
        train_medians = features.median(numeric_only=True)
        filled_features = features.fillna(train_medians)

        stage1_weights = np.ones(len(frame), dtype=float)
        stage2_weights = self._make_busy_sample_weights(target)
        if self.experiment_config.covid_mode in {"downweight", "counterfactual_downweight"}:
            covid_mask = frame["일자"].between(COVID_START, COVID_END).to_numpy().copy()
            stage1_weights[covid_mask] *= self.experiment_config.covid_weight
            stage2_weights[covid_mask] *= self.experiment_config.covid_weight

        classifier = CatBoostClassifier(**self.classifier_params)
        classifier.fit(filled_features, self._make_stage1_busy_labels(target), sample_weight=stage1_weights)
        augmented = self._append_stage1_outputs(classifier, filled_features)
        regressor = CatBoostRegressor(**self.regressor_params)
        regressor.fit(augmented, target, sample_weight=stage2_weights)
        return classifier, regressor, train_medians


def load_actual_validation() -> pd.DataFrame:
    visitors = pd.read_csv(VISITORS_FILE)
    visitors["date"] = pd.to_datetime(visitors["일자"].astype(str), format="%Y%m%d")
    visitors["actual_visitors"] = pd.to_numeric(visitors["방문인원수"], errors="coerce")
    return visitors.loc[
        visitors["date"].between(pd.Timestamp(VALIDATION_START), pd.Timestamp(VALIDATION_END))
        & visitors["actual_visitors"].notna(),
        ["date", "actual_visitors"],
    ].copy()


def evaluate_experiment(config: ExperimentConfig, actual: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame | None]:
    runtime = CovidPreprocessingRuntime(config)
    prediction_frame = runtime._build_prediction_frame_for_period(HISTORY_END, VALIDATION_END)
    evaluation = prediction_frame.merge(actual, on="date", how="inner")
    evaluation["model"] = config.name
    evaluation["error"] = evaluation["predicted_visitors"] - evaluation["actual_visitors"]
    evaluation["abs_error"] = evaluation["error"].abs()
    evaluation["ape"] = evaluation["abs_error"] / evaluation["actual_visitors"].clip(lower=1) * 100

    actual_values = evaluation["actual_visitors"].astype(float)
    predicted_values = evaluation["predicted_visitors"].astype(float)
    rmse = float(np.sqrt(np.mean((predicted_values - actual_values) ** 2)))
    ss_res = float(np.sum((actual_values - predicted_values) ** 2))
    ss_tot = float(np.sum((actual_values - actual_values.mean()) ** 2))

    actual_busy_count = max(1, int(np.ceil(len(evaluation) * mr.BUSY_LABEL_RATIO)))
    busy_threshold = evaluation["actual_visitors"].nlargest(actual_busy_count).min()
    evaluation["actual_busy"] = evaluation["actual_visitors"] >= busy_threshold
    ranked = evaluation.sort_values("predicted_visitors", ascending=False).reset_index(drop=True)
    ideal = evaluation.sort_values("actual_visitors", ascending=False).reset_index(drop=True)

    metrics: dict[str, Any] = {
        "model": config.name,
        "covid_mode": config.covid_mode,
        "covid_weight": config.covid_weight,
        "MAE": float(evaluation["abs_error"].mean()),
        "RMSE": rmse,
        "MAPE": float(evaluation["ape"].mean()),
        "R2": 1 - ss_res / ss_tot,
        "ActualMean": float(actual_values.mean()),
        "PredictedMean": float(predicted_values.mean()),
    }

    total_busy = int(evaluation["actual_busy"].sum())
    for k in (5, 10, 20):
        top_k = ranked.head(k)
        hits = int(top_k["actual_busy"].sum())
        dcg = sum(float(value) / np.log2(index + 2) for index, value in enumerate(top_k["actual_visitors"]))
        idcg = sum(float(value) / np.log2(index + 2) for index, value in enumerate(ideal.head(k)["actual_visitors"]))
        metrics[f"Precision@{k}"] = hits / min(k, len(ranked))
        metrics[f"Recall@{k}"] = hits / total_busy if total_busy else 0.0
        metrics[f"NDCG@{k}"] = dcg / idcg if idcg else 0.0

    return metrics, evaluation, runtime.counterfactual_series


def save_graphs(comparison: pd.DataFrame, predictions: pd.DataFrame) -> None:
    plot_metrics = comparison.set_index("model")[["MAE", "RMSE"]]
    ax = plot_metrics.plot(kind="bar", figsize=(10, 5), rot=20)
    ax.set_title("Validation Error by Model")
    ax.set_ylabel("Visitors")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(METRICS_GRAPH_FILE, dpi=160)
    plt.close()

    plt.figure(figsize=(14, 6))
    actual = predictions.drop_duplicates("date").sort_values("date")
    plt.plot(actual["date"], actual["actual_visitors"], label="Actual", linewidth=2.4, color="black")
    for model_name, group in predictions.groupby("model"):
        ordered = group.sort_values("date")
        plt.plot(ordered["date"], ordered["predicted_visitors"], label=model_name, linewidth=1.5)
    plt.title("Actual vs Predicted Visitors")
    plt.xlabel("Date")
    plt.ylabel("Visitors")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(ACTUAL_PREDICTED_GRAPH_FILE, dpi=160)
    plt.close()


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    actual = load_actual_validation()
    experiments = [
        ExperimentConfig(name="baseline"),
        ExperimentConfig(
            name="covid_downweight_0.25",
            covid_mode="downweight",
            covid_weight=COVID_SYNTHETIC_WEIGHT,
        ),
        ExperimentConfig(
            name="covid_counterfactual_downweight_0.25",
            covid_mode="counterfactual_downweight",
            covid_weight=COVID_SYNTHETIC_WEIGHT,
        ),
    ]

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    counterfactual_frames: list[pd.DataFrame] = []
    for experiment in experiments:
        print(f"training {experiment.name}")
        metrics, predictions, counterfactual_series = evaluate_experiment(experiment, actual)
        metric_rows.append(metrics)
        prediction_frames.append(predictions)
        if counterfactual_series is not None:
            counterfactual_frames.append(counterfactual_series)

    comparison = pd.DataFrame(metric_rows)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    comparison.to_csv(COMPARISON_FILE, index=False, encoding="utf-8-sig")
    all_predictions.to_csv(PREDICTIONS_FILE, index=False, encoding="utf-8-sig")
    if counterfactual_frames:
        pd.concat(counterfactual_frames, ignore_index=True).to_csv(
            COUNTERFACTUAL_SERIES_FILE,
            index=False,
            encoding="utf-8-sig",
        )
    save_graphs(comparison, all_predictions)

    display_columns = [
        "model",
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
        "Precision@10",
        "Recall@20",
        "NDCG@20",
        "PredictedMean",
        "ActualMean",
    ]
    print("\ncomparison")
    print(comparison[display_columns].round(4).to_string(index=False))
    print(f"\nsaved {COMPARISON_FILE}")
    print(f"saved {PREDICTIONS_FILE}")
    print(f"saved {COUNTERFACTUAL_SERIES_FILE}")
    print(f"saved {METRICS_GRAPH_FILE}")
    print(f"saved {ACTUAL_PREDICTED_GRAPH_FILE}")


if __name__ == "__main__":
    main()
