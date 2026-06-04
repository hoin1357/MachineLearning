from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from datetime import date
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


VALIDATION_START = date(2025, 5, 1)
VALIDATION_END = date(2025, 7, 31)
HISTORY_END = date(2025, 4, 30)
RANDOM_SEED = 20260513
N_TRIALS = 100

TRIAL_LOG_FILE = ARTIFACTS_DIR / "classifier_random100_trials.csv"
SUMMARY_FILE = ARTIFACTS_DIR / "classifier_random100_summary.md"
GRAPH_FILE = ARTIFACTS_DIR / "classifier_random100_ndcg_progress.png"


@dataclass(frozen=True)
class TrialSpec:
    step: int
    params: dict[str, Any]


@dataclass
class TuningContext:
    runtime: mr.PortablePredictionRuntime
    history: pd.DataFrame
    actual: pd.DataFrame


def load_actual_validation() -> pd.DataFrame:
    visitors = pd.read_csv(VISITORS_FILE)
    visitors["date"] = pd.to_datetime(visitors["일자"].astype(str), format="%Y%m%d")
    visitors["actual_visitors"] = pd.to_numeric(visitors["방문인원수"], errors="coerce")
    return visitors.loc[
        visitors["date"].between(pd.Timestamp(VALIDATION_START), pd.Timestamp(VALIDATION_END))
        & visitors["actual_visitors"].notna(),
        ["date", "actual_visitors"],
    ].copy()


def build_tuning_context() -> TuningContext:
    runtime = mr.PortablePredictionRuntime()
    visitors = pd.read_csv(VISITORS_FILE)
    visitors["일자"] = pd.to_datetime(visitors["일자"].astype(str), format="%Y%m%d")
    training_visitors = visitors.loc[visitors["일자"] <= pd.Timestamp(HISTORY_END)].copy()
    runtime.known_data_end = training_visitors.loc[training_visitors["방문인원수"].notna(), "일자"].max().date()
    filled_visitors = runtime._fill_missing_visitors(training_visitors)
    history = runtime._build_base_training_frame(filled_visitors)
    return TuningContext(runtime=runtime, history=history, actual=load_actual_validation())


def predict_validation_period(
    context: TuningContext,
    classifier: CatBoostClassifier,
    regressor: CatBoostRegressor,
    train_medians: pd.Series,
) -> pd.DataFrame:
    series_frame = context.history[["일자", "방문인원수"]].copy()
    future_rows: list[dict[str, Any]] = []
    current_date = pd.Timestamp(VALIDATION_START)
    end_date = pd.Timestamp(VALIDATION_END)

    while current_date <= end_date:
        weather = context.runtime.weather_service.feature_for_date(current_date.date())
        recurring_flags = context.runtime.event_service.recurring_flags_for_date(current_date.date())
        row = {
            "일자": current_date,
            "방문인원수": np.nan,
            "평균기온": weather.average_temperature,
            "일강수량": weather.average_precipitation,
            **recurring_flags,
        }
        row_frame = pd.DataFrame([row])
        row_frame = context.runtime._add_calendar_features(row_frame)
        row_frame = context.runtime._add_weather_derived_features(row_frame)

        history_values = series_frame["방문인원수"]
        row_frame["지연값_1"] = history_values.iloc[-1]
        row_frame["지연값_7"] = history_values.iloc[-7] if len(history_values) >= 7 else np.nan
        row_frame["지연값_30"] = history_values.iloc[-30] if len(history_values) >= 30 else np.nan
        row_frame["지연값_365"] = history_values.iloc[-365] if len(history_values) >= 365 else np.nan
        row_frame["이동평균_7"] = history_values.iloc[-7:].mean() if len(history_values) >= 7 else np.nan
        row_frame["이동평균_30"] = history_values.iloc[-30:].mean() if len(history_values) >= 30 else np.nan
        row_frame["이동표준편차_7"] = history_values.iloc[-7:].std() if len(history_values) >= 7 else np.nan
        row_frame["이동표준편차_30"] = history_values.iloc[-30:].std() if len(history_values) >= 30 else np.nan

        filled_features = row_frame[mr.FEATURE_COLUMNS].fillna(train_medians)
        stage1_augmented = context.runtime._append_stage1_outputs(classifier, filled_features)
        predicted_visitors = int(max(round(regressor.predict(stage1_augmented)[0]), 0))
        future_rows.append(
            {
                "date": current_date,
                "predicted_visitors": predicted_visitors,
                "busy_probability": float(stage1_augmented["혼잡일확률"].iloc[0]),
            }
        )
        series_frame.loc[len(series_frame)] = {"일자": current_date, "방문인원수": predicted_visitors}
        current_date += pd.Timedelta(days=1)

    return pd.DataFrame(future_rows)


def evaluate_trial(spec: TrialSpec, context: TuningContext) -> dict[str, Any]:
    features = context.history[mr.FEATURE_COLUMNS]
    target = context.history["방문인원수"]
    train_medians = features.median(numeric_only=True)
    filled_features = features.fillna(train_medians)

    classifier = CatBoostClassifier(**spec.params)
    classifier.fit(filled_features, context.runtime._make_stage1_busy_labels(target))
    augmented = context.runtime._append_stage1_outputs(classifier, filled_features)
    regressor = CatBoostRegressor(**mr.REGRESSOR_PARAMS)
    regressor.fit(augmented, target, sample_weight=context.runtime._make_busy_sample_weights(target))

    prediction_frame = predict_validation_period(context, classifier, regressor, train_medians)
    evaluation = prediction_frame.merge(context.actual, on="date", how="inner")
    actual_values = evaluation["actual_visitors"].astype(float)
    predicted_values = evaluation["predicted_visitors"].astype(float)
    errors = predicted_values - actual_values

    actual_busy_count = max(1, int(np.ceil(len(evaluation) * mr.BUSY_LABEL_RATIO)))
    busy_threshold = evaluation["actual_visitors"].nlargest(actual_busy_count).min()
    evaluation["actual_busy"] = evaluation["actual_visitors"] >= busy_threshold
    ranked = evaluation.sort_values("predicted_visitors", ascending=False).reset_index(drop=True)
    ideal = evaluation.sort_values("actual_visitors", ascending=False).reset_index(drop=True)

    result: dict[str, Any] = {
        "step": spec.step,
        "depth": spec.params["depth"],
        "learning_rate": spec.params["learning_rate"],
        "iterations": spec.params["iterations"],
        "l2_leaf_reg": spec.params["l2_leaf_reg"],
        "MAE": float(errors.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAPE": float((errors.abs() / actual_values.clip(lower=1) * 100).mean()),
    }
    ss_res = float(np.sum((actual_values - predicted_values) ** 2))
    ss_tot = float(np.sum((actual_values - actual_values.mean()) ** 2))
    result["R2"] = 1 - ss_res / ss_tot

    total_busy = int(evaluation["actual_busy"].sum())
    for k in (5, 10, 20):
        top_k = ranked.head(k)
        hits = int(top_k["actual_busy"].sum())
        dcg = sum(float(value) / np.log2(index + 2) for index, value in enumerate(top_k["actual_visitors"]))
        idcg = sum(float(value) / np.log2(index + 2) for index, value in enumerate(ideal.head(k)["actual_visitors"]))
        result[f"Precision@{k}"] = hits / min(k, len(ranked))
        result[f"Recall@{k}"] = hits / total_busy if total_busy else 0.0
        result[f"NDCG@{k}"] = dcg / idcg if idcg else 0.0
    return result


def score(row: dict[str, Any]) -> tuple[float, float, float]:
    return (-float(row["NDCG@20"]), float(row["RMSE"]), float(row["MAE"]))


def build_trial_specs() -> list[TrialSpec]:
    rng = random.Random(RANDOM_SEED)
    spaces = {
        "depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1],
        "iterations": [250, 350, 450, 500, 650, 800],
        "l2_leaf_reg": [1.5, 2.0, 3.0, 5.0, 6.0, 10.0, 12.0],
    }
    current = (
        mr.CLASSIFIER_PARAMS["depth"],
        mr.CLASSIFIER_PARAMS["learning_rate"],
        mr.CLASSIFIER_PARAMS["iterations"],
        mr.CLASSIFIER_PARAMS["l2_leaf_reg"],
    )
    combos: set[tuple[Any, ...]] = set()
    specs: list[TrialSpec] = []
    while len(specs) < N_TRIALS:
        combo = (
            rng.choice(spaces["depth"]),
            rng.choice(spaces["learning_rate"]),
            rng.choice(spaces["iterations"]),
            rng.choice(spaces["l2_leaf_reg"]),
        )
        if combo in combos or combo == current:
            continue
        combos.add(combo)
        params = {
            **mr.CLASSIFIER_PARAMS,
            "depth": combo[0],
            "learning_rate": combo[1],
            "iterations": combo[2],
            "l2_leaf_reg": combo[3],
        }
        specs.append(TrialSpec(step=len(specs) + 1, params=params))
    return specs


def markdown_table(frame: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return [header, separator, *rows]


def save_summary(results: list[dict[str, Any]], best: dict[str, Any]) -> None:
    frame = pd.DataFrame(results)
    ordered = frame.sort_values(["NDCG@20", "RMSE", "MAE"], ascending=[False, True, True]).reset_index(drop=True)
    top_table = ordered.head(15)[
        [
            "step",
            "depth",
            "learning_rate",
            "iterations",
            "l2_leaf_reg",
            "MAE",
            "RMSE",
            "R2",
            "NDCG@20",
        ]
    ].round(4)
    lines = [
        "# 1단계 혼잡일 분류 모델 100회 무작위 탐색 기록",
        "",
        f"- 검증 구간: {VALIDATION_START.isoformat()} ~ {VALIDATION_END.isoformat()}",
        f"- 탐색 횟수: {N_TRIALS}회",
        f"- random_seed: {RANDOM_SEED}",
        "- 1차 기준: NDCG@20 최대화",
        "- 동률 기준: RMSE 최소화, MAE 최소화",
        "",
        "## 최종 선택값",
        "",
        f"- depth: {best['depth']}",
        f"- learning_rate: {best['learning_rate']}",
        f"- iterations: {best['iterations']}",
        f"- l2_leaf_reg: {best['l2_leaf_reg']}",
        f"- NDCG@20: {best['NDCG@20']:.4f}",
        f"- RMSE: {best['RMSE']:.4f}",
        f"- MAE: {best['MAE']:.4f}",
        "",
        "## 상위 결과",
        "",
        *markdown_table(top_table),
    ]
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def save_graph(results: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(results).sort_values("step")
    frame["best_ndcg_so_far"] = frame["NDCG@20"].cummax()
    plt.figure(figsize=(12, 5))
    plt.plot(frame["step"], frame["NDCG@20"], marker="o", markersize=3, linewidth=1.0, label="trial NDCG@20")
    plt.plot(frame["step"], frame["best_ndcg_so_far"], linewidth=2.2, label="best NDCG@20 so far")
    plt.title("Classifier Random Search 100 Progress")
    plt.xlabel("Trial step")
    plt.ylabel("NDCG@20")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=160)
    plt.close()


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    context = build_tuning_context()
    specs = build_trial_specs()
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for spec in specs:
        result = evaluate_trial(spec, context)
        results.append(result)
        if best is None or score(result) < score(best):
            best = result
        pd.DataFrame(results).to_csv(TRIAL_LOG_FILE, index=False, encoding="utf-8-sig")
        print(
            f"step {result['step']:03d}/{N_TRIALS} "
            f"depth={result['depth']} lr={result['learning_rate']} iter={result['iterations']} "
            f"l2={result['l2_leaf_reg']} NDCG@20={result['NDCG@20']:.4f} "
            f"RMSE={result['RMSE']:.2f} MAE={result['MAE']:.2f} "
            f"best_step={best['step']} best_NDCG@20={best['NDCG@20']:.4f}",
            flush=True,
        )

    assert best is not None
    save_summary(results, best)
    save_graph(results)
    print("\nbest")
    print(
        pd.DataFrame([best])[
            ["step", "depth", "learning_rate", "iterations", "l2_leaf_reg", "MAE", "RMSE", "MAPE", "R2", "NDCG@20"]
        ].round(4).to_string(index=False)
    )
    print(f"\nsaved {TRIAL_LOG_FILE}")
    print(f"saved {SUMMARY_FILE}")
    print(f"saved {GRAPH_FILE}")


if __name__ == "__main__":
    main()
