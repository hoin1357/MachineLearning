from __future__ import annotations

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

TRIAL_LOG_FILE = ARTIFACTS_DIR / "classifier_tuning_trials.csv"
SUMMARY_FILE = ARTIFACTS_DIR / "classifier_tuning_summary.md"
GRAPH_FILE = ARTIFACTS_DIR / "classifier_tuning_rmse_progress.png"


@dataclass(frozen=True)
class TrialSpec:
    round_no: int
    step_no: int
    parameter: str
    reason: str
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
        "round": spec.round_no,
        "step": spec.step_no,
        "parameter": spec.parameter,
        "reason": spec.reason,
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
    return (float(row["RMSE"]), float(row["MAE"]), -float(row["NDCG@20"]))


def is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    return score(candidate) < score(incumbent)


def with_param(params: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    updated = dict(params)
    updated[key] = value
    return updated


def generate_candidates(best_params: dict[str, Any], round_no: int, next_step: int) -> list[TrialSpec]:
    candidates: list[tuple[str, Any, str]] = []

    depth = int(best_params["depth"])
    for value in [depth - 1, depth + 1]:
        if 3 <= value <= 10 and value != depth:
            candidates.append(("depth", value, "트리 깊이를 한 단계씩 바꿔 복잡도와 과적합 균형을 확인"))

    l2 = float(best_params["l2_leaf_reg"])
    for value in sorted({max(1.0, l2 * 0.5), l2 * 2.0}):
        if abs(value - l2) > 1e-9:
            candidates.append(("l2_leaf_reg", round(value, 4), "depth가 만드는 과적합을 규제로 보정"))

    learning_rate = float(best_params["learning_rate"])
    for value in sorted({0.03, 0.08}):
        if 0.01 <= value <= 0.12 and abs(value - learning_rate) > 1e-9:
            candidates.append(("learning_rate", value, "learning_rate와 iterations는 반비례 관계라 먼저 보폭만 확인"))

    iterations = int(best_params["iterations"])
    for value in sorted({350, 650}):
        if 200 <= value <= 1200 and value != iterations:
            candidates.append(("iterations", value, "learning_rate 변화와 연결되는 학습량을 단독 확인"))

    specs = []
    seen: set[tuple[str, Any]] = set()
    step = next_step
    for key, value, reason in candidates:
        marker = (key, value)
        if marker in seen:
            continue
        seen.add(marker)
        specs.append(
            TrialSpec(
                round_no=round_no,
                step_no=step,
                parameter=key,
                reason=reason,
                params=with_param(best_params, key, value),
            )
        )
        step += 1
    return specs


def save_summary(results: list[dict[str, Any]], best: dict[str, Any], stop_reason: str) -> None:
    frame = pd.DataFrame(results)
    ordered = frame.sort_values(["RMSE", "MAE", "NDCG@20"], ascending=[True, True, False]).reset_index(drop=True)
    top_table = ordered.head(10)[
        [
            "step",
            "parameter",
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
    header = "| " + " | ".join(top_table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(top_table.columns)) + " |"
    table_rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in top_table.itertuples(index=False, name=None)
    ]
    lines = [
        "# 1단계 혼잡일 분류 모델 튜닝 기록",
        "",
        f"- 검증 구간: {VALIDATION_START.isoformat()} ~ {VALIDATION_END.isoformat()}",
        "- 1차 기준: RMSE 최소화",
        "- 동률 기준: MAE 최소화, NDCG@20 최대화",
        f"- 중단 이유: {stop_reason}",
        "",
        "## 최종 선택값",
        "",
        f"- depth: {best['depth']}",
        f"- learning_rate: {best['learning_rate']}",
        f"- iterations: {best['iterations']}",
        f"- l2_leaf_reg: {best['l2_leaf_reg']}",
        f"- RMSE: {best['RMSE']:.4f}",
        f"- MAE: {best['MAE']:.4f}",
        f"- NDCG@20: {best['NDCG@20']:.4f}",
        "",
        "## 상위 결과",
        "",
        header,
        separator,
        *table_rows,
    ]
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def save_graph(results: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(results).sort_values("step")
    frame["best_rmse_so_far"] = frame["RMSE"].cummin()
    plt.figure(figsize=(11, 5))
    plt.plot(frame["step"], frame["RMSE"], marker="o", linewidth=1.2, label="trial RMSE")
    plt.plot(frame["step"], frame["best_rmse_so_far"], linewidth=2.4, label="best RMSE so far")
    plt.title("Classifier Hyperparameter Tuning Progress")
    plt.xlabel("Trial step")
    plt.ylabel("RMSE")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=160)
    plt.close()


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    context = build_tuning_context()
    base_params = dict(mr.CLASSIFIER_PARAMS)
    base_spec = TrialSpec(
        round_no=0,
        step_no=0,
        parameter="baseline",
        reason="현재 1단계 분류기 설정",
        params=base_params,
    )

    results: list[dict[str, Any]] = []
    best = evaluate_trial(base_spec, context)
    results.append(best)
    best_params = base_params
    pd.DataFrame(results).to_csv(TRIAL_LOG_FILE, index=False, encoding="utf-8-sig")
    print("baseline", {key: best[key] for key in ("RMSE", "MAE", "NDCG@20")}, flush=True)

    step = 1
    stop_reason = ""
    for round_no in range(1, 8):
        round_best = best
        round_best_params = best_params
        candidates = generate_candidates(best_params, round_no, step)
        print(f"round {round_no}: {len(candidates)} candidates", flush=True)
        for spec in candidates:
            step = spec.step_no + 1
            result = evaluate_trial(spec, context)
            results.append(result)
            pd.DataFrame(results).to_csv(TRIAL_LOG_FILE, index=False, encoding="utf-8-sig")
            print(
                f"step {result['step']:02d} {result['parameter']} "
                f"depth={result['depth']} lr={result['learning_rate']} "
                f"iter={result['iterations']} l2={result['l2_leaf_reg']} "
                f"RMSE={result['RMSE']:.2f} MAE={result['MAE']:.2f} NDCG@20={result['NDCG@20']:.4f}",
                flush=True,
            )
            if is_better(result, round_best):
                round_best = result
                round_best_params = spec.params

        if is_better(round_best, best):
            best = round_best
            best_params = round_best_params
            print(f"accepted step {best['step']} with RMSE={best['RMSE']:.2f}")
            continue

        stop_reason = f"round {round_no}에서 후보 {len(candidates)}개를 모두 비교했지만 기존 최고 성능을 넘지 못함"
        break
    else:
        stop_reason = "최대 탐색 라운드에 도달"

    frame = pd.DataFrame(results)
    frame.to_csv(TRIAL_LOG_FILE, index=False, encoding="utf-8-sig")
    save_summary(results, best, stop_reason)
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
    print(stop_reason)


if __name__ == "__main__":
    main()
