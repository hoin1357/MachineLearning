from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import ARTIFACTS_DIR, VISITORS_FILE  # noqa: E402
from app.services import model_runtime as mr  # noqa: E402


VALIDATION_START = date(2025, 5, 1)
VALIDATION_END = date(2025, 7, 31)
HISTORY_END = date(2025, 4, 30)

TRIAL_LOG_FILE = ARTIFACTS_DIR / "imputation_tuning_trials.csv"
SUMMARY_FILE = ARTIFACTS_DIR / "imputation_tuning_summary.md"
GRAPH_FILE = ARTIFACTS_DIR / "imputation_tuning_ndcg_progress.png"


@dataclass(frozen=True)
class TrialSpec:
    round_no: int
    step_no: int
    parameter: str
    reason: str
    params: dict[str, Any]


class ImputationTuningRuntime(mr.PortablePredictionRuntime):
    def __init__(self, imputation_params: dict[str, Any]) -> None:
        super().__init__()
        self.imputation_params = imputation_params

    def _fill_missing_visitors(self, visitors: pd.DataFrame) -> pd.DataFrame:
        frame = visitors.copy()
        historical_weather = self.weather_service.weather_history.rename(
            columns={"평균기온": "평균기온", "일강수량": "일강수량"}
        )
        frame = frame.merge(historical_weather[["일자", "평균기온", "일강수량"]], on="일자", how="left")
        frame = self.event_service.merge_historical_flags(frame)
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
        imputation_model = CatBoostRegressor(**self.imputation_params)
        imputation_model.fit(frame.loc[train_mask, mr.IMPUTATION_FEATURES], frame.loc[train_mask, "방문인원수"])
        predicted_gap_values = imputation_model.predict(frame.loc[long_gap_mask, mr.IMPUTATION_FEATURES])
        frame.loc[long_gap_mask, "방문인원수"] = np.maximum(np.round(predicted_gap_values), 0).astype(int)
        frame["방문인원수"] = frame["방문인원수"].round().astype(int)
        return frame


def load_actual_validation() -> pd.DataFrame:
    visitors = pd.read_csv(VISITORS_FILE)
    visitors["date"] = pd.to_datetime(visitors["일자"].astype(str), format="%Y%m%d")
    visitors["actual_visitors"] = pd.to_numeric(visitors["방문인원수"], errors="coerce")
    return visitors.loc[
        visitors["date"].between(pd.Timestamp(VALIDATION_START), pd.Timestamp(VALIDATION_END))
        & visitors["actual_visitors"].notna(),
        ["date", "actual_visitors"],
    ].copy()


def evaluate_trial(spec: TrialSpec, actual: pd.DataFrame) -> dict[str, Any]:
    runtime = ImputationTuningRuntime(spec.params)
    prediction_frame = runtime._build_prediction_frame_for_period(HISTORY_END, VALIDATION_END)
    evaluation = prediction_frame.merge(actual, on="date", how="inner")
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
        "min_data_in_leaf": spec.params["min_data_in_leaf"],
        "random_strength": spec.params["random_strength"],
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
            candidates.append(("depth", value, "결측 보정 모델 복잡도를 한 단계씩 조절"))

    l2 = float(best_params["l2_leaf_reg"])
    for value in sorted({max(1.0, l2 * 0.5), l2 * 2.0}):
        if abs(value - l2) > 1e-9:
            candidates.append(("l2_leaf_reg", round(value, 4), "트리 복잡도에 대한 규제 강도 조절"))

    learning_rate = float(best_params["learning_rate"])
    for value in sorted({0.03, 0.08}):
        if abs(value - learning_rate) > 1e-9:
            candidates.append(("learning_rate", value, "learning_rate와 iterations 관계를 고려해 보폭 조절"))

    iterations = int(best_params["iterations"])
    for value in sorted({500, 900}):
        if value != iterations:
            candidates.append(("iterations", value, "학습량을 줄이거나 늘려 보정 안정성 확인"))

    min_leaf = int(best_params["min_data_in_leaf"])
    for value in sorted({max(1, min_leaf - 2), min_leaf + 5}):
        if value != min_leaf:
            candidates.append(("min_data_in_leaf", value, "leaf 최소 표본 수로 과적합과 세밀함 균형 확인"))

    random_strength = float(best_params["random_strength"])
    for value in sorted({0.5, 2.0}):
        if abs(value - random_strength) > 1e-9:
            candidates.append(("random_strength", value, "분기 선택 무작위성으로 보정값 과적합 완화 확인"))

    specs: list[TrialSpec] = []
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


def markdown_table(frame: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return [header, separator, *rows]


def save_summary(results: list[dict[str, Any]], best: dict[str, Any], stop_reason: str) -> None:
    frame = pd.DataFrame(results)
    ordered = frame.sort_values(["NDCG@20", "RMSE", "MAE"], ascending=[False, True, True]).reset_index(drop=True)
    top_table = ordered.head(10)[
        [
            "step",
            "parameter",
            "depth",
            "learning_rate",
            "iterations",
            "l2_leaf_reg",
            "min_data_in_leaf",
            "random_strength",
            "MAE",
            "RMSE",
            "R2",
            "NDCG@20",
        ]
    ].round(4)
    lines = [
        "# 결측값 보정 모델 튜닝 기록",
        "",
        f"- 검증 구간: {VALIDATION_START.isoformat()} ~ {VALIDATION_END.isoformat()}",
        "- 1차 기준: NDCG@20 최대화",
        "- 동률 기준: RMSE 최소화, MAE 최소화",
        f"- 중단 이유: {stop_reason}",
        "",
        "## 최종 선택값",
        "",
        f"- depth: {best['depth']}",
        f"- learning_rate: {best['learning_rate']}",
        f"- iterations: {best['iterations']}",
        f"- l2_leaf_reg: {best['l2_leaf_reg']}",
        f"- min_data_in_leaf: {best['min_data_in_leaf']}",
        f"- random_strength: {best['random_strength']}",
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
    plt.figure(figsize=(11, 5))
    plt.plot(frame["step"], frame["NDCG@20"], marker="o", linewidth=1.2, label="trial NDCG@20")
    plt.plot(frame["step"], frame["best_ndcg_so_far"], linewidth=2.4, label="best NDCG@20 so far")
    plt.title("Imputation Hyperparameter Tuning Progress")
    plt.xlabel("Trial step")
    plt.ylabel("NDCG@20")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=160)
    plt.close()


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    actual = load_actual_validation()
    base_params = dict(mr.IMPUTATION_MODEL_PARAMS)
    base_spec = TrialSpec(
        round_no=0,
        step_no=0,
        parameter="baseline",
        reason="현재 결측값 보정 모델 설정",
        params=base_params,
    )

    results: list[dict[str, Any]] = []
    best = evaluate_trial(base_spec, actual)
    best_params = base_params
    results.append(best)
    pd.DataFrame(results).to_csv(TRIAL_LOG_FILE, index=False, encoding="utf-8-sig")
    print("baseline", {key: best[key] for key in ("NDCG@20", "RMSE", "MAE")}, flush=True)

    step = 1
    stop_reason = ""
    for round_no in range(1, 7):
        round_best = best
        round_best_params = best_params
        candidates = generate_candidates(best_params, round_no, step)
        print(f"round {round_no}: {len(candidates)} candidates", flush=True)
        for spec in candidates:
            step = spec.step_no + 1
            result = evaluate_trial(spec, actual)
            results.append(result)
            pd.DataFrame(results).to_csv(TRIAL_LOG_FILE, index=False, encoding="utf-8-sig")
            print(
                f"step {result['step']:02d} {result['parameter']} "
                f"depth={result['depth']} lr={result['learning_rate']} iter={result['iterations']} "
                f"l2={result['l2_leaf_reg']} leaf={result['min_data_in_leaf']} "
                f"random={result['random_strength']} NDCG@20={result['NDCG@20']:.4f} "
                f"RMSE={result['RMSE']:.2f} MAE={result['MAE']:.2f}",
                flush=True,
            )
            if is_better(result, round_best):
                round_best = result
                round_best_params = spec.params

        if is_better(round_best, best):
            best = round_best
            best_params = round_best_params
            print(f"accepted step {best['step']} with NDCG@20={best['NDCG@20']:.4f}", flush=True)
            continue

        stop_reason = f"round {round_no}에서 후보 {len(candidates)}개를 모두 비교했지만 기존 최고 성능을 넘지 못함"
        break
    else:
        stop_reason = "최대 탐색 라운드에 도달"

    pd.DataFrame(results).to_csv(TRIAL_LOG_FILE, index=False, encoding="utf-8-sig")
    save_summary(results, best, stop_reason)
    save_graph(results)

    print("\nbest")
    print(
        pd.DataFrame([best])[
            [
                "step",
                "depth",
                "learning_rate",
                "iterations",
                "l2_leaf_reg",
                "min_data_in_leaf",
                "random_strength",
                "MAE",
                "RMSE",
                "MAPE",
                "R2",
                "NDCG@20",
            ]
        ].round(4).to_string(index=False)
    )
    print(f"\nsaved {TRIAL_LOG_FILE}")
    print(f"saved {SUMMARY_FILE}")
    print(f"saved {GRAPH_FILE}")
    print(stop_reason)


if __name__ == "__main__":
    main()
