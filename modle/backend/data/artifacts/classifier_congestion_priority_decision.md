# 1단계 혼잡일 분류 모델 파라미터 결정 기록

- 결정일: 2026-05-13
- 목적: 방문객 수 자체의 평균 오차보다 혼잡일을 상위 위험일로 잘 올리는 성능을 우선
- 검증 구간: 2025-05-01 ~ 2025-07-31
- 우선 지표: NDCG@20
- 참고 로그:
  - `backend/data/artifacts/classifier_tuning_trials.csv`
  - `backend/data/artifacts/classifier_random100_trials.csv`

## 최종 적용값

```python
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
```

## 선택 이유

| 기준 | depth | learning_rate | iterations | l2_leaf_reg | MAE | RMSE | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 설정 | 6 | 0.05 | 500 | 3.0 | 3266.38 | 4620.77 | 0.8825 |
| 방문객 수 정확도 우선 | 6 | 0.03 | 350 | 3.0 | 2839.21 | 4188.74 | 0.8952 |
| 1차 혼잡도 랭킹 우선 | 5 | 0.03 | 500 | 3.0 | 3225.01 | 4313.09 | 0.9300 |
| 100회 무작위 탐색 최종 | 6 | 0.06 | 800 | 5.0 | 3160.34 | 4419.26 | 0.9399 |

100회 무작위 탐색 결과, `depth=6`, `learning_rate=0.06`, `iterations=800`, `l2_leaf_reg=5.0` 조합이 실제 혼잡일을 상위 20개 위험일에 가장 잘 배치했습니다. 방문객 수 정확도만 보면 더 낮은 RMSE 조합이 있지만, 이 프로젝트의 목표를 혼잡 위험일 탐지로 둘 경우 `NDCG@20`이 가장 높은 이 설정을 사용합니다.
