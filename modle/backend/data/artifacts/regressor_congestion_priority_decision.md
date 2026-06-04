# 2단계 방문객 수 예측 모델 파라미터 결정 기록

- 결정일: 2026-05-13
- 목적: 최종 예측 방문객 수의 순위가 실제 혼잡일을 상위 위험일로 잘 올리도록 조정
- 검증 구간: 2025-05-01 ~ 2025-07-31
- 우선 지표: NDCG@20
- 참고 로그: `backend/data/artifacts/regressor_random100_trials.csv`
- 전제: 1단계 혼잡일 분류 모델은 `depth=6`, `learning_rate=0.06`, `iterations=800`, `l2_leaf_reg=5.0`으로 고정

## 최종 적용값

```python
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
```

## 선택 이유

| 기준 | depth | learning_rate | iterations | l2_leaf_reg | MAE | RMSE | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 회귀 설정 | 8 | 0.05 | 800 | 5.0 | - | - | - |
| 100회 무작위 탐색 최종 | 10 | 0.04 | 1200 | 8.0 | 3214.38 | 4509.65 | 0.9478 |

100회 무작위 탐색 결과, `depth=10`, `learning_rate=0.04`, `iterations=1200`, `l2_leaf_reg=8.0` 조합이 실제 혼잡일을 상위 20개 위험일에 가장 잘 배치했습니다. 이 설정은 방문객 수 평균 오차만 최소화하는 값은 아니지만, 혼잡 위험일 탐지를 우선하는 현재 목표에 가장 적합합니다.
