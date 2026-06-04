# 2-2-2

이 폴더는 2모델 방식으로 구성한 혼잡일 중심 CatBoost 모델을 보관합니다.

구성:
- 1단계: `CatBoostClassifier`로 혼잡일 여부와 혼잡일 확률 예측
- 2단계: `CatBoostRegressor`가 1단계 출력(`혼잡일확률`, `혼잡일예측라벨`)을 추가 특징으로 받아 방문객 수 예측

포함 파일:
- `second_improvement_two_stage_busy_focus.py`
- `second_improvement_two_stage_busy_focus_graph.png`
- `second_improvement_two_stage_busy_focus_results.csv`
- `catboost_info/`

비교 구간:
- `2025-05-01 ~ 2025-07-31`

공용 데이터는 상위 폴더를 참조합니다.
- `../seoul_grand_park_daily_visitors_with_missing_dates.csv`
- `../seoul_grand_park_events_2015_2025_aug.csv`
- `../날씨 데이터/`

실행:
```powershell
python .\second_improvement_two_stage_busy_focus.py
```
