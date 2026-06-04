# 서울대공원 눈치싸움

`머신러닝3`는 서울대공원 방문자 수 예측 모델을 웹 서비스로 감싼 이식형 시연 패키지입니다. 이 폴더만 다른 기기로 옮겨도 실행할 수 있도록 데이터, 백엔드, 프론트엔드, 실행 파일을 함께 넣었습니다.

## 구성

- `backend/app/main.py`
  - FastAPI 서버 진입점
- `backend/app/services/model_runtime.py`
  - 기존 2단계 CatBoost 구조를 바탕으로 모델 학습과 미래 예측을 수행
- `backend/model_tuning_settings.py`
  - 모델 하이퍼파라미터와 혼잡일 가중치를 한 곳에서 수정
- `backend/app/services/weather_service.py`
  - 공공 날씨 API와 기본기후 fallback 처리
- `backend/app/services/event_season_service.py`
  - 행사 CSV를 시즌 feature로 가공
- `frontend`
  - React 기반 달력 UI
- `scripts/prepare_artifacts.py`
  - 예측 캐시와 아티팩트 생성
- `scripts/run_demo.py`
  - 시연 서버 실행
- `모델_튜닝_가이드.md`
  - 하이퍼파라미터 수정 방법과 캐시 재생성 순서

## 핵심 동작

- 제목: `서울대공원 눈치싸움`
- 달력에서 날짜 선택 시 예측 방문객 수와 혼잡도 출력
- 선택일은 연한 초록색 단일 선택 처리
- 오늘 이전 날짜와 기존 학습/평가 날짜는 선택 불가
- 토요일, 일요일, 공휴일은 빨간 글자 표시
- `위험도 보기` 토글로 선택일 기준 주변 5일 위험도 표시
- 혼잡도 기준
  - 한산함: 3,000명 미만
  - 보통: 3,000명 이상 ~ 7,999명 이하
  - 붐빔: 8,000명 이상 ~ 14,999명 이하
  - 매우 붐빔: 15,000명 이상

## 실행 방법

Windows에서는 `클릭해서_실행.cmd`를 더블클릭하면 됩니다. 필요한 Python 패키지가 없으면 자동으로 설치한 뒤 서버를 실행합니다.

### 1. Python 패키지 설치

Linux/macOS:

```bash
python3 -m pip install --user --break-system-packages -r backend/requirements.txt
```

Windows:

```bat
py -m pip install -r backend\requirements.txt
```

### 2. 프론트엔드 의존성 설치 및 빌드

```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. 예측 캐시 생성

```bash
python3 scripts/prepare_artifacts.py
```

Windows:

```bat
py scripts\prepare_artifacts.py
```

### 4. 시연 서버 실행

```bash
python3 scripts/run_demo.py
```

Windows:

```bat
클릭해서_실행.cmd
```

브라우저에서 `http://127.0.0.1:8765` 에 접속합니다.

## 테스트

백엔드 테스트:

```bash
cd backend
python3 -m pytest
cd ..
```

프론트엔드 빌드 확인:

```bash
cd frontend
npm run build
cd ..
```

## 예시 입력/출력

- 입력: `2026-04-20`
- 출력 예시
  - 예상 방문객 수: `2,814명`
  - 혼잡도: `붐빔`
  - 날짜 코멘트: 평일/공휴일/시즌/날씨 source를 반영한 설명
  - 랜덤 코멘트: 혼잡도 단계에 맞는 자연어 문구

## 다른 기기로 보낼 때

다음만 함께 보내면 됩니다.

- `backend`
- `frontend`
- `scripts`
- `클릭해서_실행.cmd`
- `README.md`

`node_modules`와 Python 캐시는 보내지 않아도 됩니다. 대신 다른 기기에서 `npm install`, `pip install`, `npm run build`, `python scripts/prepare_artifacts.py`를 한 번 실행하면 됩니다.
