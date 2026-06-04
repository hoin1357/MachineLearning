# 서울대공원 눈치싸움 실행 패키지

이 폴더는 서울대공원 방문객/혼잡도 예측 웹 서비스를 GitHub에서 바로 내려받아 실행할 수 있게 묶은 패키지입니다. 백엔드, 프론트엔드 빌드 결과, 모델 학습/예측 데이터, 날씨/행사 데이터가 모두 `modle` 폴더 안에 들어 있습니다.

## 바로 실행

```bash
cd modle
python -m pip install -r backend/requirements.txt
python scripts/run_demo.py
```

브라우저에서 `http://127.0.0.1:8765`로 접속합니다.

## 클라우드/Docker 실행

Docker를 지원하는 Render, Railway, Fly.io, VM 등에서는 이 폴더를 Docker 컨텍스트로 배포하면 됩니다.

```bash
cd modle
docker build -t seoul-grand-park-forecast .
docker run --rm -p 8765:8765 seoul-grand-park-forecast
```

클라우드 플랫폼이 `PORT` 환경변수를 주입하면 서버가 자동으로 그 포트를 사용합니다.

## 환경변수

- `PORT`: 클라우드 플랫폼 포트. 있으면 `APP_PORT`보다 우선합니다.
- `APP_PORT`: 로컬 실행 포트. 기본값은 `8765`입니다.
- `APP_HOST`: 기본값은 `0.0.0.0`입니다.
- `OPEN_BROWSER`: 클라우드에서는 `0`으로 둡니다.
- `WEATHER_API_SERVICE_KEY`: 선택값입니다. 없으면 날씨 API 호출 없이 과거 기후 평균값으로 예측합니다.

## 포함된 주요 파일

- `backend/app`: FastAPI API와 모델 런타임
- `backend/data`: 방문객, 행사, 날씨 이력, 예측 캐시, 외부 교통 분석 데이터
- `frontend/dist`: 서버가 바로 제공하는 프론트엔드 빌드 결과
- `frontend/src`: React 프론트엔드 소스
- `scripts/run_demo.py`: 웹 서버 실행 스크립트
- `scripts/prepare_artifacts.py`: 예측 캐시 재생성 스크립트

## 새 예측 캐시 생성

데이터나 모델 설정을 바꾼 뒤에는 다음을 실행합니다.

```bash
cd modle
python scripts/prepare_artifacts.py
```

그 뒤 다시 `python scripts/run_demo.py`로 서버를 띄우면 갱신된 캐시를 사용합니다.
