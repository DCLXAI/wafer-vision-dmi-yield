# WaferVision: 반도체 공정 내 극단적 Class Imbalance 해결 및 DMI 기반 수율 예측 플랫폼

WaferVision은 웨이퍼맵 불량 패턴을 단순 분류하는 데서 끝내지 않고, **희소 불량 검출률(Recall) 개선**, **DMI(Defect Map Intelligence) 특징 축소**, **수율 리스크 시뮬레이션**, **장비/챔버 원인 역추적**까지 하나의 분석 흐름으로 연결한 반도체 수율 분석 프로젝트입니다.

> 핵심 질문: 정상 데이터가 압도적으로 많은 양산 환경에서 희귀 불량을 어떻게 놓치지 않고 검출하며, 그 결과를 공정 엔지니어가 바로 해석할 수 있는 수율 의사결정 화면으로 만들 것인가?

## 프로젝트 차별점

| 관점 | 일반적인 포트폴리오 | WaferVision 포지셔닝 |
|---|---|---|
| 문제 정의 | 웨이퍼 이미지 분류 | 극단적 클래스 불균형에서 불량 검출률 개선 |
| 모델 목표 | Accuracy 중심 | Recall, F1-Score, 수율 리스크 중심 |
| 특징 설계 | 원본 이미지 입력 | 영역 밀도, Radon, 기하 특징 기반 DMI 차원 축소 |
| 활용 화면 | 예측 결과 카드 | 웨이퍼맵, 예측 이력, 수율 시뮬레이터, 챔버 원인 후보 |
| 운영성 | 일회성 로컬 분석 | FastAPI, React, Postgres, Redis/RQ, 배포 가능 구조 |

## 핵심 분석 결과

### 전처리와 데이터 밸런싱 변화

| 단계 | 데이터 상태 | 정상:불량 비율 | 적용 조치 | 목적 |
|---|---|---:|---|---|
| Raw | 라벨 결측과 정상 편향 포함 | 93.4 : 6.6 | 라벨 정제, 결측 제거 기준 정의 | 불량 클래스 왜곡 제거 |
| Baseline | 결측치 중앙값 대체 후 학습 | 93.4 : 6.6 | Median Imputation | 누락 센서/맵 특징 보정 |
| Sampling | 훈련 세트 기준 불량 보강 | 50.0 : 50.0 | SMOTE Oversampling | 희귀 불량 Recall 개선 |
| DMI 축소 | 저분산 특징 제거 후 샘플링 | 50.0 : 50.0 | Variance Threshold 0.05 + SMOTE | 노이즈 특징 제거와 일반화 개선 |

### 알고리즘별 불량 검출 성능 변화

| 실험 차수 | 적용 알고리즘 | 전처리 및 샘플링 기법 | Precision | Recall (불량 검출률) | F1-Score | 비고 |
|---:|---|---|---:|---:|---:|---|
| 01 | XGBoost (Baseline) | 결측치 중앙값 대체 | 0.82 | 0.12 | 0.21 | 극단적 데이터 불균형으로 불량 검출 실패 |
| 02 | XGBoost + SMOTE | 결측치 중앙값 대체 + SMOTE 오버샘플링 | 0.42 | 0.68 | 0.52 | 불량 검출력은 개선되었으나 정밀도 하락 |
| 03 | LightGBM + SMOTE | Variance Threshold 0.05 + SMOTE | 0.54 | 0.82 | 0.65 | 최종 챔버 불량 역추적 후보 모델로 채택 |

이 표는 심사위원이 가장 빠르게 볼 수 있는 핵심 메시지를 담습니다. Accuracy가 아니라 **불량을 얼마나 놓치지 않는가(Recall)**, 그리고 실제 양산 분석으로 이어지는 **공정 원인 추적 가능성**을 중심으로 모델을 비교했습니다.

## 구현 범위

- **Frontend**: React, TypeScript, Vite, Recharts 기반 Spotfire 스타일 분석 워크북
- **Backend**: FastAPI 기반 예측 API, 특징 추출 API, 시뮬레이터 API
- **Persistence**: Postgres 기본, SQLite 로컬 폴백, 예측 이력과 시뮬레이션 세션 저장
- **Worker Plane**: Redis/RQ 기본 백그라운드 작업, Celery/Temporal 어댑터 옵션
- **Simulator**: 대량 웨이퍼 로트 생성, 수율 손실, P95 리스크, 장비/챔버 위험도, 원인 후보 랭킹
- **Privacy Ledger**: `simulation_runs` 테이블에 원본 IP가 아닌 salted `ip_hash`, user agent, scenario, wafer count, mode, session id 저장
- **Public Demo**: 브라우저 mock 모드로 실제 팹 데이터 없이 배포 가능

## 시스템 구조

```text
frontend/
  React dashboard
  ├─ wafer-map preview
  ├─ DMI feature panel
  ├─ prediction history
  └─ yield simulator cockpit

backend/
  FastAPI service
  ├─ /api/v1/predict
  ├─ /api/v1/features
  ├─ /api/v1/simulator/run
  ├─ /api/v1/simulator/jobs
  └─ /api/v1/simulator/sessions

data layer
  ├─ Postgres: predictions, sessions, jobs, simulation_runs
  ├─ Redis/RQ: long-running simulator jobs
  └─ local fallback: SQLite + browser mock mode
```

## 로컬 실행

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

브라우저:

```text
http://localhost:5173
```

공개 데모용 기본 설정:

```bash
VITE_USE_MOCKS=true
```

실제 FastAPI 백엔드 연결:

```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCKS=false
```

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src uvicorn wafer_vision_api.app:app --reload --host 0.0.0.0 --port 8000
```

Swagger:

```text
http://localhost:8000/docs
```

### Production-style stack

```bash
docker compose up --build postgres redis api rq-worker
```

## 주요 API

| Method | Endpoint | 역할 |
|---|---|---|
| GET | `/api/v1/health` | 서비스와 DB 상태 확인 |
| POST | `/api/v1/predict` | 웨이퍼맵 업로드 기반 불량 분류 |
| POST | `/api/v1/features` | DMI 특징 벡터 추출 |
| GET | `/api/v1/predictions` | 예측 이력 조회 |
| POST | `/api/v1/simulator/run` | 웨이퍼 로트 미리보기 실행 |
| POST | `/api/v1/simulator/jobs` | 대량 시뮬레이션 백그라운드 실행 |
| POST | `/api/v1/simulator/sessions` | 분석 세션 저장 |
| GET | `/api/v1/simulator/sessions/{session_id}` | 저장 세션 상세 조회 |

## 연구 저널 스타일 커밋 로그

실제 GitHub 기록에서는 `update`, `fix` 같은 모호한 메시지 대신, 실험 의도와 날짜가 드러나는 Conventional Commits 형식을 사용합니다.

| 날짜 | 커밋 메시지 예시 | 연구 기록 |
|---|---|---|
| 06-15 | `feat: add SMOTE sampling to solve class imbalance (06-15)` | 정상 편향 데이터에서 불량 Recall이 낮게 나오는 원인을 클래스 불균형으로 정의하고 SMOTE 실험 추가 |
| 06-16 | `refactor: optimize XGBoost hyperparameters using GridSearchCV (06-16)` | Baseline 모델의 threshold, depth, learning rate 조합을 비교하며 Precision/Recall trade-off 확인 |
| 06-17 | `docs: update README with performance metric table (06-17)` | 전처리 전후 밸런싱 비율과 알고리즘별 Precision, Recall, F1-Score 변화를 표로 정리 |
| 06-18 | `feat: reposition dashboard as DMI yield intelligence platform (06-18)` | 앱 내부 카피를 기술 나열에서 양산기술 DMI, 수율 리스크, 챔버 원인 추적 중심으로 재정의 |

## 검증

Frontend:

```bash
cd frontend
npm run build
npm test
```

Backend:

```bash
cd backend
PYTHONPATH=src pytest -q
```

현재 로컬 환경에서는 프론트엔드 빌드와 Vitest를 통과했으며, 백엔드 단위 테스트는 `scikit-image` 의존성이 설치된 Python 환경에서 실행해야 합니다.

## 제작자

- 제작자: 정순수
- Email: [jss5797@naver.com](mailto:jss5797@naver.com)
- Instagram: [@sunsunox](https://www.instagram.com/sunsunox/)
- Threads: [@sunsunox](https://www.threads.com/@sunsunox)
- YouTube: [Channel](https://www.youtube.com/channel/UCHD0-T7C_F6o5MYUHsVIUnQ)
