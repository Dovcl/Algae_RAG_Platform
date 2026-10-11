# 인수인계: 녹조 예측 + RAG 챗봇 웹 서비스 (KCI)

> 작성: 2026-10-12 (같은 날 밤 진행 상황 갱신) · 이전 대화의 흐름과 결정 사항을 다음 작업자(AI)가 이어받기 위한 문서.
> 상세 설계는 같은 폴더의 `DEV_PLAN.md`를 함께 읽을 것. 이 문서는 그 이후 바뀐 점과 현재 진행 상황을 담는다.

## 0. 사용자와 일하는 방식

- 사용자: 도현 (서울시립대 연구실, KCI 논문 준비 중). 한국어로 대화.
- **명령어를 직접 입력하며 배우고 싶어 함.** 폴더나 파일을 AI가 대신 만들지 말고, 순서와 명령어를 알려주면 사용자가 실행한다. 사용자가 명시적으로 부탁할 때만 직접 만든다.
- 개념 설명을 좋아함. 왜 그렇게 하는지(이유)를 함께 설명할 것. 현업에서 쓰는 일반적인 방식으로 만들고 싶어 함.
- 이전에 `~/Projects/NowGo_Busan` 프로젝트에서 CLAUDE.md / GPT 지침서(하네스 엔지니어링)를 써 봤고, 이번에도 할 예정.

## 1. 프로젝트 한 줄 요약

연구실 서버에 있는 **TFT 기반 남조류(녹조) 예측모델**과 **RAG 챗봇**을 웹 서비스로 만들어, Render/Vercel 없이 **연구실 서버 한 대에서 Docker Compose로 배포**한다.

## 2. 지금까지의 핵심 결정

| 결정 | 내용 | 이유 |
|---|---|---|
| 배포 위치 | 연구실 서버 | 모델·데이터가 이미 서버에 있음 |
| Render / Vercel | 사용 안 함 | 백엔드가 어차피 서버에 있어야 함. 배포 지점 2곳·CORS 관리 회피 |
| 구조 | Nginx가 React 빌드 파일 서빙 + `/api`를 FastAPI로 프록시 | 같은 도메인 → **CORS 불필요**. 현업 표준 |
| 개발 시 API 연결 | Vite proxy (`/api` → `localhost:8000`) | 배포 때 Nginx와 같은 역할 |
| 외부 공개 | Cloudflare Tunnel | 학교 방화벽 개방 불필요 |
| 컨테이너 | Docker + Docker Compose | 환경 재현, 자원 제한, 재시작 |
| 언어 | 프론트 TypeScript | 사용자가 선택 |
| DB | PostgreSQL + pgvector | 대화 기록 + 벡터 검색을 DB 하나로 |
| **개발 장소** | **맥에서 프론트·백엔드 둘 다 개발** → Docker 권한 받으면 git으로 서버 이전 | 서버 Docker 권한이 아직 없음 |
| 현재 목표 | **웹페이지만 먼저 띄우기** (프론트에서 `/api/health` 호출 성공까지) | 모델·DB·RAG는 그 다음 |

중간에 "백엔드는 서버, 프론트는 맥(Figma 활용)"으로 나누는 안도 논의했으나, 최종적으로 **둘 다 맥에서 먼저 만들기**로 정함.

## 3. 기술 스택

| 영역 | 스택 |
|---|---|
| Frontend | React + Vite + TypeScript |
| Backend | FastAPI + Pydantic, 운영은 gunicorn + uvicorn workers |
| DB | PostgreSQL + pgvector, SQLAlchemy + Alembic |
| 웹 서버 | Nginx |
| 컨테이너 | Docker Compose (공통 `docker-compose.yml` + 서버 전용 `docker-compose.gpu.yml`) |
| 외부 공개 | Cloudflare Tunnel |
| 개발 도구 | Ruff(린트+포맷), pytest, ESLint, Prettier, pre-commit |
| 나중에 | GitHub Actions self-hosted runner (자동 배포) |

미정: 임베딩 모델(`bge-m3` / `multilingual-e5` 후보), LLM(외부 API vs 서버 GPU 직접 실행 — GPU를 다른 연구원과 공유하므로 외부 API가 현실적이라는 의견까지 나눔).

## 4. 확정 버전

예측모델 학습 환경(서버의 `~/np2-environment.yml`, conda 환경 `np2`) 기준.

| 항목 | 버전 | 비고 |
|---|---|---|
| Python | 3.11 | 학습 환경 3.11.13 |
| torch | 2.5.1 | 서버 `+cu121`, 맥 CPU. 2.6부터 `torch.load` 기본값 변경 → 2.5.1 고정 |
| numpy | 1.26.4 | **2.x 금지** |
| pandas | 2.1.4 | **3.x 금지** |
| scikit-learn | 1.7.2 | |
| pykalman | 미확인 (최신 0.11.2를 임시로 기재) | 서버 np2에서 `pip show pykalman` 확인 필요 |
| CUDA | 12.1 빌드 | 서버 드라이버 535 / CUDA 12.2 |
| Node.js | 24 LTS | 맥에 설치된 v24.13.0 사용. nvm은 설치하지 않음 |
| PostgreSQL | `pgvector/pgvector:pg16` | |

### backend/requirements.txt (확정안, 2026-10-12)

웹 패키지 버전은 PyPI 최신 안정 버전으로, Python 3.11 / macOS arm64에서 충돌 없이 설치되는 것을 확인함.

```txt
# Python 3.11.x (예측모델 학습 환경 np2와 동일)
# 설치: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# --- 웹 서버 ---
fastapi==0.143.0
uvicorn[standard]==0.54.0

# --- 설정 (.env 로딩) ---
pydantic-settings==2.15.0
python-dotenv==1.2.4

# --- DB (5단계에서 주석 해제) ---
# sqlalchemy
# alembic
# psycopg2-binary
# pgvector

# --- 예측모델 TFT (6단계에서 주석 해제) ---
# 학습 환경(np2-environment.yml)과 버전을 반드시 맞출 것. 임의로 올리지 말 것.
# torch==2.5.1                  # 서버(GPU)는 --extra-index-url https://download.pytorch.org/whl/cu121
# numpy==1.26.4
# pandas==2.1.4
# scikit-learn==1.7.2
# pykalman==0.11.2              # 서버 np2 환경 버전 확인 후 맞추기
```

### backend/requirements-dev.txt

```txt
-r requirements.txt

ruff==0.17.0
pytest==9.1.1
httpx==0.28.1
```

- 사용자가 예전 NowGo_Busan의 requirements.txt를 재사용하려 했으나 **numpy 2.4.6 / pandas 3.0.3이 모델 환경과 충돌**하고, 지도·좌표·로그인 관련 불필요 패키지가 많아서 새로 만들기로 함. 주석으로 용도별 묶는 형식만 차용.

## 5. 현재 진행 상황 (2026-10-12 밤 기준)

첫 목표였던 **"프론트에서 `/api/health` 호출 성공"까지 완료**하고 첫 커밋을 GitHub에 올렸다.

- [x] 서버 환경 확인 (아래 6장)
- [x] 예측모델 코드 분석 (`DEV_PLAN.md` 5장)
- [x] 버전 확정, requirements 확정
- [ ] **관리자에게 Docker 권한 요청** (메시지 초안은 `DEV_PLAN.md` 7장)
- [x] GitHub 저장소: `Dovcl/Algae_RAG_Platform` (Private), SSH로 연결, `main`에 첫 커밋 push
- [x] 로컬 프로젝트 폴더: `~/Projects/Algae_RAG_Platform` 으로 확정
- [x] 백엔드: `backend/.venv` (Python 3.11.4), requirements 설치, `/api/health`
- [x] 프론트: Vite react-ts (React 19, Vite 8, TypeScript 6, ESLint), proxy 설정
- [x] 프론트에서 `/api/health` 호출 성공 (화면에 "백엔드 상태: ok")
- [x] 루트 `.gitignore`
- [ ] `README.md`, `.editorconfig`, `.env.example`, `docs/` 정리

### 이번에 정한 것

| 항목 | 결정 | 이유 |
|---|---|---|
| 백엔드 구조 | `backend/app/` 패키지, 실행은 `uvicorn app.main:app --reload --port 8000` | import가 실행 위치에 덜 의존, Dockerfile에서 `COPY app/`만 하면 됨 |
| 라우터 폴더 이름 | `app/routers/` (`api/` 아님) | FastAPI 공식 문서 명칭, URL의 `/api`와 혼동 방지 |
| `/api` 접두사 | `app/routers/__init__.py`의 `api_router = APIRouter(prefix="/api")`에서 한 번만 붙임 | 각 라우터 파일은 `/health`처럼 짧게 작성 |
| requirements 분리 | `requirements.txt`(실행용) + `requirements-dev.txt`(개발용) | 운영 Docker 이미지에 개발 도구를 넣지 않기 위해 |
| Node 버전 | 24 LTS | 맥에 이미 설치돼 있고, Node는 모델 환경과 맞출 제약이 없음 |
| 린터 | ESLint (Oxlint 아님) | 현업 표준, 계획한 도구 구성 |
| 연구실 코드 | `TFT_*.py`, `관련파일/`, `*.docx`, `*.pptx`는 **git에 올리지 않음** (`.gitignore`) | 연구실 자산 |
| 커밋 | 사용자가 직접 입력. `Co-Authored-By` 줄 없이 | 사용자 요청 |

### 알아 둘 것

- 맥의 `python3`는 anaconda 3.12.7이다. venv는 반드시 `python3.11`로 만든다.
- pytest로 FastAPI `TestClient`를 쓰면 "`httpx` 대신 `httpx2`를 설치하라"는 Starlette deprecated 경고가 뜬다. 동작은 정상. 첫 테스트를 쓸 때 `requirements-dev.txt`를 정리한다.
- 모델 변환 스크립트(`scripts/export_models.py`)와 `app/ml/`의 TFT 클래스는 학습 코드의 모델 정의를 옮겨 와야 한다. **그 단계 전에 모델 구조 코드를 Private 저장소에 올려도 되는지 연구실에 확인**할 것. 안 되면 서버에만 두고 volume으로 연결한다.
- `src/App.css`는 템플릿에 딸려 온 파일로 현재 쓰지 않는다.

### 다음 순서

1. AI 지침서: 본문은 `AGENTS.md`, `CLAUDE.md`는 `@AGENTS.md`로 불러오고 Claude 전용만 추가. 짧게 시작해 실수할 때마다 한 줄씩 추가. 넣을 것: 프로젝트 요약, 확정 버전(임의 업그레이드 금지), 실행·테스트 명령, 코드 규칙, 금지사항(.env 커밋·연구실 코드 커밋·모델 파일 수정·서버 GPU 1 사용·무단 패키지 추가), 완료 기준
2. Docker Compose 뼈대 (backend + db 먼저). Docker Desktop 메모리 상한 설정
3. 첫 테스트 (`backend/tests/test_health.py`) — 예측 API mock을 만들 때 함께 해도 됨
4. 기본 파일: README, .editorconfig, .env.example, docs/
5. Prettier, pre-commit, Vitest
6. 커밋 규칙: Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`)

## 6. 연구실 서버 정보 (2026-10-08 확인)

| 항목 | 값 |
|---|---|
| 접속 | `uoswem926@uos-wem` (VS Code Remote-SSH 사용) |
| 홈 실제 경로 | `/data/s3/uoswem926` |
| sudo | **없음** |
| Docker | 설치돼 있으나 **권한 없음** (docker 그룹 미소속) → 관리자 요청 |
| GPU | NVIDIA A40 46GB × 2. GPU 0 사용 권장 (GPU 1은 다른 연구원이 98% 사용 중) |
| 드라이버 / CUDA | 535.161.07 / 12.2 |
| 기타 | git 2.34.1, 시스템 Python 3.10.12(사용 안 함), conda 23.7.4, **Node 없음**(필요 시 nvm) |

서버 이전 시 챙길 것: `.env` 새로 작성, 모델·데이터는 volume(`:ro`)으로 연결, 이미지는 서버에서 빌드(맥 ARM ≠ 서버 x86), GPU는 `docker-compose.gpu.yml`, 포트 충돌 확인(`ss -tlnp`), 컨테이너가 쓴 파일의 root 소유 문제(`user:` 지정), 서버에서 Docker Hub·PyPI·npm 접근 가능한지 확인.

## 7. 예측모델 요약 (상세는 DEV_PLAN.md 5장)

- 코드: `TFT_train_each_location.py`, `TFT_TL_PLOT.py` (이 폴더에 복사본 있음)
- PyTorch TFT, **22개 지점별 개별 모델** (낙동강 12, 한강 5, 영산강 2, 금강 3), 전이학습 버전 사용
- 예측 대상: 남조류 세포수(Cyanocell), 분위수 [0.2, 0.5, 0.8], 출력은 `log10(1+x)` → 단계 0~4 (1천/1만/10만/100만 cells/mL 기준)
- 입력 10종(Cyanocell, WT, Chla, TN, TP, WL, Discharge, Temp, Prec, Forecast), 과거 2 + 미래 1시점, 고정 변수 season
- 전처리: 칼만 필터 보간 → log10(1+x) → MinMaxScaler(학습구간 ~2021 fit) → 타깃 열 맨 앞
- 모델 파일: `/data/s3/uoswem926/LLM_RAG/selected pickle/{강}/{지점}_final/*.pt` (지점당 1개), 성능표 `MODEL3_LOC_final_results.csv`
- 웹 연동 이슈: scaler 미저장, `torch.save(model)` 통째 저장(`__main__` 클래스 의존), 모델 내부에 `cuda:0` 고정 → **변환 스크립트로 weights + config.json + scaler.joblib 추출** 예정

## 8. 미결정·확인 필요

- 서버 Docker 권한, 컨테이너 GPU(NVIDIA Container Toolkit)
- pykalman 버전
- 데이터 시점 간격(일/주) — `~/LLM_RAG/DATA/{강}/` CSV로 확인
- `Forecast` 변수 의미, 웹에서 최신 입력 데이터를 어디서 받을지 → "오늘 기준 예측" vs "과거 기간 결과 조회" 화면 결정
- LLM, 임베딩 모델
- API 약속(엔드포인트·요청·응답 형태) — 예: `GET /api/locations`, `POST /api/predict`. FastAPI OpenAPI + openapi-typescript로 타입 공유 제안함
- 도메인, 외부 공개 범위와 연구실 승인

## 9. 이 폴더의 관련 파일

| 파일 | 내용 |
|---|---|
| `DEV_PLAN.md` | 전체 개발 계획 (아키텍처, 결정 이유, 모델 분석, 서버 환경, 폴더 구조, 진행 단계) |
| `HANDOFF.md` | 이 문서 |
| `TFT_TL_PLOT.py`, `TFT_train_each_location.py` | 예측모델 코드 사본 (git 제외) |
| `#. RAG-녹조예측/` | 참고 논문, 미팅 자료(KCI 논문화 설계 보고서 등) |
| `RQ 취합 및 실험 설계안_26-09-04_TS (1).docx` | 연구 질문·실험 설계 |
