# 인수인계: 녹조 예측 + RAG 챗봇 웹 서비스 (KCI)

> 작성: 2026-10-12 (진행 상황 갱신: 2026-10-13) · 이전 대화의 흐름과 결정 사항을 다음 작업자(AI)가 이어받기 위한 문서.
> 상세 설계는 같은 폴더의 `DEV_PLAN.md`를 함께 읽을 것. 이 문서는 그 이후 바뀐 점과 현재 진행 상황을 담는다.

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

미정: 임베딩 모델(`bge-m3` / `multilingual-e5` 후보), Ollama에서 쓸 모델. LLM은 당분간 Ollama로 정함(5장). GPU를 다른 연구원과 공유하므로 서버에서 불안정하면 외부 API provider를 추가한다.

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

### backend/requirements.txt (확정안, 2026-10-12 · DB 부분 2026-10-13 갱신)

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

# --- DB ---
sqlalchemy==2.1.4
alembic==1.20.0
psycopg[binary]==3.3.6
# pgvector                      # RAG 단계에서 주석 해제

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
httpx2==2.13.1                  # TestClient용. httpx는 Starlette에서 deprecated
```

- 사용자가 예전 NowGo_Busan의 requirements.txt를 재사용하려 했으나 **numpy 2.4.6 / pandas 3.0.3이 모델 환경과 충돌**하고, 지도·좌표·로그인 관련 불필요 패키지가 많아서 새로 만들기로 함. 주석으로 용도별 묶는 형식만 차용.

## 5. 현재 진행 상황 (2026-10-14 기준)

`DEV_PLAN.md` 11장의 **6단계(예측 API mock)까지 완료**. 다음은 프론트 뼈대와 RAG. 이 서비스는 **KCI 실험 설계안의 실험 1~3을 돌리는 평가 플랫폼**이기도 하다 (`DEV_PLAN.md` 13장).

- [x] 서버 환경 확인 (아래 6장)
- [x] 예측모델 코드 분석 (`DEV_PLAN.md` 5장)
- [x] 버전 확정, requirements 확정
- [ ] **관리자에게 Docker 권한 요청** (메시지 초안은 `DEV_PLAN.md` 7장)
- [x] GitHub 저장소: `Dovcl/Algae_RAG_Platform` (Private), SSH로 연결
- [x] 로컬 프로젝트 폴더: `~/Projects/Algae_RAG_Platform` 으로 확정
- [x] 백엔드: `backend/.venv` (Python 3.11.4), `/api/health`
- [x] 프론트: Vite react-ts (React 19, Vite 8, TypeScript 6, ESLint), proxy 설정
- [x] 프론트에서 `/api/health` 호출 성공 (화면에 "백엔드 상태: ok")
- [x] 루트 `.gitignore`, `.env.example`
- [x] AI 지침서: 본문은 `AGENTS.md`, `CLAUDE.md`는 `@AGENTS.md`로 불러옴. 짧게 유지하고 AI가 실수할 때마다 한 줄씩 추가
- [x] Docker Compose 뼈대: `backend` + `db`(`pgvector/pgvector:pg16`), healthcheck, `docker-compose.override.yml`로 개발용 `--reload`
- [x] SQLAlchemy + Alembic 연결, `/api/health/db`
- [x] `chat_logs` 테이블 (`question`, `answer`, `created_at`) — 검색 문서·응답 시간 열은 RAG 단계에서 추가
- [x] 예측 API mock: `GET /api/locations`(21개 지점), `POST /api/predict`(분위수 3개 + 단계). `PREDICT_BACKEND=mock`. 구조는 `app/ml/`(`locations.py`, `predictor.py`, `mock.py`) + `app/routers/predict.py`
- [x] 첫 테스트: `backend/tests/` 17개 (health, 예측 주소, 단계 분류 경계값). `pytest.ini`, `conftest.py`
- [x] 화면 디자인: Stitch로 4개 화면 제작, 넣을 것과 뺄 것 확정 (`DEV_PLAN.md` 13장 "화면별 확정 구성")
- [ ] 프론트 뼈대: 사이드바(대화 기록) + 채팅 + 시각화 탭(지도 / 시계열 / 상세 플롯)
- [ ] RAG: 문서 확정 → chunking·적재 스크립트 → 검색 → LLM 응답
- [ ] DB 조회 모듈, Case 전환(C1~C4), 구조화된 출력, 실험 실행 도구 (`DEV_PLAN.md` 13장)
- [ ] `frontend/Dockerfile`, `nginx/nginx.conf`, `docker-compose.gpu.yml`
- [ ] `README.md`, `.editorconfig`, `docs/` 정리

### 지금까지 정한 것

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
| DB 드라이버 | `psycopg[binary]` 3 (`postgresql+psycopg`) | psycopg2의 후속. SQLAlchemy 2.x 권장 |
| DB 설정 | `app/core/config.py`의 `Settings`가 `POSTGRES_*`를 읽어 `database_url` 생성 | 접속 정보를 코드에서 분리, 비밀번호 특수문자 안전 |
| 개발 실행 | `docker compose up` (override가 `./backend`를 마운트하고 `--reload`) | DB가 컨테이너 안에만 있음 |
| 백엔드 포트 | `127.0.0.1:8000`으로만 공개 | 같은 네트워크의 다른 기기에서 접근 차단 |
| 이미지 라벨 | Dockerfile에 `LABEL project=algae-rag` | 서버에서 이 프로젝트 이미지만 정리 (`DEV_PLAN.md` 8장) |
| 학습 기록 | `STUDY_LOG.md` (git 제외, AI가 직접 써도 됨) | 파일별 용도와 이유를 남기기 위해 |
| RAG 개발 장소 | 맥에서 chunking·적재·검색까지 만들고, 서버에서는 적재 스크립트를 다시 실행 | 서버로 넘어가는 것은 코드와 원본 문서. 벡터는 같은 임베딩 모델이면 재현됨 |
| LLM·예측모델 전환 | `.env`의 `LLM_PROVIDER`(`mock`/`ollama`) + `LLM_MODEL`, `PREDICT_BACKEND`(`mock`/`tft`)로 구현을 고른다. true/false 스위치는 쓰지 않음 | 값 하나만 보면 무엇이 도는지 알 수 있고, 분기가 한 곳에만 생김. `mock`일 때는 torch 없이 실행됨 (아래 "LLM·예측모델 전환 구조") |
| 화면 구성 | 이전 졸업 프로젝트(`rag-chatbot`)를 잇는다: 사이드바 + 채팅 + 시각화 탭 3개. 게시판은 넣지 않는다. 디자인만 새로 | 기능이 검증돼 있고, 달라지는 것은 모델과 실험 목적 |
| 디자인 | Stitch 결과물(`~/Downloads/stitch_ai/`)을 참고로 React 부품을 다시 만든다. 코드는 붙여 넣지 않는다. 화면 속 수치·문구·기관명은 예시라 쓰지 않는다 | 차트·지도가 정적 그림이고, 우리 모델과 다른 내용(Eco-LSTM, 95% 신뢰구간, 여러 주 예측)이 섞여 있음 |
| 스타일 | Tailwind CSS | Stitch 결과물이 Tailwind라 클래스를 거의 그대로 옮길 수 있음 |
| 실험 기준 문서 | `RQ 취합 및 실험 설계안_26-09-04_TS (1).docx`. 요약과 구현 요구사항은 `DEV_PLAN.md` 13장 | KCI 논문의 실험 1~3을 이 서비스로 돌린다 |
| 예측 범위 | 2023년까지의 과거 주. 통합 DB에서 최신 입력을 만드는 변환은 KCI 범위 밖 | 설계안이 예측 질문을 시험 구간(2023년)에서 뽑으라고 함 |
| LLM (당분간) | Ollama (무료, 로컬 실행) | 비용 없이 시작. provider 구조라 나중에 외부 API를 값 하나로 추가 가능 |

### 팀 과업과의 대응

| 팀 과업 | 맥에서 | 서버에서 |
|---|---|---|
| 1. 관리문서 정리·chunking·적재 (출처: 문서명, 페이지, section, chunk ID) | 전부 | PDF 복사 후 적재 스크립트 재실행 |
| 2. 검색 파이프라인 (임베딩 → 검색 → 컨텍스트 → LLM) | 전부 | 그대로 동작 |
| 3. DB·예측모델 연결 | API 형태, mock 예측, 샘플 데이터, LLM 컨텍스트 구획 | 모델 변환, mock → 실제 예측, 실제 데이터 |

과업 3의 "현재값 / 미래 예측값 / 관리문서 정보를 LLM이 구분해서 사용"은 프롬프트에 구획을 나눠 넣는 일이라 mock 값으로 맥에서 완성할 수 있다. 서버에서는 값의 출처만 바뀐다.

### LLM·예측모델 전환 구조 (2026-10-13 결정. `PREDICT_BACKEND=mock`은 구현됨, `tft`와 `LLM_PROVIDER`는 구현 전)

| 설정 | 값 | 맥 | 서버 |
|---|---|---|---|
| `LLM_PROVIDER` | `mock` / `ollama` | 둘 다 가능 | `ollama` |
| `LLM_MODEL` | 예: `llama3.2:3b` | | |
| `OLLAMA_BASE_URL` | 맥: `http://host.docker.internal:11434` | | 서버: Ollama 컨테이너 주소 |
| `PREDICT_BACKEND` | `mock` / `tft` | `mock`만 | `tft` |

- 값은 `app/core/config.py`의 `Settings`에 넣는다. 파일마다 `os.getenv`를 쓰지 않는다.
- 분기는 한 곳에서만 한다. 시작할 때 설정값으로 구현을 하나 골라 끼우고, 함수마다 `if mock:`을 넣지 않는다. `mock`도 구현 중 하나로 취급한다.
- `torch`는 `tft` 구현 안에서만 import한다. `PREDICT_BACKEND=mock`이면 torch가 설치돼 있지 않아도 실행돼야 한다.
- `mock`과 실제 구현은 같은 입력·출력 형태를 지킨다. 테스트는 `mock`으로 돌린다.
- `chat_logs`에 provider와 모델 이름 열을 추가한다. mock 답변이 논문 실험 데이터에 섞이면 나중에 가려낼 수 없다.
- 맥에서는 Ollama를 맥에 직접 설치해 돌린다. 컨테이너 안에서 `localhost:11434`는 통하지 않으므로 `host.docker.internal`을 쓴다.
- 서버에서는 Ollama도 컨테이너로 올려 GPU 0에 묶는다. Docker 권한과 NVIDIA Container Toolkit이 필요하다.
- Ollama 호출에 HTTP 클라이언트가 필요하다. `httpx2`가 지금은 `requirements-dev.txt`에만 있으므로 실행용(`requirements.txt`)으로 옮길지 사용자에게 먼저 묻는다.

### 참고 프로젝트: `~/Documents/GitHub/careerfit_ai_new` (사용자의 이전 RAG 프로젝트)

**가져올 것**

| 무엇 | 그쪽 위치 | 우리 쪽 쓰임 |
|---|---|---|
| SSE 이벤트 순서: `sources` → `token` 반복 → `done`, 실패 시 `error` | `backend/routers/analyze.py`의 `analyze_career_stream`, `_sse_event` | `/api/chat` 스트리밍. 출처를 먼저 보내 답변이 흐르는 동안 근거가 화면에 먼저 뜸 |
| `X-Accel-Buffering: no` 응답 헤더 (`Cache-Control: no-cache`와 함께) | 같은 함수의 `StreamingResponse` | Nginx가 스트리밍을 모았다가 한꺼번에 보내는 것을 막음. 빠뜨리면 스트리밍이 안 되는 것처럼 보임 |
| 프롬프트와 출처 목록을 따로 만드는 구조 | `backend/services/llm_service.py`의 `build_rag_prompt`, `build_sources` | 출처(문서명, 페이지, section, chunk ID) 응답. "현재값 / 예측값 / 관리문서" 구획도 프롬프트 쪽에 같은 방식으로 |
| "자료에 없으면 지어내지 말라" 규칙, 검색 결과가 없을 때의 분기 | `build_rag_prompt` | 관리문서 기반 답변의 환각 방지 |
| 사용자용 오류 문구 변환 (Ollama 꺼짐, 시간 초과) | `get_user_facing_error`, `call_ollama`의 예외 처리 | 챗봇 화면의 오류 안내 |
| Ollama 호출 방식 (`/api/generate`, `stream=True`, timeout 120초) | `stream_ollama`, `call_ollama` | `ollama` provider 구현 |
| 모델 벤치마크: 같은 질문 세트로 모델별 응답 시간·품질 기록 | `backend/data/benchmark_models.py`, `docs/MODEL_BENCHMARK.md` | KCI 논문 실험. 질문 세트와 평가 기준표 형식 참고 |
| `.env.example`에 선택 가능한 모델 값을 주석으로 나열 | `.env.example` | `LLM_PROVIDER`, `LLM_MODEL` 예시 |

**가져오지 않을 것**

- `MOCK_MODE=true/false`를 함수마다 확인하는 방식 → 위 "전환 구조"로 대체
- CORS 설정, `FRONTEND_ORIGINS`, `VITE_API_BASE_URL` → 우리는 같은 주소 구조라 불필요
- ChromaDB → pgvector로 결정함
- 버전 없는 requirements, `*.backup.py` 파일
- provider 4종(Gemini, Mistral, HuggingFace, Ollama) → 지금은 `mock`과 `ollama`만
- 모듈 최상단의 `os.getenv` → `Settings` 사용

### 참고 프로젝트 2: `~/Desktop/UOS/grad_proj/grad_proj1/rag-chatbot` (이전 졸업 프로젝트)

화면과 기능의 원형이다. Vanilla JS + FastAPI + OpenAI + LangChain(PGVector) + TimeSeriesTransformer 1개. 구조 설명은 그 폴더의 `ARCHITECTURE.md`, `RAG.md`, `TRANSFORMER.md`.

**이번에 달라지는 것**

| | 이전 | 이번 |
|---|---|---|
| 예측모델 | TimeSeriesTransformer 1개, 과거 7주로 5종 변수 | 지점별 TFT 21개, 과거 2주로 남조류 세포수 분위수 3개 |
| 목적 | 동작하는 챗봇 | 실험 1~3을 돌리는 평가 플랫폼 |
| 프론트 | Vanilla JS (HTML 3개) | React + Vite + TypeScript |
| LLM / RAG | OpenAI, LangChain | Ollama, 직접 구현 (pgvector) |

**가져올 것**

| 무엇 | 그쪽 위치 | 우리 쪽 쓰임 |
|---|---|---|
| 화면 구성 | `frontend/index.html`, `visualization.html` (탭: 녹조 예측 지도 / 시계열 차트 / 상세 플롯) | 디자인의 출발점. 스크린샷을 디자인 도구에 참고로 넣는다 |
| 채팅 응답에 시각화 데이터를 싣는 형태 | `backend/app/schemas/chat.py`의 `VisualizationData` (`timeseries`, `map_points`, `plot_points`, `query_context`) | `/api/chat` 응답 설계. 예측은 분위수 범위를 더한다 |
| 지도 재료 | `frontend/static/data/rivers.geojson`, `watershed.geojson` | 지도 배경 |
| 지점 좌표 | `monitoring_stations` 테이블 (`backend/app/models/monitoring_station.py`, `db_dump/`, `scripts/load_stations_from_csv.py`) | 21개 지점 좌표를 여기서 찾을 수 있는지 확인. 없으면 연구실에 요청 |
| 대화 기록 API의 형태 | `backend/app/api/history.py` (조회 / 추가 / 수정 / 삭제) | 사이드바의 대화 기록 |
| 평가 재료 | `eval/rag_eval_questions.json`, `scripts/eval_rag_recall.py`, `scripts/export_chunks_for_eval.py` | pilot 질문과 검색 평가의 참고 |
| 지도·차트 구현 참고 | `frontend/visualization.js` (Leaflet, Chart.js) | 라이브러리 선택은 따로 묻고 정한다 |

**가져오지 않을 것**

- 질문에서 키워드를 찾아 분기하는 방식 (`chat_service.py`의 `_detect_prediction_request` 등) → Case마다 동작이 달라질 수 있어 실험 통제와 맞지 않는다. DB 조회와 예측을 독립 모듈로 두고 같은 방식으로 부른다
- 게시판 (`board.*`)
- CORS 설정, `frontend/config.js`의 API 주소 → 우리는 같은 주소 구조
- LangChain, OpenAI 호출
- 그 저장소의 compose·설정 파일에 적힌 값(계정, 비밀번호 등). 새로 정한다
- `>=`로만 적힌 패키지 버전

### 알아 둘 것

- 맥의 `python3`는 anaconda 3.12.7이다. venv는 반드시 `python3.11`로 만든다.
- **venv에서 `uvicorn`을 직접 띄우면 DB에 붙지 못한다.** `db`가 포트를 밖으로 열지 않고 `postgres_host` 기본값이 `db`이기 때문. `/api/health`는 되지만 `/api/health/db`는 실패한다. DB가 필요한 작업은 `docker compose up`으로 한다.
- 마이그레이션은 컨테이너 안에서 실행한다 (`docker compose exec backend alembic upgrade head`).
- pgvector는 이미지에만 들어 있다. Python 패키지는 주석 상태이고 `CREATE EXTENSION vector`도 아직 안 했다. RAG 첫 마이그레이션에서 함께 한다.
- 임베딩 모델을 정하면 벡터 차원이 테이블 정의에 고정된다. 바꾸면 전부 다시 적재해야 하므로 적재 전에 확정한다.
- 적재는 반드시 다시 실행할 수 있는 스크립트로 만든다 (중복 없이). 서버 이전 때 그대로 돌린다.
- `TestClient`용 HTTP 클라이언트는 `httpx2`를 쓴다 (2026-10-13 교체). `httpx`를 쓰면 Starlette deprecated 경고가 뜬다. venv에 `httpx`가 남아 있으면 `pip uninstall httpx`.
- 모델 변환 스크립트(`scripts/export_models.py`)와 `app/ml/`의 TFT 클래스는 학습 코드의 모델 정의를 옮겨 와야 한다. **그 단계 전에 모델 구조 코드를 Private 저장소에 올려도 되는지 연구실에 확인**할 것. 안 되면 서버에만 두고 volume으로 연결한다.
- `src/App.css`는 템플릿에 딸려 온 파일로 현재 쓰지 않는다.

### 다음 순서

1. 서버·연구실 대기 항목 보내기: Docker 권한 요청, `pip show pykalman`, 모델 코드 저장소 반입 가능 여부, 지점 좌표와 한글 이름, 위험등급 기준
2. 프론트 뼈대: Tailwind 설치와 디자인 토큰 → 공통 틀(사이드바, 상단 띠) → 채팅 화면(고정 예시 답변) → 시각화 탭. 시각화는 `/api/locations`, `/api/predict`(mock)로 먼저 연결
3. 아이콘·차트·지도 라이브러리 결정 (각 화면을 만들 때 묻는다)
4. RAG 결정: 사용할 공식 문서, 임베딩 모델, Ollama 모델과 버전
5. RAG 적재: `documents`/`chunks` 테이블(출처 정보 + 문서 버전 메타데이터 + 벡터), chunking·적재 스크립트
6. DB 조회 모듈 (기준일까지만 조회). 통합 DB를 PostgreSQL에 넣을지 여기서 정한다
7. `/api/chat`: Case 전환(C1~C4), 구조화된 출력, 스트리밍. `LLM_PROVIDER` 설정을 이때 넣는다. `chat_logs`에 실험용 열 추가
8. 프론트 채팅 화면을 `/api/chat`에 연결
9. 실험 실행 도구 (일괄 실행, 근거 조작, 블라인드 내보내기)
10. Nginx + 프론트 Dockerfile로 맥에서 전체 구조 실행
11. 기본 파일(README, .editorconfig), Prettier, pre-commit, Vitest
12. `.claude/settings.json`의 `permissions.deny`로 `.env` 읽기 차단 (지침서 글 대신 도구로 강제)

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
- PyTorch TFT, **21개 지점별 개별 모델** (낙동강 12, 한강 4, 영산강 2, 금강 3), 전이학습 버전 사용. 2026-10-13 서버 폴더로 확인. 코드의 지점 목록에는 한강 `HGDG`가 더 있지만 최종 모델은 없다. 지점 코드는 `DEV_PLAN.md` 5장
- 예측 대상: 남조류 세포수(Cyanocell), 분위수 [0.2, 0.5, 0.8], 출력은 `log10(1+x)` → 단계 0~4 (1천/1만/10만/100만 cells/mL 기준)
- 입력 10종(Cyanocell, WT, Chla, TN, TP, WL, Discharge, Temp, Prec, Forecast), 과거 2 + 미래 1시점, 고정 변수 season. **주 단위** 데이터, `Forecast`는 예보 최고기온. 운영 입력은 과거 2주 관측값 + 예측 주의 예보값
- 지점별 입력 CSV는 2023-12-25까지. 통합 DB 6종(`INTEGRATED_DB/`, 2026년 4~6월까지)이 따로 있으나 여기서 모델 입력을 만드는 코드는 없다 → 당장은 **과거 주 조회**만 가능
- 전처리: 칼만 필터 보간 → log10(1+x) → MinMaxScaler(학습구간 ~2021 fit) → 타깃 열 맨 앞
- 모델 파일: `/data/s3/uoswem926/LLM_RAG/selected pickle/{강}/{지점}_final/*.pt` (지점당 1개), 성능표 `MODEL3_LOC_final_results.csv`
- 웹 연동 이슈: scaler 미저장, `torch.save(model)` 통째 저장(`__main__` 클래스 의존), 모델 내부에 `cuda:0` 고정 → **변환 스크립트로 weights + config.json + scaler.joblib 추출** 예정

## 8. 미결정·확인 필요

- 서버 Docker 권한, 컨테이너 GPU(NVIDIA Container Toolkit)
- pykalman 버전
- 위험등급 기준: 모델의 0~4단계를 쓸지, 조류경보제 공식 기준을 따로 적용할지
- C2·C3에 넣을 고정 management context를 어떻게 만들지
- 지점 좌표와 한글 이름 (지도용)
- 대화 기록 범위 (로그인 없이 브라우저별 구분을 제안함)
- 아이콘·지도·차트 라이브러리
- 단계의 공식 명칭
- 통합 DB 6종을 PostgreSQL에 넣을지, CSV를 그대로 읽을지
- 데이터 QA: JSCG 강수 컬럼 이름 등 (`DEV_PLAN.md` 5장)
- 임베딩 모델, Ollama에서 쓸 모델 (LLM은 당분간 Ollama로 결정)
- API 약속(엔드포인트·요청·응답 형태) — 예: `GET /api/locations`, `POST /api/predict`. FastAPI OpenAPI + openapi-typescript로 타입 공유 제안함
- 도메인, 외부 공개 범위와 연구실 승인

## 9. 이 폴더의 관련 파일

| 파일 | 내용 |
|---|---|
| `DEV_PLAN.md` | 전체 개발 계획 (아키텍처, 결정 이유, 모델 분석, 서버 환경, 폴더 구조, 진행 단계) |
| `HANDOFF.md` | 이 문서 |
| `TFT_TL_PLOT.py`, `TFT_train_each_location.py` | 예측모델 코드 사본 (git 제외) |
| 서버 `LLM_RAG/algae_forecast_project_overview.html` | 데이터·모델·코드 인수인계서 (기준일 2026-09-04). 맥 사본은 `~/Downloads/`. 핵심은 `DEV_PLAN.md` 5장에 옮김 |
| `#. RAG-녹조예측/` | 참고 논문, 미팅 자료(KCI 논문화 설계 보고서 등) |
| `RQ 취합 및 실험 설계안_26-09-04_TS (1).docx` | 연구 질문·실험 설계. **구현의 기준 문서.** 요약은 `DEV_PLAN.md` 13장 |
