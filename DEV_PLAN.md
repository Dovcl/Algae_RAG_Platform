# KCI RAG 챗봇 웹 서비스 개발 계획

> 최종 수정: 2026-10-13 · 지금까지 논의한 내용 정리. 폴더와 파일은 명령어를 직접 입력하면서 단계별로 같이 만든다.

## 1. 개요

- KCI 논문용 RAG 챗봇과 녹조(남조류) 예측모델을 웹 서비스로 만든다.
- Render나 Vercel 같은 외부 배포 서비스 없이 **연구실 서버 한 대에서 Docker Compose로 배포**한다.
- 예측모델과 데이터는 이미 서버에 있으므로 옮기지 않고, 서비스를 데이터 옆에 둔다.
- 코드는 현업에서 쓰는 일반적인 구조로 작성한다.
- 서버 Docker 권한이 아직 없으므로 **맥에서 Docker로 먼저 개발하고, 권한을 받으면 서버로 옮긴다** (6장).

## 2. 최종 아키텍처

```
사용자 브라우저
   │  https://kci.도메인.com
   ▼
Cloudflare (HTTPS 처리)
   │  터널 (서버가 바깥으로 먼저 연결 → 학교 방화벽 개방 불필요)
   ▼
연구실 서버 ── Docker Compose ─────────────────────────────
 │
 ├─ cloudflared : 외부 요청을 nginx로 전달
 ├─ nginx       : /      → React 빌드 파일 (HTML/JS/CSS)
 │                /api/  → backend 로 전달 (리버스 프록시)
 ├─ backend     : FastAPI (gunicorn + uvicorn workers)
 │                  ├─ 예측모델 TFT (서버 폴더를 volume으로 연결)
 │                  ├─ RAG: 임베딩 → 벡터 검색 → LLM 응답
 │                  └─ db 접근
 └─ db          : PostgreSQL + pgvector (대화 기록 + 문서 임베딩)
```

화면과 API가 **같은 주소**에서 나오므로 CORS 설정이 필요 없다.

## 3. 기술 스택

| 영역 | 스택 | 역할 |
|---|---|---|
| Frontend | React + Vite + TypeScript | 화면. Vite는 개발 서버와 빌드 도구 |
| Backend | FastAPI + Pydantic | API, 모델 호출, RAG 파이프라인 |
| 운영 실행 | gunicorn + uvicorn workers | FastAPI를 여러 프로세스로 안정적으로 실행 |
| DB | PostgreSQL + pgvector | 일반 데이터 + 벡터 검색을 DB 하나로 |
| ORM | SQLAlchemy + Alembic | DB 접근, 테이블 변경 이력(마이그레이션) |
| 웹 서버 | Nginx | 정적 파일 서빙 + `/api` 리버스 프록시 |
| 컨테이너 | Docker + Docker Compose | 전체 서비스를 한 번에 실행, 재시작 관리 |
| 외부 공개 | Cloudflare Tunnel | HTTPS 공개 주소 |
| 설정 | `.env` 환경변수 | 모델 경로, API 키 등을 코드에서 분리 |
| 자동 배포 (나중) | GitHub Actions self-hosted runner | push하면 서버에서 자동 반영 |

### 확정 버전

예측모델 학습 환경(`np2-environment.yml`)을 기준으로 맞춘다. Docker 이미지와 맥 venv 모두 이 버전을 쓴다.

| 항목 | 버전 | 비고 |
|---|---|---|
| Python | 3.11 | 학습 환경 3.11.13 |
| PyTorch | 2.5.1 | 서버 `+cu121`, 맥은 CPU 빌드. 2.6부터 `torch.load` 기본값이 바뀌므로 2.5.1 고정 |
| numpy | 1.26.4 | |
| pandas | 2.1.4 | |
| scikit-learn | 1.7.2 | |
| pykalman | 확인 필요 | yml에 없음 |
| CUDA | 12.1 빌드 | 서버 드라이버 535 (CUDA 12.2 지원)와 호환 |
| Node.js | 24 LTS | 맥에 설치된 v24.13.0 기준. 프론트 Dockerfile도 `node:24` 사용 |
| PostgreSQL + pgvector | `pgvector/pgvector:pg16` | |
| Nginx | 버전 명시 (예: `nginx:1.27-alpine`) | `latest` 사용 안 함 |
| Python 패키지 | `requirements.txt`에 `==`로 고정 | |

**아직 정할 것 (RAG)**
- 임베딩 모델: 한국어 문서 → `bge-m3` 또는 `multilingual-e5` 계열 후보
- LLM: 외부 API(OpenAI, Claude 등) vs 서버 GPU로 직접 실행(vLLM, Ollama). GPU를 다른 연구원과 나눠 쓰므로 안정적인 데모가 중요하면 외부 API가 현실적
- 답변 스트리밍: SSE (FastAPI `StreamingResponse`)

## 4. 주요 결정과 이유

| 결정 | 이유 |
|---|---|
| Render 사용 안 함 | 모델·데이터가 서버에 있음. Render로 옮기면 메모리/GPU 부족, 데이터 반출 문제 |
| Vercel 사용 안 함 | 배포 지점이 두 곳이 되고 CORS·API 주소 관리가 생김. 백엔드가 어차피 서버에 있어야 함 |
| React도 서버에서 서빙 | 주소 하나, 터널 하나, 배포 한 번. CORS 불필요 |
| FastAPI 대신 Nginx가 정적 파일 서빙 | 현업 표준 구조. 정적 파일 처리와 프록시를 Nginx가 담당 |
| Docker 사용 | 환경 재현(Dockerfile = 환경 설명서), 자원 제한 가능, 재시작 자동화 |
| TypeScript 사용 | API 응답 타입을 미리 정의해 오타·필드 오류를 실행 전에 잡음. 현업 표준 |
| 맥에서 먼저 개발 | 서버 Docker 권한 대기 중. 코드 + Dockerfile을 git으로 옮겨 서버에서 다시 빌드 |

### CORS가 필요 없는 이유
- **개발할 때:** Vite proxy가 `/api` 요청을 FastAPI(`localhost:8000`)로 대신 전달한다. 브라우저는 Vite(`localhost:5173`)하고만 통신하므로 같은 출처로 인식한다.
- **배포할 때:** 같은 역할을 Nginx가 한다.
- 프론트 코드는 어느 환경에서나 `fetch("/api/...")`로 동일하게 작성한다.

```ts
// frontend/vite.config.ts (개발용)
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
})
```

```nginx
# nginx/nginx.conf (배포용)
server {
  listen 80;
  root /usr/share/nginx/html;
  location /      { try_files $uri /index.html; }
  location /api/  { proxy_pass http://backend:8000; }
}
```

## 5. 예측모델 분석 (TFT)

코드: `TFT_train_each_location.py`(학습), `TFT_TL_PLOT.py`(예측·평가). 웹에서 쓰는 모델은 전이학습(transfer learning) 버전이다.

### 무엇을 예측하나
- **남조류 세포수(Cyanocell)**를 지점별로 예측한다.
- **21개 지점, 지점마다 별도 모델** (2026-10-13 서버 `selected pickle` 폴더로 확인)

| 수계 | 모델 수 | 지점 코드 |
|---|---|---|
| 낙동강 NAK | 12 | CGB, CHB, CS, DSB, GJGR, GJGRB, GMB, HCB, HP, MGMR, NDB, SJB |
| 한강 HAN | 4 | GDDG, GJG, JSCG, MSDG |
| 영산강 YOUNG | 2 | JSB, SCB |
| 금강 GUM | 3 | BJB, GJB, SaJB |

  `TFT_TL_PLOT.py`의 지점 목록에는 한강 `HGDG`가 더 있어 22개로 보이지만, 최종 모델 폴더에는 없다. 학습용 CSV는 38개 지점분이 있으나 모델이 있는 것은 위 21개뿐이다.
- 출력: 분위수 3개 `[0.2, 0.5, 0.8]`. 0.5를 예측값으로, 0.2~0.8을 불확실성 범위로 화면에 표시할 수 있다.
- 출력은 `log10(1+x)` 값 → `10^x − 1`로 cells/mL 복원 (`inverse_log10_1p`, 상한 8로 clip)
- 단계 분류 `classify_stage` (log 값 기준)

| 단계 | log 값 | cells/mL |
|---|---|---|
| 0 | < 3 | < 1,000 |
| 1 | 3 ~ 4 | 1,000 ~ 10,000 |
| 2 | 4 ~ 5 | 10,000 ~ 100,000 |
| 3 | 5 ~ 6 | 100,000 ~ 1,000,000 |
| 4 | ≥ 6 | ≥ 1,000,000 |

### 입력
- 변수 10종: `Cyanocell`(남조류 세포수), `WT`(수온), `Chla`(클로로필-a), `TN`, `TP`, `WL`(수위), `Discharge`(유량), `Temp`(기온), `Prec`(강수), `Forecast`(예보 최고기온 기반 변수 `Forecast_Temp_*`. 미래 값을 아는 변수)
- 코드는 정확한 컬럼명이 아니라 위 이름이 **부분문자열로 들어간 컬럼**을 고른다. 지점마다 연결된 상·하류 시설명이 달라 실제 컬럼명과 개수가 다르다
- 시퀀스: 과거 2시점(`Input_Sequence=2`) + 미래 1시점(`Output_Sequence=1`) = 3시점 입력 → 마지막 시점 예측
- 고정 변수: `season` (6~10월 = 여름, 나머지 = 그 외)
- 시점 간격: **주 단위** (한 행 = 한 주). 지점별 CSV는 2023-12-25에 끝난다
- 운영 입력: **과거 2주의 관측 변수 전체 + 예측 주의 기상예보값**. 예측 주의 실측값은 `observed`로 분류돼 미래 디코더에서 빠지므로 계산에 쓰이지 않는다. `season`은 시퀀스 첫 시점의 값을 쓴다

### 전처리 순서 (웹에서 똑같이 재현해야 함)
1. 지점별 CSV 읽기 (`cp949`, `Date` 인덱스)
2. 칼만 필터로 결측치 보간 (`pykalman`, EM 10회)
3. 입력 변수 열만 선택, `season` 생성 후 LabelEncoder
4. `Cyanocell`에 `log10(1+x)`
5. **MinMaxScaler를 학습 구간(~2021-12-31) 데이터로 fit** 후 적용
6. `Cyanocell`을 맨 앞 열로 정렬

데이터 분할: 학습 ~2021, 검증 2022, 테스트 2023

### 모델 파일
- 위치: `/data/s3/uoswem926/LLM_RAG/selected pickle/{강}/{지점}_final/*.pt` (지점당 1개)
- 하이퍼파라미터는 파일 이름에 있음 (예: `dr0.1_lr0.0005_nl8_hid16_emb8_atten4_...`)
- 지점별 성능표: `selected pickle/MODEL3_LOC_final_results.csv`
- 원본 데이터: `~/LLM_RAG/DATA/{강}/`

### 서버의 자료 (`/data/s3/uoswem926/LLM_RAG/`, 인수인계서 `algae_forecast_project_overview.html` 기준일 2026-09-04)

| 폴더 | 내용 |
|---|---|
| `DATA/{수계}/{지점}_{수계소문자}_total.csv` | 모델 입력용 주 단위 CSV. CP949, `Date` 열. 38개 지점, 2023-12-25까지 |
| `selected pickle/{수계}/{지점}_final/*.pt` | 최종 모델 21개와 평가 플롯 |
| `CODE/` | `TFT_train_each_location.py`, `TFT_TL_PLOT.py` |
| `INTEGRATED_DB/` | 환경 측정망 통합 DB 6종 (약 252MB). 아래 표 |

| 통합 DB 파일 | 규모 | 기간 |
|---|---|---|
| `WAMIS_댐보자료_통합.csv` (수문) | 373,551행 · 79시설 | 2012-01-01 ~ 2025-12-31 |
| `기상자료개방포털_통합.csv` (기상 실측) | 502,297행 · 97지점 | 2012-01-01 ~ 2026-06-25 |
| `기상예보자료_통합.csv` (예보 최고기온) | 750,303행 · 213지점 | 2012-01-03 ~ 2026-04-29 |
| `수질일반측정망_통합.csv` | 489,172행 · 2,176지점 | 2012-01-01 ~ 2026-04-30 |
| `수질자동측정망_통합.csv` | 321,587행 · 80지점 | 2012-01-01 ~ 2026-04-30 |
| `조류측정망_통합.csv` (남조류 세포수 등) | 43,786행 · 40지점 | 2012-01-03 ~ 2026-06-22 |

- 통합 DB는 조회·근거 제공·최신 입력 생성용이다. 실시간이 아니라 받아 둔 파일이다.
- **통합 DB에서 지점별 주 단위 모델 입력을 만드는 코드는 없다.** 지점마다 연결 시설·측정소가 달라 별도 작업이 필요하다. 그 전까지 예측은 2023년까지의 과거 주만 가능하다.

**데이터에서 확인할 것**
- `DATA/HAN/JSCG_han_total.csv`의 강수 컬럼 이름이 `Prec_JSCG\`로 끝난다 (JSCG는 모델이 있는 지점)
- `DATA/HAN/SSANDG_han_total.csv`에 `TP_MSDG`가 들어 있다 (SSANDG는 모델 없음)
- 수문 컬럼(`WL_*`, `Discharge_*`)은 있어도 결측이거나 0으로만 채워진 지점이 있다. 추론 전에 비율 확인

### 웹에 붙일 때 문제와 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| scaler가 저장돼 있지 않음 | 예측할 때마다 학습 데이터로 다시 fit | 지점별 scaler를 한 번 만들어 파일로 저장 |
| 모델을 통째로 저장 | `torch.save(model)` → `TFT` 클래스가 `__main__`에 있어야 불러와짐 | 가중치(`state_dict`) + 설정값(json) 추출 |
| GPU 주소가 모델 안에 저장됨 | `self.device = cuda:0`을 forward에서 사용 | 추출할 때 device 정리, CPU에서도 동작하게 |
| PyTorch 버전 의존 | 2.6부터 `torch.load` 기본값 변경 | 2.5.1 고정 + 가중치 추출로 이중 대비 |
| 계절 인코딩도 저장돼 있지 않음 | LabelEncoder를 매번 새로 fit | 인코딩 값을 `config.json`에 고정 |
| 코드에 옛 서버 경로가 박혀 있음 | `/home/uoswem4/TFT_TL/...` | 변환 스크립트에서 현재 경로를 설정값으로 받음 |

→ **모델 변환 스크립트(`scripts/export_models.py`)**를 학습 환경에서 한 번 실행해 지점별로 아래 세 파일을 만든다. 웹은 이 파일만 읽는다.

```
models/{강}/{지점}/
├─ weights.pt      # state_dict
├─ config.json     # 하이퍼파라미터, col_to_idx, 변수 구성
└─ scaler.joblib   # MinMaxScaler (학습 구간 fit)
```

## 6. 개발 흐름

### 1단계: 맥에서 개발 (지금)
- 맥에서 Docker Compose로 전체 구조(React, FastAPI, DB, Nginx)를 만든다.
- 맥에는 모델·데이터가 없으므로 예측과 RAG는 **가짜 응답(mock)이나 샘플 데이터**로 먼저 연결한다.
- 모델·데이터 경로는 `.env`로 분리 (맥: 샘플, 서버: 실제 경로)
- 맥에도 venv(`backend/.venv`)와 Node를 설치한다. 실행은 Docker에서 하지만 VS Code 자동완성·타입 체크용.

### 병행: 서버에서 모델만 확인
- Docker 없이 conda 환경에서 모델 변환 스크립트를 실행하고, "입력 → 예측값" 함수가 동작하는지 확인한다. 웹은 띄우지 않으므로 공용 서버 부담이 거의 없다.

### 2단계: Docker 권한을 받으면 서버로 이전
```
로컬: 코드 + Dockerfile + docker-compose.yml → git push
서버: git pull → .env를 서버 경로로 → docker compose up -d --build
```
- Docker "환경"을 옮기는 게 아니라 코드와 Dockerfile을 옮겨 **서버에서 다시 빌드**한다.
- mock 자리에 실제 모델 함수를 끼운다.

**주의**
- 맥이 Apple Silicon(ARM)이면 맥에서 빌드한 이미지는 서버(x86)에서 안 돌 수 있음 → 서버에서 빌드
- GPU 설정은 서버용 compose 파일로 분리 (맥은 GPU 없이)
- 모델·데이터는 이미지에 넣지 않고 volume으로 연결
- `.env`, `.venv`, `node_modules`, 모델 파일, 데이터는 git에 올리지 않음 (`.gitignore`)
- 맥 Docker Desktop은 가상머신이라 메모리를 많이 씀 → 설정 > Resources에서 상한 지정

## 7. 서버 환경 (2026-10-08 확인)

| 항목 | 결과 |
|---|---|
| 계정 | `uoswem926`, 홈 실제 경로 `/data/s3/uoswem926` |
| sudo | 없음 |
| Docker | 설치돼 있으나 **권한 없음** (docker 그룹 미소속) → 관리자 요청 필요 |
| GPU | NVIDIA A40 46GB × 2. 메모리 여유 많음. GPU 0 연산 58%, GPU 1 98%(다른 연구원 사용 중) → **GPU 0 사용 권장** |
| 드라이버 / CUDA | 535.161.07 / 12.2 |
| git | 2.34.1 |
| Python (시스템) | 3.10.12 — 사용 안 함, conda 환경 사용 |
| conda | 23.7.4 |
| Node.js | 없음 → 필요하면 nvm으로 내 계정에 설치 (sudo 불필요) |

### 관리자 요청 메시지
> 안녕하세요, KCI 논문용 RAG 챗봇 웹 서비스를 연구실 서버에 Docker로 배포하려고 합니다.
> 1. `uoswem926` 계정을 docker 그룹에 추가해 주실 수 있을까요? (`sudo usermod -aG docker uoswem926`)
> 2. Docker 컨테이너에서 GPU를 쓸 수 있도록 NVIDIA Container Toolkit이 설치돼 있는지도 확인 부탁드립니다.
>
> 메모리·CPU 상한과 GPU 번호를 지정해서 다른 분들 작업에 영향이 없도록 운영하겠습니다.

권한을 받은 뒤 확인:
```bash
docker ps
docker run --rm --gpus device=0 nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## 8. 공용 서버 운영 수칙

- 리눅스에서 Docker 자체의 메모리 부담은 작다(수십 MB). 메모리를 쓰는 건 안에서 도는 모델이다.
- 다른 사람과 같이 쓰는 서버이므로 상한을 건다.

```yaml
# docker-compose.yml 일부
services:
  backend:
    mem_limit: 8g
    cpus: 4
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]     # 사용할 GPU 번호만
              capabilities: [gpu]
```

- 다른 사람이 쓰는 GPU와 겹치지 않게 번호 지정
- 오래된 이미지는 **이 프로젝트 것만** 정리한다. Dockerfile에 `LABEL project=algae-rag`를 넣고 `docker image prune --filter "label=project=algae-rag"`로 지운다
- `docker system prune`, `docker builder prune`은 쓰지 않는다. Docker 데몬을 서버 전체가 같이 쓰므로 다른 연구원의 중지된 컨테이너·미사용 네트워크·dangling 이미지·빌드 캐시까지 지운다
- 연구실에 "이 서비스를 이 GPU로 상시 실행 중" 공유
- 외부 공개 전 교수님/관리자에게 확인
- 홈 폴더 권한 확인 (`ls -ld ~`), 필요하면 `chmod 700 ~` — `.env`(API 키), SSH 키 보호
- GitHub 인증은 SSH 키로. `credential.helper store`(토큰 평문 저장) 사용 안 함

## 9. 배포 원리: 왜 연구실 서버만으로 되는가

웹사이트 = **어떤 컴퓨터에서 실행 중인 프로그램**. `uvicorn`을 띄우는 순간 웹 서버는 이미 동작한다. 배포란 아래 세 문제를 푸는 것이다.

| 문제 | Render가 해 주던 방식 | 연구실 서버에서 |
|---|---|---|
| 남들이 어떻게 접속하나 | `xxx.onrender.com` 자동 제공 | Cloudflare Tunnel |
| 프로그램이 계속 켜져 있으려면 | 자동 재시작 | Docker `restart: always` |
| 코드 수정을 어떻게 반영하나 | push 감지 후 자동 배포 | 수동 `git pull` → 나중에 self-hosted runner |

처음에 한 번만 설정하면 이후에는 "수정 → push → 반영" 흐름이 된다.

## 10. 폴더 구조 (backend/app, frontend까지 생성됨)

```
Algae_RAG_Platform/
├─ frontend/              # React + Vite + TypeScript
│  ├─ src/
│  ├─ vite.config.ts
│  ├─ package.json
│  └─ Dockerfile          # 빌드 → nginx 이미지에 결과물 복사
├─ backend/               # FastAPI
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ routers/         # 라우터 (/api/health, /api/chat, /api/predict ...)
│  │  ├─ core/            # 설정(.env 로딩)
│  │  ├─ db/              # SQLAlchemy 모델, 세션
│  │  ├─ rag/             # 임베딩, 검색, LLM 호출
│  │  └─ ml/              # TFT 클래스, 전처리, 예측 (mock ↔ 실제 교체)
│  ├─ alembic/
│  ├─ requirements.txt      # 실행용 (Docker 이미지에 설치)
│  ├─ requirements-dev.txt  # 개발용 (ruff, pytest, httpx)
│  └─ Dockerfile
├─ scripts/
│  └─ export_models.py    # .pt → weights + config + scaler 변환 (서버에서 1회)
├─ nginx/
│  └─ nginx.conf
├─ docker-compose.yml     # 공통
├─ docker-compose.gpu.yml # 서버 전용 (GPU 설정)
├─ .env.example
├─ .gitignore
└─ README.md
```

## 11. 진행 단계

- [x] **0. 서버 환경 확인** (7장) — Docker 권한 없음, GPU 확인 완료
- [ ] 0-1. 관리자에게 Docker 권한 요청
- [x] 0-2. 맥 환경 확인 — Docker 29.1.3, Node v24.13.0, Python 3.11.4(`python3.11`), arm64. `python3`는 anaconda 3.12라 venv는 `python3.11`로 생성
- [x] 1. GitHub 저장소 생성, 기본 폴더와 `.gitignore` — `Dovcl/Algae_RAG_Platform` (Private)
- [x] 2. Frontend: Vite로 React + TypeScript 프로젝트 생성, proxy 설정
- [x] 3. Backend: FastAPI 기본 앱, `/api/health` 엔드포인트, venv
- [x] 4. 프론트에서 `/api/health` 호출해 연결 확인
- [x] 5. Docker Compose에 db(PostgreSQL + pgvector) 추가, SQLAlchemy + Alembic 연결 — `chat_logs` 테이블, `/api/health/db`. pgvector extension은 RAG 단계에서
- [x] 6. 예측 API를 mock으로 연결 (`/api/predict`, 21개 지점 목록, 분위수·단계 응답 형태) — 요청은 지점 + 선택적 `target_date`, 테스트 17개
- [ ] 6-1. (서버) 모델 변환 스크립트 실행 → weights / config / scaler
- [ ] 6-2. mock을 실제 TFT 예측으로 교체
- [ ] 7. RAG 파이프라인 (문서 적재 → 임베딩 → 검색 → LLM 응답, 스트리밍)
- [ ] 8. 대화 로그 테이블 (질문, 답변, 검색 문서, 응답 시간) — 논문 실험·평가용
- [ ] 9. Nginx + Docker Compose로 서버에서 전체 실행, 자원 제한 설정
- [ ] 10. Cloudflare Tunnel로 외부 공개
- [ ] 11. (선택) GitHub Actions self-hosted runner로 자동 배포

## 12. 미결정 사항

- [ ] 서버 Docker 권한 (관리자 응답 대기)
- [ ] 컨테이너에서 GPU 사용 가능 여부 (NVIDIA Container Toolkit)
- [ ] pykalman 버전
- [x] 데이터 시점 간격 → 주 단위. 화면은 "다음 주 예측"으로 표현
- [x] `Forecast` 변수의 의미 → 예보 최고기온 (`Forecast_Temp_*`)
- [x] 모델 수 → 21개 (한강 4)
- [ ] 예측 입력 데이터: 우선 기존 지점별 CSV(2023년까지)로 과거 주 조회. 통합 DB에서 최신 입력을 만드는 변환을 할지, 한다면 누가 할지
- [ ] 통합 DB 6종을 PostgreSQL에 넣을지, CSV를 그대로 읽을지 (팀 과업 3의 수질·기상·녹조 조회)
- [ ] JSCG 강수 컬럼 이름(`Prec_JSCG\`) 등 데이터 QA 항목을 연구실에 확인
- [ ] LLM: 외부 API vs 서버에서 직접 실행
- [ ] 임베딩 모델 선택
- [ ] 도메인 구입 여부 (테스트는 임시 터널 주소로 가능)
- [ ] 외부 공개 범위와 연구실 승인
