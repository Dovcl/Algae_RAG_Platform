# AGENTS.md — Algae RAG Platform

TFT 남조류 예측모델과 RAG 챗봇을 연구실 서버 한 대에서 Docker Compose로 배포하는 웹 서비스.
설계는 `DEV_PLAN.md`, 현재 진행 상황은 `HANDOFF.md`. 작업 전에 HANDOFF 5장을 먼저 읽는다.

## 일하는 방식
- 한국어로 답한다. 코드를 제안하기 전에 이유를 먼저 설명한다.
- 사용자가 명령어를 직접 입력하며 배운다. 파일·폴더를 대신 만들지 말고 순서와 명령어를 알려준다. 명시적으로 부탁받았을 때만 직접 만든다.
- 커밋은 사용자가 한다. Conventional Commits(`feat:`, `fix:`, `chore:`, `docs:`), `Co-Authored-By` 줄 없이.
- 지금 필요한 것만 만든다. 기존 코드는 최소 범위로 고친다.
- 파일을 새로 만들거나 크게 바꾸는 단계가 끝나면 `STUDY_LOG.md` 맨 아래에 순서대로 추가한다 (파일별 용도 → 없으면 생기는 문제 → 핵심 줄의 이유). 이 파일은 개인 학습 기록이라 git에 올리지 않으며, AI가 직접 써도 된다.

## 구조
- `backend/app/` FastAPI. 라우터는 `app/routers/`, `/api` 접두사는 `routers/__init__.py`에서 한 번만 붙인다.
- `frontend/` React + Vite + TypeScript. API 호출은 항상 `fetch("/api/...")` 상대 경로. CORS 설정을 추가하지 않는다.

## 명령어
- 백엔드 실행: 프로젝트 루트에서 `docker compose up` (backend + db, 코드 수정 시 자동 재시작). venv에서 `uvicorn`을 직접 띄우면 DB에 붙지 못한다.
- DB 마이그레이션: `docker compose exec backend alembic upgrade head`
- 백엔드 검사: `cd backend && source .venv/bin/activate && ruff check . && ruff format --check . && pytest` (테스트가 아직 없으면 `pytest`는 생략)
- 프론트 실행: `cd frontend && npm run dev`
- 프론트 검사: `npm run lint && npm run build`

## 버전 (임의로 올리지 않는다)
- Python 3.11 (venv는 반드시 `python3.11`로. 맥의 `python3`는 anaconda 3.12)
- torch 2.5.1, numpy 1.26.4(2.x 금지), pandas 2.1.4(3.x 금지), scikit-learn 1.7.2
- Python 패키지(`requirements*.txt`)는 `==`로 고정한다.
- 프론트 패키지는 `package-lock.json`으로 고정한다. lock 파일을 커밋하고, 지우거나 새로 만들지 않는다.
- 새 패키지(Python·npm 모두)는 먼저 묻고 추가한다.

## 하지 말 것
- `.env`, 모델 파일(`*.pt`, `*.joblib`), 데이터, 연구실 코드(`TFT_*.py`, `관련파일/`)를 커밋하지 않는다.
- 서버의 원본 모델·데이터를 수정하지 않는다 (읽기 전용).
- 서버 GPU 1을 쓰지 않는다 (다른 연구원 사용 중). GPU 0만.
- 예측 전처리 순서와 단계 분류 기준(`DEV_PLAN.md` 5장)을 바꾸지 않는다.
- Docker 이미지 태그에 `latest`를 쓰지 않는다.
- 서버에서 `docker system prune`, `docker builder prune`을 쓰지 않는다 (같은 Docker 데몬을 쓰는 다른 연구원의 자원까지 지운다). 정리는 `DEV_PLAN.md` 8장 방식으로.

## 작업별로 읽을 문서
| 작업 | 문서 |
|---|---|
| 예측 API, 모델 변환 | `DEV_PLAN.md` 5장 |
| Docker, 서버 이전 | `DEV_PLAN.md` 6~8장, `HANDOFF.md` 6장 |
| 결정을 바꾸려 할 때 | `DEV_PLAN.md` 4장 (이유 확인 후 사용자에게 묻기) |

## 완료 기준
- 위 검사 명령이 통과한다. 실행하지 못했으면 못 했다고 말한다.
- 결정이 바뀌거나 단계가 끝나면 `HANDOFF.md` 5장을 갱신하자고 제안한다.
