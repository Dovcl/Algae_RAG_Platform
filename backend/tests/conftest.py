import os

import pytest

# 테스트는 실제 .env에 기대지 않는다. app을 import하기 전에 값을 정해 둔다.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ["PREDICT_BACKEND"] = "mock"
os.environ["LLM_PROVIDER"] = "mock"


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
