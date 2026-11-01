from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "db"
    postgres_port: int = 5432
    predict_backend: Literal["mock", "precomputed", "tft"] = "mock"
    llm_provider: Literal["mock", "ollama"] = "mock"
    # 연구실 관측 CSV와 저장 예측이 있는 곳. PREDICT_BACKEND=precomputed일 때만 읽는다
    lab_data_dir: Path = Path("/data/lab")

    # 맥: 컨테이너에서 호스트의 Ollama로. 서버에서는 Ollama 컨테이너 주소
    ollama_base_url: str = "http://host.docker.internal:11434"
    embed_model: str = "bge-m3"
    # 검색할 컬렉션. 비어 있으면 문서 검색을 하지 않는다
    rag_collection: str | None = None
    rag_top_k: int = 6
    llm_model: str = "exaone3.5:7.8b"
    llm_num_ctx: int = 8192
    llm_num_predict: int = 1024  # 같은 항목을 끝없이 반복해도 제한 시간 전에 끊는다
    llm_timeout: float = 300

    @property
    def database_url(self) -> URL:
        return URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


settings = Settings()
