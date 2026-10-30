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
    predict_backend: Literal["mock", "tft"] = "mock"
    llm_provider: Literal["mock", "ollama"] = "mock"

    # 맥: 컨테이너에서 호스트의 Ollama로. 서버에서는 Ollama 컨테이너 주소
    ollama_base_url: str = "http://host.docker.internal:11434"
    embed_model: str = "bge-m3"
    # 검색할 컬렉션. 비어 있으면 문서 검색을 하지 않는다
    rag_collection: str | None = None

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
