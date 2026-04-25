from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    OPENAI_API_KEY: str | None = None

    POSTGRES_USER: str  | None= "postgres"
    POSTGRES_PASSWORD: str | None = "postgres"
    POSTGRES_HOST: str | None = "localhost"
    POSTGRES_PORT: int | None = 5432
    POSTGRES_DB: str | None = "memory_db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()