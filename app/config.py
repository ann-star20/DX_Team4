"""
애플리케이션 설정 (환경변수 → Pydantic Settings).

.env에는 감정 모델 학습/추론용 변수(pretrained_model 등)가 같이 있을 수 있으므로
Settings에 없는 키는 무시한다.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gcp_project_id: str
    gcp_location: str = "asia-northeast3"
    google_application_credentials: str = ""

    supabase_db_url: str
    supabase_url: str = ""
    supabase_key: str = ""
    cron_session_check: str = "interval:minutes=5"
    cron_midnight_reset: str = "cron:hour=0,minute=0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
