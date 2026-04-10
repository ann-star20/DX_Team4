"""
애플리케이션 설정 (환경변수 → Pydantic Settings).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gcp_project_id: str
    gcp_location: str = "asia-northeast3"
    google_application_credentials: str = ""

    supabase_db_url: str
    supabase_url: str = ""
    supabase_key: str = ""

    # Shadow pipeline / model settings
    pretrained_model: str = "beomi/kcelectra-base-v2022"
    text_model_checkpoint: str = "models/text/best_kcelectra_lora.pt"
    audio_model_checkpoint: str = "models/audio/best_hubert_lora.pt"
    soft_voting_csv_path: str = "models/soft_voting.csv"
    neutral_margin_default: float = 0.08
    w_text_default: float = 0.55
    w_audio_default: float = 0.45
    voice_emotion_cache_ttl_seconds: int = 600
    # 섀도우 정산 시 음성 감정 캐시를 기다리는 최대 시간(초). HuBERT·ffmpeg가 느리면 여유 있게.
    voice_emotion_shadow_wait_seconds: float = 60.0

    cron_midnight_reset: str = "cron:hour=0,minute=0"
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
    

@lru_cache
def get_settings() -> Settings:
    return Settings()


# request_id -> {"voice_emotion": dict, "stored_at": float} — Settings 밖에서 공유(가변 캐시)
VOICE_EMOTION_CACHE: dict[str, dict] = {}
