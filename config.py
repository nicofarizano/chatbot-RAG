from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache

class Settings(BaseSettings):
    user_agent: str = "promtior-bot"
    model_name: str = "llama2"
    chunk_size: int = 1500
    chunk_overlap: int = 100
    max_words_response: int = 200
    max_context_length: int = 5000
    port: int = 8000
    urls: List[str] = [
        "https://www.promtior.ai",
        "https://www.promtior.ai/service",
        "https://www.promtior.ai/use-cases",
    ]
    pdf_path: str = "AI Engineer.pdf"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()