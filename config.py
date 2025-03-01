import os
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache

class Settings(BaseSettings):
    user_agent: str = os.getenv("USER_AGENT", "promtior-bot")
    model_name: str = "llama2"
    chunk_size: int = 1500
    chunk_overlap: int = 100
    max_words_response: int = 200
    max_context_length: int = 5000
    port: int = 11435  
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    ##ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    preload_model: bool = False
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    use_openai: bool = os.getenv("USE_OPENAI", "").lower() in ["true", "1", "yes"]
    urls: List[str] = [
        "https://www.promtior.ai",
        "https://www.promtior.ai/service",
        "https://www.promtior.ai/use-cases",
    ]
    pdf_path: str = "doc/AI Engineer.pdf"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()