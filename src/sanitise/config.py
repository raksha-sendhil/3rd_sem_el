"""Runtime configuration via pydantic-settings.

Values come from, in order of precedence: explicit keyword arguments, the
process environment, a ``.env`` file in the working directory, then the
defaults below. Every environment variable is prefixed with ``SANITISE_``
(see ``.env.example``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SANITISE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM backend
    llm_backend: Literal["ollama", "mock"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    model: str = "llama3:8b"
    fallback_model: str = "qwen2.5:7b-instruct"
    llm_timeout_s: float = Field(default=120.0, gt=0)
    llm_num_ctx: int = Field(default=4096, ge=512)
    # Fixed sampling seed so repeated runs (and the metrics report) are reproducible.
    # None lets Ollama pick a random seed.
    llm_seed: int | None = 42
    # Upper bound on generated tokens per call, so a runaway generation cannot stall a run.
    llm_num_predict: int = Field(default=1024, ge=64)

    # ---- Validation retry loop: "retries up to three times with an escalating temperature"
    llm_retries: int = Field(default=3, ge=0, le=10)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_temperature_step: float = Field(default=0.2, ge=0.0, le=1.0)
    llm_temperature_max: float = Field(default=1.0, ge=0.0, le=2.0)
    # Extra attempts on a refused connection only (Ollama still starting up).
    # Timeouts are not retried.
    transport_retries: int = Field(default=1, ge=0, le=5)

    # ---- Document load
    max_doc_chars: int = Field(default=2000, ge=1)
    # ChatDoctor records hold the patient's question in "input" and the doctor's reply in
    # "output". By default only the patient text is used as the document.
    chatdoctor_include_answer: bool = False

    # ---- Phase 3a: storage, retrieval and attacks (docs/phase3_plan.md). Paths are relative
    # to the working directory; mock mode ignores them and uses the repository's data/sample/.
    store_path: str = "data/store.sqlite"
    attacks_dir: str = "data/attacks"
    people_path: str = "data/corpus/people.json"
    retrieval_k: int = Field(default=4, ge=1, le=20)
    # "hybrid" = BM25 + bge-m3 dense vectors (needs `pip install -e .[retrieval]`, falls back
    # to BM25 with a warning when the packages are missing); "bm25" = keyword search only.
    retrieval_mode: Literal["hybrid", "bm25"] = "hybrid"
    index_dir: str = "data/index"
    embedding_model: str = "BAAI/bge-m3"

    # ---- HTTP API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Origins allowed to call the API from a browser (the Phase 3 Next.js dashboard).
    api_cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once. Tests build ``Settings(...)`` directly instead."""
    return Settings()
