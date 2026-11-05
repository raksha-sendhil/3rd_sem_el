"""Runtime configuration via pydantic-settings.

Values come from, in order of precedence: explicit keyword arguments, the
process environment, a ``.env`` file in the working directory, then the
defaults below. Every environment variable is prefixed with ``SANITISE_``
(see ``.env.example``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SANITISE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Domain: which prompts, data and models the pipeline uses (sanitise.domain). The
    # default for requests that do not choose one; profiles live in domains_file.
    domain: str = "legal"
    domains_file: str = "config/domains.json"

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
    # Ollama's "think" option for models with a thinking mode. Gemma 4 can spend its whole token
    # allowance thinking and return no JSON, so the legal profile sets false. None = not sent.
    llm_think: bool | None = None

    # ---- Legal build spec, "Model choice": Gemini answers from cleaned text only; Gemma 4
    # (Ollama, or transformers with the LoRA adapter for Agent 1) sees raw text.
    # answer_backend "same" answers with the agents' model (the medical set-up).
    answer_backend: Literal["same", "gemini", "ollama", "mock"] = "same"
    gemini_model: str = "gemini-3.5-flash"  # or "gemini-3.8-flash": one-word change
    # Keys are rotated: when one hits its rate limit or quota, or uses up its daily token
    # budget, the next one is used. Read from here, then GEMINI_API_KEYS (comma-separated),
    # GEMINI_API_KEY, GEMINI_API_KEY_2 .. GEMINI_API_KEY_9 (environment or Kaggle secrets).
    gemini_api_key: str | None = None
    gemini_api_keys: str | None = None  # comma-separated
    gemini_key_cooldown_s: float = Field(default=60.0, ge=0)  # after a per-minute limit
    gemini_daily_token_budget: int = Field(default=0, ge=0)  # per key; 0 = no budget
    gemini_cache_dir: str = "data/cache/gemini"  # every answer cached on disk (rate limits)
    gemini_rpm: int = Field(default=10, ge=1)  # calls per minute
    # Agent 1's LLM pass: "same" uses the agents' model; "transformers" loads the base Gemma
    # model plus the fine-tuned LoRA adapter (Part 3) on its own GPU.
    agent1_backend: Literal["same", "transformers"] = "same"
    agent1_base_model: str = "google/gemma-4-E4B-it"
    agent1_adapter: str | None = None  # e.g. /kaggle/working/agent1_lora
    agent1_device: str = "cuda:1"

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
    # None = the domain's default: top 4 for legal (legal build spec), top 3 for medical.
    retrieval_k: int | None = Field(default=None, ge=1, le=20)
    # "hybrid" = BM25 + bge-m3 dense vectors (needs `pip install -e .[retrieval]`, falls back
    # to BM25 with a warning when the packages are missing); "bm25" = keyword search only.
    retrieval_mode: Literal["hybrid", "bm25"] = "hybrid"
    index_dir: str = "data/index"
    embedding_model: str = "BAAI/bge-m3"
    # ---- Phase 3b: session privacy monitor (legal build spec, step 12). "candidate_pool"
    # blocks an answer that would leave fewer than monitor_min_pool clients in the client
    # table matching what the session has disclosed; it falls back to "off" while the table
    # does not exist yet.
    monitor: Literal["off", "candidate_pool"] = "candidate_pool"
    monitor_min_pool: int = Field(default=5, ge=1)
    clients_path: str = "data/legal/clients.json"
    # A dashboard bundle exported from a Kaggle run (sanitise.legal.export). When set, the API
    # replays the recorded answers and attack runs instead of calling a model.
    replay_dir: str | None = None

    # ---- HTTP API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Origins allowed to call the API from a browser (the Phase 3 Next.js dashboard).
    api_cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _domain_defaults(self) -> Settings:
        """Paths left at their medical defaults point at the legal data in the legal domain."""
        if self.retrieval_k is None:
            self.retrieval_k = 4 if self.domain == "legal" else 3
        if self.domain == "legal":
            for name, legal in _LEGAL_PATHS.items():
                if getattr(self, name) == type(self).model_fields[name].default:
                    setattr(self, name, legal)
        return self


_LEGAL_PATHS = {
    "store_path": "data/legal/store.sqlite",
    "attacks_dir": "data/legal/attacks",
    "people_path": "data/legal/clients.json",
    "index_dir": "data/legal/index",
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once. Tests build ``Settings(...)`` directly instead."""
    return Settings()
