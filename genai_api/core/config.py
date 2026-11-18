from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Embeddings / chunking (specific to the Qwen3-Embedding-4B model served by LiteLLM).
# If you change the embedding model, existing Milvus collections must be re-created.
EMBEDDINGS_MODEL = "AI4EOSC/Qwen/Qwen3-Embedding-4B"
EMBEDDINGS_SIZE = 2560
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 256


class Settings(BaseSettings):
    database_url: str
    keycloak_issuer: str
    keycloak_audience: str = "genai-api"
    litellm_base_url: str
    litellm_api_key: str
    papi_base_url: str
    genai_key_name: str
    cors_origins: list[str] = [
        "https://genai.cloud.ai4eosc.eu",
        "https://genai.dev.ai4eosc.eu",
        "http://localhost:5173",
    ]

    # 15 MB. If this changes, update MAX_ATTACHMENTS_BYTES in the UI (lib/constants.ts).
    max_images_total_bytes: int = 15_000_000

    milvus_uri: str = "https://milvus.k8s.cloud.ai4eosc.eu"
    milvus_user: str = "root"
    milvus_pwd: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("cors_origins")
    @classmethod
    def _strip_trailing_slash(cls, origins: list[str]) -> list[str]:
        return [o.rstrip("/") for o in origins]


settings = Settings()
