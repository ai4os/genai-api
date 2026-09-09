from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    keycloak_issuer: str
    litellm_base_url: str
    litellm_api_key: str
    papi_base_url: str
    max_images_total_bytes: int = 15_000_000  # 15 MB - if this variable changes, update the corresponding UI variable.

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
