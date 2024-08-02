from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    server_host: str = "http://localhost"
    server_port: int = 20213
    data: str = "./output"
    lancedb_uri: str = (
        "./lancedb"
    )
    api_key: str = ""
    api_base: str = ""
    api_version: str = ""
    api_type: str = ""
    llm_model: str = ""
    max_retries: int = 3
    embedding_model: str = ""
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
