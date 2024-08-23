import os
from pathlib import Path

import yaml
from azure.identity import get_bearer_token_provider, DefaultAzureCredential
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from graphrag.config import create_graphrag_config, GraphRagConfig, LLMParameters, TextEmbeddingConfig, LocalSearchConfig, GlobalSearchConfig, LLMType
from graphrag.query.llm.oai import OpenaiApiType


class Settings(BaseSettings):
    server_port: int = 20213
    website_address: str = f"http://127.0.0.1:{server_port}"
    cors_allowed_origins: list = ["*"] # Edit the list to restrict access.
    data: str = (
        "./output"
    )
    lancedb_uri: str = (
        "./lancedb"
    )
    llm: LLMParameters
    embeddings: TextEmbeddingConfig
    global_search: GlobalSearchConfig
    local_search: LocalSearchConfig
    encoding_model: str = "o200k_base"

    def is_azure_client(self):
        return self.llm.type == LLMType.AzureOpenAIChat or settings.llm.type == LLMType.AzureOpenAI

    def get_api_type(self):
        return OpenaiApiType.AzureOpenAI if self.is_azure_client() else OpenaiApiType.OpenAI

    def azure_ad_token_provider(self):
        if self.llm.cognitive_services_endpoint is None:
            cognitive_services_endpoint = "https://cognitiveservices.azure.com/.default"
        else:
            cognitive_services_endpoint = settings.llm.cognitive_services_endpoint
        if self.is_azure_client() and not settings.llm.api_key:
            return get_bearer_token_provider(DefaultAzureCredential(), cognitive_services_endpoint)
        else:
            return None


def load_settings_from_yaml(file_path: str) -> Settings:
    config = file_path
    _root = Path(config)
    settings_yaml = (
        Path(config)
        if config and Path(config).suffix in [".yaml", ".yml"]
        else _root / "settings.yaml"
    )

    with settings_yaml.open("rb") as file:
        data = yaml.safe_load(file.read().decode(encoding="utf-8", errors="strict"))
        parameters: GraphRagConfig = create_graphrag_config(data, "./")

    return Settings(
        llm=parameters.llm,
        embeddings=parameters.embeddings,
        global_search=parameters.global_search,
        local_search=parameters.local_search,
        encoding_model=parameters.encoding_model
    )


settings = load_settings_from_yaml("settings.yaml")
