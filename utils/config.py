"""
Configuration management for RAG chatbot.
Loads and validates environment variables.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


load_dotenv()


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    # Azure OpenAI
    azure_openai_endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(..., alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment_name: str = Field(..., alias="AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_api_version: str = Field(
        default="2024-08-01-preview",
        alias="AZURE_OPENAI_API_VERSION"
    )
    
    # Groq
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model_name: str = Field(
        default="llama-3.3-70b-versatile",
        alias="GROQ_MODEL_NAME"
    )
    
    # Elasticsearch
    elasticsearch_url: str = Field(
        default="http://localhost:9200",
        alias="ELASTICSEARCH_URL"
    )
    elasticsearch_index_name: str = Field(..., alias="ELASTICSEARCH_INDEX_NAME")
    # Optional Elasticsearch auth from legacy .env keys
    es_user: Optional[str] = Field(default=None, alias="ES_USER")
    es_password: Optional[str] = Field(default=None, alias="ES_PASSWORD")
    
    # LangSmith
    langchain_tracing_v2: str = Field(default="false", alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: Optional[str] = Field(
        default="rag-chatbot",
        alias="LANGCHAIN_PROJECT"
    )
    
    # RAG Parameters
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    confidence_threshold: float = Field(default=0.35, alias="CONFIDENCE_THRESHOLD")  # Changed from 0.65
    max_query_words: int = Field(default=2000, alias="MAX_QUERY_WORDS")
    
    # Embedding Model
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME"
    )
    
    @field_validator("langchain_tracing_v2")
    @classmethod
    def validate_tracing(cls, v: str, info) -> str:
        """Validate LangSmith tracing configuration."""
        if v.lower() == "true":
            api_key = os.getenv("LANGCHAIN_API_KEY")
            if not api_key:
                raise ValueError(
                    "LANGCHAIN_TRACING_V2 is enabled but LANGCHAIN_API_KEY is missing. "
                    "Either set LANGCHAIN_API_KEY or disable tracing."
                )
        return v
    
    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Ensure confidence threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
        if v > 0.5:
            logger.warning(f"CONFIDENCE_THRESHOLD={v} is very high. Typical range is 0.2-0.4")
        return v
    
    @field_validator("retrieval_top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """Ensure retrieval_top_k is positive."""
        if v <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be positive")
        return v
    
    @field_validator("max_query_words")
    @classmethod
    def validate_max_query_words(cls, v: int) -> int:
        """Ensure max_query_words is positive."""
        if v <= 0:
            raise ValueError("MAX_QUERY_WORDS must be positive")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        # Allow extra environment variables so loading doesn't fail
        # when unrelated keys are present in the environment (e.g. from
        # Streamlit or legacy .env files). We will normalize important
        # legacy names before creating the settings instance.
        extra = "ignore"


def get_config() -> Config:
    """
    Load and validate configuration.
    Fails fast if required variables are missing.
    """
    # Normalize a few legacy env var names to the canonical aliases expected
    # by this Config class. This avoids validation errors when older
    # .env files or other parts of the app use different names.
    if not os.getenv("ELASTICSEARCH_INDEX_NAME"):
        # Common legacy/alternate name used in this project
        alt_index = os.getenv("index_name") or os.getenv("INDEX_NAME")
        if alt_index:
            os.environ["ELASTICSEARCH_INDEX_NAME"] = alt_index

    # Map ES_HOST (used in top-level config.py) into ELASTICSEARCH_URL so
    # both configs can read the same setting regardless of which variable
    # the .env file contains.
    if not os.getenv("ELASTICSEARCH_URL") and os.getenv("ES_HOST"):
        os.environ["ELASTICSEARCH_URL"] = os.getenv("ES_HOST")

    try:
        config = Config()
        return config
    except Exception as e:
        # Preserve original exception message for easier debugging
        raise RuntimeError(f"Configuration validation failed: {e}")


# Global config instance
config = get_config()