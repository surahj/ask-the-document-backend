"""
Configuration settings for DocuMind AI Assistant with PostgreSQL pgvector support
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Application
    app_name: str = "DocuMind AI Assistant"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database - PostgreSQL with pgvector
    database_url: str = "postgresql://postgres:postgres@localhost:5432/documind"
    # Use PostgreSQL with pgvector
    use_postgresql: bool = True

    # File upload
    upload_dir: str = "./uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB in bytes
    allowed_extensions: list = [".pdf", ".docx", ".txt", ".md"]

    # Cloudinary Configuration
    cloudinary_cloud_name: Optional[str] = None
    cloudinary_api_key: Optional[str] = None
    cloudinary_api_secret: Optional[str] = None
    cloudinary_folder: str = "documind"  # Folder in Cloudinary
    use_cloudinary: bool = True  # Toggle between local and cloud storage

    # AI Services
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"  # Keep original model
    )
    huggingface_api_key: Optional[str] = None

    # Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_document: int = 100

    # Vector Search
    top_k_results: int = 5
    similarity_threshold: float = 0.1  # Lower threshold for testing
    embedding_dimension: int = 384  # For all-MiniLM-L6-v2
    vector_index_lists: int = 100  # For IVFFlat index

    # Performance
    max_concurrent_uploads: int = 5
    max_concurrent_queries: int = 10
    cache_ttl: int = 3600  # 1 hour

    # Security
    secret_key: str = "your-secret-key-here"
    access_token_expire_minutes: int = 60 * 24 * 30  # 30 days

    # Email Provider
    email_provider: str = "gmail"
    gmail_sender_email: Optional[str] = None
    gmail_app_password: Optional[str] = None

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Only set PostgreSQL database URL
        if not self.database_url:
            self.database_url = "postgresql://user:password@localhost/documind"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
