"""
Application Configuration

This module contains all configurable settings used throughout the
Enterprise AI Travel Reimbursement Approval Agent.
"""

from pathlib import Path
from dataclasses import dataclass
import os

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application configuration."""

    # ==========================
    # Project Paths
    # ==========================
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    POLICY_PATH: Path = BASE_DIR / "data" / "policies" / "travel_policy.md"

    CHROMA_DB_PATH: Path = BASE_DIR / "chroma_db"

    LOGS_PATH: Path = BASE_DIR / "logs"

    OUTPUTS_PATH: Path = BASE_DIR / "outputs"

    # ==========================
    # API Configuration
    # ==========================
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    LLM_MODEL: str = "gemini-2.5-flash"

    # ==========================
    # Embedding Model
    # ==========================
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # ==========================
    # Text Splitting
    # ==========================
    CHUNK_SIZE: int = 1000

    CHUNK_OVERLAP: int = 200

    # ==========================
    # Retriever Configuration
    # ==========================
    TOP_K_RESULTS: int = 5


# Singleton configuration object
settings = Settings()