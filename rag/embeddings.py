"""
Embedding Manager

Loads and manages the embedding model used throughout the RAG pipeline.

Features:
- Singleton pattern
- Lazy loading
- Health check
- Centralized embedding model management
"""

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Singleton manager for HuggingFace embedding models.
    """

    _instance = None
    _embedding_model: Optional[HuggingFaceEmbeddings] = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating EmbeddingManager instance...")
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self) -> HuggingFaceEmbeddings:
        """
        Returns the embedding model.
        Loads it only once.
        """

        if self._embedding_model is None:

            logger.info(
                f"Loading embedding model: {settings.EMBEDDING_MODEL}"
            )

            self._embedding_model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            logger.info("Embedding model loaded successfully.")

        return self._embedding_model

    def health_check(self) -> bool:
        """
        Verify that the embedding model can generate embeddings.
        """

        try:
            model = self.get_model()

            vector = model.embed_query("Health Check")

            return len(vector) > 0

        except Exception as e:
            logger.error(f"Embedding health check failed: {e}")

            return False

    def embedding_dimension(self) -> int:
        """
        Returns embedding vector dimension.
        """

        model = self.get_model()

        vector = model.embed_query("Dimension Test")

        return len(vector)