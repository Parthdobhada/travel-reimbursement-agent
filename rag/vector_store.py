"""
Vector Store Manager

Creates and manages the persistent Chroma vector database.
"""

import logging
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from rag.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Handles creation and loading of the Chroma vector database.
    """

    def __init__(self):
        self.persist_directory = str(settings.CHROMA_DB_PATH)

        self.embedding_model = (
            EmbeddingManager()
            .get_model()
        )

    def create_vector_store(
        self,
        documents: List[Document],
    ) -> Chroma:
        """
        Create and persist a new Chroma vector database.
        """

        logger.info("Creating Chroma Vector Database...")

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
        )

        logger.info(
            "Vector database created successfully."
        )

        return vector_store

    def load_vector_store(self) -> Chroma:
        """
        Load an existing Chroma vector database.
        """

        logger.info("Loading existing Chroma database...")

        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
        )

    def similarity_search(
        self,
        query: str,
        k: int = None,
    ) -> List[Document]:
        """
        Search similar policy chunks.
        """

        if k is None:
            k = settings.TOP_K_RESULTS

        vector_store = self.load_vector_store()

        return vector_store.similarity_search(
            query=query,
            k=k,
        )

    def delete_database(self):
        """
        Delete the Chroma database.
        """

        db_path = Path(self.persist_directory)

        if db_path.exists():

            shutil.rmtree(db_path)

            logger.info(
                "Existing vector database deleted."
            )

    def database_exists(self) -> bool:
        """
        Check whether the database exists.
        """

        return Path(self.persist_directory).exists()
        