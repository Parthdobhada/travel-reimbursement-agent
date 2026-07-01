"""
Document Loader

Loads the enterprise travel policy from disk and converts it into
LangChain Document objects for downstream RAG processing.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from config.settings import settings

# Configure logger
logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Responsible for loading the travel policy document.
    """

    def __init__(self):
        self.policy_path: Path = settings.POLICY_PATH

    def load_policy(self) -> list[Document]:
        """
        Load the travel policy markdown document.

        Returns:
            list[Document]: Loaded LangChain documents.

        Raises:
            FileNotFoundError:
                If the policy document does not exist.
        """

        logger.info("Loading travel policy...")

        if not self.policy_path.exists():
            logger.error(f"Policy file not found: {self.policy_path}")
            raise FileNotFoundError(
                f"Policy document not found:\n{self.policy_path}"
            )

        loader = TextLoader(
            str(self.policy_path),
            encoding="utf-8"
        )

        documents = loader.load()

        logger.info(
            f"Successfully loaded {len(documents)} document(s)."
        )

        return documents