"""
Retriever

Provides a simple interface for retrieving relevant policy chunks
from the Chroma vector database.
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from config.settings import settings
from rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class PolicyRetriever:
    """
    Retrieves relevant policy documents from the vector database.
    """

    def __init__(self):
        self.vector_store = VectorStoreManager()

    def retrieve(
        self,
        query: str,
        k: int = None,
    ) -> List[Document]:
        """
        Retrieve the most relevant policy chunks.

        Args:
            query: User question.
            k: Number of chunks to retrieve.

        Returns:
            List of relevant documents.
        """

        if k is None:
            k = settings.TOP_K_RESULTS

        logger.info(f"Searching policy for: {query}")

        return self.vector_store.similarity_search(
            query=query,
            k=k,
        )

    def retrieve_with_scores(
        self,
        query: str,
        k: int = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents along with relevance scores.

        Returns:
            List of (Document, Score)
        """

        if k is None:
            k = settings.TOP_K_RESULTS

        store = self.vector_store.load_vector_store()

        return store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )

    @staticmethod
    def format_context(
        documents: List[Document]
    ) -> str:
        """
        Convert retrieved chunks into a single context string
        for the LLM.
        """

        context = []

        for doc in documents:

            section = doc.metadata.get("section", "Unknown Section")

            policy_id = doc.metadata.get(
                "policy_id",
                "UNKNOWN"
            )

            context.append(
                f"""
Section : {section}
Policy ID : {policy_id}

{doc.page_content}
"""
            )

        return "\n\n".join(context)