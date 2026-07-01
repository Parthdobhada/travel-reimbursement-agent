"""
Text Splitter

Splits loaded policy documents into optimized chunks for Retrieval-Augmented
Generation (RAG).

Features:
- Markdown-aware chunking
- Recursive character splitting
- Metadata enrichment
- Chunk validation
- Chunk statistics
"""

import logging
from typing import List
import re
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from torch import chunk

from config.settings import settings

logger = logging.getLogger(__name__)


class PolicyTextSplitter:
    """
    Splits travel policy documents into RAG-friendly chunks.
    """

    def __init__(self):
        """
        Initialize both markdown and recursive splitters.
        """

        # Split first based on markdown headers
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

        # Further split large sections
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    @staticmethod
    def extract_policy_id(text: str) -> str:
        """
        Extract policy ID from a heading or text.

        Example:
        Appendix N: AI Output Schema [POL-AGENT-002]
        """

        if not text:
            return "UNKNOWN"

        match = re.search(r"(POL-[A-Z0-9-]+)", text)

        if match:
            return match.group(1)

        return "UNKNOWN"

    @staticmethod
    def determine_policy_type(policy_id: str) -> str:
        """
        Determine policy type from policy ID.
        """

        mapping = {
            "HOTEL": "hotel",
            "FLIGHT": "flight",
            "MEAL": "meal",
            "TRANS": "transport",
            "RECEIPT": "receipt",
            "DUP": "duplicate",
            "LIMIT": "limit",
            "LIM": "limit",
            "APPROVAL": "approval",
            "MANUAL": "manual_review",
            "AGENT": "ai",
            "SAMP": "sample_case",
            "GLOSS": "glossary",
        }

        for key, value in mapping.items():
            if key in policy_id:
                return value

        return "general"

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into optimized chunks.

        Args:
            documents: Loaded LangChain documents.

        Returns:
            List of chunked documents.
        """

        logger.info("Starting document splitting...")

        final_chunks = []

        chunk_counter = 1

        for document in documents:

            markdown_sections = self.markdown_splitter.split_text(
                document.page_content
            )

            for section in markdown_sections:

                recursive_chunks = self.recursive_splitter.split_documents(
                    [section]
                )

                for chunk in recursive_chunks:

                    section_name = (
                    chunk.metadata.get("h3")
                    or chunk.metadata.get("h2")
                    or chunk.metadata.get("h1")
                    or ""
                    )

                    policy_id = self.extract_policy_id(section_name)

                    policy_type = self.determine_policy_type(policy_id)

                    chunk.metadata["chunk_id"] = chunk_counter
                    chunk.metadata["chunk_length"] = len(chunk.page_content)
                    chunk.metadata["word_count"] = len(
                    chunk.page_content.split()
                    )

                    chunk.metadata["source"] = document.metadata.get(
                        "source",
                        "unknown"
                    )

                    chunk.metadata["document_type"] = "travel_policy"
                    chunk.metadata["retrieval_source"] = "policy"
                    chunk.metadata["version"] = "1.0.1"
                    chunk.metadata["indexed_by"] = "PolicyTextSplitter"

                    # New business metadata
                    chunk.metadata["policy_id"] = policy_id
                    chunk.metadata["policy_type"] = policy_type
                    section = chunk.metadata.get("h2", "")

                    clean_section = re.sub(
                    r"\s*\[POL-[A-Z0-9-]+\]",
                    "",
                    section
                    ).strip()

                    chunk.metadata["section"] = clean_section

                    final_chunks.append(chunk)

                    chunk_counter += 1

        logger.info(f"Created {len(final_chunks)} chunks.")

        self.validate_chunks(final_chunks)

        return final_chunks

    @staticmethod
    def validate_chunks(chunks: List[Document]) -> None:
        """
        Validate generated chunks.

        Raises:
            ValueError if an empty chunk is found.
        """

        for chunk in chunks:

            if not chunk.page_content.strip():
                raise ValueError("Empty chunk detected.")

            if len(chunk.page_content) < 50:
                logger.warning(
                    f"Very small chunk detected: {chunk.metadata.get('chunk_id')}"
                )

        logger.info("Chunk validation successful.")

    @staticmethod
    def get_statistics(chunks: List[Document]) -> dict:
        """
        Return chunk statistics.

        Returns:
            Dictionary containing chunk statistics.
        """

        if not chunks:
            return {
                "total_chunks": 0,
                "average_length": 0,
                "max_length": 0,
                "min_length": 0,
                "total_characters": 0,
            }

        lengths = [len(chunk.page_content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "average_length": round(sum(lengths) / len(lengths), 2),
            "max_length": max(lengths),
            "min_length": min(lengths),
            "total_characters": sum(lengths),
        }