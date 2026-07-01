"""
Build Vector Database

This script builds the Chroma vector database from the enterprise
travel policy document.

Pipeline:
Travel Policy
    ↓
Document Loader
    ↓
Text Splitter
    ↓
Embedding Model
    ↓
Chroma Vector Database
"""

import logging

from rag.document_loader import DocumentLoader
from rag.text_splitter import PolicyTextSplitter
from rag.vector_store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def build_vector_database() -> None:
    """
    Build the complete vector database.
    """

    logger.info("=" * 60)
    logger.info("Starting Vector Database Build")
    logger.info("=" * 60)

    # Load policy document
    loader = DocumentLoader()
    documents = loader.load_policy()

    logger.info(f"Documents Loaded : {len(documents)}")

    # Split into chunks
    splitter = PolicyTextSplitter()
    chunks = splitter.split_documents(documents)

    stats = splitter.get_statistics(chunks)

    logger.info(f"Total Chunks      : {stats['total_chunks']}")
    logger.info(f"Average Length    : {stats['average_length']}")
    logger.info(f"Maximum Length    : {stats['max_length']}")
    logger.info(f"Minimum Length    : {stats['min_length']}")

    # Create vector database
    vector_store = VectorStoreManager()

    if vector_store.database_exists():
        logger.info("Existing database found. Removing...")
        vector_store.delete_database()

    vector_store.create_vector_store(chunks)

    logger.info("=" * 60)
    logger.info("Vector Database Created Successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    build_vector_database()