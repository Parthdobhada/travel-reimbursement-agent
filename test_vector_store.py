from rag.document_loader import DocumentLoader
from rag.text_splitter import PolicyTextSplitter
from rag.vector_store import VectorStoreManager


def test_vector_store_create_and_search(tmp_path):
    loader = DocumentLoader()
    documents = loader.load_policy()

    splitter = PolicyTextSplitter()
    chunks = splitter.split_documents(documents)

    manager = VectorStoreManager()
    manager.persist_directory = str(tmp_path / "chroma_db")

    vector_store = manager.create_vector_store(chunks)

    print("=" * 60)
    print("Vector Database Created Successfully")
    print("=" * 60)

    results = manager.similarity_search(
        "What is the hotel reimbursement limit?"
    )

    print(f"Retrieved {len(results)} documents\n")

    print(results[0].metadata)

    print()

    print(results[0].page_content[:600])

    assert vector_store is not None
    assert results
