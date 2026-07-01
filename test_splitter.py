from rag.document_loader import DocumentLoader
from rag.text_splitter import PolicyTextSplitter

loader = DocumentLoader()

documents = loader.load_policy()

splitter = PolicyTextSplitter()

chunks = splitter.split_documents(documents)

stats = splitter.get_statistics(chunks)

print("=" * 60)
print("RAG CHUNK STATISTICS")
print("=" * 60)

for key, value in stats.items():
    print(f"{key}: {value}")

print("\nFirst Chunk Metadata:\n")
print(chunks[0].metadata)

print("\nFirst Chunk Preview:\n")
print(chunks[0].page_content[:500])