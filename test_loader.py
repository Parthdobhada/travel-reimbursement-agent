from rag.document_loader import DocumentLoader

loader = DocumentLoader()
documents = loader.load_policy()

print("=" * 60)
print("Documents Loaded:", len(documents))
print("=" * 60)

print("Document Type:", type(documents[0]))
print("Metadata:", documents[0].metadata)
print("Content Length:", len(documents[0].page_content))

print("\nFirst 500 Characters:\n")
print(repr(documents[0].page_content[:500]))