from rag.embeddings import EmbeddingManager

manager = EmbeddingManager()

print("=" * 60)
print("Embedding Health:", manager.health_check())

print("=" * 60)

print("Embedding Dimension:", manager.embedding_dimension())

print("=" * 60)

model = manager.get_model()

vector = model.embed_query(
    "What is the hotel reimbursement limit?"
)

print("Vector Length:", len(vector))

print("First 10 Values:")

print(vector[:10])