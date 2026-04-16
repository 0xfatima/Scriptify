
# =========================
# retriever.py
# =========================
from embedder import model


def retrieve(query, vector_store, k=3):
    query_embedding = model.encode([query])
    return vector_store.search(query_embedding, k)

