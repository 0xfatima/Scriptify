from __future__ import annotations

from sentence_transformers import SentenceTransformer


_EMBEDDER = None


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    # Offline-friendly: will use local cache; set local_files_only=True to hard-enforce offline
    _EMBEDDER = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _EMBEDDER


def embed_texts(texts):
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()