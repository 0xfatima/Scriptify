from __future__ import annotations

import chromadb

from config_manager import APP_DIR, ensure_app_dirs


class ChromaPerDocStore:
    def __init__(self) -> None:
        ensure_app_dirs()
        self._client = chromadb.PersistentClient(path=str(APP_DIR / "chroma_db"))

    def get_collection(self, doc_id: str):
        # One collection per document (keeps papers separated)
        return self._client.get_or_create_collection(name=f"doc_{doc_id}")

    def add_documents(self, *, doc_id: str, source_name: str, texts, embeddings, pages):
        col = self.get_collection(doc_id)
        # stable-ish ids inside doc scope
        ids = [f"{doc_id}_{i}" for i in range(len(texts))]
        col.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"page": pages[i], "source": source_name} for i in range(len(texts))],
        )

    def search(self, *, doc_id: str, query_embedding, k: int = 4):
        col = self.get_collection(doc_id)
        count = col.count()
        if count == 0:
            return []
        actual_k = min(k, count)
        results = col.query(query_embeddings=[query_embedding], n_results=actual_k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [
            {"text": docs[i], "page": metas[i].get("page"), "source": metas[i].get("source")}
            for i in range(len(docs))
        ]

    def delete_collection(self, doc_id: str) -> None:
        try:
            self._client.delete_collection(name=f"doc_{doc_id}")
        except Exception:
            pass