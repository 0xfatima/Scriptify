from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from model_singleton import generate_pdf_explain, generate_rag_answer, generate_writing_assist
from guardrails import gated_mode_refusal_message


@dataclass
class LlmTask:
    kind: str  # "writing" | "rag"
    text: str
    mode: str
    context: str = ""
    task_id: float = 0.0


@dataclass
class LlmResult:
    task_id: float
    ok: bool
    text: str
    error: str = ""

# thread_manager.py
from dataclasses import dataclass
from typing import Optional

# ... existing imports/queues ...

@dataclass
class RagRetrieveTask:
    text: str
    doc_ids: list[str]
    task_id: float

@dataclass
class RagRetrieveResult:
    task_id: float
    ok: bool
    context: str = ""
    sources: str = ""
    error: str = ""

rag_in: "queue.Queue[RagRetrieveTask]" = queue.Queue()
rag_out: "queue.Queue[RagRetrieveResult]" = queue.Queue()

llm_in: "queue.Queue[LlmTask]" = queue.Queue()
llm_out: "queue.Queue[LlmResult]" = queue.Queue()

spell_in: "queue.Queue[Dict[str, Any]]" = queue.Queue()
spell_out: "queue.Queue[Dict[str, Any]]" = queue.Queue()

index_in: "queue.Queue[Dict[str, Any]]" = queue.Queue()
index_out: "queue.Queue[Dict[str, Any]]" = queue.Queue()

# PDF text extraction (for selectable PDF view)
pdf_in: "queue.Queue[Dict[str, Any]]" = queue.Queue()
pdf_out: "queue.Queue[Dict[str, Any]]" = queue.Queue()


_started = False




def start_workers(*, spell_checker) -> None:
    global _started
    if _started:
        return
    _started = True

    def llm_worker():
        while True:
            task: Optional[LlmTask] = llm_in.get()
            if task is None:
                break
            try:
                if task.kind == "rag":
                    out = generate_rag_answer(query=task.text, context=task.context)
                elif task.kind == "pdf":
                    out = generate_pdf_explain(question=task.text, selected_text=task.context)
                elif task.mode in ("LaTeX", "Email"):
                    refusal = gated_mode_refusal_message(task.text, task.mode)
                    if refusal is not None:
                        llm_out.put(LlmResult(task_id=task.task_id, ok=True, text=refusal))
                        continue
                    out = generate_writing_assist(text=task.text, mode=task.mode)
                else:
                    out = generate_writing_assist(text=task.text, mode=task.mode)
                llm_out.put(LlmResult(task_id=task.task_id, ok=True, text=out))
            except Exception as e:
                llm_out.put(LlmResult(task_id=task.task_id, ok=False, text="", error=str(e)))

    def spell_worker():
        while True:
            item = spell_in.get()
            if item is None:
                break
            try:
                text = item.get("text", "")
                req_id = item.get("id", 0.0)
                errors = spell_checker.check(text)
                spell_out.put({"id": req_id, "errors": errors})
            except Exception as e:
                spell_out.put({"id": item.get("id", 0.0), "errors": [], "error": str(e)})

    def index_worker():
        while True:
            item = index_in.get()
            if item is None:
                break
            try:
                from pathlib import Path
                from RAG.indexer import index_docx, index_pdf

                path = Path(item["path"])
                if path.suffix.lower() == ".pdf":
                    res = index_pdf(path)
                elif path.suffix.lower() in {".docx", ".doc"}:
                    if path.suffix.lower() == ".doc":
                        raise RuntimeError("Legacy .doc not supported offline. Please upload .docx or PDF.")
                    res = index_docx(path)
                else:
                    raise RuntimeError("Unsupported file type. Upload PDF or DOCX.")
                index_out.put({"ok": True, "doc_id": res.doc_id, "saved_path": res.saved_path, "chunks": res.chunks_indexed})
            except Exception as e:
                index_out.put({"ok": False, "error": str(e)})

    def pdf_worker():
        while True:
            item = pdf_in.get()
            if item is None:
                break
            try:
                path = item.get("path", "")
                req_id = item.get("id", 0.0)
                title = item.get("title", "")
                if not path:
                    pdf_out.put({"id": req_id, "ok": False, "error": "Missing PDF path."})
                    continue

                import fitz  # PyMuPDF

                doc = fitz.open(path)
                parts = []
                for i in range(len(doc)):
                    page = doc.load_page(i)
                    parts.append(f"\n\n--- Page {i + 1} ---\n\n")
                    parts.append(page.get_text("text") or "")
                doc.close()

                pdf_out.put({"id": req_id, "ok": True, "path": path, "title": title, "text": "".join(parts)})
            except Exception as e:
                pdf_out.put(
                    {
                        "id": item.get("id", 0.0),
                        "ok": False,
                        "error": str(e),
                        "path": item.get("path", ""),
                        "title": item.get("title", ""),
                    }
                )
                
    # def rag_worker():
    #     while True:
    #         task: Optional[RagRetrieveTask] = rag_in.get()
    #         if task is None:
    #             break
    #         try:
    #             from RAG.embedder import embed_texts
    #             from RAG.vector_store import ChromaPerDocStore
    #             from guardrails import filter_rag_contexts
    #             q_emb = embed_texts([task.text])[0]
    #             store = ChromaPerDocStore()
    #             all_hits = []
    #             for did in task.doc_ids:
    #                 hits = store.search(doc_id=did, query_embedding=q_emb, k=10)
    #                 all_hits.extend(hits)
    #             filtered_hits, has_enough = filter_rag_contexts(all_hits)
    #             if not has_enough:
    #                 rag_out.put(RagRetrieveResult(
    #                     task_id=task.task_id,
    #                     ok=False,
    #                     error="Not enough relevant content in uploaded documents."
    #                 ))
    #                 continue
    #             chunks = []
    #             seen_sources: dict[str, set[str]] = {}
    #             for idx, h in enumerate(filtered_hits):
    #                 txt = (h.get("text", "") or "").strip()
    #                 if not txt:
    #                     continue
    #                 src = h.get("source", "document")
    #                 pg = str(h.get("page", "?"))
    #                 chunks.append(
    #                     f"--- Chunk {idx+1} ---\n"
    #                     f"Document: {src}\n"
    #                     f"Page: {pg}\n"
    #                     f"Content: {txt}"
    #                 )
    #                 seen_sources.setdefault(src, set()).add(pg)
    #             context = "\n\n".join(chunks).strip()
    #             if not context:
    #                 rag_out.put(RagRetrieveResult(
    #                     task_id=task.task_id, ok=False, error="No usable context found."
    #                 ))
    #                 continue
    #             parts = []
    #             for s, pages in seen_sources.items():
    #                 # keep your existing sort logic if you want
    #                 parts.append(f"{s} (p. {', '.join(sorted(pages))})")
    #             sources = " | ".join(parts)
    #             rag_out.put(RagRetrieveResult(
    #                 task_id=task.task_id, ok=True, context=context, sources=sources
    #             ))
    #         except Exception as e:
    #             rag_out.put(RagRetrieveResult(task_id=task.task_id, ok=False, error=str(e)))

    # threading.Thread(target=rag_worker, daemon=True).start()
    threading.Thread(target=llm_worker, daemon=True).start()
    threading.Thread(target=spell_worker, daemon=True).start()
    threading.Thread(target=index_worker, daemon=True).start()
    threading.Thread(target=pdf_worker, daemon=True).start()


def next_task_id() -> float:
    return time.time()

