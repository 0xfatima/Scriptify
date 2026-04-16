from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from model_singleton import generate_rag_answer, generate_writing_assist


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


llm_in: "queue.Queue[LlmTask]" = queue.Queue()
llm_out: "queue.Queue[LlmResult]" = queue.Queue()

spell_in: "queue.Queue[Dict[str, Any]]" = queue.Queue()
spell_out: "queue.Queue[Dict[str, Any]]" = queue.Queue()

index_in: "queue.Queue[Dict[str, Any]]" = queue.Queue()
index_out: "queue.Queue[Dict[str, Any]]" = queue.Queue()


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

    threading.Thread(target=llm_worker, daemon=True).start()
    threading.Thread(target=spell_worker, daemon=True).start()
    threading.Thread(target=index_worker, daemon=True).start()


def next_task_id() -> float:
    return time.time()

