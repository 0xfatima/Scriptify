from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config_manager import APP_DIR, ensure_app_dirs
from RAG.embedder import embed_texts
from RAG.pdf_loader import extract_text_with_metadata
from RAG.vector_store import ChromaPerDocStore


def compute_doc_id(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def copy_into_uploads(src_path: Path) -> Path:
    ensure_app_dirs()
    uploads = APP_DIR / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    dst = uploads / src_path.name
    if dst.resolve() != src_path.resolve():
        shutil.copy2(src_path, dst)
    return dst


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    # simple sentence-ish chunking (offline, lightweight)
    parts = text.replace("\r\n", "\n").split(". ")
    chunks: List[str] = []
    cur = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        add = (p + ". ") if not p.endswith(".") else (p + " ")
        if len(cur) + len(add) <= chunk_size:
            cur += add
        else:
            if cur.strip():
                chunks.append(cur.strip())
            # overlap: keep tail of previous chunk
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = (tail + " " + add).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


@dataclass
class IndexResult:
    doc_id: str
    saved_path: str
    chunks_indexed: int


def index_pdf(pdf_path: Path) -> IndexResult:
    ensure_app_dirs()
    saved = copy_into_uploads(pdf_path)
    doc_id = compute_doc_id(saved)

    store = ChromaPerDocStore()

    data = extract_text_with_metadata(str(saved))
    all_chunks: List[str] = []
    pages: List[int] = []
    for d in data:
        page = int(d.get("page", 0) or 0)
        chunks = chunk_text(d.get("text", ""))
        for c in chunks:
            if len(c) < 30:
                continue
            all_chunks.append(c)
            pages.append(page)

    if not all_chunks:
        return IndexResult(doc_id=doc_id, saved_path=str(saved), chunks_indexed=0)

    embeddings = embed_texts(all_chunks)
    store.add_documents(doc_id=doc_id, source_name=saved.name, texts=all_chunks, embeddings=embeddings, pages=pages)
    return IndexResult(doc_id=doc_id, saved_path=str(saved), chunks_indexed=len(all_chunks))


def _docx_to_pdf_best_effort(docx_path: Path) -> Path:
    """
    Best-effort conversion for Windows:
    - Uses docx2pdf if installed (often requires MS Word).
    - If conversion fails, raises and caller can fall back to text extraction.
    """
    try:
        from docx2pdf import convert  # type: ignore
    except Exception as e:
        raise RuntimeError("docx2pdf not available") from e

    out_pdf = docx_path.with_suffix(".pdf")
    convert(str(docx_path), str(out_pdf))
    if not out_pdf.exists():
        raise RuntimeError("DOCX->PDF conversion failed")
    return out_pdf


def index_docx(docx_path: Path) -> IndexResult:
    ensure_app_dirs()
    saved = copy_into_uploads(docx_path)

    # Preferred path: convert to PDF then index pages
    try:
        pdf = _docx_to_pdf_best_effort(saved)
        return index_pdf(pdf)
    except Exception:
        pass

    # Fallback: extract text from DOCX and index without page numbers
    try:
        from docx import Document  # type: ignore
    except Exception as e:
        raise RuntimeError("python-docx not available to read DOCX") from e

    doc = Document(str(saved))
    text = "\n".join([p.text for p in doc.paragraphs if p.text])
    chunks = chunk_text(text)

    doc_id = compute_doc_id(saved)
    store = ChromaPerDocStore()

    filtered = [c for c in chunks if len(c) >= 30]
    if not filtered:
        return IndexResult(doc_id=doc_id, saved_path=str(saved), chunks_indexed=0)
    embeddings = embed_texts(filtered)
    pages = [0 for _ in filtered]
    store.add_documents(doc_id=doc_id, source_name=saved.name, texts=filtered, embeddings=embeddings, pages=pages)
    return IndexResult(doc_id=doc_id, saved_path=str(saved), chunks_indexed=len(filtered))

