"""
Output-validation and RAG-context-filtering guardrails.
* Writing modes (Spell & Grammar, Academic): ensures the model returns a
  correction / rewrite and not an explanation or new content.
* RAG mode: filters retrieved contexts by relevance, ranks them, and
  validates that the generated answer is grounded in context.
"""
from __future__ import annotations
import numpy as np
import re

from typing import Dict, List, Tuple, Optional
# ── RAG context filtering ────────────────────────────────────────────────
# L2² = 2·(1 − cos_sim)  →  threshold 0.8 ≈ cosine_sim > 0.60
RAG_DISTANCE_THRESHOLD = 0.8


# ── Writing output validation ────────────────────────────────────────────
SPELL_MAX_LEN_RATIO = 2.0
SPELL_MIN_SIM = 0.50
RAG_TOP_K = 5
RAG_MIN_CONTEXTS = 1
ACADEMIC_MAX_LEN_RATIO = 3.0
ACADEMIC_MIN_SIM = 0.35
SIM_SKIP_CHARS = 5  # skip embedding similarity for very short inputs
# ── RAG answer validation ────────────────────────────────────────────────
RAG_ANS_CTX_MIN_SIM = 0.20
RAG_ANS_QUERY_MIN_SIM = 0.20



# ── helpers ──────────────────────────────────────────────────────────────
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0

def _text_similarity(t1: str, t2: str) -> float:
    """Cosine similarity between two texts using the sentence embedder."""
    from RAG.embedder import embed_texts
    vecs = embed_texts([t1, t2])
    return _cosine(np.array(vecs[0]), np.array(vecs[1]))



# ── RAG context filtering ────────────────────────────────────────────────
def filter_rag_contexts(
    hits: List[Dict],
    threshold: float = RAG_DISTANCE_THRESHOLD,
    top_k: int = RAG_TOP_K,
) -> Tuple[List[Dict], bool]:
    """Keep only hits whose distance < *threshold*, rank by distance, return
    the top-*k*.  The second element signals whether enough contexts remain."""
    good = [h for h in hits if h.get("distance", float("inf")) < threshold]
    good.sort(key=lambda h: h.get("distance", float("inf")))
    top = good[:top_k]
    return top, len(top) >= RAG_MIN_CONTEXTS


# ── RAG context filtering ────────────────────────────────────────────────
def filter_rag_contexts(
    hits: List[Dict],
    threshold: float = RAG_DISTANCE_THRESHOLD,
    top_k: int = RAG_TOP_K,
) -> Tuple[List[Dict], bool]:
    """Keep only hits whose distance < *threshold*, rank by distance, return
    the top-*k*.  The second element signals whether enough contexts remain."""
    good = [h for h in hits if h.get("distance", float("inf")) < threshold]
    good.sort(key=lambda h: h.get("distance", float("inf")))
    top = good[:top_k]
    return top, len(top) >= RAG_MIN_CONTEXTS

def validate_writing(inp: str, out: str, mode: str) -> bool:

    """Return *True* if the output looks like a correction / rewrite rather
    than an explanation or newly-generated content."""
    i_stripped, o_stripped = inp.strip(), out.strip()
    if not o_stripped:
        return False
    if not i_stripped:
        return True
    ratio = len(o_stripped) / max(len(i_stripped), 1)
    if mode == "Spell & Grammar":
        if ratio > SPELL_MAX_LEN_RATIO:
            return False
        if len(i_stripped) >= SIM_SKIP_CHARS:
            if _text_similarity(i_stripped, o_stripped) < SPELL_MIN_SIM:
                return False
    elif mode == "Academic":
        if ratio > ACADEMIC_MAX_LEN_RATIO:
            return False
        if len(i_stripped) >= SIM_SKIP_CHARS:
            if _text_similarity(i_stripped, o_stripped) < ACADEMIC_MIN_SIM:
                return False
    return True


# ── RAG answer guardrails ────────────────────────────────────────────────
def validate_rag_answer(answer: str, context: str, query: str) -> bool:
    """Return *True* if the answer is grounded in the context and relevant to
    the user query."""
    if not answer.strip():
        return False
    if _text_similarity(answer, context[:2000]) < RAG_ANS_CTX_MIN_SIM:
        return False
    if _text_similarity(answer, query) < RAG_ANS_QUERY_MIN_SIM:
        return False
    return True
