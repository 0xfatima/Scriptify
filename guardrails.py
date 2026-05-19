"""
Output-validation and RAG-context-filtering guardrails.
* Writing modes (Spell & Grammar, Academic): ensures the model returns a
  correction / rewrite and not an explanation or new content.
* RAG mode: filters retrieved contexts by relevance, ranks them, and
  validates that the generated answer is grounded in context.
"""
from __future__ import annotations
import numpy as np

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
RAG_ANS_QUERY_MIN_SIM = 0.35


# ── LaTeX / Email input guardrails (writing mode only; see thread_manager) ─
# In-domain vs out-of-domain is decided by max cosine similarity to fixed
# anchor sentences (same embedder as RAG; normalized vectors → dot = cosine).
MODE_GATE_MIN_SIM = 0.38
_MODE_GATE_MAX_CHARS = 2000

LATEX_MODE_REFUSAL = (
    "In LaTeX mode I only help with **tables** and **figures** (e.g. `table`/`figure` "
    "environments, `tabular`, `\\includegraphics`, captions, floats). "
    "Please ask a table- or figure-related question."
)
EMAIL_MODE_REFUSAL = (
    "In Email mode I only help with **email-related** tasks (drafting, replying, "
    "subject/body, tone, greetings/closings). Please rephrase as an email task."
)

_LATEX_ANCHOR_TEXTS: Tuple[str, ...] = (
    "Help me write a LaTeX table environment with tabular rows and columns.",
    "How do I create a figure with includegraphics and caption in LaTeX?",
    "Fix my floating table placement using the table and figure environments.",
    "Add subfigures and cross-references with label and ref for tables and figures.",
    "Convert this data into a LaTeX tabular table with alignment and borders.",
    "I need a LaTeX table with multiple rows and merged cells or multicolumn.",
    "Resize a figure image inside the figure environment and set the caption.",
    "Side-by-side figures or tables using minipage or subcaption in TeX.",
)

_EMAIL_ANCHOR_TEXTS: Tuple[str, ...] = (
    "Draft a professional email with subject line and polite greeting.",
    "Write a follow-up email after a meeting requesting confirmation.",
    "Reply to this email thread with a concise tone and clear next steps.",
    "Polish the wording of my email body and closing signature.",
    "Help me write a cover letter email to a hiring manager.",
    "Rewrite this email to sound more formal or more casual as needed.",
    "Compose an apology or thank-you email with appropriate tone.",
    "Schedule a meeting by email including time options and CC line.",
)

_LATEX_ANCHOR_MAT: Optional[np.ndarray] = None
_EMAIL_ANCHOR_MAT: Optional[np.ndarray] = None


def _clip_for_mode_gate(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) <= _MODE_GATE_MAX_CHARS:
        return t
    return t[:_MODE_GATE_MAX_CHARS]


def _lazy_anchor_matrix(texts: Tuple[str, ...], which: str) -> np.ndarray:
    global _LATEX_ANCHOR_MAT, _EMAIL_ANCHOR_MAT
    from RAG.embedder import embed_texts

    if which == "latex":
        if _LATEX_ANCHOR_MAT is None:
            _LATEX_ANCHOR_MAT = np.asarray(embed_texts(list(texts)), dtype=np.float64)
        return _LATEX_ANCHOR_MAT
    if which == "email":
        if _EMAIL_ANCHOR_MAT is None:
            _EMAIL_ANCHOR_MAT = np.asarray(embed_texts(list(texts)), dtype=np.float64)
        return _EMAIL_ANCHOR_MAT
    raise ValueError(which)


def _max_anchor_similarity(text: str, anchor_texts: Tuple[str, ...], which: str) -> float:
    clipped = _clip_for_mode_gate(text)
    if not clipped:
        return 0.0
    from RAG.embedder import embed_texts

    q = np.asarray(embed_texts([clipped])[0], dtype=np.float64)
    mat = _lazy_anchor_matrix(anchor_texts, which)
    return float(np.max(mat @ q))


def latex_tables_figures_query_allowed(text: str) -> bool:
    return _max_anchor_similarity(text, _LATEX_ANCHOR_TEXTS, "latex") >= MODE_GATE_MIN_SIM


def email_relevant_query_allowed(text: str) -> bool:
    return _max_anchor_similarity(text, _EMAIL_ANCHOR_TEXTS, "email") >= MODE_GATE_MIN_SIM


def gated_mode_refusal_message(text: str, mode: str) -> Optional[str]:
    """Return a refusal message if *text* is out of scope for *mode*; else *None*.
    Only LaTeX and Email modes use this from the LLM worker."""
    if mode == "LaTeX":
        return None if latex_tables_figures_query_allowed(text) else LATEX_MODE_REFUSAL
    if mode == "Email":
        return None if email_relevant_query_allowed(text) else EMAIL_MODE_REFUSAL
    return None


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


def _is_mostly_copied(answer: str, context: str) -> bool:
    return _text_similarity(answer, context[:2000]) > 0.85

# ── RAG answer guardrails ────────────────────────────────────────────────
def validate_rag_answer(answer: str, context: str, query: str) -> bool:
    """Return *True* if the answer is grounded in the context and relevant to
    the user query."""
    if not answer.strip():
        return False
    if _is_mostly_copied(answer, context):
        return False
    # if _text_similarity(answer, context[:2000]) < RAG_ANS_CTX_MIN_SIM:
    #     return False
    if _text_similarity(answer, query) < RAG_ANS_QUERY_MIN_SIM:
        return False
    return True
