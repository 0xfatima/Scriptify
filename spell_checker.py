from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SpanError:
    start: int
    end: int
    type: str  # "spell" | "grammar"
    message: str
    suggestions: List[str]


class SpellGrammarChecker:
    """
    Lightweight offline checker.

    - Tries SymSpell if available (fast, offline).
    - Always runs a few cheap heuristic "grammar-ish" checks.
    """

    def __init__(self) -> None:
        self._symspell = None
        self._symspell_max_edit = 2
        self._init_symspell_optional()

    def _init_symspell_optional(self) -> None:
        try:
            from symspellpy import SymSpell  # type: ignore
        except Exception:
            return

        sym = SymSpell(max_dictionary_edit_distance=self._symspell_max_edit, prefix_length=7)

        # Try to load a frequency dictionary if user provides one.
        # You can place it at ./data/frequency_dictionary_en_82_765.txt (SymSpell standard file)
        # This file is optional; the app still runs without it.
        from pathlib import Path

        dict_path = Path("./data/frequency_dictionary_en_82_765.txt")
        if dict_path.exists():
            try:
                sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
                self._symspell = sym
            except Exception:
                self._symspell = None

    def check(self, text: str) -> List[SpanError]:
        errs: List[SpanError] = []
        if not text:
            return errs

        errs.extend(self._heuristics(text))
        if self._symspell is not None:
            errs.extend(self._symspell_spans(text))

        # Keep spans valid and non-empty
        out: List[SpanError] = []
        for e in errs:
            if 0 <= e.start < e.end <= len(text):
                out.append(e)
        return out

    def _heuristics(self, text: str) -> List[SpanError]:
        errs: List[SpanError] = []

        # Double spaces
        for m in re.finditer(r"[^\S\r\n]{2,}", text):
            errs.append(
                SpanError(
                    start=m.start(),
                    end=m.end(),
                    type="grammar",
                    message="Multiple spaces",
                    suggestions=[" "],
                )
            )

        # Space before punctuation
        for m in re.finditer(r"\s+([,.;:!?])", text):
            errs.append(
                SpanError(
                    start=m.start(),
                    end=m.end(),
                    type="grammar",
                    message="Extra space before punctuation",
                    suggestions=[m.group(1)],
                )
            )

        return errs

    def _symspell_spans(self, text: str) -> List[SpanError]:
        # Very simple tokenization → mark single-token "unknown" words.
        # This avoids heavy NLP; it’s good enough for MVP highlighting.
        from symspellpy import Verbosity  # type: ignore

        errs: List[SpanError] = []
        for m in re.finditer(r"[A-Za-z]{3,}", text):
            word = m.group(0)
            # Skip capitalized words (often names) to reduce false positives.
            if word[0].isupper():
                continue

            suggestions = self._symspell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if not suggestions:
                continue
            best = suggestions[0].term
            if best.lower() == word.lower():
                continue
            errs.append(
                SpanError(
                    start=m.start(),
                    end=m.end(),
                    type="spell",
                    message="Possible misspelling",
                    suggestions=[best],
                )
            )
        return errs

