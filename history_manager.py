from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from config_manager import APP_DIR, ensure_app_dirs


HISTORY_DIR = APP_DIR / "chat_history"


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant" | "system"
    content: str
    mode: str
    ts: float


@dataclass
class SessionInfo:
    session_id: str  # file stem
    title: str
    ts: float


def _sanitize_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"[^A-Za-z0-9 _-]", "", title).strip()
    title = title.replace(" ", "_")
    return title[:40] or "Chat"


def derive_title_from_text(text: str, *, min_words: int = 3, max_words: int = 5) -> str:
    words = re.findall(r"[A-Za-z0-9]+", (text or "").strip())
    if not words:
        return "New chat"
    n = max(min_words, min(max_words, len(words)))
    return " ".join(words[:n])


def new_session_id(title: str) -> str:
    # filename format: YYYYMMDD_HHMMSS__Title_Words.json
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe = _sanitize_title(title)
    return f"{ts}__{safe}"


def session_path(session_id: str) -> Path:
    ensure_app_dirs()
    return HISTORY_DIR / f"{session_id}.json"


def save_session(session_id: str, title: str, messages: List[ChatMessage]) -> None:
    payload = {
        "title": title,
        "messages": [asdict(m) for m in messages],
    }
    p = session_path(session_id)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_session(session_id: str) -> List[ChatMessage]:
    p = session_path(session_id)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("messages", data)  # backward compatible
    msgs: List[ChatMessage] = []
    for item in raw:
        try:
            msgs.append(ChatMessage(**item))
        except Exception:
            continue
    return msgs


def load_session_title(session_id: str) -> str:
    p = session_path(session_id)
    if not p.exists():
        return "Chat"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return (data.get("title") or "Chat").strip() or "Chat"
    except Exception:
        return "Chat"


def list_sessions() -> List[SessionInfo]:
    ensure_app_dirs()
    files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)
    out: List[SessionInfo] = []
    for f in files:
        sid = f.stem
        title = load_session_title(sid)
        ts = f.stat().st_mtime
        out.append(SessionInfo(session_id=sid, title=title, ts=ts))
    return out


def prune_sessions(keep: int) -> None:
    ensure_app_dirs()
    files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)
    for f in files[keep:]:
        try:
            f.unlink()
        except Exception:
            pass

