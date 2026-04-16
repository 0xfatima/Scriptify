from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


APP_DIR = Path.home() / ".spell_grammar_offline"
CONFIG_PATH = APP_DIR / "settings.json"


@dataclass
class Settings:
    theme: str = "dark"  # "dark" | "light"
    history_keep_sessions: int = 5
    history_keep_messages: int = 5
    font_family: str = "Georgia"
    font_size: int = 15


def ensure_app_dirs() -> None:
    (APP_DIR / "chat_history").mkdir(parents=True, exist_ok=True)
    (APP_DIR / "uploads").mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    ensure_app_dirs()
    if not CONFIG_PATH.exists():
        s = Settings()
        save_settings(s)
        return s
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        s = Settings(**{k: data.get(k) for k in Settings().__dict__.keys() if k in data})
        # fill defaults for missing keys
        base = Settings()
        for k, v in base.__dict__.items():
            if getattr(s, k, None) is None:
                setattr(s, k, v)
        return s
    except Exception:
        s = Settings()
        save_settings(s)
        return s


def save_settings(settings: Settings) -> None:
    ensure_app_dirs()
    CONFIG_PATH.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


DOC_REGISTRY_PATH = APP_DIR / "doc_registry.json"


def load_doc_registry() -> dict[str, str]:
    """Return {doc_id: display_name} persisted from previous sessions."""
    if not DOC_REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_doc_registry(registry: dict[str, str]) -> None:
    ensure_app_dirs()
    DOC_REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")

