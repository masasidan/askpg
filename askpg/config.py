from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = Path(os.environ.get("ASKPG_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
ESSAYS_DIR = DATA_DIR / "essays"
MANIFEST_PATH = DATA_DIR / "manifest.json"
TWEETS_DIR = DATA_DIR / "tweets"
TWEETS_PATH = TWEETS_DIR / "tweets.jsonl"
TWEETS_MANIFEST_PATH = TWEETS_DIR / "manifest.json"
DB_PATH = DATA_DIR / "askpg.sqlite3"

CHAT_MODEL = os.environ.get("ASKPG_MODEL", "gpt-5.6-terra")
REASONING_EFFORT = os.environ.get("ASKPG_REASONING_EFFORT", "medium")
RETRIEVAL_MODEL = os.environ.get("ASKPG_RETRIEVAL_MODEL", "gpt-5.6-luna")
EMBEDDING_MODEL = os.environ.get("ASKPG_EMBEDDING_MODEL", "text-embedding-3-small")


def api_key() -> str | None:
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if value:
        return value
    if ENV_PATH.exists():
        try:
            for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
                key, separator, stored = line.partition("=")
                if separator and key.strip() == "OPENAI_API_KEY":
                    stored = stored.strip()
                    if len(stored) >= 2 and stored[0] == stored[-1] and stored[0] in "\"'":
                        stored = stored[1:-1]
                    return stored or None
        except OSError:
            return None
    return None


def ensure_data_dirs() -> None:
    ESSAYS_DIR.mkdir(parents=True, exist_ok=True)
    TWEETS_DIR.mkdir(parents=True, exist_ok=True)
