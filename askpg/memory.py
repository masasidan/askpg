from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from typing import Sequence


CONVERSATION_ID = "main"

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversation_messages_lookup
ON conversation_messages(conversation_id, id);
CREATE VIRTUAL TABLE IF NOT EXISTS conversation_messages_fts USING fts5(
    message_id UNINDEXED,
    content,
    tokenize='porter unicode61'
);
"""

STOP_WORDS = {
    "a", "about", "and", "are", "as", "at", "be", "but", "do", "for",
    "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "so", "that", "the", "this", "to", "was", "we", "what", "when", "who",
    "why", "with", "would", "you", "your",
}


def ensure_memory(connection: sqlite3.Connection) -> None:
    connection.executescript(MEMORY_SCHEMA)


def load_recent_history(
    connection: sqlite3.Connection,
    *,
    limit: int = 12,
    conversation_id: str = CONVERSATION_ID,
) -> list[dict[str, str]]:
    ensure_memory(connection)
    rows = connection.execute(
        """
        SELECT role, content FROM conversation_messages
        WHERE conversation_id=? ORDER BY id DESC LIMIT ?
        """,
        (conversation_id, limit),
    ).fetchall()
    return [
        {"role": row["role"], "content": row["content"]}
        for row in reversed(rows)
    ]


def save_turn(
    connection: sqlite3.Connection,
    question: str,
    answer: str,
    *,
    conversation_id: str = CONVERSATION_ID,
) -> None:
    ensure_memory(connection)
    timestamp = datetime.now(timezone.utc).isoformat()
    with connection:
        for role, content in (("user", question), ("assistant", answer)):
            cursor = connection.execute(
                """
                INSERT INTO conversation_messages(
                    conversation_id, role, content, created_at
                ) VALUES(?, ?, ?, ?)
                """,
                (conversation_id, role, content, timestamp),
            )
            connection.execute(
                """
                INSERT INTO conversation_messages_fts(rowid, message_id, content)
                VALUES(?, ?, ?)
                """,
                (cursor.lastrowid, str(cursor.lastrowid), content),
            )


def search_memories(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 6,
    recent_message_count: int = 12,
    conversation_id: str = CONVERSATION_ID,
) -> list[str]:
    ensure_memory(connection)
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+", query.lower())
    terms = list(dict.fromkeys(token for token in tokens if token not in STOP_WORDS))[:12]
    if not terms:
        return []
    fts_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
    newest = connection.execute(
        "SELECT COALESCE(MAX(id), 0) FROM conversation_messages WHERE conversation_id=?",
        (conversation_id,),
    ).fetchone()[0]
    rows = connection.execute(
        """
        SELECT m.role, m.content, bm25(conversation_messages_fts) AS rank
        FROM conversation_messages_fts f
        JOIN conversation_messages m
          ON m.id = CAST(f.message_id AS INTEGER)
        WHERE conversation_messages_fts MATCH ?
          AND m.conversation_id=?
          AND m.id <= ?
        ORDER BY rank, m.id DESC
        LIMIT ?
        """,
        (fts_query, conversation_id, max(0, newest - recent_message_count), limit),
    ).fetchall()
    return [f"{row['role'].title()}: {row['content']}" for row in rows]


def clear_memory(
    connection: sqlite3.Connection,
    *,
    conversation_id: str = CONVERSATION_ID,
) -> None:
    ensure_memory(connection)
    ids = [
        (str(row["id"]),)
        for row in connection.execute(
            "SELECT id FROM conversation_messages WHERE conversation_id=?",
            (conversation_id,),
        ).fetchall()
    ]
    with connection:
        if ids:
            connection.executemany(
                "DELETE FROM conversation_messages_fts WHERE message_id=?", ids
            )
        connection.execute(
            "DELETE FROM conversation_messages WHERE conversation_id=?",
            (conversation_id,),
        )


def memory_count(
    connection: sqlite3.Connection,
    *,
    conversation_id: str = CONVERSATION_ID,
) -> int:
    ensure_memory(connection)
    return int(
        connection.execute(
            "SELECT COUNT(*) FROM conversation_messages WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()[0]
    )


def format_memory(memories: Sequence[str]) -> str:
    return "\n".join(f"- {memory}" for memory in memories)
