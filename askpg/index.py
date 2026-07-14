from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .config import (
    DB_PATH,
    EMBEDDING_MODEL,
    MANIFEST_PATH,
    TWEETS_MANIFEST_PATH,
    api_key,
    ensure_data_dirs,
)


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS essays (
    slug TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published TEXT,
    sha256 TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'essay'
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    essay_slug TEXT NOT NULL REFERENCES essays(slug) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    text TEXT NOT NULL,
    chunk_hash TEXT NOT NULL UNIQUE,
    embedding BLOB,
    embedding_norm REAL,
    embedding_model TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    title,
    text,
    tokenize='porter unicode61'
);
"""

STOP_WORDS = {
    "a", "about", "am", "an", "and", "are", "as", "at", "be", "been", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "paul", "pg", "say",
    "she", "should", "so", "that", "the", "their", "them", "there", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
}


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    essay_slug: str
    chunk_index: int
    title: str
    url: str
    text: str
    score: float
    source_type: str = "essay"
    published: str | None = None


class IndexError(RuntimeError):
    pass


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_data_dirs()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(essays)").fetchall()
    }
    if "source_type" not in columns:
        connection.execute(
            "ALTER TABLE essays ADD COLUMN source_type TEXT NOT NULL DEFAULT 'essay'"
        )
        connection.commit()
    return connection


def _word_windows(text: str, *, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def _is_heading(paragraph: str) -> bool:
    words = paragraph.split()
    if not words or len(words) > 9 or len(paragraph) > 80:
        return False
    if re.fullmatch(r"(?:19|20)\d{2}", paragraph):
        return False
    if re.fullmatch(
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(?:19|20)\d{2}",
        paragraph,
        re.IGNORECASE,
    ):
        return False
    if paragraph[-1] in ".?!,:;":
        return False
    letters = [word for word in words if any(character.isalpha() for character in word)]
    if not letters:
        return False
    return len(words) == 1 or all(word[0].isupper() for word in letters)


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    if len(paragraph.split()) <= max_words:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", paragraph)
    if len(sentences) == 1:
        return _word_windows(paragraph, size=max_words, overlap=min(45, max_words // 5))
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and count + sentence_words > max_words:
            pieces.append(" ".join(current))
            current = []
            count = 0
        current.append(sentence)
        count += sentence_words
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(
    text: str,
    *,
    size: int = 320,
    overlap: int = 70,
    max_size: int = 430,
) -> list[str]:
    """Split prose on idea and section boundaries, with paragraph-level overlap.

    Plain text without paragraph breaks uses word windows as a compatibility fallback.
    Scraped essays preserve paragraph breaks and therefore take the idea-aware path.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    normalized = text.strip()
    if not normalized:
        return []
    if not re.search(r"\n\s*\n", normalized):
        return _word_windows(normalized, size=size, overlap=overlap)

    raw_paragraphs = [
        " ".join(part.split())
        for part in re.split(r"\n\s*\n", normalized)
        if part.strip()
    ]
    sections: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    paragraphs: list[str] = []
    for paragraph in raw_paragraphs:
        if _is_heading(paragraph):
            if paragraphs:
                sections.append((heading, paragraphs))
            heading = paragraph
            paragraphs = []
        else:
            paragraphs.extend(_split_long_paragraph(paragraph, max_size))
    if paragraphs:
        sections.append((heading, paragraphs))

    chunks: list[str] = []
    for section_heading, section_paragraphs in sections:
        current: list[str] = []
        current_words = 0
        for paragraph in section_paragraphs:
            paragraph_words = len(paragraph.split())
            if current and current_words + paragraph_words > size:
                body = "\n\n".join(current)
                chunks.append(
                    f"Section: {section_heading}\n\n{body}" if section_heading else body
                )
                carry: list[str] = []
                carry_words = 0
                for previous in reversed(current):
                    previous_words = len(previous.split())
                    if previous_words > overlap or carry_words + previous_words > overlap:
                        break
                    carry.insert(0, previous)
                    carry_words += previous_words
                current = carry
                current_words = carry_words
            current.append(paragraph)
            current_words += paragraph_words
        if current:
            body = "\n\n".join(current)
            chunks.append(
                f"Section: {section_heading}\n\n{body}" if section_heading else body
            )
    return chunks


def _read_body(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    parts = content.split("---\n", 2)
    if len(parts) == 3:
        content = parts[2]
    lines = content.lstrip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    if lines and not lines[0].strip():
        lines = lines[1:]
    if lines and lines[0].startswith("Source: "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _rebuild_fts(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM chunks_fts")
    connection.execute(
        """
        INSERT INTO chunks_fts(rowid, chunk_id, title, text)
        SELECT id, CAST(id AS TEXT), title, text FROM chunks
        """
    )


def sync_corpus(
    connection: sqlite3.Connection,
    *,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[int, int]:
    if not manifest_path.exists():
        raise IndexError(f"Corpus manifest not found at {manifest_path}. Run `askpg scrape` first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("failures"):
        raise IndexError("The scrape manifest contains failures. Rerun `askpg scrape`.")

    records = manifest.get("essays", [])
    if not records:
        raise IndexError("The corpus manifest contains no essays.")

    desired_hashes: list[tuple[str]] = []
    with connection:
        connection.execute("CREATE TEMP TABLE IF NOT EXISTS desired_chunks (hash TEXT PRIMARY KEY)")
        connection.execute("DELETE FROM desired_chunks")

        for record in records:
            connection.execute(
                """
                INSERT INTO essays(
                    slug, title, url, published, sha256, word_count, source_type
                )
                VALUES(?, ?, ?, ?, ?, ?, 'essay')
                ON CONFLICT(slug) DO UPDATE SET
                    title=excluded.title,
                    url=excluded.url,
                    published=excluded.published,
                    sha256=excluded.sha256,
                    word_count=excluded.word_count,
                    source_type='essay'
                """,
                (
                    record["slug"], record["title"], record["url"], record.get("published"),
                    record["sha256"], record["word_count"],
                ),
            )
            essay_path = manifest_path.parent / record["filename"]
            if not essay_path.exists():
                raise IndexError(f"Missing essay file: {essay_path}")
            for chunk_index, text in enumerate(chunk_text(_read_body(essay_path))):
                digest = hashlib.sha256(
                    f"{record['title']}\n{text}".encode("utf-8")
                ).hexdigest()
                desired_hashes.append((digest,))
                connection.execute(
                    """
                    INSERT INTO chunks(
                        essay_slug, chunk_index, title, url, text, chunk_hash
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_hash) DO UPDATE SET
                        essay_slug=excluded.essay_slug,
                        chunk_index=excluded.chunk_index,
                        title=excluded.title,
                        url=excluded.url,
                        text=excluded.text
                    """,
                    (
                        record["slug"], chunk_index, record["title"], record["url"],
                        text, digest,
                    ),
                )

        connection.executemany("INSERT OR IGNORE INTO desired_chunks(hash) VALUES(?)", desired_hashes)
        connection.execute(
            """
            DELETE FROM chunks
            WHERE essay_slug IN (SELECT slug FROM essays WHERE source_type='essay')
              AND chunk_hash NOT IN (SELECT hash FROM desired_chunks)
            """
        )
        connection.execute(
            """
            DELETE FROM essays
            WHERE source_type='essay'
              AND slug NOT IN (SELECT DISTINCT essay_slug FROM chunks)
            """
        )

        _rebuild_fts(connection)
        fingerprint = hashlib.sha256(
            "".join(sorted(record["sha256"] for record in records)).encode("ascii")
        ).hexdigest()
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('corpus_fingerprint', ?)",
            (fingerprint,),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('scraped_at', ?)",
            (manifest.get("scraped_at", "unknown"),),
        )

    essay_count = connection.execute(
        "SELECT COUNT(*) FROM essays WHERE source_type='essay'"
    ).fetchone()[0]
    chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return essay_count, chunk_count


def sync_tweets(
    connection: sqlite3.Connection,
    *,
    manifest_path: Path = TWEETS_MANIFEST_PATH,
) -> tuple[int, int]:
    if not manifest_path.exists():
        return 0, connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filename = manifest.get("filename")
    if not filename:
        raise IndexError(f"Tweet manifest at {manifest_path} has no filename.")
    tweets_path = manifest_path.parent / filename
    if not tweets_path.exists():
        raise IndexError(f"Tweet archive not found at {tweets_path}. Run `askpg scrape-tweets`.")

    desired_slugs: list[tuple[str]] = []
    with connection:
        connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS desired_tweets (slug TEXT PRIMARY KEY)"
        )
        connection.execute("DELETE FROM desired_tweets")
        with tweets_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IndexError(
                        f"Invalid tweet JSON at {tweets_path}:{line_number}: {exc}"
                    ) from exc
                tweet_id = str(record["id"])
                slug = f"tweet-{tweet_id}"
                text = " ".join(str(record["text"]).split())
                created_at = str(record["created_at"])
                published = created_at[:10]
                title = f"Tweet — {published}"
                url = str(record.get("url") or f"https://x.com/paulg/status/{tweet_id}")
                digest = hashlib.sha256(f"{tweet_id}\n{text}".encode("utf-8")).hexdigest()
                desired_slugs.append((slug,))
                connection.execute(
                    """
                    INSERT INTO essays(
                        slug, title, url, published, sha256, word_count, source_type
                    ) VALUES(?, ?, ?, ?, ?, ?, 'tweet')
                    ON CONFLICT(slug) DO UPDATE SET
                        title=excluded.title,
                        url=excluded.url,
                        published=excluded.published,
                        sha256=excluded.sha256,
                        word_count=excluded.word_count,
                        source_type='tweet'
                    """,
                    (slug, title, url, published, digest, len(text.split())),
                )
                connection.execute(
                    """
                    INSERT INTO chunks(
                        essay_slug, chunk_index, title, url, text, chunk_hash
                    ) VALUES(?, 0, ?, ?, ?, ?)
                    ON CONFLICT(chunk_hash) DO UPDATE SET
                        essay_slug=excluded.essay_slug,
                        chunk_index=0,
                        title=excluded.title,
                        url=excluded.url,
                        text=excluded.text
                    """,
                    (slug, title, url, text, digest),
                )

        connection.executemany(
            "INSERT OR IGNORE INTO desired_tweets(slug) VALUES(?)", desired_slugs
        )
        connection.execute(
            """
            DELETE FROM essays
            WHERE source_type='tweet'
              AND slug NOT IN (SELECT slug FROM desired_tweets)
            """
        )
        _rebuild_fts(connection)
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('tweets_scraped_at', ?)",
            (manifest.get("scraped_at", "unknown"),),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('tweets_fingerprint', ?)",
            (manifest.get("sha256", "unknown"),),
        )

    tweet_count = connection.execute(
        "SELECT COUNT(*) FROM essays WHERE source_type='tweet'"
    ).fetchone()[0]
    chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return tweet_count, chunk_count


def _batched(items: Sequence[sqlite3.Row], size: int) -> Iterable[Sequence[sqlite3.Row]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def embed_missing(
    connection: sqlite3.Connection,
    *,
    model: str = EMBEDDING_MODEL,
    progress: Callable[[str], None] | None = None,
    batch_size: int = 128,
) -> int:
    key = api_key()
    if not key:
        raise IndexError("OPENAI_API_KEY is not set; cannot create semantic embeddings.")
    from openai import OpenAI

    emit = progress or (lambda _: None)
    current_model_row = connection.execute(
        "SELECT value FROM metadata WHERE key='embedding_model'"
    ).fetchone()
    current_model = current_model_row[0] if current_model_row else None
    if current_model and current_model != model:
        with connection:
            connection.execute(
                "UPDATE chunks SET embedding=NULL, embedding_norm=NULL, embedding_model=NULL"
            )

    rows = connection.execute(
        "SELECT id, title, text FROM chunks WHERE embedding IS NULL ORDER BY id"
    ).fetchall()
    if not rows:
        return 0

    client = OpenAI(api_key=key)
    completed = 0
    for batch in _batched(rows, batch_size):
        inputs = [f"Title: {row['title']}\n\n{row['text']}" for row in batch]
        response = client.embeddings.create(model=model, input=inputs)
        vectors = sorted(response.data, key=lambda item: item.index)
        if len(vectors) != len(batch):
            raise IndexError("Embedding API returned an unexpected number of vectors.")
        values = []
        for row, item in zip(batch, vectors):
            vector = array("f", item.embedding)
            norm = math.sqrt(sum(value * value for value in vector))
            values.append((vector.tobytes(), norm, model, row["id"]))
        with connection:
            connection.executemany(
                """
                UPDATE chunks
                SET embedding=?, embedding_norm=?, embedding_model=?
                WHERE id=?
                """,
                values,
            )
        completed += len(batch)
        emit(f"Embedded {completed}/{len(rows)} chunks")

    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('embedding_model', ?)",
            (model,),
        )
    return completed


def has_embeddings(connection: sqlite3.Connection) -> bool:
    row = connection.execute("SELECT 1 FROM chunks WHERE embedding IS NOT NULL LIMIT 1").fetchone()
    return row is not None


def _search_terms(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+", query.lower())
    useful = [token for token in tokens if token not in STOP_WORDS]
    selected = useful or tokens
    return list(dict.fromkeys(selected))[:16]


def lexical_search(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 40,
    source_type: str | None = None,
) -> list[SearchResult]:
    terms = _search_terms(query)
    if not terms:
        return []
    fts_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
    source_clause = "AND e.source_type = ?" if source_type else ""
    parameters: tuple[object, ...] = (
        (fts_query, source_type, limit) if source_type else (fts_query, limit)
    )
    rows = connection.execute(
        """
        SELECT c.*, e.source_type, e.published,
               bm25(chunks_fts, 3.0, 1.0) AS fts_score
        FROM chunks_fts
        JOIN chunks c ON c.id = CAST(chunks_fts.chunk_id AS INTEGER)
        JOIN essays e ON e.slug = c.essay_slug
        WHERE chunks_fts MATCH ?
        """ + source_clause + """
        ORDER BY fts_score
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    return [
        SearchResult(
            chunk_id=row["id"], essay_slug=row["essay_slug"],
            chunk_index=row["chunk_index"], title=row["title"], url=row["url"],
            text=row["text"], score=-float(row["fts_score"]),
            source_type=row["source_type"], published=row["published"],
        )
        for row in rows
    ]


def _query_vector(query: str, *, model: str = EMBEDDING_MODEL) -> array:
    key = api_key()
    if not key:
        raise IndexError("OPENAI_API_KEY is not set; semantic search is unavailable.")
    from openai import OpenAI

    response = OpenAI(api_key=key).embeddings.create(model=model, input=[query])
    return array("f", response.data[0].embedding)


def semantic_search(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 40,
    query_vector: array | None = None,
    source_type: str | None = None,
) -> list[SearchResult]:
    vector = query_vector if query_vector is not None else _query_vector(query)
    query_norm = math.sqrt(sum(value * value for value in vector))
    source_clause = "WHERE e.source_type = ?" if source_type else ""
    rows = connection.execute(
        """
        SELECT c.id, c.essay_slug, c.chunk_index, c.title, c.url, c.text,
               c.embedding, c.embedding_norm, e.source_type, e.published
        FROM chunks c JOIN essays e ON e.slug = c.essay_slug
        """ + source_clause + (" AND" if source_type else " WHERE") + """
        c.embedding IS NOT NULL
        """,
        (source_type,) if source_type else (),
    ).fetchall()
    scored: list[SearchResult] = []
    for row in rows:
        document = array("f")
        document.frombytes(row["embedding"])
        denominator = query_norm * float(row["embedding_norm"] or 1.0)
        score = sum(left * right for left, right in zip(vector, document)) / denominator
        scored.append(
            SearchResult(
                chunk_id=row["id"], essay_slug=row["essay_slug"],
                chunk_index=row["chunk_index"], title=row["title"], url=row["url"],
                text=row["text"], score=score,
                source_type=row["source_type"], published=row["published"],
            )
        )
    scored.sort(key=lambda result: result.score, reverse=True)
    return scored[:limit]


def retrieve(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 7,
    use_semantic: bool = True,
) -> list[SearchResult]:
    ranked_lists: list[tuple[float, list[SearchResult]]] = []
    for kind in ("essay", "tweet"):
        results = lexical_search(connection, query, limit=50, source_type=kind)
        if results:
            ranked_lists.append((1.0, results))
    if use_semantic and api_key() and has_embeddings(connection):
        vector = _query_vector(query)
        for kind in ("essay", "tweet"):
            results = semantic_search(
                connection, query, limit=50, query_vector=vector, source_type=kind
            )
            if results:
                ranked_lists.append((1.15, results))

    if not ranked_lists:
        return []
    by_id = {
        result.chunk_id: result
        for _weight, results in ranked_lists
        for result in results
    }
    fused: dict[int, float] = {}
    for weight, results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + weight / (60 + rank)
    candidates = [
        SearchResult(**{**by_id[chunk_id].__dict__, "score": score})
        for chunk_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)
    ]

    diversified: list[SearchResult] = []
    per_essay: dict[str, int] = {}
    tweet_count = 0
    tweet_limit = max(3, limit // 2)
    for result in candidates:
        if per_essay.get(result.essay_slug, 0) >= 2:
            continue
        if result.source_type == "tweet" and tweet_count >= tweet_limit:
            continue
        diversified.append(result)
        per_essay[result.essay_slug] = per_essay.get(result.essay_slug, 0) + 1
        if result.source_type == "tweet":
            tweet_count += 1
        if len(diversified) >= limit:
            break
    return diversified


def stats(connection: sqlite3.Connection) -> dict[str, int | str | None]:
    essay_count = connection.execute(
        "SELECT COUNT(*) FROM essays WHERE source_type='essay'"
    ).fetchone()[0]
    tweet_count = connection.execute(
        "SELECT COUNT(*) FROM essays WHERE source_type='tweet'"
    ).fetchone()[0]
    chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded_count = connection.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    scraped = connection.execute(
        "SELECT value FROM metadata WHERE key='scraped_at'"
    ).fetchone()
    model = connection.execute(
        "SELECT value FROM metadata WHERE key='embedding_model'"
    ).fetchone()
    return {
        "essays": essay_count,
        "tweets": tweet_count,
        "chunks": chunk_count,
        "embedded_chunks": embedded_count,
        "scraped_at": scraped[0] if scraped else None,
        "embedding_model": model[0] if model else None,
    }
