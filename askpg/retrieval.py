from __future__ import annotations

import json
from collections.abc import Sequence

from .config import RETRIEVAL_MODEL
from .index import SearchResult


def _structured_response(client, *, instructions: str, prompt: str, schema: dict) -> dict:
    response = client.responses.create(
        model=RETRIEVAL_MODEL,
        instructions=instructions,
        input=prompt,
        reasoning={"effort": "none"},
        text={
            "format": {
                "type": "json_schema",
                "name": "retrieval_result",
                "strict": True,
                "schema": schema,
            }
        },
        max_output_tokens=600,
        store=False,
    )
    return json.loads(response.output_text)


def rewrite_question(
    client,
    question: str,
    *,
    history: Sequence[dict[str, str]] = (),
    memories: Sequence[str] = (),
) -> str:
    recent = "\n".join(
        f"{message['role']}: {message['content']}" for message in history[-8:]
    )
    remembered = "\n".join(memories[:5])
    prompt = (
        f"Conversation:\n{recent or '(none)'}\n\n"
        f"Relevant older context:\n{remembered or '(none)'}\n\n"
        f"Newest question:\n{question}"
    )
    schema = {
        "type": "object",
        "properties": {"search_query": {"type": "string"}},
        "required": ["search_query"],
        "additionalProperties": False,
    }
    try:
        parsed = _structured_response(
            client,
            instructions=(
                "Rewrite the newest conversational question as one precise standalone search "
                "query for a corpus of Paul Graham essays and tweets. Resolve pronouns and "
                "implicit references from the conversation. Preserve names, constraints, and "
                "the user's actual intent. Do not answer the question."
            ),
            prompt=prompt,
            schema=schema,
        )
        rewritten = str(parsed.get("search_query") or "").strip()
        if rewritten:
            return rewritten
    except Exception:
        pass
    previous_user = next(
        (
            message["content"]
            for message in reversed(history)
            if message.get("role") == "user"
        ),
        "",
    )
    return f"{previous_user} {question}".strip()


def _balanced_selection(
    ranked: Sequence[SearchResult],
    *,
    limit: int,
) -> list[SearchResult]:
    tweet_limit = max(3, limit // 2)
    selected: list[SearchResult] = []
    per_document: dict[str, int] = {}
    tweets = 0
    for result in ranked:
        if per_document.get(result.essay_slug, 0) >= 2:
            continue
        if result.source_type == "tweet" and tweets >= tweet_limit:
            continue
        selected.append(result)
        per_document[result.essay_slug] = per_document.get(result.essay_slug, 0) + 1
        if result.source_type == "tweet":
            tweets += 1
        if len(selected) >= limit:
            break
    return selected


def rerank_sources(
    client,
    question: str,
    candidates: Sequence[SearchResult],
    *,
    limit: int = 7,
) -> list[SearchResult]:
    if not candidates:
        return []
    passages = []
    for result in candidates:
        excerpt = result.text[:1200]
        passages.append(
            f"ID {result.chunk_id} | {result.source_type} | {result.title} | "
            f"{result.published or 'unknown date'}\n{excerpt}"
        )
    prompt = f"Question:\n{question}\n\nCandidate passages:\n\n" + "\n\n".join(passages)
    schema = {
        "type": "object",
        "properties": {
            "ranked_ids": {
                "type": "array",
                "items": {"type": "integer"},
            }
        },
        "required": ["ranked_ids"],
        "additionalProperties": False,
    }
    by_id = {candidate.chunk_id: candidate for candidate in candidates}
    ranked: list[SearchResult] = []
    try:
        parsed = _structured_response(
            client,
            instructions=(
                "Rank passages by how directly and reliably they help answer the question. "
                "Prefer exact topical evidence over merely similar language. Essays establish "
                "developed views; tweets add concise views and conversational voice. Return "
                "only candidate IDs, most useful first, without duplicates."
            ),
            prompt=prompt,
            schema=schema,
        )
        seen: set[int] = set()
        for raw_id in parsed.get("ranked_ids", []):
            chunk_id = int(raw_id)
            if chunk_id in by_id and chunk_id not in seen:
                ranked.append(by_id[chunk_id])
                seen.add(chunk_id)
    except Exception:
        ranked = []
    ranked_ids = {result.chunk_id for result in ranked}
    ranked.extend(result for result in candidates if result.chunk_id not in ranked_ids)
    return _balanced_selection(ranked, limit=limit)
