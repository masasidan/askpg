from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from .config import (
    CHAT_MODEL,
    DB_PATH,
    EMBEDDING_MODEL,
    MANIFEST_PATH,
    REASONING_EFFORT,
    RETRIEVAL_MODEL,
    TWEETS_MANIFEST_PATH,
    api_key,
)
from .index import (
    IndexError,
    connect,
    embed_missing,
    retrieve,
    stats,
    sync_corpus,
    sync_tweets,
)
from .images import (
    ImageAttachment,
    ImageError,
    load_clipboard_image,
    load_images,
    validate_image_collection,
)
from .memory import (
    clear_memory,
    load_recent_history,
    memory_count,
    save_turn,
    search_memories,
)
from .rag import RagError, generate_answer
from .retrieval import rerank_sources, rewrite_question
from .scraper import ScrapeError, scrape_all
from .tweets import TweetScrapeError, scrape_tweets
from .ui import create_chat_prompt, thinking, user_prompt


console = Console(highlight=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="askpg",
        description="Chat with Paul over a local RAG index of his essays and tweets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser("scrape", help="Download all essays from paulgraham.com")
    scrape.add_argument("--delay", type=float, default=0.15, help="Delay between requests")

    scrape_tweets_parser = subparsers.add_parser(
        "scrape-tweets", help="Download and normalize the historical tweet archive"
    )
    scrape_tweets_parser.add_argument(
        "--refresh", action="store_true", help="Redownload the source archive"
    )

    index = subparsers.add_parser("index", help="Build/update the local search index")
    index.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Build only the free local full-text index",
    )

    subparsers.add_parser("embed", help="Add missing OpenAI semantic embeddings")

    search = subparsers.add_parser("search", help="Search the corpus without generating an answer")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=7)
    search.add_argument("--lexical", action="store_true", help="Disable semantic retrieval")

    ask = subparsers.add_parser("ask", help="Ask one question")
    ask.add_argument("question")
    ask.add_argument("--lexical", action="store_true", help="Disable semantic retrieval")
    ask.add_argument(
        "--research", action="store_true", help="Show inline citations and the source list"
    )
    ask.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="PATH",
        help="Attach a screenshot or image (repeat for multiple images)",
    )

    chat = subparsers.add_parser("chat", help="Start an interactive terminal conversation")
    chat.add_argument("--lexical", action="store_true", help="Disable semantic retrieval")
    chat.add_argument(
        "--research", action="store_true", help="Start with citations and sources visible"
    )

    subparsers.add_parser("stats", help="Show corpus and index statistics")
    subparsers.add_parser("doctor", help="Check local setup without exposing secrets")
    return parser


def _ready_connection():
    if not DB_PATH.exists():
        raise IndexError("Search index not found. Run `askpg index` first.")
    connection = connect()
    if stats(connection)["chunks"] == 0:
        connection.close()
        raise IndexError("Search index is empty. Run `askpg index` first.")
    return connection


def _openai_client():
    key = api_key()
    if not key:
        raise IndexError(
            "OPENAI_API_KEY is not set. Export it in your shell, then rerun this command."
        )
    from openai import OpenAI

    return OpenAI(api_key=key)


def _print_sources(sources) -> None:
    console.print("\n[dim]Sources:[/dim]")
    for number, source in enumerate(sources, start=1):
        label = "tweet" if source.source_type == "tweet" else "essay"
        console.print(
            f"  S{number}. [{label}] {source.title} — {source.url}", markup=False
        )


def _answer(
    connection,
    client,
    question,
    history,
    *,
    lexical: bool,
    research: bool,
    images: tuple[ImageAttachment, ...] = (),
):
    console.print("\n[bold cyan]Paul[/bold cyan]")
    with thinking(console):
        memories = search_memories(connection, question)
        retrieval_query = rewrite_question(
            client,
            question,
            history=history,
            memories=memories,
            images=images,
        )
        candidates = retrieve(
            connection,
            retrieval_query,
            limit=28,
            use_semantic=not lexical,
        )
        if not candidates:
            raise IndexError("No relevant passages were found for that question.")
        sources = rerank_sources(client, retrieval_query, candidates, limit=7)
        answer = generate_answer(
            client,
            question,
            sources,
            history=history,
            memories=memories,
            images=images,
            cite_sources=research,
        )
    console.print(answer, markup=False)
    if research:
        _print_sources(sources)
    return answer, sources


def command_scrape(args) -> None:
    manifest = scrape_all(delay=max(0.0, args.delay), progress=console.print)
    console.print(
        f"[green]Complete:[/green] {manifest['essay_count']} essays saved under "
        f"{MANIFEST_PATH.parent / 'essays'}"
    )


def command_scrape_tweets(args) -> None:
    manifest = scrape_tweets(progress=console.print, refresh=args.refresh)
    console.print(
        f"[green]Complete:[/green] {manifest['tweet_count']:,} authored tweets "
        f"({manifest['first_date']} to {manifest['last_date']})"
    )


def command_index(args) -> None:
    connection = connect()
    try:
        essay_count, _chunk_count = sync_corpus(connection)
        tweet_count, chunk_count = sync_tweets(connection)
        console.print(
            f"Local index: {essay_count} essays, {tweet_count:,} tweets, "
            f"{chunk_count:,} passages"
        )
        if not TWEETS_MANIFEST_PATH.exists():
            console.print("Historical tweets are not downloaded; run `askpg scrape-tweets`.")
        if args.skip_embeddings:
            console.print("Skipped semantic embeddings; full-text retrieval is ready.")
        elif not api_key():
            console.print(
                "OPENAI_API_KEY is not set, so only the local full-text index was built. "
                "Set the key and run `askpg embed` for semantic retrieval."
            )
        else:
            count = embed_missing(connection, progress=console.print)
            console.print(f"[green]Semantic index ready.[/green] Added {count} embeddings.")
    finally:
        connection.close()


def command_embed(_args) -> None:
    connection = _ready_connection()
    try:
        count = embed_missing(connection, progress=console.print)
        console.print(f"[green]Semantic index ready.[/green] Added {count} embeddings.")
    finally:
        connection.close()


def command_search(args) -> None:
    connection = _ready_connection()
    try:
        results = retrieve(
            connection, args.query, limit=max(1, args.limit), use_semantic=not args.lexical
        )
    finally:
        connection.close()
    for position, result in enumerate(results, start=1):
        excerpt = result.text[:380].rsplit(" ", 1)[0] + "…"
        console.print(f"\n[bold]{position}. {result.title}[/bold]")
        console.print(result.url, markup=False)
        console.print(excerpt, markup=False)


def command_ask(args) -> None:
    connection = _ready_connection()
    try:
        client = _openai_client()
        history = load_recent_history(connection)
        images = tuple(load_images(args.image))
        answer, _sources = _answer(
            connection,
            client,
            args.question,
            history,
            lexical=args.lexical,
            research=args.research,
            images=images,
        )
        save_turn(connection, args.question, answer)
    finally:
        connection.close()


def command_chat(args) -> None:
    pending_images: list[ImageAttachment] = []

    def queue_clipboard_image() -> None:
        image = load_clipboard_image()
        combined = [*pending_images, image]
        validate_image_collection(combined)
        pending_images.append(image)

    def remove_last_attachment() -> None:
        if pending_images:
            pending_images.pop()

    chat_prompt = create_chat_prompt(
        on_image_paste=queue_clipboard_image,
        on_attachment_delete=remove_last_attachment,
    )
    connection = _ready_connection()
    client = _openai_client()
    history = load_recent_history(connection)
    last_sources = []
    research = bool(args.research)
    remembered_turns = memory_count(connection) // 2
    console.print(
        f"[bold cyan]Paul[/bold cyan] — continuing with {remembered_turns} remembered turns\n"
        "Commands: /sources, /memory, /research, /immersive, /clear, /quit"
    )
    try:
        while True:
            try:
                console.print()
                question = chat_prompt.prompt(
                    lambda: user_prompt(len(pending_images))
                ).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nBye.")
                break
            if not question:
                continue
            command = question.lower()
            if command in {"/quit", "/exit", "quit", "exit"}:
                console.print("Bye.")
                break
            if command == "/clear":
                clear_memory(connection)
                history.clear()
                last_sources = []
                console.print("Persistent conversation memory cleared.")
                continue
            if command == "/memory":
                console.print(
                    f"Remembering {memory_count(connection) // 2} conversation turns."
                )
                continue
            if command == "/research":
                research = True
                console.print("Research mode: citations and source lists are visible.")
                continue
            if command == "/immersive":
                research = False
                console.print("Immersive mode: sources are hidden unless you use /sources.")
                continue
            if command == "/sources":
                if last_sources:
                    _print_sources(last_sources)
                else:
                    console.print("No sources retrieved yet.")
                continue
            answer, last_sources = _answer(
                connection,
                client,
                question,
                history,
                lexical=args.lexical,
                research=research,
                images=tuple(pending_images),
            )
            pending_images.clear()
            save_turn(connection, question, answer)
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
            history = history[-12:]
    finally:
        connection.close()


def command_stats(_args) -> None:
    connection = _ready_connection()
    try:
        information = stats(connection)
    finally:
        connection.close()
    table = Table(show_header=False)
    for key, value in information.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)


def command_doctor(_args) -> None:
    manifest_count = 0
    if MANIFEST_PATH.exists():
        try:
            manifest_count = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get(
                "essay_count", 0
            )
        except (json.JSONDecodeError, OSError):
            pass
    console.print(f"Project data: {MANIFEST_PATH.parent}")
    console.print(f"Corpus manifest: {'ready' if manifest_count else 'missing'} ({manifest_count} essays)")
    console.print(f"Search database: {'ready' if DB_PATH.exists() else 'missing'}")
    console.print(f"OpenAI key: {'set' if api_key() else 'not set'}")
    console.print(f"Chat model: {CHAT_MODEL}")
    console.print(f"Reasoning effort: {REASONING_EFFORT}")
    console.print(f"Embedding model: {EMBEDDING_MODEL}")
    console.print(f"Retrieval model: {RETRIEVAL_MODEL}")
    console.print(
        f"Tweet archive: {'ready' if TWEETS_MANIFEST_PATH.exists() else 'missing'}"
    )


COMMANDS = {
    "scrape": command_scrape,
    "scrape-tweets": command_scrape_tweets,
    "index": command_index,
    "embed": command_embed,
    "search": command_search,
    "ask": command_ask,
    "chat": command_chat,
    "stats": command_stats,
    "doctor": command_doctor,
}


def main() -> None:
    args = _parser().parse_args()
    try:
        COMMANDS[args.command](args)
    except (ImageError, IndexError, RagError, ScrapeError, TweetScrapeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        console.print("\nCancelled.")
        raise SystemExit(130)
