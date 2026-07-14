from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import TWEETS_DIR, TWEETS_MANIFEST_PATH, TWEETS_PATH, ensure_data_dirs


ARCHIVE_PAGE_URL = "https://huggingface.co/datasets/aaahmet/paulg-tweets"
ARCHIVE_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/aaahmet/paulg-tweets/resolve/main/"
    "tweets.jsonl?download=true"
)
USER_AGENT = "AskPG/0.2 (+local personal research RAG)"


class TweetScrapeError(RuntimeError):
    pass


def _download(destination: Path, *, progress: Callable[[str], None]) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(
        ARCHIVE_DOWNLOAD_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/jsonl"},
    )
    try:
        with urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            next_report = 5 * 1024 * 1024
            while block := response.read(1024 * 1024):
                handle.write(block)
                written += len(block)
                if written >= next_report:
                    if total:
                        progress(f"Downloaded {written / 1_048_576:.0f}/{total / 1_048_576:.0f} MB")
                    else:
                        progress(f"Downloaded {written / 1_048_576:.0f} MB")
                    next_report += 5 * 1024 * 1024
        os.replace(temporary, destination)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise TweetScrapeError(f"Could not download the public tweet archive: {exc}") from exc


def scrape_tweets(
    *,
    progress: Callable[[str], None] | None = None,
    refresh: bool = False,
) -> dict:
    """Download and normalize Paul Graham's public authored-tweet archive."""
    ensure_data_dirs()
    emit = progress or (lambda _message: None)
    raw_path = TWEETS_DIR / "archive.jsonl"
    if refresh or not raw_path.exists():
        emit(f"Fetching historical posts from {ARCHIVE_PAGE_URL}")
        _download(raw_path, progress=emit)
    else:
        emit("Using the previously downloaded historical archive")

    temporary = TWEETS_PATH.with_suffix(".jsonl.part")
    source_count = 0
    kept_count = 0
    retweet_count = 0
    invalid_count = 0
    first_date: str | None = None
    last_date: str | None = None
    digest = hashlib.sha256()
    seen: set[str] = set()

    with raw_path.open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            source_count += 1
            try:
                record = json.loads(line)
                tweet_id = str(record["id"])
                text = " ".join(str(record.get("text") or "").split())
                created_at = str(record["created_at"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                invalid_count += 1
                continue
            if tweet_id in seen or not text or not any(character.isalpha() for character in text):
                invalid_count += 1
                continue
            if str(record.get("type") or "").lower() == "retweet":
                retweet_count += 1
                continue
            seen.add(tweet_id)
            normalized = {
                "id": tweet_id,
                "text": text,
                "created_at": created_at,
                "url": record.get("url") or f"https://x.com/paulg/status/{tweet_id}",
                "type": record.get("type") or "original",
                "conversation_id": record.get("conversation_id"),
                "in_reply_to_tweet_id": record.get("in_reply_to_tweet_id"),
                "in_reply_to_username": record.get("in_reply_to_username"),
            }
            encoded = (
                json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            destination.write(encoded.decode("utf-8"))
            digest.update(encoded)
            kept_count += 1
            date = created_at[:10]
            first_date = min(first_date, date) if first_date else date
            last_date = max(last_date, date) if last_date else date
            if line_number % 10000 == 0:
                emit(f"Normalized {line_number:,} archive records")

    if kept_count < 1000:
        temporary.unlink(missing_ok=True)
        raise TweetScrapeError(
            f"Only found {kept_count} usable posts; the archive format may have changed."
        )
    os.replace(temporary, TWEETS_PATH)
    manifest = {
        "source_dataset_url": ARCHIVE_PAGE_URL,
        "source_download_url": ARCHIVE_DOWNLOAD_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "filename": TWEETS_PATH.name,
        "tweet_count": kept_count,
        "source_record_count": source_count,
        "excluded_retweets": retweet_count,
        "excluded_invalid_or_duplicate": invalid_count,
        "first_date": first_date,
        "last_date": last_date,
        "sha256": digest.hexdigest(),
        "redistribution_note": (
            "Local research copy. Preserve tweet URLs and review X/Twitter terms before "
            "redistributing the archive."
        ),
    }
    TWEETS_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    emit(
        f"Saved {kept_count:,} authored posts spanning {first_date} through {last_date}"
    )
    return manifest
