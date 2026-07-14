from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup

from .config import ESSAYS_DIR, MANIFEST_PATH, ensure_data_dirs


INDEX_URL = "https://paulgraham.com/articles.html"
ROBOTS_URL = "https://paulgraham.com/robots.txt"
USER_AGENT = "AskPG/0.1 (+local personal research RAG)"
EXCLUDED_PAGES = {"articles.html", "index.html", "rss.html"}
DATE_RE = re.compile(
    r"^(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Spring|Summer|Fall|Winter)\s+(?:19|20)\d{2}$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EssayLink:
    slug: str
    title: str
    url: str


@dataclass(frozen=True)
class EssayRecord:
    slug: str
    title: str
    url: str
    published: str | None
    filename: str
    sha256: str
    word_count: int


class ScrapeError(RuntimeError):
    pass


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "www.paulgraham.com":
        host = "paulgraham.com"
    return urlunparse(("https", host, parsed.path, "", "", ""))


def fetch_html(url: str, *, retries: int = 3, timeout: float = 30.0) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
            with urlopen(request, timeout=timeout) as response:
                encoding = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(encoding, errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise ScrapeError(f"Could not fetch {url}: {last_error}")


def parse_essay_links(index_html: str, base_url: str = INDEX_URL) -> list[EssayLink]:
    soup = BeautifulSoup(index_html, "html.parser")
    links: list[EssayLink] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        title = " ".join(anchor.get_text(" ", strip=True).split())
        if not title:
            continue
        absolute = _canonical_url(urljoin(base_url, anchor["href"]))
        parsed = urlparse(absolute)
        filename = Path(parsed.path).name
        if parsed.hostname != "paulgraham.com" or not filename.endswith(".html"):
            continue
        if filename in EXCLUDED_PAGES:
            continue
        slug = Path(filename).stem
        if slug in seen:
            continue
        seen.add(slug)
        links.append(EssayLink(slug=slug, title=title, url=absolute))

    if len(links) < 150:
        raise ScrapeError(
            f"Only found {len(links)} essay links; the source page structure may have changed."
        )
    return links


def _clean_text(node) -> str:
    fragment = BeautifulSoup(str(node), "html.parser")
    for unwanted in fragment.find_all(["script", "style", "noscript"]):
        unwanted.decompose()
    for br in fragment.find_all("br"):
        br.replace_with("\n")
    text = fragment.get_text("", strip=False).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if line:
            cleaned.append(line)
        elif cleaned and cleaned[-1] != "":
            cleaned.append("")
    return "\n".join(cleaned).strip()


def parse_essay(essay_html: str, fallback_title: str) -> tuple[str, str | None, str]:
    soup = BeautifulSoup(essay_html, "html.parser")
    title = fallback_title
    if soup.title:
        candidate_title = " ".join(soup.title.get_text(" ", strip=True).split())
        if candidate_title:
            title = candidate_title

    candidates = []
    for node in soup.find_all("font"):
        text = _clean_text(node)
        if len(text) >= 100:
            candidates.append((len(text), text))
    if not candidates:
        for node in soup.find_all("td"):
            text = _clean_text(node)
            if len(text) >= 100:
                candidates.append((len(text), text))
    if not candidates:
        raise ScrapeError(f"Could not locate the article body for {title}")

    body = max(candidates, key=lambda item: item[0])[1]
    published = next((line for line in body.splitlines()[:8] if DATE_RE.match(line)), None)
    if len(body.split()) < 40:
        raise ScrapeError(f"Parsed article body is suspiciously short for {title}")
    return title, published, body


def _markdown(record: EssayRecord, body: str) -> str:
    published = json.dumps(record.published) if record.published else "null"
    return (
        "---\n"
        f"title: {json.dumps(record.title, ensure_ascii=False)}\n"
        f"source: {json.dumps(record.url)}\n"
        f"published: {published}\n"
        "author: \"Paul Graham\"\n"
        "---\n\n"
        f"# {record.title}\n\n"
        f"Source: {record.url}\n\n"
        f"{body}\n"
    )


def _robots_parser() -> RobotFileParser:
    parser = RobotFileParser()
    parser.set_url(ROBOTS_URL)
    parser.read()
    return parser


def scrape_all(
    *,
    delay: float = 0.15,
    progress: Callable[[str], None] | None = None,
) -> dict:
    ensure_data_dirs()
    emit = progress or (lambda _: None)

    robots = _robots_parser()
    if not robots.can_fetch(USER_AGENT, INDEX_URL):
        raise ScrapeError(f"robots.txt does not permit fetching {INDEX_URL}")

    emit(f"Fetching essay index: {INDEX_URL}")
    index_html = fetch_html(INDEX_URL)
    links = parse_essay_links(index_html)
    emit(f"Found {len(links)} canonical essay pages")

    records: list[EssayRecord] = []
    failures: list[dict[str, str]] = []
    for position, link in enumerate(links, start=1):
        try:
            if not robots.can_fetch(USER_AGENT, link.url):
                raise ScrapeError("Blocked by robots.txt")
            html = fetch_html(link.url)
            title, published, body = parse_essay(html, link.title)
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            record = EssayRecord(
                slug=link.slug,
                title=title,
                url=link.url,
                published=published,
                filename=f"essays/{link.slug}.md",
                sha256=digest,
                word_count=len(body.split()),
            )
            destination = ESSAYS_DIR / f"{link.slug}.md"
            destination.write_text(_markdown(record, body), encoding="utf-8")
            records.append(record)
            if position == 1 or position % 10 == 0 or position == len(links):
                emit(f"[{position}/{len(links)}] {title}")
        except Exception as exc:  # Continue so the manifest reports every failed URL.
            failures.append({"url": link.url, "error": str(exc)})
            emit(f"FAILED [{position}/{len(links)}] {link.url}: {exc}")
        if delay > 0 and position < len(links):
            time.sleep(delay)

    manifest = {
        "source_index_url": INDEX_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "essay_count": len(records),
        "expected_count": len(links),
        "failures": failures,
        "essays": [asdict(record) for record in records],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise ScrapeError(
            f"Scraped {len(records)} of {len(links)} essays; see {MANIFEST_PATH} for failures."
        )
    return manifest
