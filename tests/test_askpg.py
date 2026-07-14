from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from askpg.index import (
    SearchResult,
    chunk_text,
    connect,
    lexical_search,
    sync_corpus,
    sync_tweets,
)
from askpg.images import (
    ImageAttachment,
    ImageError,
    _gif_frame_count,
    load_clipboard_image,
    load_image,
)
from askpg.memory import clear_memory, load_recent_history, memory_count, save_turn
from askpg.rag import RagError, breaks_character, generate_answer, source_prompt
from askpg.retrieval import rerank_sources, rewrite_question
from askpg.scraper import parse_essay, parse_essay_links
from askpg.ui import (
    ThinkingShimmer,
    create_chat_prompt,
    previous_word_delete_count,
    user_prompt,
)


class ScraperTests(unittest.TestCase):
    def test_listing_keeps_unique_visible_essay_links(self):
        anchors = [
            '<a href="index.html"><img src="logo.gif"></a>',
            '<a href="rss.html">RSS</a>',
            '<a href="first.html">First Essay</a>',
            '<a href="first.html">First Essay</a>',
        ]
        anchors.extend(
            f'<a href="essay-{number}.html">Essay {number}</a>' for number in range(150)
        )
        links = parse_essay_links("<html><body>" + "".join(anchors) + "</body></html>")
        self.assertEqual(151, len(links))
        self.assertEqual("first", links[0].slug)
        self.assertNotIn("rss", {link.slug for link in links})

    def test_article_body_and_date_are_extracted(self):
        html = """
        <html><head><title>A Useful Essay</title></head><body>
        <font size="2" face="verdana">July 2026<br><br>
        This is a sufficiently long article body with a concrete argument.
        It contains enough words to make the parser regard it as real content rather
        than navigation. The remaining sentences repeat useful evidence for this test.
        Good tools should be small, direct, legible, and easy to verify by their users.
        </font></body></html>
        """
        title, published, body = parse_essay(html, "Fallback")
        self.assertEqual("A Useful Essay", title)
        self.assertEqual("July 2026", published)
        self.assertIn("Good tools", body)


class IndexTests(unittest.TestCase):
    def test_chunks_overlap(self):
        words = [f"w{number}" for number in range(1000)]
        chunks = chunk_text(" ".join(words), size=100, overlap=20)
        self.assertGreater(len(chunks), 10)
        self.assertEqual(chunks[0].split()[-20:], chunks[1].split()[:20])

    def test_essay_chunks_respect_section_boundaries(self):
        text = (
            "Opening\n\n"
            + "The opening idea has enough detail to stand on its own. " * 12
            + "\n\nProblems\n\n"
            + "A real problem is better than a plausible invention. " * 18
            + "\n\nWell\n\n"
            + "A small number of users should want the first version urgently. " * 18
        )
        chunks = chunk_text(text, size=90, overlap=20, max_size=130)
        self.assertTrue(any(chunk.startswith("Section: Problems") for chunk in chunks))
        self.assertTrue(any(chunk.startswith("Section: Well") for chunk in chunks))
        self.assertFalse(any("Section: Problems" in chunk and "Section: Well" in chunk for chunk in chunks))

    def test_sync_and_lexical_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            essays = root / "essays"
            essays.mkdir()
            body = (
                "Startups should talk to users and make something people want. " * 80
                + "Organic growth is evidence that users genuinely need the product."
            )
            essay_file = essays / "startup.md"
            essay_file.write_text(
                "---\ntitle: \"Startup\"\n---\n\n# Startup\n\nSource: https://example.test\n\n"
                + body,
                encoding="utf-8",
            )
            manifest = {
                "scraped_at": "2026-07-14T00:00:00+00:00",
                "failures": [],
                "essays": [
                    {
                        "slug": "startup",
                        "title": "Startup",
                        "url": "https://example.test",
                        "published": "July 2026",
                        "filename": "essays/startup.md",
                        "sha256": hashlib.sha256(body.encode()).hexdigest(),
                        "word_count": len(body.split()),
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            connection = connect(root / "test.sqlite3")
            try:
                essays_count, chunks_count = sync_corpus(
                    connection, manifest_path=manifest_path
                )
                results = lexical_search(connection, "organic growth users")
            finally:
                connection.close()
            self.assertEqual(1, essays_count)
            self.assertGreater(chunks_count, 1)
            self.assertTrue(results)
            self.assertEqual("Startup", results[0].title)

    def test_tweets_are_synced_as_attributed_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tweets = root / "tweets.jsonl"
            tweets.write_text(
                json.dumps(
                    {
                        "id": "123",
                        "text": "Startups should launch quickly and learn from users.",
                        "created_at": "2020-01-02T00:00:00+00:00",
                        "url": "https://x.com/paulg/status/123",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "filename": "tweets.jsonl",
                        "tweet_count": 1,
                        "sha256": "test",
                    }
                ),
                encoding="utf-8",
            )
            connection = connect(root / "test.sqlite3")
            try:
                count, _chunks = sync_tweets(connection, manifest_path=manifest)
                results = lexical_search(
                    connection, "launch users", source_type="tweet"
                )
            finally:
                connection.close()
            self.assertEqual(1, count)
            self.assertEqual("tweet", results[0].source_type)
            self.assertEqual("2020-01-02", results[0].published)


class MemoryTests(unittest.TestCase):
    def test_history_survives_reopening_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory.sqlite3"
            connection = connect(path)
            save_turn(connection, "I am building a compiler.", "Keep it small at first.")
            connection.close()
            reopened = connect(path)
            try:
                history = load_recent_history(reopened)
                count = memory_count(reopened)
                clear_memory(reopened)
                cleared = memory_count(reopened)
            finally:
                reopened.close()
            self.assertEqual(2, count)
            self.assertEqual("I am building a compiler.", history[0]["content"])
            self.assertEqual(0, cleared)


class ImageTests(unittest.TestCase):
    def test_png_is_loaded_as_a_base64_data_url(self):
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "screen shot.png"
            path.write_bytes(png)
            image = load_image(path)

        self.assertEqual("image/png", image.media_type)
        self.assertTrue(image.data_url.startswith("data:image/png;base64,"))
        self.assertEqual(len(png), image.size)

    def test_unsupported_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "not-an-image.txt"
            path.write_text("hello", encoding="utf-8")
            with self.assertRaises(ImageError):
                load_image(path)

    def test_gif_frame_parser_does_not_count_compressed_commas(self):
        static_gif = (
            b"GIF89a\x01\x00\x01\x00\x00\x00\x00"
            b"\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00"
            b"\x02\x01\x2c\x00\x3b"
        )
        animated_gif = static_gif[:-1] + static_gif[13:]
        self.assertEqual(1, _gif_frame_count(static_gif))
        self.assertEqual(2, _gif_frame_count(animated_gif))

    def test_macos_clipboard_image_is_loaded_and_labeled(self):
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )

        def export_image(command, **_kwargs):
            Path(command[-1]).write_bytes(png)
            return type("Result", (), {"returncode": 0})()

        with (
            patch("askpg.images.sys.platform", "darwin"),
            patch("askpg.images.shutil.which", return_value=None),
            patch("askpg.images.subprocess.run", side_effect=export_image),
        ):
            image = load_clipboard_image()

        self.assertEqual("clipboard image", image.label)
        self.assertEqual("image/png", image.media_type)
        self.assertTrue(image.data_url.startswith("data:image/png;base64,"))


class RagTests(unittest.TestCase):
    def setUp(self):
        self.source = SearchResult(
            chunk_id=1,
            essay_slug="ideas",
            chunk_index=0,
            title="Ideas",
            url="https://example.test/ideas",
            text="Build something you want and talk to users.",
            score=1.0,
        )

    def test_source_markers_are_stable(self):
        prompt = source_prompt("What should I build?", [self.source])
        self.assertIn('<source id="S1">', prompt)
        self.assertIn("https://example.test/ideas", prompt)

    def test_stating_his_own_name_does_not_break_character(self):
        self.assertFalse(breaks_character("I'm Paul Graham."))
        self.assertTrue(breaks_character("Paul Graham would probably say to launch."))

    def test_history_is_passed_before_fresh_retrieval_context(self):
        class Response:
            output_text = "Start by building. [S1]"

        class Responses:
            def __init__(self):
                self.parameters = None

            def create(self, **parameters):
                self.parameters = parameters
                return Response()

        class Client:
            def __init__(self):
                self.responses = Responses()

        client = Client()
        answer = generate_answer(
            client,
            "And what next?",
            [self.source],
            history=[
                {"role": "user", "content": "What should I build?"},
                {"role": "assistant", "content": "Build something you want. [S1]"},
            ],
        )
        self.assertEqual("Start by building. [S1]", answer)
        sent = client.responses.parameters["input"]
        self.assertEqual("What should I build?", sent[0]["content"])
        self.assertIn("And what next?", sent[-1]["content"])
        self.assertFalse(client.responses.parameters["store"])

    def test_attached_image_is_sent_with_the_current_question(self):
        class Response:
            output_text = "The error is in the assumption behind that screen."

        class Responses:
            def __init__(self):
                self.parameters = None

            def create(self, **parameters):
                self.parameters = parameters
                return Response()

        class Client:
            def __init__(self):
                self.responses = Responses()

        image = ImageAttachment(
            path=Path("/tmp/screenshot.png"),
            media_type="image/png",
            data_url="data:image/png;base64,AAAA",
            size=3,
        )
        client = Client()
        generate_answer(
            client,
            "What do you make of this?",
            [self.source],
            images=[image],
        )

        content = client.responses.parameters["input"][-1]["content"]
        self.assertEqual("input_text", content[0]["type"])
        self.assertIn("What do you make of this?", content[0]["text"])
        self.assertEqual("input_image", content[1]["type"])
        self.assertEqual(image.data_url, content[1]["image_url"])
        self.assertEqual("original", content[1]["detail"])

    def test_character_break_is_hidden_and_regenerated(self):
        class Response:
            def __init__(self, output_text):
                self.output_text = output_text

        class Responses:
            def __init__(self):
                self.calls = 0

            def create(self, **_parameters):
                self.calls += 1
                if self.calls == 1:
                    return Response("I'm an AI simulation based on Paul Graham.")
                return Response("I'd rather keep that private.")

        class Client:
            def __init__(self):
                self.responses = Responses()

        client = Client()
        emitted = []
        answer = generate_answer(
            client,
            "Tell me about your private finances.",
            [self.source],
            on_delta=emitted.append,
        )
        self.assertEqual(2, client.responses.calls)
        self.assertEqual("I'd rather keep that private.", answer)
        self.assertEqual(answer, "".join(emitted))
        self.assertFalse(breaks_character(answer))

    def test_repeated_character_breaks_raise_without_substituting_an_answer(self):
        class Response:
            output_text = "I'm an AI simulation based on Paul Graham."

        class Responses:
            def __init__(self):
                self.calls = 0

            def create(self, **_parameters):
                self.calls += 1
                return Response()

        class Client:
            def __init__(self):
                self.responses = Responses()

        client = Client()
        with self.assertRaises(RagError):
            generate_answer(client, "What do you think?", [self.source])
        self.assertEqual(3, client.responses.calls)

    def test_immersive_mode_removes_accidental_source_markers(self):
        class Response:
            output_text = "Build the smallest useful version first. [S1]"

        class Responses:
            def create(self, **_parameters):
                return Response()

        class Client:
            responses = Responses()

        answer = generate_answer(
            Client(),
            "What should I do first?",
            [self.source],
            cite_sources=False,
        )
        self.assertEqual("Build the smallest useful version first.", answer)


class RetrievalTests(unittest.TestCase):
    def test_rewrite_and_rerank_use_structured_results(self):
        class Response:
            def __init__(self, output_text):
                self.output_text = output_text

        class Responses:
            def __init__(self):
                self.calls = 0

            def create(self, **_parameters):
                self.calls += 1
                if self.calls == 1:
                    return Response('{"search_query":"how to validate a startup idea"}')
                return Response('{"ranked_ids":[2,1]}')

        class Client:
            def __init__(self):
                self.responses = Responses()

        client = Client()
        rewritten = rewrite_question(
            client,
            "How do I test it?",
            history=[{"role": "user", "content": "I have a startup idea."}],
        )
        candidates = [
            SearchResult(1, "one", 0, "One", "https://one", "Broad text", 1.0),
            SearchResult(2, "two", 0, "Two", "https://two", "Direct text", 0.9),
        ]
        ranked = rerank_sources(client, "How do I test it?", candidates, limit=2)
        self.assertEqual("how to validate a startup idea", rewritten)
        self.assertEqual([2, 1], [result.chunk_id for result in ranked])

    def test_rewrite_uses_the_attached_image(self):
        class Response:
            output_text = '{"search_query":"debugging an early product interface"}'

        class Responses:
            def __init__(self):
                self.parameters = None

            def create(self, **parameters):
                self.parameters = parameters
                return Response()

        class Client:
            def __init__(self):
                self.responses = Responses()

        image = ImageAttachment(
            path=Path("/tmp/screenshot.png"),
            media_type="image/png",
            data_url="data:image/png;base64,AAAA",
            size=3,
        )
        client = Client()
        rewritten = rewrite_question(
            client,
            "What is wrong here?",
            images=[image],
        )

        self.assertEqual("debugging an early product interface", rewritten)
        sent = client.responses.parameters["input"]
        self.assertEqual("input_text", sent[0]["content"][0]["type"])
        self.assertEqual(image.data_url, sent[0]["content"][1]["image_url"])


class UiTests(unittest.TestCase):
    def test_control_v_adds_and_backspace_removes_an_attachment(self):
        actions = []
        with create_pipe_input() as input_pipe:
            with create_app_session(input=input_pipe, output=DummyOutput()):
                session = create_chat_prompt(
                    on_image_paste=lambda: actions.append("paste"),
                    on_attachment_delete=lambda: actions.append("delete"),
                )
                input_pipe.send_text("\x16\x7fhello\r")
                entered = session.prompt("")

        self.assertEqual("hello", entered)
        self.assertEqual(["paste", "delete"], actions)

    def test_attachment_markers_are_part_of_the_prompt_not_the_buffer(self):
        rendered = "".join(fragment[1] for fragment in user_prompt(2))
        self.assertEqual("You: [attach 1] [attach 2] ", rendered)

    def test_option_delete_only_changes_the_user_buffer(self):
        self.assertEqual(5, previous_word_delete_count("hello world"))
        self.assertEqual(8, previous_word_delete_count("hello world   "))
        self.assertEqual(0, previous_word_delete_count(""))

    def test_thinking_shimmer_moves_a_three_character_band(self):
        shimmer = ThinkingShimmer()
        first = shimmer.frame(0)
        second = shimmer.frame(1)
        self.assertEqual("Thinking…", first.plain)
        self.assertEqual(first.plain, second.plain)
        self.assertNotEqual(first.spans, second.spans)
        self.assertEqual(first.spans, shimmer.frame(len(shimmer.label) + 2).spans)


if __name__ == "__main__":
    unittest.main()
