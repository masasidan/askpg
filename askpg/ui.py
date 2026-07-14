from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from rich.console import Console, RenderableType
from rich.live import Live
from rich.text import Text


USER_PROMPT = FormattedText([("bold ansigreen", "You:"), ("", " ")])


def previous_word_delete_count(text_before_cursor: str) -> int:
    """Count user-buffer characters removed by Option/Alt+Delete."""
    remaining = text_before_cursor
    while remaining and remaining[-1].isspace():
        remaining = remaining[:-1]
    while remaining and not remaining[-1].isspace():
        remaining = remaining[:-1]
    return len(text_before_cursor) - len(remaining)


def _delete_previous_word(event: KeyPressEvent) -> None:
    buffer = event.current_buffer
    count = previous_word_delete_count(buffer.document.text_before_cursor)
    if count:
        buffer.delete_before_cursor(count=count)


def create_chat_prompt() -> PromptSession[str]:
    """Create an input editor whose prompt is separate from the editable buffer."""
    bindings = KeyBindings()
    bindings.add(Keys.Escape, Keys.ControlH, eager=True)(_delete_previous_word)
    return PromptSession(key_bindings=bindings)


class ThinkingShimmer:
    """A moving white band across dim terminal text."""

    def __init__(
        self,
        label: str = "Thinking…",
        *,
        band_width: int = 3,
        seconds_per_step: float = 0.105,
        gap_steps: int = 2,
    ) -> None:
        self.label = label
        self.band_width = max(1, band_width)
        self.seconds_per_step = seconds_per_step
        self.gap_steps = max(0, gap_steps)
        self.started_at = time.monotonic()

    def frame(self, step: int) -> Text:
        head = step % (len(self.label) + self.gap_steps)
        result = Text()
        for index, character in enumerate(self.label):
            if head <= index < head + self.band_width:
                style = "bold #ffffff" if index == head + 1 else "#eeeeee"
            else:
                style = "#727277"
            result.append(character, style=style)
        return result

    def __rich__(self) -> RenderableType:
        elapsed = time.monotonic() - self.started_at
        step = int(elapsed / self.seconds_per_step)
        return self.frame(step)


@contextmanager
def thinking(console: Console) -> Iterator[None]:
    """Show the shimmer only in an interactive terminal and clear it afterward."""
    if not console.is_terminal or getattr(console, "is_dumb_terminal", False):
        yield
        return
    shimmer = ThinkingShimmer()
    with Live(
        shimmer,
        console=console,
        refresh_per_second=16,
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    ):
        yield
