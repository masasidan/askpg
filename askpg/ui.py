from __future__ import annotations

import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Iterator

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from rich.console import Console, RenderableType
from rich.live import Live
from rich.text import Text

from .images import ClipboardImageUnavailable, ImageError


def user_prompt(attachment_count: int = 0) -> FormattedText:
    fragments = [("bold ansigreen", "You:"), ("", " ")]
    for number in range(1, attachment_count + 1):
        fragments.append(("fg:#777777", f"[attach {number}] "))
    return FormattedText(fragments)


USER_PROMPT = user_prompt()


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


def _system_clipboard_text() -> str:
    command = None
    if sys.platform == "darwin":
        command = ["pbpaste"]
    elif shutil.which("wl-paste"):
        command = ["wl-paste", "--no-newline", "--type", "text"]
    elif shutil.which("xclip"):
        command = ["xclip", "-selection", "clipboard", "-o"]
    if command is None:
        return ""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def create_chat_prompt(
    on_image_paste: Callable[[], None] | None = None,
    on_attachment_delete: Callable[[], None] | None = None,
) -> PromptSession[str]:
    """Create an input editor whose prompt is separate from the editable buffer."""
    bindings = KeyBindings()
    bindings.add(Keys.Escape, Keys.ControlH, eager=True)(_delete_previous_word)

    @bindings.add(Keys.ControlV, eager=True)
    def _paste(event: KeyPressEvent) -> None:
        if on_image_paste is not None:
            try:
                on_image_paste()
            except ClipboardImageUnavailable:
                pass
            except ImageError as exc:
                run_in_terminal(lambda: print(f"Could not paste image: {exc}"))
                return
            else:
                event.app.invalidate()
                return
        text = _system_clipboard_text()
        if text:
            event.current_buffer.insert_text(text)

    @bindings.add(Keys.Backspace, eager=True)
    def _backspace(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        if buffer.selection_state is not None:
            buffer.cut_selection()
        elif buffer.document.text_before_cursor:
            buffer.delete_before_cursor(count=1)
        elif not buffer.text and on_attachment_delete is not None:
            on_attachment_delete()
            event.app.invalidate()

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
