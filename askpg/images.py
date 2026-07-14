from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4


MAX_IMAGE_COUNT = 1500
# Base64 adds roughly one third to the request body. Keeping the raw files below
# 380 MiB leaves room inside the API's 512 MiB total payload limit.
MAX_TOTAL_IMAGE_BYTES = 380 * 1024 * 1024


class ImageError(RuntimeError):
    pass


class ClipboardImageUnavailable(ImageError):
    pass


@dataclass(frozen=True)
class ImageAttachment:
    path: Path
    media_type: str
    data_url: str
    size: int
    label: str | None = None


def _detect_media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _gif_frame_count(data: bytes) -> int:
    if len(data) < 13:
        return 0
    offset = 13
    packed = data[10]
    if packed & 0x80:
        offset += 3 * (2 ** ((packed & 0x07) + 1))

    frames = 0
    while offset < len(data):
        marker = data[offset]
        if marker == 0x3B:
            break
        if marker == 0x2C:
            if offset + 10 > len(data):
                break
            frames += 1
            local_packed = data[offset + 9]
            offset += 10
            if local_packed & 0x80:
                offset += 3 * (2 ** ((local_packed & 0x07) + 1))
            if offset >= len(data):
                break
            offset += 1
        elif marker == 0x21:
            if offset + 2 > len(data):
                break
            offset += 2
        else:
            break

        while offset < len(data):
            block_size = data[offset]
            offset += 1
            if block_size == 0:
                break
            offset += block_size
    return frames


def load_image(raw_path: str | Path) -> ImageAttachment:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise ImageError(f"Image not found: {path}")
    if not path.is_file():
        raise ImageError(f"Image path is not a file: {path}")

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ImageError(f"Could not inspect image: {path}") from exc
    if size > MAX_TOTAL_IMAGE_BYTES:
        raise ImageError("Image is too large to attach.")

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ImageError(f"Could not read image: {path}") from exc

    media_type = _detect_media_type(data)
    if media_type is None:
        raise ImageError("Unsupported image. Use PNG, JPEG, WebP, or GIF.")
    if media_type == "image/gif" and _gif_frame_count(data) > 1:
        raise ImageError("Animated GIFs are not supported.")

    encoded = base64.b64encode(data).decode("ascii")
    return ImageAttachment(
        path=path,
        media_type=media_type,
        data_url=f"data:{media_type};base64,{encoded}",
        size=size,
    )


def validate_image_collection(images: Sequence[ImageAttachment]) -> None:
    if len(images) > MAX_IMAGE_COUNT:
        raise ImageError(f"A maximum of {MAX_IMAGE_COUNT} images can be attached at once.")
    if sum(image.size for image in images) > MAX_TOTAL_IMAGE_BYTES:
        raise ImageError("Attached images are too large. Keep their combined size under 380 MiB.")


def load_images(paths: Sequence[str | Path]) -> list[ImageAttachment]:
    images = [load_image(path) for path in paths]
    validate_image_collection(images)
    return images


def _export_macos_clipboard_type(path: Path, image_class: str) -> bool:
    script = """
on run argv
    set outputPath to item 1 of argv
    set imageData to the clipboard as «class IMAGE_CLASS»
    set outputFile to open for access POSIX file outputPath with write permission
    try
        set eof outputFile to 0
        write imageData to outputFile
        close access outputFile
    on error errorMessage
        try
            close access outputFile
        end try
        error errorMessage
    end try
end run
""".replace("IMAGE_CLASS", image_class)
    try:
        result = subprocess.run(
            ["osascript", "-e", script, str(path)],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and path.exists() and path.stat().st_size > 0


def load_clipboard_image() -> ImageAttachment:
    if sys.platform != "darwin":
        raise ClipboardImageUnavailable(
            "Clipboard image paste is currently supported on macOS. Use --image PATH instead."
        )

    with tempfile.TemporaryDirectory(prefix="askpg-clipboard-") as temporary:
        root = Path(temporary)
        png_path = root / "clipboard.png"
        pngpaste = shutil.which("pngpaste")
        if pngpaste:
            try:
                result = subprocess.run(
                    [pngpaste, str(png_path)],
                    capture_output=True,
                    timeout=8,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                result = None
            if result is not None and result.returncode != 0:
                png_path.unlink(missing_ok=True)

        if not png_path.exists() and not _export_macos_clipboard_type(
            png_path, "PNGf"
        ):
            tiff_path = root / "clipboard.tiff"
            if not _export_macos_clipboard_type(tiff_path, "TIFF"):
                raise ClipboardImageUnavailable("The clipboard does not contain an image.")
            try:
                converted = subprocess.run(
                    ["sips", "-s", "format", "png", str(tiff_path), "--out", str(png_path)],
                    capture_output=True,
                    timeout=12,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ImageError("Could not convert the clipboard image to PNG.") from exc
            if converted.returncode != 0 or not png_path.exists():
                raise ImageError("Could not convert the clipboard image to PNG.")

        image = load_image(png_path)
        return replace(
            image,
            path=Path(f"clipboard-{uuid4().hex}.png"),
            label="clipboard image",
        )
