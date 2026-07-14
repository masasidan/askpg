from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


MAX_IMAGE_COUNT = 1500
# Base64 adds roughly one third to the request body. Keeping the raw files below
# 380 MiB leaves room inside the API's 512 MiB total payload limit.
MAX_TOTAL_IMAGE_BYTES = 380 * 1024 * 1024


class ImageError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageAttachment:
    path: Path
    media_type: str
    data_url: str
    size: int


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
