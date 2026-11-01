"""Dependency-light image fingerprints for overlap auditing."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageOps


FINGERPRINT_SCHEMA_VERSION = "jmoshivis.image-fingerprint.v1"


def fingerprint_provenance() -> dict[str, str | int]:
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "file_hash": "sha256(raw bytes)",
        "pixel_hash": "sha256(width x height + decoded RGB bytes)",
        "dhash": "64-bit grayscale difference hash; Lanczos resize",
        "phash": "64-bit DCT perceptual hash; 4x high-frequency factor; Lanczos resize",
        "pillow_version": PIL.__version__,
        "numpy_version": np.__version__,
    }


@lru_cache(maxsize=4)
def _dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * size))
    matrix[0] *= np.sqrt(1 / size)
    matrix[1:] *= np.sqrt(2 / size)
    return matrix


def _bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    width = (bits.size + 3) // 4
    return f"{value:0{width}x}"


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    gray = ImageOps.grayscale(image).resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.int16)
    return _bits_to_hex(values[:, 1:] > values[:, :-1])


def phash(image: Image.Image, hash_size: int = 8, high_frequency_factor: int = 4) -> str:
    size = hash_size * high_frequency_factor
    gray = ImageOps.grayscale(image).resize((size, size), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float64)
    dct = _dct_matrix(size) @ values @ _dct_matrix(size).T
    low = dct[:hash_size, :hash_size]
    median = np.median(low.reshape(-1)[1:])
    return _bits_to_hex(low > median)


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def image_fingerprints(path: str | Path) -> dict[str, str | int]:
    file_digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            file_digest.update(chunk)
    with Image.open(path) as image:
        image.load()
        return {
            "file_sha256": file_digest.hexdigest(),
            "pixel_sha256": pixel_sha256(image),
            "dhash": dhash(image),
            "phash": phash(image),
            "width": image.width,
            "height": image.height,
        }


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()
