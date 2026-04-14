"""Common normalization helpers for inference utilities."""

from typing import Optional
from urllib.parse import unquote, urlparse

import re


def normalize_title(title: Optional[str]) -> str:
    """Return a whitespace-trimmed title, defaulting missing values to empty."""
    if title is None:
        return ""
    return title.strip()


def normalize_url(url: Optional[str]) -> str:
    """Extract domain and short path tokens from a URL for matching."""
    if url is None:
        return ""
    url = url.strip()
    if not url:
        return ""

    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        path = parsed.path or ""
    except ValueError:
        domain = ""
        path = ""

    segments = re.split(r"[-/\s_]+", unquote(path))
    filtered_segments = [segment for segment in segments if 0 < len(segment) < 32]

    if not domain:
        return " ".join(filtered_segments).strip()

    if not filtered_segments:
        return domain

    return f"{domain} {' '.join(filtered_segments)}"








