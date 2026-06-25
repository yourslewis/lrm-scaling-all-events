#!/usr/bin/env python3
"""Shared v3 vocabulary normalization and bucketing utilities."""
import hashlib
from urllib.parse import urlparse


EVENT_TO_DOMAIN = {
    "SearchClick": 0,
    "NativeClick": 0,
    "EdgePageTitle": 1,
    "MSN": 1,
    "ChromePageTitle": 1,
    "UET": 1,
    "UETShoppingView": 1,
    "OrganicSearchQuery": 2,
    "EdgeSearchQuery": 2,
    "UETShoppingCart": 3,
    "AbandonCart": 3,
    "EdgeShoppingCart": 3,
    "EdgeShoppingPurchase": 3,
    "OutlookSenderDomain": 4,
}
NUM_DOMAINS = 5
MIN_ITEM_ID = 20
NORMALIZER_VERSION = "v3_url_domain_20260624"


def normalize_url_to_domain(text):
    text = text.strip()
    if not text:
        return ""
    if "://" in text or text.startswith("www."):
        if not text.startswith("http"):
            text = "https://" + text
        try:
            parsed = urlparse(text)
            domain = parsed.netloc or parsed.path.split("/")[0]
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            pass
    if "." in text and " " not in text and "/" not in text:
        if text.startswith("www."):
            text = text[4:]
        return text
    return text


def extract_text_normalized(event):
    texts = event.get("Texts", ["", ""])
    t0 = str(texts[0]).strip() if len(texts) > 0 and texts[0] else ""
    t1 = str(texts[1]).strip() if len(texts) > 1 and texts[1] else ""
    if t1:
        t1 = normalize_url_to_domain(t1)
    if t0 and t1:
        return f"{t0} {t1}"
    if t0:
        return t0
    if t1:
        return t1
    return event.get("Type", "UNK")


def stable_bucket_hash(text):
    """Stable, process-independent 64-bit hash for vocab bucketing."""
    h = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(h, "big")


def bucket_of(text, num_buckets):
    return stable_bucket_hash(text) % num_buckets
