from __future__ import annotations

import re


MODALITY_TAG_RE = re.compile(r"<\s*(?:image|video|audio|text)\s*>", flags=re.IGNORECASE)
PROMPT_ECHO_RE = re.compile(r"^user\b(?P<user_block>[\s\S]*?)\n+assistant\b[:\s]*", flags=re.IGNORECASE)


def strip_modality_tags(text: str) -> str:
    return MODALITY_TAG_RE.sub(" ", text or "").strip()


def canonicalize_text_block(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def clean_prompt_for_echo(prompt: str) -> str:
    return canonicalize_text_block(MODALITY_TAG_RE.sub(" ", prompt or ""))
