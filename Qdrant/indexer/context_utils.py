from __future__ import annotations

import re

__all__ = ["_sanitize_context"]


def _sanitize_context(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s*==+\s*([^=]+?)\s*==+\s*", r"\n\1\n", text)
    text = re.sub(r"\|\-+", " ", text)
    text = re.sub(r"\|{1,}", " ", text)
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)
    text = " ".join(text.split())
    return text.strip()
