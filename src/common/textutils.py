import re

WS_RE = re.compile(r"\s+")
HYPHEN_RE = re.compile(r"(\w)-
(\w)")

def normalize_ws(text: str) -> str:
    return WS_RE.sub(" ", text).strip()

def fix_hyphens(text: str) -> str:
    return HYPHEN_RE.sub(r"\1\2", text.replace("-
", ""))
