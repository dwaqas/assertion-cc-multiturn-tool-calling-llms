import re

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def inject_assertion(original: str, assertion: str, position: str) -> str:
    assertion = assertion.strip()
    if not assertion:
        return original
    if not assertion.endswith((".", "!", "?")):
        assertion += "."
    if position == "end":
        if original.endswith((" ", "\n")):
            return original + assertion
        return original + " " + assertion
    # middle insertion (rough heuristic by sentence);
    parts = SENT_SPLIT_RE.split(original.strip())
    if len(parts) <= 1:
        # Fallback: treat as end;
        return original + (" " if not original.endswith((" ", "\n")) else "") + assertion
    mid = len(parts) // 2
    new_parts = parts[:mid] + [assertion] + parts[mid:]
    return " ".join(new_parts)
