import re
from typing import Any, Dict, List

PROMPT_RE = re.compile(r"\d+")

def prompt_number(bfcl_id: str) -> str:
    matches = PROMPT_RE.findall(bfcl_id or "")
    return matches[-1] if matches else (bfcl_id or "")

def turn_user_text(item: Dict[str, Any], turn_idx: int) -> str:
    question = item.get("question", []) or [] # Sequence of multi-turn messages;
    if turn_idx < 0:
        turn_idx = 0
    if turn_idx >= len(question):
        return ""
    turn = question[turn_idx]
    if isinstance(turn, list):
        parts: List[str] = [] # Collect user-role messages only;
        for message in turn:
            if isinstance(message, dict) and message.get("role") == "user":
                parts.append(str(message.get("content", "")))
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(turn, dict) and turn.get("role") == "user":
        return str(turn.get("content", "")).strip()
    return str(turn).strip()
