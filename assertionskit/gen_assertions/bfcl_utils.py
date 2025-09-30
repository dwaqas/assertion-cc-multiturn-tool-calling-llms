import re
from typing import Any, Dict, List

def prompt_number(bfcl_id: str) -> str:
    m = re.findall(r"\d+", bfcl_id or "")
    return m[-1] if m else (bfcl_id or "")

def first_user_text(item: Dict[str, Any]) -> str:
    q = item.get("question", [])
    if not q: return ""
    turn0 = q[0]
    if isinstance(turn0, list):
        return "\n".join([str(m.get("content","")) for m in turn0 if isinstance(m, dict)])
    return str(turn0)
