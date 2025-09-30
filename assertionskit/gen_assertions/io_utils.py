from pathlib import Path
import json
from typing import Any, Dict, List

def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    # Try JSON (list or object);
    try:
        data = json.loads(text)
        if isinstance(data, list): return data
        if isinstance(data, dict): return [data]
    except Exception:
        pass
    # Try JSONL;
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln: continue
        out.append(json.loads(ln))
    return out
