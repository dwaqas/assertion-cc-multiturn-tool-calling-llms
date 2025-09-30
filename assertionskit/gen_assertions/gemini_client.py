import os, json, re, time
from typing import Any, Dict, Optional

import google.generativeai as genai

def get_model(model_name: str):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in your environment.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def call_gemini_json(model,
                     system_preamble: str,
                     user_prompt: str,
                     num_candidates: int,
                     retries: int = 1) -> Dict[str, Any]:
    """Call Gemini forcing JSON, enforcing a 1..N pick; one gentle retry if invalid."""
    gen_config = {
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 40,
        "response_mime_type": "application/json",
    }

    def _once(reminder: str = "") -> Dict[str, Any]:
        content = [
            {"role": "user", "parts": [{"text": system_preamble}]},
            {"role": "user", "parts": [{"text": user_prompt}]},
        ]
        if reminder:
            content.append({"role": "user", "parts": [{"text": reminder}]})
        resp = model.generate_content(content, generation_config=gen_config)
        txt = resp.text or "{}"
        try:
            data = json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
        return {
            "selected_index": data.get("selected_index"),
            "selected_function_name": data.get("selected_function_name"),
            "justification": (data.get("justification") or "").strip(),
            "assertion": (data.get("assertion") or "").strip(),
            "raw": data
        }

    out = _once()
    N = int(num_candidates)

    def _valid(o: Dict[str, Any]) -> bool:
        try:
            idx = int(o["selected_index"])
        except Exception:
            return False
        if not (1 <= idx <= N):
            return False
        name = o.get("selected_function_name") or ""
        return name != "" and len((o.get("assertion") or "").strip()) > 0

    if not _valid(out) and retries > 0:
        reminder = (
            "REMINDER: You MUST pick exactly one candidate. "
            "Return JSON ONLY with {selected_index: 1..N, selected_function_name matching the list, justification, assertion}. "
            "Assertion must be ONE sentence, <=20 words, and non-empty."
        )
        out = _once(reminder=reminder)

    # Final coercion to ensure determinism
    try:
        idx = int(out.get("selected_index", 1))
    except Exception:
        idx = 1
    if not (1 <= idx <= N):
        idx = 1
    out["selected_index"] = idx
    # If name missing, at least set to a placeholder; the caller will map to list if needed;
    out["selected_function_name"] = out.get("selected_function_name") or ""
    out["assertion"] = str(out.get("assertion") or "").strip()
    return out
