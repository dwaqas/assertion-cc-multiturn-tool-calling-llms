import os
import json
import re
from typing import Any, Dict, Iterable, List

import google.generativeai as genai

_GENERATION_CONFIG = {
    "temperature": 0.45,
    "top_p": 0.8,
    "top_k": 20,
    "response_mime_type": "application/json",
}


def get_model(model_name: str):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in your environment.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def _response_to_text(payload: Any) -> str:
    """Best-effort extraction of JSON text from Gemini responses."""
    if payload is None:
        return "{}"
    if isinstance(payload, dict):
        for key in ("text", "output", "response"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val
        candidates = payload.get("candidates") or []
    else:
        txt = getattr(payload, "text", None)
        if isinstance(txt, str) and txt.strip():
            return txt
        candidates = getattr(payload, "candidates", [])

    for cand in candidates or []:
        content = getattr(cand, "content", None) or cand.get("content") if isinstance(cand, dict) else None
        parts = getattr(content, "parts", None) or content.get("parts") if isinstance(content, dict) else None
        if not parts:
            continue
        fragments: List[str] = []
        for part in parts:
            part_text = getattr(part, "text", None) if not isinstance(part, dict) else part.get("text")
            if part_text:
                fragments.append(part_text)
        if fragments:
            return "".join(fragments)
    return "{}"


def _parse_json_blob(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
        return json.loads(match.group(0)) if match else {}


def _normalise_choice(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "selected_index": data.get("selected_index"),
        "selected_function_name": data.get("selected_function_name"),
        "justification": (data.get("justification") or "").strip(),
        "assertion": (data.get("assertion") or "").strip(),
        "raw": data,
    }


def _valid_choice(choice: Dict[str, Any], n: int) -> bool:
    try:
        idx = int(choice.get("selected_index"))
    except Exception:
        return False
    if not (1 <= idx <= n):
        return False
    if not (choice.get("selected_function_name") or "").strip():
        return False
    return bool((choice.get("assertion") or "").strip())


def _enforce_choice(choice: Dict[str, Any], n: int) -> Dict[str, Any]:
    try:
        idx = int(choice.get("selected_index", 1))
    except Exception:
        idx = 1
    if not (1 <= idx <= n):
        idx = 1
    choice["selected_index"] = idx
    choice["selected_function_name"] = (choice.get("selected_function_name") or "").strip()
    choice["assertion"] = (choice.get("assertion") or "").strip()
    return choice


def _call_single(model,
                 system_preamble: str,
                 user_prompt: str,
                 num_candidates: int,
                 retries: int) -> Dict[str, Any]:
    def _once(reminder: str = "") -> Dict[str, Any]:
        content = [
            {"role": "user", "parts": [{"text": system_preamble}]},
            {"role": "user", "parts": [{"text": user_prompt}]},
        ]
        if reminder:
            content.append({"role": "user", "parts": [{"text": reminder}]})
        response = model.generate_content(content, generation_config=_GENERATION_CONFIG)
        text = _response_to_text(response)
        return _normalise_choice(_parse_json_blob(text))

    choice = _once()
    if not _valid_choice(choice, num_candidates) and retries > 0:
        reminder = (
            "REMINDER: Pick exactly one listed candidate. Return JSON only with "
            "{selected_index:1..N, selected_function_name, justification, assertion}. "
            "Assertion must be one sentence, <=30 words, non-empty."
        )
        choice = _once(reminder)
    return _enforce_choice(choice, num_candidates)


def call_gemini_json(model,
                     system_preamble: str,
                     user_prompt: str,
                     num_candidates: int,
                     retries: int = 1) -> Dict[str, Any]:
    return _call_single(model, system_preamble, user_prompt, num_candidates, retries)


def call_gemini_json_batch(model,
                           system_preamble: str,
                           requests: Iterable[Dict[str, Any]],
                           retries: int = 1) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for req in requests:
        user_prompt = req.get("user_prompt", "")
        num_candidates = int(req.get("num_candidates", 1))
        outputs.append(_call_single(model, system_preamble, user_prompt, num_candidates, retries))
    return outputs
