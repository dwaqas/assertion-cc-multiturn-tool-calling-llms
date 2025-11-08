import os
import json
import re
from typing import Any, Dict

import google.generativeai as genai

_GENERATION_CONFIG = {
    "temperature": 0.35,
    "top_p": 0.8,
    "top_k": 20,
    "response_mime_type": "application/json",
} # Slightly more deterministic than `gen_assertions` due to the nature of F-SAs

def get_model(model_name: str):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in your environment.")
    genai.configure(api_key=api_key) # Configure client;
    return genai.GenerativeModel(model_name) # Return model handle;

def _response_to_text(payload: Any) -> str:
    if payload is None:
        return "{}"
    if isinstance(payload, dict): # Handle REST-like response;
        for key in ("text", "output", "response"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        candidates = payload.get("candidates") or []
    else: # SDK object path;
        direct_text = getattr(payload, "text", None)
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text
        candidates = getattr(payload, "candidates", [])
    for candidate in candidates or []:
        content = getattr(candidate, "content", None) or candidate.get("content") if isinstance(candidate, dict) else None
        parts = getattr(content, "parts", None) or content.get("parts") if isinstance(content, dict) else None
        if not parts:
            continue
        fragments = [] # Collect text parts;
        for part in parts:
            text = getattr(part, "text", None) if not isinstance(part, dict) else part.get("text")
            if text:
                fragments.append(text)
        if fragments:
            return "".join(fragments)
    return "{}"

def _parse_json_blob(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text) # Direct parse;
    except Exception:
        match = re.search(r"\{.*\}", text or "", flags=re.DOTALL) # Fallback extraction;
        return json.loads(match.group(0)) if match else {} # Graceful fallback;

def _normalise_choice(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "selected_index": data.get("selected_index"),
        "selected_function_name": (data.get("selected_function_name") or "").strip(),
        "justification": (data.get("justification") or "").strip(),
        "system_policy_note": (data.get("system_policy_note") or "").strip(),
        "raw": data,
    }

def _valid_choice(choice: Dict[str, Any], num_candidates: int) -> bool:
    try:
        idx = int(choice.get("selected_index")) # Validate index;
    except Exception:
        return False # Non-integer index;
    if not (1 <= idx <= num_candidates):
        return False
    if not choice.get("selected_function_name"):
        return False
    if not choice.get("system_policy_note"):
        return False
    return True

def _enforce_choice(choice: Dict[str, Any], num_candidates: int) -> Dict[str, Any]:
    try:
        idx = int(choice.get("selected_index", 1))
    except Exception:
        idx = 1
    if not (1 <= idx <= num_candidates):
        idx = 1 # Clamp out-of-range;
    choice["selected_index"] = idx
    choice["selected_function_name"] = (choice.get("selected_function_name") or "").strip()
    choice["system_policy_note"] = (choice.get("system_policy_note") or "").strip()
    choice["justification"] = (choice.get("justification") or "").strip()
    return choice

def call_gemini_json(model,
                     system_preamble: str,
                     user_prompt: str,
                     num_candidates: int,
                     retries: int = 1) -> Dict[str, Any]:
    def _invoke(reminder: str = "") -> Dict[str, Any]: # Single Gemini call;
        content = [
            {"role": "user", "parts": [{"text": system_preamble}]},
            {"role": "user", "parts": [{"text": user_prompt}]},
        ]
        if reminder:
            content.append({"role": "user", "parts": [{"text": reminder}]})
        response = model.generate_content(content, generation_config=_GENERATION_CONFIG) # Invoke model;
        text = _response_to_text(response) # Extract raw text;
        return _normalise_choice(_parse_json_blob(text)) # Canonicalise fields;

    choice = _invoke()
    if not _valid_choice(choice, num_candidates) and retries > 0: # Retry with reminder;
        reminder = (
            "REMINDER: Pick exactly one candidate function. Output JSON only with "
            "{selected_index, selected_function_name, justification, system_policy_note}. "
            "Hint must be one sentence (<=30 words)."
        )
        choice = _invoke(reminder)
    return _enforce_choice(choice, num_candidates)
