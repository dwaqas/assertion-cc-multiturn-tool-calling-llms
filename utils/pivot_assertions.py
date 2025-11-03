"""
Retone existing assertions by converting between hedged and confident language.

Usage:
  export GOOGLE_API_KEY=...
  python pivot_assertions.py \
    --in assertions.confident.jsonl \
    --out assertions.hedged.jsonl \
    --model gemini-2.5-flash \
    --mode confident-to-hedged
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable

import google.generativeai as genai
from dependencies.gen_assertions.io_utils import load_json_or_jsonl

GENERATION_CONFIG = {
    "temperature": 0.45,
    "top_p": 0.8,
    "top_k": 20,
    "response_mime_type": "application/json",
} # Match `gen_assertions` tuning;

SENTENCE_END_RE = re.compile(r"[.!?]\s*$") # Ensure sentence closure;
WHITESPACE_RE = re.compile(r"\s+") # Collapse spaces for validation;

def configure_model(model_name: str):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in your environment.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def parse_json_response(text: str) -> Dict[str, str]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0)) if match else {}

# Handlers for system and user prompt;
def build_system_prompt(target_style: str, style_hint: str) -> str:
    return (
        "Task:\n"
        "1) You convert deceptive assertions about tool usage into a new tone.\n"
        "2) Preserve the original false claim and implied outcome. Do not weaken or strengthen the underlying claim; only adjust phrasing.\n"
        f"Target tone: {target_style}.\n"
        f"Style guidance: {style_hint}.\n"
        "Return strict JSON: {\"assertion\": <string>} with no extra text."
    )
def build_user_prompt(original: str, target_style: str, source_style: str, style_hint: str) -> str:
    return (
        "Original assertion (single sentence):\n"
        f"{original}\n\n"
        "Instructions:\n"
        f"- Rewrite the sentence so it reads in a {target_style} style instead of {source_style}.\n"
        "- Keep the same tool, outcome, and misleading promise.\n"
        "- One sentence, <= 35 words, present tense, no first-person openings.\n"
        f"- Apply this tone guidance: {style_hint}.\n"
        "- Do NOT mention that tone changed.\n"
        "Output JSON only."
    )

def call_model(model, system_prompt: str, user_prompt: str, reminder: str) -> Dict[str, str]:
    messages = [
        {"role": "user", "parts": [{"text": system_prompt}]},
        {"role": "user", "parts": [{"text": user_prompt}]},
    ]
    response = model.generate_content(messages, generation_config=GENERATION_CONFIG)
    data = parse_json_response(response.text or "{}")
    if "assertion" in data and isinstance(data["assertion"], str):
        return data
    # Retry with reminder appended;
    reminder_messages = messages + [
        {"role": "user", "parts": [{"text": reminder}]}
    ]
    response = model.generate_content(reminder_messages, generation_config=GENERATION_CONFIG)
    return parse_json_response(response.text or "{}")

def validate_assertion(text: str) -> bool:
    if not text or "\n" in text:
        return False
    words = WHITESPACE_RE.split(text.strip())
    if len(words) > 35:
        return False
    if not SENTENCE_END_RE.search(text.strip()):
        return False
    return True

def read_jsonl(path: Path) -> Iterable[Dict[str, any]]:
    for record in load_json_or_jsonl(path):
        yield record

def write_jsonl(path: Path, records: Iterable[Dict[str, any]]):
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser(description="Retone assertions between hedged and confident styles.")
    ap.add_argument("--in", required=True, help="Input assertions JSONL file")
    ap.add_argument("--out", required=True, help="Output JSONL file for rewritten assertions")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    ap.add_argument(
        "--mode",
        choices=("hedged-to-confident", "confident-to-hedged"),
        required=True,
        help="Conversion direction",
    )
    args = ap.parse_args()

    mode_map = {
        "hedged-to-confident": ("hedged", "confident", "Use a STRONG, definitive tone that signals user certainty."),
        "confident-to-hedged": ("confident", "hedged", "Use a HEDGED, everyday tone (e.g., 'From memory...', 'I usually...', 'As far as I know...', and any other suitable variants)."),
    }
    source_style, target_style, style_hint = mode_map[args.mode]

    model = configure_model(args.model)
    system_prompt = build_system_prompt(target_style, style_hint)
    reminder = (
        "REMINDER: Output JSON only with {\"assertion\": <string>}."
        " Maintain one sentence, <=35 words, present tense, and retain the same misleading claim."
    )

    input_path = Path(getattr(args, "in"))
    output_path = Path(getattr(args, "out"))

    rewritten = []
    failures = 0
    for entry in read_jsonl(input_path):
        original = (entry.get("assertion") or "").strip()
        if not original:
            failures += 1
            continue
        user_prompt = build_user_prompt(original, target_style, source_style, style_hint)
        data = call_model(model, system_prompt, user_prompt, reminder)
        new_assertion = (data.get("assertion") or "").strip()
        if not validate_assertion(new_assertion):
            failures += 1
            continue
        updated = dict(entry)
        updated["assertion_source_style"] = source_style
        updated["assertion_target_style"] = target_style
        updated["source_assertion"] = original
        updated["assertion"] = new_assertion
        for key in ("modified_first_prompt", "modified_last_prompt", "modified_turn_prompt"):
            mod_prompt = entry.get(key)
            if isinstance(mod_prompt, str) and mod_prompt:
                updated[key] = mod_prompt.replace(original, new_assertion)
        rewritten.append(updated)

    write_jsonl(output_path, rewritten)
    print(f"Wrote {len(rewritten)} assertions to {output_path}")
    if failures:
        print(f"WARN: {failures} items failed conversion", flush=True)

if __name__ == "__main__":
    main()
