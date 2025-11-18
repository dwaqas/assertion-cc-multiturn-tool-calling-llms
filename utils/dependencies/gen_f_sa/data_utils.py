from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from dependencies.gen_assertions.io_utils import load_json_or_jsonl

EXCLUDED_PROMPT_NUMBERS: Set[int] = {47, 57, 157}
FUNC_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(")
def _split_sentences(text: str) -> List[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text or "") if sentence.strip()]

def _normalise_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()

def _shared_sentence_count(descriptions: List[str]) -> int:
    if not descriptions:
        return 0
    sentence_lists = [_split_sentences(desc) for desc in descriptions]
    if any(not sentences for sentences in sentence_lists):
        return 0
    shared = 0
    shortest = min(len(sentences) for sentences in sentence_lists)
    for idx in range(shortest):
        pivot = _normalise_sentence(sentence_lists[0][idx])
        if all(_normalise_sentence(sentence[idx]) == pivot for sentence in sentence_lists):
            shared += 1
        else:
            break
    return shared

CANONICAL_CLASSES = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
} # Map BFCL class names to documentation stems, similar process to `gen_assertions`;

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower()) # Normalise for comparisons;

def extract_func_name(call: str) -> Optional[str]:
    match = FUNC_RE.match(call or "") # Attempt to match leading identifier;
    return match.group(1) if match else None # Return clean function name;

def load_possible_answer_turns(path: Path) -> Dict[str, List[List[Dict[str, str]]]]:
    entries = load_json_or_jsonl(path)
    index: Dict[str, List[List[Dict[str, str]]]] = {} # id -> per-turn call list;
    for entry in entries:
        case_id = entry.get("id")
        if not case_id:
            continue
        turns: List[List[Dict[str, str]]] = []
        for calls in entry.get("ground_truth", []) or []:
            turn_calls: List[Dict[str, str]] = []
            if isinstance(calls, list):
                for call in calls:
                    name = extract_func_name(call)
                    if not name:
                        continue
                    turn_calls.append({"call": call, "name": name}) # Retain raw call text;
            turns.append(turn_calls) # Append even if empty to keep alignment;
        index[case_id] = turns # Record mapping;
    return index

def load_function_catalog(doc_dir: Path,
                          read_only: Optional[Set[str]] = None) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Dict[str, str]]]:
    by_stem: Dict[str, List[Dict[str, str]]] = {} # Stem -> documents;
    by_function: Dict[str, Dict[str, str]] = {} # Function -> metadata;
    read_only = read_only or set()
    for path in doc_dir.glob("*.json"):
        records = load_json_or_jsonl(path)
        raw_entries: List[Tuple[str, str]] = []
        for record in records:
            name = record.get("name")
            if not name:
                continue
            description = (record.get("description") or "").strip()
            raw_entries.append((name, description))
        if not raw_entries:
            continue
        descriptions = [desc for _, desc in raw_entries]
        shared = _shared_sentence_count(descriptions)
        functions: List[Dict[str, str]] = []
        for name, description in raw_entries:
            trimmed = description
            if shared:
                sentences = _split_sentences(description)
                if len(sentences) > shared:
                    trimmed = " ".join(sentences[shared:]).strip()
            doc_entry = {
                "name": name,
                "description": trimmed or description,
                "read_only": name in read_only,
            }
            functions.append(doc_entry)
            if name not in by_function:
                by_function[name] = {
                    "description": doc_entry["description"],
                    "stem": path.stem,
                    "read_only": name in read_only,
                }
        if functions:
            by_stem[path.stem] = functions
    return by_stem, by_function # Provide both lookup tables;

def resolve_class_to_stem(class_name: str, available: Iterable[str]) -> Optional[str]:
    target = _norm(class_name) # Canonical key;
    for canonical, stem in CANONICAL_CLASSES.items():
        if _norm(canonical) == target and stem in available:
            return stem
    for stem in available:
        normed = _norm(stem)
        if normed == target or normed == _norm(stem.replace("_", "")):
            return stem
    if target.endswith("api"):
        trimmed = target[:-3] # Allow dropping "API" suffix;
        for stem in available:
            normed = _norm(stem)
            if normed == trimmed or normed == _norm(stem.replace("_", "")):
                return stem
    return None

def load_read_only_functions(path: Path) -> Set[str]:
    if not path:
        return set() # No exclusions configured;
    try:
        payload = load_json_or_jsonl(path) # Accept dict or JSONL;
    except Exception:
        return set()
    mapping = payload if isinstance(payload, dict) else payload[0] if payload else {} # Unwrap dict payload;
    read_only: Set[str] = set()
    if isinstance(mapping, dict):
        for funcs in mapping.values():
            if isinstance(funcs, list):
                read_only.update(funcs)
    return read_only # Final exclusion set;

def load_turn_index_map(path: Path, key: str = "write_heavy") -> Dict[int, int]:
    payload = load_json_or_jsonl(path) # Load dataset map;
    data = payload if isinstance(payload, dict) else payload[0] if payload else {} # Extract dict body;
    if not isinstance(data, dict):
        return {}
    values = data.get(key, [])
    index: Dict[int, int] = {}
    if isinstance(values, list):
        for idx, value in enumerate(values):
            try:
                index[idx] = int(value) # Turn index aligned to prompt_number;
            except (TypeError, ValueError):
                continue # Skip malformed entries;
    elif isinstance(values, dict):
        for raw_idx, raw_value in values.items():
            try:
                index[int(raw_idx)] = int(raw_value) # Accept string-keyed maps;
            except (TypeError, ValueError):
                continue # Ignore invalid entries;
    return index
