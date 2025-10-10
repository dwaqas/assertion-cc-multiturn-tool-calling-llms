"""Helpers for assembling candidate function sets from BFCL metadata."""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .io_utils import load_json_or_jsonl

FUNC_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(") # Extract function name from call string;
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+") # Sentence boundary heuristic;

CANONICAL_CLASSES = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
} # Map canonical dataset classes to doc stems;


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower()) # Normalise class names;


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()] # Split description into sentences;


def _normalise_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()  # Canonical form for sentence comparison;


def _shared_sentence_count(descriptions: List[str]) -> int:
    if not descriptions:
        return 0 # No descriptions to compare;
    sentence_lists = [_split_sentences(desc) for desc in descriptions]
    if any(not sentences for sentences in sentence_lists):
        return 0 # Skip trimming when a description lacks sentences;
    shared = 0
    shortest = min(len(sentences) for sentences in sentence_lists)
    for idx in range(shortest):
        pivot = _normalise_sentence(sentence_lists[0][idx])
        if all(_normalise_sentence(sentences[idx]) == pivot for sentences in sentence_lists):
            shared += 1
        else:
            break
    return shared # Number of leading sentences shared across all descriptions;


def extract_func_name(call: str) -> Optional[str]:
    match = FUNC_RE.match(call or "")
    return match.group(1) if match else None  # Return function identifier or None;


def load_possible_answer_index(path: Path) -> Dict[str, Dict[str, List[str]]]:
    entries = load_json_or_jsonl(path)
    index: Dict[str, Dict[str, List[str]]] = {}
    for entry in entries:
        _id = entry.get("id")
        if not _id:
            continue
        first_turn: List[str] = []
        all_funcs: List[str] = []
        for turn_idx, turn_calls in enumerate(entry.get("ground_truth", []) or []):
            if not isinstance(turn_calls, list):
                continue
            for call in turn_calls:
                name = extract_func_name(call)
                if not name:
                    continue
                if turn_idx == 0:
                    first_turn.append(name)
                all_funcs.append(name)
        index[_id] = {
            "first_turn": list(dict.fromkeys(first_turn)), # Preserve order, drop duplicates;
            "all_funcs": list(dict.fromkeys(all_funcs)), # All gold functions across the plan;
        }
    return index


def load_function_catalog(doc_dir: Path) -> Tuple[
    Dict[str, Dict[str, Dict[str, str]]],
    Dict[str, Dict[str, str]],
]:
    by_stem: Dict[str, Dict[str, Dict[str, str]]] = {} # stem -> {func: {description}};
    flat: Dict[str, Dict[str, str]] = {} # func -> {description, stem};
    for path in doc_dir.glob("*.json"):
        try:
            records = load_json_or_jsonl(path)
        except Exception:
            continue
        funcs: Dict[str, Dict[str, str]] = {}
        names: List[str] = []
        descriptions: List[str] = []
        for rec in records:
            name = rec.get("name")
            if not name:
                continue
            names.append(name)
            descriptions.append(rec.get("description", "") or "")
        shared = _shared_sentence_count(descriptions)
        for name, desc_text in zip(names, descriptions):
            sentences = _split_sentences(desc_text)
            if shared and len(sentences) > shared:
                trimmed = " ".join(sentences[shared:]).strip()
            else:
                trimmed = desc_text.strip()
            desc = trimmed or desc_text.strip()
            funcs[name] = {"description": desc}
            if name not in flat:
                flat[name] = {"description": desc, "stem": path.stem}
        by_stem[path.stem] = funcs
    return by_stem, flat


def resolve_class_to_stem(cls: str, available_stems: Iterable[str]) -> Optional[str]:
    key = _norm(cls)
    # Direct canonical mapping;
    for canon, stem in CANONICAL_CLASSES.items():
        if _norm(canon) == key and stem in available_stems:
            return stem
    for stem in available_stems:
        if key == _norm(stem) or key == _norm(stem.replace("_", "")):
            return stem
    if key.endswith("api"):
        trimmed = key[:-3]
        for stem in available_stems:
            if trimmed == _norm(stem) or trimmed == _norm(stem.replace("_", "")):
                return stem
    return None
