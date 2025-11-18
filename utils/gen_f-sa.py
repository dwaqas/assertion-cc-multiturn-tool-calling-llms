"""Generate function-sourced system policy notes (assertions) for BFCL multi-turn tasks."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dependencies.gen_assertions.io_utils import load_json_or_jsonl
from dependencies.gen_f_sa import (
    EXCLUDED_PROMPT_NUMBERS,
    build_system_prompt,
    build_user_prompt,
    call_gemini_json,
    get_model,
    load_function_catalog,
    load_possible_answer_turns,
    load_read_only_functions,
    load_turn_index_map,
    prompt_number,
    resolve_class_to_stem,
    turn_user_text,
)

DEFAULT_INJECTION_FIELD = "system_policy_note"
DEFAULT_MODEL = "gemini-2.5-pro"
PROMPT_LOG_PATH = "gen_f_sa_prompts.temp.log"
CSV_HEADER = [
    "id",
    "turn_idx",
    "target_function",
    "followup_function",
    "system_policy_note",
    "justification",
]

def parse_items(spec: str) -> List[int]:
    text = (spec or "").strip() # Normalise input spec;
    if not text:
        raise SystemExit("Error: --items-idxs is required (e.g., '0-199').")
    segments = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            try:
                start = int(start_s.strip())
                end = int(end_s.strip())
            except ValueError:
                raise SystemExit(f"Invalid range token '{token}'.")
            if start > end:
                raise SystemExit(f"Range start > end in '{token}'.")
            segments.extend(range(start, end + 1))
        else:
            try:
                segments.append(int(token))
            except ValueError:
                raise SystemExit(f"Invalid index token '{token}'.")
    return sorted(set(segments))

def _load_bfcl(path: Path) -> Dict[int, Dict[str, object]]:
    items = load_json_or_jsonl(path) # Entire BFCL corpus;
    by_number: Dict[int, Dict[str, object]] = {} # prompt_number -> item;
    for item in items:
        case_id = item.get("id")
        if not case_id:
            continue
        try:
            number = int(prompt_number(case_id))
        except ValueError:
            continue
        by_number[number] = item # Keep last-seen variant;
    return by_number

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate function-sourced system policy notes.")
    parser.add_argument("--bfcl", required=True)
    parser.add_argument("--possible-answer", required=True)
    parser.add_argument("--func-doc-dir", required=True)
    parser.add_argument("--turn-index-map", required=True)
    parser.add_argument("--read-only-map", required=True)
    parser.add_argument("--items-idxs", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-json", default="assertions.function.write.confident.jsonl")
    parser.add_argument("--out-meta", default="assertions.function.write.confident.metadata.jsonl")
    parser.add_argument("--out-csv", default="assertions.function.write.confident.review.csv")
    args = parser.parse_args()

    bfcl_map = _load_bfcl(Path(args.bfcl))
    possible_answers = load_possible_answer_turns(Path(args.possible_answer))
    read_only_funcs = load_read_only_functions(Path(args.read_only_map))
    docs_by_stem, funcs_by_name = load_function_catalog(Path(args.func_doc_dir), read_only_funcs)
    turn_index_map = load_turn_index_map(Path(args.turn_index_map), key="write_heavy")
    requested = parse_items(args.items_idxs)

    model = get_model(args.model) # Gemini model;
    system_preamble = build_system_prompt() # Shared system message;

    prompt_log = Path(PROMPT_LOG_PATH).open("w", encoding="utf-8") # Prompt audit log;
    out_json = Path(args.out_json).open("w", encoding="utf-8")
    out_meta = Path(args.out_meta).open("w", encoding="utf-8")
    csv_file = Path(args.out_csv).open("w", encoding="utf-8", newline="") # Review table;
    csv_writer = csv.writer(csv_file, lineterminator="\n")
    csv_writer.writerow(CSV_HEADER)

    processed = 0 # Successful generations;
    skipped = 0 # Skipped due to errors;

    for idx in requested:
        if idx in EXCLUDED_PROMPT_NUMBERS:
            skipped += 1
            out_meta.write(json.dumps({
                "prompt_number": idx,
                "error": "excluded_prompt_number",
            }, ensure_ascii=False) + "\n")
            continue
        item = bfcl_map.get(idx) # Lookup BFCL entry;
        if item is None:
            skipped += 1
            out_meta.write(json.dumps({
                "prompt_number": idx,
                "error": "bfcl_item_not_found",
            }, ensure_ascii=False) + "\n")
            continue
        case_id = item.get("id") # Stable identifier;
        turn_idx = turn_index_map.get(idx) # Target turn index;
        if turn_idx is None:
            skipped += 1
            out_meta.write(json.dumps({
                "id": case_id,
                "prompt_number": idx,
                "error": "write_turn_missing",
            }, ensure_ascii=False) + "\n")
            continue
        user_turns = []
        for pos, turn in enumerate(item.get("question", []) or []):
            lines = []
            if isinstance(turn, list):
                for message in turn:
                    if isinstance(message, dict) and message.get("role") == "user":
                        content = (message.get("content", "") or "").strip()
                        if content:
                            prefix = "* " if pos == turn_idx else "  "
                            lines.append(f"{prefix}Turn {pos + 1}: {content}")
            elif isinstance(turn, dict):
                if turn.get("role") == "user":
                    content = (turn.get("content", "") or "").strip()
                    if content:
                        prefix = "* " if pos == turn_idx else "  "
                        lines.append(f"{prefix}Turn {pos + 1}: {content}")
            text_block = "\n".join(lines).strip()
            if text_block:
                user_turns.append(text_block)
        turn_text = turn_user_text(item, turn_idx) # User text for turn;
        answer_entry = possible_answers.get(case_id, []) # Ground-truth plan;
        if turn_idx >= len(answer_entry):
            skipped += 1
            out_meta.write(json.dumps({
                "id": case_id,
                "prompt_number": idx,
                "turn_idx": turn_idx,
                "error": "turn_answers_missing",
            }, ensure_ascii=False) + "\n")
            continue
        raw_calls = answer_entry[turn_idx] # List of call dicts;
        candidates: List[Dict[str, str]] = [] # Filtered write tools;
        for call_bundle in raw_calls:
            name = call_bundle.get("name")
            if not name or name in read_only_funcs:
                continue
            info = funcs_by_name.get(name, {})
            candidates.append({
                "name": name,
                "call": call_bundle.get("call", ""),
                "description": info.get("description", ""),
                "stem": info.get("stem"),
            })
        if not candidates:
            skipped += 1
            out_meta.write(json.dumps({
                "id": case_id,
                "prompt_number": idx,
                "turn_idx": turn_idx,
                "error": "no_write_functions",
            }, ensure_ascii=False) + "\n")
            continue
        stem_label_map: Dict[str, str] = {}
        available_stems = docs_by_stem.keys()
        for class_name in item.get("involved_classes", []) or []:
            stem = resolve_class_to_stem(class_name, available_stems)
            if stem and stem not in stem_label_map:
                stem_label_map[stem] = class_name

        turn_stems: List[str] = []
        seen_turn_stems: Set[str] = set()
        for call_bundle in raw_calls:
            info = funcs_by_name.get(call_bundle.get("name"), {})
            stem = info.get("stem")
            if stem and stem not in seen_turn_stems:
                seen_turn_stems.add(stem)
                turn_stems.append(stem)

        class_docs: List[Tuple[str, List[Dict[str, str]]]] = []
        followup_options: List[Dict[str, object]] = []
        followup_seen: Set[str] = set()
        for stem in turn_stems:
            docs = docs_by_stem.get(stem, [])
            if not docs:
                continue
            label = stem_label_map.get(stem, stem)
            class_docs.append((label, docs))
            for record in docs:
                raw_name = record.get("name")
                name = raw_name.strip() if isinstance(raw_name, str) else raw_name
                if not name or name in followup_seen:
                    continue
                if record.get("read_only", False):
                    continue
                followup_options.append({
                    "name": name,
                    "description": record.get("description", ""),
                    "stem": label,
                    "stem_id": stem,
                    "read_only": record.get("read_only", False),
                })
                followup_seen.add(name)

        if not followup_options:
            skipped += 1
            out_meta.write(json.dumps({
                "id": case_id,
                "prompt_number": idx,
                "turn_idx": turn_idx,
                "error": "no_followup_functions",
            }, ensure_ascii=False) + "\n")
            continue

        user_prompt = build_user_prompt(
            turn_text,
            candidates,
            class_docs,
            user_turns,
            followup_options,
        ) # Compose prompt;
        prompt_log.write("==== PROMPT START ====\n")
        prompt_log.write(f"ID: {case_id}\n")
        prompt_log.write("-- SYSTEM --\n")
        prompt_log.write(system_preamble + "\n")
        prompt_log.write("-- USER --\n")
        prompt_log.write(user_prompt + "\n")
        prompt_log.write("==== PROMPT END ====\n")
        followup_lookup: Dict[str, Dict[str, object]] = {
            opt["name"]: opt for opt in followup_options
        }
        followup_names = list(followup_lookup.keys())
        choice = call_gemini_json(
            model,
            system_preamble,
            user_prompt,
            len(candidates),
            followup_names,
            retries=1,
        ) # LLM call;
        try:
            selected_index = int(choice.get("selected_index", 1))
        except (TypeError, ValueError):
            selected_index = 1
        fn_name = (choice.get("selected_function_name") or "").strip()
        matched_idx = None
        if fn_name:
            for pos, candidate in enumerate(candidates, 1):
                if candidate["name"] == fn_name:
                    matched_idx = pos
                    break
        if matched_idx is not None:
            selected_index = matched_idx
        if selected_index < 1 or selected_index > len(candidates):
            selected_index = 1
        selected = candidates[selected_index - 1] # Candidate record;
        if fn_name and fn_name != selected["name"]:
            skipped += 1
            out_meta.write(json.dumps({
                "id": case_id,
                "prompt_number": idx,
                "turn_idx": turn_idx,
                "error": "name_mismatch",
                "candidates": candidates,
                "raw_llm": choice.get("raw"),
            }, ensure_ascii=False) + "\n")
            continue
        system_policy_note = choice.get("system_policy_note", "") # Proposed note;
        if not system_policy_note:
            skipped += 1
            out_meta.write(json.dumps({
                "id": case_id,
                "prompt_number": idx,
                "turn_idx": turn_idx,
                "error": "empty_system_policy_note",
                "candidates": candidates,
                "raw_llm": choice.get("raw"),
            }, ensure_ascii=False) + "\n")
            continue
        followup_name = (choice.get("followup_function") or "").strip()
        followup_entry = followup_lookup.get(followup_name)
        if not followup_entry:
            skipped += 1
            out_meta.write(json.dumps({
                "id": case_id,
                "prompt_number": idx,
                "turn_idx": turn_idx,
                "error": "invalid_followup_function",
                "followup_function": followup_name,
                "available_followups": followup_options,
                "raw_llm": choice.get("raw"),
            }, ensure_ascii=False) + "\n")
            continue
        justification = choice.get("justification", "") # LLM explanation;
        minimal = {
            "id": case_id,
            "turn_idx": turn_idx,
            "target_function": selected["name"],
            "injection_field": DEFAULT_INJECTION_FIELD,
            "system_policy_note": system_policy_note,
            "assertion": system_policy_note,
            "justification": justification,
        } # Minimal output record;
        out_json.write(json.dumps(minimal, ensure_ascii=False) + "\n")
        meta_record = {
            "id": case_id,
            "prompt_number": idx,
            "turn_idx": turn_idx,
            "candidates": candidates,
            "selected_index": selected_index,
            "selected_function_name": selected["name"],
            "followup_function_name": followup_name,
            "followup_function": followup_entry,
            "followup_options": followup_options,
            "system_policy_note": system_policy_note,
            "justification": justification,
            "user_prompt": user_prompt,
            "raw_llm": choice.get("raw"),
        } # Detailed metadata;
        out_meta.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
        csv_writer.writerow([
            case_id,
            turn_idx,
            selected["name"],
            followup_name,
            system_policy_note,
            justification,
        ]) # Review entry;
        processed += 1

    prompt_log.close()
    out_json.close()
    out_meta.close()
    csv_file.close()
    print(f"Completed: {processed} items written; skipped={skipped}.") # Summary;

if __name__ == "__main__":
    main()
