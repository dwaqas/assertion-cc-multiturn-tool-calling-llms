from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .structures import AssertionInfo, EvalCase, TurnSummary, ScoreCase

LOGGER = logging.getLogger(__name__)
TOOL_NAME_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')
FUNC_CALL_PATTERN = re.compile(r'([A-Za-z_][\w\.]*)\s*\(')
ERROR_TYPE_PATTERN = re.compile(r'"error_type"\s*:\s*"([^"]+)"')

def load_json_or_jsonl(path: Path) -> List[dict]:
    # Load data from JSON or JSONL sources;
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    records: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records

def _normalise_tool_name(name: str) -> str:
    # Reduce fully qualified tool names to their callable identifier; 
    cleaned = name.strip().strip('"\'')
    return cleaned.split('.')[-1]

def _ensure_list_of_lists(obj: Iterable) -> List[List[str]]:
    turns: List[List[str]] = []
    for item in obj:
        if isinstance(item, (list, tuple)):
            turns.append([str(x) for x in item])
        else:
            turns.append([str(item)])
    return turns

def _selected_turn_indices(total_turns: int, focal: str) -> List[int]:
    if total_turns <= 0:
        return []
    if focal == "first":
        return [0]
    if focal == "last":
        return [total_turns - 1]
    return list(range(total_turns))

def parse_eval_file(path: Path, focal_turn: str = "none", excluded_ids: Optional[set[str]] = None) -> Dict[str, EvalCase]:
    # Parse evaluation traces into canonical case structures; 
    records = load_json_or_jsonl(path)
    cases: Dict[str, EvalCase] = {}
    excluded_ids = excluded_ids or set()

    for rec in records:
        case_id = str(rec.get("id") or rec.get("prompt_id") or rec.get("case_id") or "")
        if not case_id:
            LOGGER.warning("Skipping record without id in %s", path)
            continue
        if case_id in excluded_ids:
            # Exclude baseline-controlled cases so metrics align with 197-count denominator;
            continue

        raw_turns = rec.get("result") or rec.get("model_result") or rec.get("turns" )
        if not raw_turns:
            LOGGER.warning("Record %s missing result/model_result; skipping", case_id)
            continue

        full_turns = _ensure_list_of_lists(raw_turns)
        selected_indices = _selected_turn_indices(len(full_turns), focal_turn)
        turn_summaries: List[TurnSummary] = []
        tool_call_sequence: List[str] = []
        tool_call_turns: List[int] = []
        error_counter: Counter[str] = Counter()
        total_tool_calls = 0
        total_errors = 0
        text_fragments: List[str] = []

        for idx, messages in enumerate(full_turns):
            if idx not in selected_indices:
                continue
            tool_calls: List[str] = []
            text_messages: List[str] = []
            error_types: List[str] = []
            had_error = False

            for message in messages:
                tool_matches = TOOL_NAME_PATTERN.findall(message)
                if not tool_matches:
                    call_matches = FUNC_CALL_PATTERN.findall(message)
                    call_matches = [name for name in call_matches if name and name[0].isalpha()]
                    tool_matches = call_matches

                if tool_matches:
                    for name in tool_matches:
                        normalised = _normalise_tool_name(name)
                        tool_calls.append(normalised)
                        tool_call_sequence.append(normalised)
                        tool_call_turns.append(idx)
                        total_tool_calls += 1

                else:
                    stripped = message.strip()
                    if stripped:
                        text_messages.append(stripped)
                        text_fragments.append(stripped)

                if "error" in message.lower():
                    had_error = True
                    err_match = ERROR_TYPE_PATTERN.search(message)
                    if err_match:
                        error_type = err_match.group(1)
                    else:
                        error_type = "unknown"
                    error_types.append(error_type)
                    error_counter[error_type] += 1

            if error_types:
                total_errors += len(error_types)

            turn_summaries.append(
                TurnSummary(
                    index=idx,
                    tool_calls=tool_calls,
                    text_messages=text_messages,
                    error_types=error_types,
                    had_error=had_error,
                )
            )

        case = EvalCase(
            case_id=case_id,
            turns=turn_summaries,
            total_steps=len(turn_summaries),
            total_tool_calls=total_tool_calls,
            total_errors=total_errors,
            error_types=error_counter,
            all_text=" \n".join(text_fragments),
            tool_call_sequence=tool_call_sequence,
            tool_call_turns=tool_call_turns,
        )
        cases[case_id] = case

    return cases

def load_score_file(path: Path) -> Dict[str, ScoreCase]:
    # Load scoring outcomes for success/failure tagging; 
    records = load_json_or_jsonl(path)
    outcomes: Dict[str, ScoreCase] = {}

    for rec in records:
        case_id = rec.get("id")
        if not case_id:
            continue
        success = bool(rec.get("valid", False))
        error_type = None
        error_info = rec.get("error")
        if isinstance(error_info, dict):
            error_type = error_info.get("error_type")
        outcomes[case_id] = ScoreCase(
            case_id=case_id,
            success=success,
            error_type=error_type,
        )

    return outcomes

def load_assertion_metadata(path: Path) -> Dict[str, AssertionInfo]:
    records = load_json_or_jsonl(path)
    metadata: Dict[str, AssertionInfo] = {}

    for rec in records:
        case_id = rec.get("id")
        if not case_id:
            continue
        target = rec.get("selected_function_name") or rec.get("selected_function")
        assertion_text = str(rec.get("assertion") or "")

        turn_idx_value = rec.get("turn_idx")
        turn_idx: Optional[int] = None
        if turn_idx_value is not None:
            try:
                turn_idx = int(turn_idx_value)
            except (TypeError, ValueError):
                LOGGER.debug("Unable to parse turn_idx for %s: %r", case_id, turn_idx_value)

        injection_position = rec.get("injection_position")
        if injection_position is None and rec.get("injection_pos"):
            injection_position = rec.get("injection_pos")

        if turn_idx is None and isinstance(injection_position, str):
            pos_norm = injection_position.lower()
            if pos_norm in {"first", "initial", "init"}:
                turn_idx = 0

        if turn_idx is None:
            if "modified_first_prompt" in rec:
                turn_idx = 0
            elif "modified_last_prompt" in rec:
                turn_idx = -1

        followup_name = rec.get("followup_function_name")
        followup_info = rec.get("followup_function")
        if not followup_name and isinstance(followup_info, dict):
            followup_name = followup_info.get("name")

        metadata[case_id] = AssertionInfo(
            case_id=case_id,
            target_function=target,
            assertion_text=assertion_text,
            turn_idx=turn_idx,
            injection_position=injection_position,
            followup_function_name=followup_name,
            followup_function=followup_info if isinstance(followup_info, dict) else None,
        )

    return metadata
