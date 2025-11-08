"""
# !: PATCHED CONTENT, this entire file is not part of the original BFCL codebase
"""

import json
import logging
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from bfcl_eval.constants.eval_config import PROJECT_ROOT

_LOGGER = logging.getLogger(__name__)

FSA_CONFIG_ENABLE: bool = False
FSA_CONFIG_FILE: Optional[Path] = None
FSA_CONFIG_REQUESTED: bool = False
FSA_CONFIG_REQUEST_PATH: Optional[str] = None

_LOCK = threading.Lock()
_SUMMARY_EMITTED: bool = False

@dataclass
class FSAInstruction:
    turn_idx: int
    target_function: str
    note: str
    field: str
    applied: bool = False

class _NullRegistry:
    enabled = False
    def start_case(self, _case_id: str) -> None:
        return None
    def maybe_inject(
        self,
        _case_id: str,
        _turn_idx: int,
        _decoded_calls: Iterable[str],
        _execution_results: List[str],
    ) -> bool:
        return False

    def finish_case(self, _case_id: str) -> None:
        return None
    def emit_summary(self, _logger: Optional[logging.Logger] = None) -> None:
        return None
    def log_case_summary(self, _case_id: str, _logger: logging.Logger) -> None:
        return None

class FSARegistry:
    def __init__(self, source_path: Path, entries: Dict[str, Dict[int, List[FSAInstruction]]]):
        self.enabled = True
        self._source_path = source_path
        self._entries = entries
        self._lock = threading.Lock()
        self._cases_started: set[str] = set()
        self._cases_finished: set[str] = set()
        self._cases_with_hits: set[str] = set()
        self._planned_counts: Dict[str, int] = {
            case_id: sum(len(instrs) for instrs in turn_map.values())
            for case_id, turn_map in entries.items()
        }
        self._applied_counts: Dict[str, int] = {case_id: 0 for case_id in entries}
        self._summary_logged = False

    def start_case(self, case_id: str) -> None:
        if case_id not in self._entries:
            return
        with self._lock:
            self._cases_started.add(case_id)

    def maybe_inject(
        self,
        case_id: str,
        turn_idx: int,
        decoded_calls: Iterable[str],
        execution_results: List[str],
    ) -> bool:
        if case_id not in self._entries:
            return False
        turn_instructions = self._entries[case_id].get(turn_idx)
        if not turn_instructions:
            return False
        call_list = list(decoded_calls or [])
        any_change = False
        for instruction in turn_instructions:
            if self._apply_instruction(case_id, instruction, call_list, execution_results):
                any_change = True
        return any_change

    def finish_case(self, case_id: str) -> None:
        if case_id not in self._entries:
            return
        with self._lock:
            self._cases_finished.add(case_id)

    def emit_summary(self, logger: Optional[logging.Logger] = None) -> None:
        logger = logger or _LOGGER
        with self._lock:
            if self._summary_logged:
                return
            total_cases = len(self._entries)
            seen_cases = len(self._cases_started)
            finished_cases = len(self._cases_finished)
            injected_cases = len(self._cases_with_hits)
            planned_instructions = sum(self._planned_counts.values())
            applied_instructions = sum(self._applied_counts.values())
            missing_after_finish = sorted(
                case_id
                for case_id in self._cases_finished
                if self._applied_counts.get(case_id, 0) == 0
            )
            unseen_cases = sorted(set(self._entries.keys()) - self._cases_started)
            logger.warning(
                "[F-SA] Summary: configured_cases=%d, seen=%d, finished=%d, injected_cases=%d, instructions_applied=%d/%d",  # noqa: E501
                total_cases,
                seen_cases,
                finished_cases,
                injected_cases,
                applied_instructions,
                planned_instructions,
            )
            if missing_after_finish:
                preview = ", ".join(missing_after_finish[:10])
                suffix = "" if len(missing_after_finish) <= 10 else " ..."
                logger.warning(
                    "[F-SA] Summary: finished_without_injection=%s%s",
                    preview,
                    suffix,
                )
            if unseen_cases:
                preview = ", ".join(unseen_cases[:10])
                suffix = "" if len(unseen_cases) <= 10 else " ..."
                logger.warning(
                    "[F-SA] Summary: cases_not_seen=%s%s",
                    preview,
                    suffix,
                )
            logger.warning(
                "[F-SA] Summary: source=%s",
                self._source_path,
            )
            self._summary_logged = True

    def log_case_summary(self, case_id: str, logger: logging.Logger) -> None:
        self.finish_case(case_id)

    def _apply_instruction(
        self,
        case_id: str,
        instruction: FSAInstruction,
        decoded_calls: Sequence[str],
        execution_results: List[str],
    ) -> bool:
        matched = False
        for index, call_text in enumerate(decoded_calls):
            if instruction.target_function not in _extract_function_names(call_text):
                continue
            matched = True
            result_text = execution_results[index]
            if not _is_successful_result(result_text):
                continue
            execution_results[index] = _inject_field(
                result_text,
                instruction.field,
                instruction.note,
            )
        if matched:
            self._mark_instruction_applied(case_id, instruction)
        return matched

    def _mark_instruction_applied(self, case_id: str, instruction: FSAInstruction) -> None:
        with self._lock:
            if instruction.applied:
                return
            instruction.applied = True
            self._applied_counts[case_id] += 1
            self._cases_with_hits.add(case_id)

def _resolve_custom_path(path_like: str) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / candidate).resolve()

def _load_instructions(path: Path) -> Dict[str, Dict[int, List[FSAInstruction]]]:
    case_map: Dict[str, Dict[int, List[FSAInstruction]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, 1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                _LOGGER.warning("[F-SA] Failed to parse %s line %d: %s", path, line_num, exc)
                continue
            case_id = payload.get("id")
            turn_idx = payload.get("turn_idx")
            target_function = payload.get("target_function")
            note = (
                payload.get("system_policy_note")
                or payload.get("operator_hint")
                or payload.get("assertion")
            )
            field = payload.get("injection_field") or "system_policy_note"
            if not isinstance(case_id, str):
                _LOGGER.warning("[F-SA] Invalid case id at %s line %d", path, line_num)
                continue
            try:
                turn_idx = int(turn_idx)
            except (TypeError, ValueError):
                _LOGGER.warning("[F-SA] Invalid turn index for %s line %d", case_id, line_num)
                continue
            if not isinstance(target_function, str) or not target_function:
                _LOGGER.warning("[F-SA] Invalid target function for %s line %d", case_id, line_num)
                continue
            if not isinstance(note, str) or not note:
                _LOGGER.warning("[F-SA] Missing note for %s line %d", case_id, line_num)
                continue
            instruction = FSAInstruction(
                turn_idx=turn_idx,
                target_function=target_function,
                note=note,
                field=str(field),
            )
            case_map[case_id][turn_idx].append(instruction)
    return {
        case_id: {turn_idx: list(instructions) for turn_idx, instructions in turn_map.items()}
        for case_id, turn_map in case_map.items()
    }

def _extract_function_names(call_text: str) -> List[str]:
    if not call_text:
        return []
    call_text = call_text.strip()
    if not call_text:
        return []
    if call_text[0] in "[{":
        try:
            parsed = json.loads(call_text)
        except json.JSONDecodeError:
            parsed = None
        else:
            names: List[str] = []
            _collect_names(parsed, names)
            if names:
                return names
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()", call_text)
    if match:
        return [match.group(1)]
    return []

def _collect_names(payload, names: List[str]) -> None:
    if isinstance(payload, dict):
        name = payload.get("name")
        if isinstance(name, str):
            names.append(name)
        for value in payload.values():
            _collect_names(value, names)
    elif isinstance(payload, list):
        for item in payload:
            _collect_names(item, names)

def _inject_field(original: Optional[str], field: str, note: str) -> str:
    if original is None:
        original = ""
    stripped = original.strip()
    if not stripped:
        return json.dumps({field: note}, ensure_ascii=False)
    try:
        payload = json.loads(original)
    except json.JSONDecodeError:
        if note in original:
            return original
        suffix = f"\n{field}: {note}" if original else f"{field}: {note}"
        return f"{original}{suffix}" if original else suffix
    if isinstance(payload, dict):
        payload[field] = note
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(payload, list):
        payload.append({field: note})
        return json.dumps(payload, ensure_ascii=False)
    if note in original:
        return original
    suffix = f"\n{field}: {note}" if original else f"{field}: {note}"
    return f"{original}{suffix}" if original else suffix

def _is_successful_result(result: Optional[str]) -> bool:
    if result is None:
        return False
    text = str(result).strip()
    if not text:
        return False
    if text.startswith("Error during execution:"):
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return True
    if isinstance(payload, dict):
        for key in payload.keys():
            lowered = str(key).lower()
            if lowered == "error" or lowered.endswith("_error"):
                return False
    return True

_REGISTRY: object = _NullRegistry()

def configure_fsa(file_path: Optional[str] = None) -> None:
    global FSA_CONFIG_ENABLE, FSA_CONFIG_FILE, FSA_CONFIG_REQUESTED, FSA_CONFIG_REQUEST_PATH, _REGISTRY, _SUMMARY_EMITTED
    with _LOCK:
        _SUMMARY_EMITTED = False
        FSA_CONFIG_REQUEST_PATH = file_path
        FSA_CONFIG_REQUESTED = bool(file_path)
        if not file_path:
            FSA_CONFIG_ENABLE = False
            FSA_CONFIG_FILE = None
            _REGISTRY = _NullRegistry()
            _LOGGER.warning("[F-SA] configure_fsa disabled")
            return
        resolved = _resolve_custom_path(file_path)
        if not resolved.exists():
            FSA_CONFIG_ENABLE = False
            FSA_CONFIG_FILE = resolved
            _REGISTRY = _NullRegistry()
            _LOGGER.warning("[F-SA] configure_fsa file not found: %s", resolved)
            return
        entries = _load_instructions(resolved)
        if not entries:
            FSA_CONFIG_ENABLE = False
            FSA_CONFIG_FILE = resolved
            _REGISTRY = _NullRegistry()
            _LOGGER.warning("[F-SA] configure_fsa loaded 0 valid entries from %s", resolved)
            return
        FSA_CONFIG_ENABLE = True
        FSA_CONFIG_FILE = resolved
        _REGISTRY = FSARegistry(resolved, entries)
        _LOGGER.warning(
            "[F-SA] configure_fsa enabled with %d cases (%d instructions) from %s",
            len(entries),
            sum(len(instrs) for turn_map in entries.values() for instrs in turn_map.values()),
            resolved,
        )

def get_registry() -> object:
    return _REGISTRY

def emit_summary() -> None:
    global _SUMMARY_EMITTED
    with _LOCK:
        if _SUMMARY_EMITTED or not FSA_CONFIG_REQUESTED:
            return
        registry = get_registry()
        if getattr(registry, "enabled", False):
            registry.emit_summary(_LOGGER)
        else:
            path_hint = FSA_CONFIG_FILE or (
                _resolve_custom_path(FSA_CONFIG_REQUEST_PATH)
                if FSA_CONFIG_REQUEST_PATH
                else None
            )
            _LOGGER.warning(
                "[F-SA] Summary: registry inactive (file=%s)",
                path_hint,
            )
        _SUMMARY_EMITTED = True

__all__ = [
    "FSAInstruction",
    "configure_fsa",
    "emit_summary",
    "get_registry",
    "FSA_CONFIG_ENABLE",
    "FSA_CONFIG_FILE",
]
