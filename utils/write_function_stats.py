import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

FUNC_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(")


def load_json_or_jsonl(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def extract_func_name(call: str) -> Optional[str]:
    match = FUNC_RE.match(call or "")
    return match.group(1) if match else None


def _count_calls(turns: Iterable[Iterable[str]], read_only: set[str], mode: str) -> List[int]:
    counts: List[int] = []
    for turn in turns or []:
        total = 0
        if isinstance(turn, list):
            for call in turn:
                fn = extract_func_name(call)
                if not fn:
                    continue
                is_read = fn in read_only
                if mode == "write" and not is_read:
                    total += 1
                elif mode == "read" and is_read:
                    total += 1
        counts.append(total)
    return counts


def _turn_label(index: int, total: int) -> str:
    if index == 0:
        phase = "first"
    elif index == total - 1:
        phase = "last"
    else:
        phase = "mid"
    return f"Turn {index + 1} ({phase})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise read/write function usage per BFCL multi-turn item.")
    ap.add_argument("--possible-answers", required=True,
                    help="Path to BFCL possible answers JSON/JSONL")
    ap.add_argument("--read-only-map", required=True,
                    help="Path to JSON map of read-only functions")
    ap.add_argument("--mode", choices=("write", "read"), default="write",
                    help="Report per-turn write counts (default) or read counts")
    args = ap.parse_args()

    read_only_data = json.loads(Path(args.read_only_map).read_text(encoding="utf-8"))
    read_only = {fn for fns in read_only_data.values() for fn in fns}

    records = load_json_or_jsonl(Path(args.possible_answers))

    for item in records:
        item_id = item.get("id", "<unknown>")
        turns = item.get("ground_truth", [])
        counts = _count_calls(turns, read_only, args.mode)
        counts_str = ", ".join(str(c) for c in counts)
        focus = "write" if args.mode == "write" else "read"
        label = "None"
        if counts:
            max_count = max(counts)
            if max_count > 0:
                best_indexes = [idx for idx, val in enumerate(counts) if val == max_count]
                if len(best_indexes) == 1:
                    label = _turn_label(best_indexes[0], len(counts))
                else:
                    latest = best_indexes[-1]
                    label = f"Multiple; latest {_turn_label(latest, len(counts))}"
        print(f"{item_id}: [{counts_str}] | Most {focus}-heavy turn: {label}")


if __name__ == "__main__":
    main()
