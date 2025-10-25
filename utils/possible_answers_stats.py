"""Inspect first-turn API class usage in BFCL possible answers."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from dependencies.gen_assertions.io_utils import load_json_or_jsonl

def load_json(path: Path) -> Iterable[Dict]:
    # Read possible answers;
    data = load_json_or_jsonl(path)
    return data if isinstance(data, list) else []

def build_func_to_class(doc_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for doc in doc_dir.glob("*.json"):
        try:
            entries = json.loads(doc.read_text(encoding="utf-8")) # Parse func doc;
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            name = entry.get("name") # Function identifier;
            if name:
                mapping[name] = doc.stem # Record owning class;
    return mapping

def extract_classes(entry: Dict, func_to_class: Dict[str, str]) -> Tuple[str, Set[str]]:
    _id = entry.get("id", "")  # Prompt identifier;
    turns = entry.get("ground_truth")
    first_turn = turns[0] if isinstance(turns, list) and turns else [] # First sequence;
    classes: Set[str] = set()
    for call in first_turn or []:
        func = call.split("(", 1)[0].strip() # Extract function name;
        cls = func_to_class.get(func) # Lookup class stem;
        if cls:
            classes.add(cls)
    return _id, classes

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--possible-answer", required=True)
    ap.add_argument("--func-doc-dir", required=True)
    args = ap.parse_args()

    entries = load_json(Path(args.possible_answer)) # Load possible answers;
    func_to_class = build_func_to_class(Path(args.func_doc_dir)) # Build mapping;

    multi: List[Tuple[str, List[str]]] = []
    single: List[Tuple[str, List[str]]] = []
    for entry in entries:
        _id, classes = extract_classes(entry, func_to_class)
        if not _id:
            continue
        (multi if len(classes) > 1 else single).append((_id, sorted(classes)))

    print(f"Total entries: {len(entries)}")
    print(f"Single-class first turns: {len(single)}")
    print(f"Multi-class first turns: {len(multi)}")
    if multi:
        print("Examples with multiple classes:")
        for _id, classes in multi[:10]:
            print(f"  {_id}: {', '.join(classes)}")

if __name__ == "__main__":
    main()
