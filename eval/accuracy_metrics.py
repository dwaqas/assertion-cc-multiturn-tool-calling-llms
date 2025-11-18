from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LOGGER = logging.getLogger(__name__)
SPECIAL_CASE_IDS = {
    "multi_turn_base_47",
    "multi_turn_base_57",
    "multi_turn_base_157",
}
DEFAULT_RESULTS_ROOT = Path("data/results")
DEFAULT_OUTPUT_CSV = Path("eval/output/accuracy_summary.csv")
SCORE_FILENAME = "BFCL_v4_multi_turn_base_score.json"

@dataclass(frozen=True)
class AttemptInfo:
    name: str
    path: Path
    models: List[str]
    order: int

@dataclass(frozen=True)
class ModelAccuracy:
    model: str
    accuracy: float
    total_count: int
    correct_count: int

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

def _parse_attempt_number(name: str) -> int:
    if name.startswith("att") and name[3:].isdigit():
        return int(name[3:])
    raise ValueError(f"Unexpected attempt directory: {name}")

# Discover per-condition attempts before filtering;
def _discover_attempts(category_dir: Path) -> List[AttemptInfo]:
    attempts: List[AttemptInfo] = []
    for child in category_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("att"):
            continue
        order = _parse_attempt_number(name)
        models = sorted(
            subdir.name
            for subdir in child.iterdir()
            if subdir.is_dir()
        )
        if not models:
            LOGGER.warning("No model directories found under %s", child)
            continue
        attempts.append(AttemptInfo(name=name, path=child, models=models, order=order))
    attempts.sort(key=lambda item: item.order)
    return attempts

# Keep attempts whose model roster matches the baseline, trimming to odd count;
def _filter_attempts(attempts: List[AttemptInfo]) -> Tuple[List[AttemptInfo], List[str]]:
    if not attempts:
        return [], []
    baseline_models = attempts[0].models
    filtered: List[AttemptInfo] = []
    for attempt in attempts:
        if attempt.models != baseline_models:
            LOGGER.warning(
                "Skipping %s because model set %s does not match baseline %s",
                attempt.name,
                attempt.models,
                baseline_models,
            )
            continue
        filtered.append(attempt)
    while filtered and len(filtered) % 2 == 0:
        removed = filtered.pop()
        LOGGER.warning(
            "Dropping attempt %s to maintain odd attempt count", removed.name
        )
    return filtered, baseline_models

def _load_score_file(score_path: Path) -> Tuple[int, int, set[str]]:
    total = None
    correct = None
    failures: set[str] = set()
    with score_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "accuracy" in data:
                total = int(data.get("total_count", 0))
                correct = int(data.get("correct_count", 0))
            elif "id" in data:
                failures.add(str(data["id"]))
    if total is None or correct is None:
        raise ValueError(f"Summary counts missing in {score_path}")
    return total, correct, failures

# Bring counts down to the shared 197-case denominator;
def _adjust_counts(total: int, correct: int, failures: Iterable[str]) -> Tuple[int, int]:
    adjusted_total = total
    adjusted_correct = correct
    failure_set = set(failures)
    for case_id in SPECIAL_CASE_IDS:
        if adjusted_total <= 197:
            break
        if case_id in failure_set:
            adjusted_total -= 1
        else:
            adjusted_total -= 1
            adjusted_correct -= 1
    if adjusted_total > 197:
        LOGGER.warning(
            "Total count remains %d after adjustments; expected 197",
            adjusted_total,
        )
    return adjusted_total, adjusted_correct

def _compute_accuracy(score_path: Path) -> ModelAccuracy:
    total, correct, failures = _load_score_file(score_path)
    adjusted_total, adjusted_correct = _adjust_counts(total, correct, failures)
    if adjusted_total <= 0:
        raise ValueError(f"Adjusted total count invalid for {score_path}")
    accuracy = adjusted_correct / adjusted_total
    return ModelAccuracy(
        model=score_path.parent.name,
        accuracy=accuracy,
        total_count=adjusted_total,
        correct_count=adjusted_correct,
    )

def _median_accuracy(values: List[ModelAccuracy]) -> ModelAccuracy:
    if not values:
        raise ValueError("Cannot compute median accuracy for empty list")
    accuracies = [value.accuracy for value in values]
    median_accuracy = statistics.median(accuracies)
    sample = values[0]
    return ModelAccuracy(
        model=sample.model,
        accuracy=median_accuracy,
        total_count=sample.total_count,
        correct_count=sample.correct_count,
    )

def _collect_category_metrics(category_dir: Path) -> Dict[str, ModelAccuracy]:
    attempts = _discover_attempts(category_dir)
    valid_attempts, models = _filter_attempts(attempts)
    if not valid_attempts:
        LOGGER.warning("No valid attempts remain for %s", category_dir.name)
        return {}
    model_runs: Dict[str, List[ModelAccuracy]] = {model: [] for model in models}
    for attempt in valid_attempts:
        for model in models:
            score_path = attempt.path / model / SCORE_FILENAME
            if not score_path.exists():
                LOGGER.warning("Missing score file: %s", score_path)
                continue
            accuracy = _compute_accuracy(score_path)
            model_runs[model].append(accuracy)
    aggregated: Dict[str, ModelAccuracy] = {}
    for model, entries in model_runs.items():
        if not entries:
            LOGGER.warning(
                "No accuracy entries collected for %s in %s", model, category_dir.name
            )
            continue
        aggregated[model] = _median_accuracy(entries)
    return aggregated

def _determine_category_order(categories: Iterable[str]) -> List[str]:
    ordered = list(categories)
    ordered.sort()
    if "noassert" in ordered:
        ordered.remove("noassert")
        ordered.insert(0, "noassert")
    return ordered

def _write_csv(
    output_path: Path,
    table: Dict[str, Dict[str, ModelAccuracy]],
    category_order: List[str],
) -> None:
    models = sorted({model for category in table.values() for model in category})
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = table.get("noassert", {})

    def _format_cell(model: str, category: str) -> str:
        entry = table.get(category, {}).get(model)
        if entry is None:
            return ""
        accuracy_pct = entry.accuracy * 100
        value = f"{accuracy_pct:.2f}%"
        if category != "noassert" and model in baseline:
            base_accuracy_pct = baseline[model].accuracy * 100
            delta = accuracy_pct - base_accuracy_pct
            value += f" ({delta:+.2f}%)"
        return value

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *category_order])
        for model in models:
            row = [model]
            for category in category_order:
                row.append(_format_cell(model, category))
            writer.writerow(row)
    LOGGER.info("Wrote accuracy summary to %s", output_path)

def _aggregate_accuracy(root: Path) -> Dict[str, Dict[str, ModelAccuracy]]:
    table: Dict[str, Dict[str, ModelAccuracy]] = {}
    for category_dir in root.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        metrics = _collect_category_metrics(category_dir)
        if not metrics:
            continue
        table[category] = metrics
    return table

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate model accuracy across assertion conditions"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing per-condition results (default: data/results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for aggregated accuracies",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    if not args.results_root.exists():
        raise SystemExit(f"Results root not found: {args.results_root}")

    table = _aggregate_accuracy(args.results_root)
    if not table:
        raise SystemExit("No accuracy data collected")

    category_order = _determine_category_order(table.keys())
    _write_csv(args.output, table, category_order)

if __name__ == "__main__":
    main()
