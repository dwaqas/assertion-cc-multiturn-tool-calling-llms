from __future__ import annotations
import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from eval.dependencies.non_accuracy_metrics.loader import load_assertion_metadata, load_score_file, parse_eval_file
from eval.dependencies.non_accuracy_metrics.metrics import (
    bundle_metrics,
    compute_compliance_metrics,
    compute_followup_compliance_metrics,
)
from eval.dependencies.non_accuracy_metrics.plots import plot_compliance
from eval.dependencies.non_accuracy_metrics.structures import (
    AssertionInfo,
    CaseCompliance,
    CaseOutcome,
    EvalCase,
    MetricsBundle,
    OutcomeGroupMetrics,
    ScoreCase,
)
LOGGER = logging.getLogger(__name__)
OUTCOME_DEFINITIONS = {(True, True): ("success_to_success", "Resilient (success→success)"), (True, False): ("success_to_failure", "Regressed (success→failure)"), (False, True): ("failure_to_success", "Improved (failure→success)"), (False, False): ("failure_to_failure", "Persistent failure (failure→failure)")}
OUTCOME_ORDER = ["success_to_success", "success_to_failure", "failure_to_success", "failure_to_failure"]

DEFAULT_RESULTS_ROOT = Path("data/results")
DEFAULT_METADATA_ROOT = Path("data/assertions")
DEFAULT_OUTPUT_DIR = Path("eval/output/non_accuracy")

MODES = {"u-sa", "f-sa", "inter"}

FSA_METADATA_PATH = Path("data/assertions/f-sa-ablation/f-sa-ablation.metadata.jsonl")
SUMMARY_CSV_NAME = "non_accuracy_summary.csv"
SUMMARY_JSON_NAME = "non_accuracy_summary.json"
DETAIL_SUMMARY_FILENAME = "median_summary.json"
DETAIL_CASES_FILENAME = "median_cases.json"
DETAIL_REPORT_FILENAME = "median_report.txt"
AGGREGATED_MODEL_FILENAME = "aggregated_metrics.json"
PLOTS_SUBDIR = "plots"
SPECIAL_CASE_IDS = {
    "multi_turn_base_47",
    "multi_turn_base_57",
    "multi_turn_base_157",
} # Maintain consistent 197-case denominator;

@dataclass(frozen=True)
class AttemptInfo:
    name: str
    path: Path
    models: List[str]
    order: int

@dataclass(frozen=True)
class ScalarMetrics:
    compliance_rate: float
    non_compliance_rate: float
    considered_cases: int
    total_cases: int

@dataclass
class AttemptResult:
    attempt: str
    model: str
    metrics: ScalarMetrics
    bundle: Optional[MetricsBundle] = None
    compliance_cases: Optional[List[CaseCompliance]] = None
    outcome_groups: Optional[List[OutcomeGroupMetrics]] = None
    outcome_cases: Optional[List[CaseOutcome]] = None
    mode: str = "u-sa"

@dataclass
class AggregatedModelMetrics:
    compliance_rate: float
    non_compliance_rate: float
    considered_cases: float
    total_cases: float
    attempts: List[str]
    attempt_metrics: List[Tuple[str, ScalarMetrics]]
    median_attempt: Optional[AttemptResult]

def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

def _parse_attempt_number(name: str) -> int:
    if name.startswith("att") and name[3:].isdigit():
        return int(name[3:])
    raise ValueError(f"Unexpected attempt directory: {name}")

# Discover per-condition attempts before filtering;
def _discover_attempts(category_dir: Path) -> List[AttemptInfo]:
    attempts: List[AttemptInfo] = []
    if not category_dir.exists():
        return attempts
    for child in category_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("att"):
            continue
        try:
            order = _parse_attempt_number(name)
        except ValueError:
            LOGGER.debug("Skipping unexpected directory %s", child)
            continue
        models = sorted(
            subdir.name for subdir in child.iterdir() if subdir.is_dir()
        )
        if not models:
            LOGGER.warning("No model directories found under %s", child)
            continue
        attempts.append(AttemptInfo(name=name, path=child, models=models, order=order))
    attempts.sort(key=lambda item: item.order)
    return attempts

# Locate the metadata file matching a given condition directory;
def _find_metadata_path(metadata_root: Path, category: str) -> Optional[Path]:
    candidate_names = [category]
    if category.endswith("-hedged"):
        candidate_names.append(category[:-7] + "-confident")

    for idx, name in enumerate(candidate_names):
        category_dir = metadata_root / name
        if not category_dir.exists() or not category_dir.is_dir():
            continue
        candidates = sorted(category_dir.glob("*.metadata.json*"))
        if not candidates:
            continue
        if idx > 0:
            LOGGER.info(
                "Metadata for %s not found; falling back to %s",
                category,
                name,
            )
        if len(candidates) > 1:
            LOGGER.warning(
                "Multiple metadata files found for %s; using %s",
                name,
                candidates[0],
            )
        return candidates[0]

    LOGGER.warning("No metadata files found for %s", category)
    return None


def _load_metadata(metadata_root: Path, category: str, mode: str) -> Optional[Dict[str, AssertionInfo]]:
    if mode == "f-sa":
        if not FSA_METADATA_PATH.exists():
            LOGGER.warning("F-SA metadata missing at %s", FSA_METADATA_PATH)
            return None
        return load_assertion_metadata(FSA_METADATA_PATH)

    if mode == "inter":
        if category == "assert_f-sa-interaction-confident":
            return load_assertion_metadata(Path("data/assertions/writeHeavy-confident/write-heavy.confident.metadata.jsonl"))
        if category == "assert_f-sa-interaction-hedged":
            hedged = Path("data/assertions/writeHeavy-hedged/write-heavy.hedged.metadata.jsonl")
            if hedged.exists():
                return load_assertion_metadata(hedged)
            confident = Path("data/assertions/writeHeavy-confident/write-heavy.confident.metadata.jsonl")
            if confident.exists():
                LOGGER.info(
                    "Hedged write-heavy metadata missing; falling back to %s",
                    confident,
                )
                return load_assertion_metadata(confident)
            LOGGER.warning("No metadata found for interaction hedged condition")
            return None

    lookup = category[7:] if category.startswith("assert_") else category
    meta_path = _find_metadata_path(metadata_root, lookup)
    if not meta_path:
        return None
    return load_assertion_metadata(meta_path)

# Pair category attempts with matching baseline attempts before metric aggregation;
def _align_attempts(
    category: str,
    category_attempts: List[AttemptInfo],
    baseline_attempts: List[AttemptInfo],
) -> Tuple[List[Tuple[AttemptInfo, AttemptInfo]], List[str]]:
    baseline_map = {attempt.name: attempt for attempt in baseline_attempts}
    aligned: List[Tuple[AttemptInfo, AttemptInfo]] = []

    for attempt in category_attempts:
        baseline = baseline_map.get(attempt.name)
        if baseline is None:
            LOGGER.warning(
                "Skipping %s attempt %s: baseline attempt missing",
                category,
                attempt.name,
            )
            continue
        if attempt.models != baseline.models:
            LOGGER.warning(
                "Skipping %s attempt %s: model set %s does not match baseline %s",
                category,
                attempt.name,
                attempt.models,
                baseline.models,
            )
            continue
        aligned.append((attempt, baseline))

    aligned.sort(key=lambda pair: pair[0].order)

    while aligned and len(aligned) % 2 == 0:
        removed = aligned.pop()
        LOGGER.warning(
            "Dropping attempt %s for %s to maintain odd attempt count",
            removed[0].name,
            category,
        )

    models = aligned[0][0].models if aligned else []
    return aligned, models

def _validate_case_sets(assert_cases: Dict[str, EvalCase], no_assert_cases: Dict[str, EvalCase]) -> None:
    assert_ids = set(assert_cases)
    no_assert_ids = set(no_assert_cases)
    if assert_ids != no_assert_ids:
        missing_in_assert = sorted(no_assert_ids - assert_ids)
        missing_in_no_assert = sorted(assert_ids - no_assert_ids)
        if missing_in_assert:
            LOGGER.warning("Cases missing in assert run: %s", ", ".join(missing_in_assert[:10]))
        if missing_in_no_assert:
            LOGGER.warning("Cases missing in no-assert run: %s", ", ".join(missing_in_no_assert[:10]))
        raise SystemExit("Mismatch between assert and no-assert case IDs")

    if len(assert_cases) != len(no_assert_cases):
        LOGGER.warning(
            "Case count mismatch: assert=%d, no-assert=%d",
            len(assert_cases),
            len(no_assert_cases),
        )

def _validate_score_sets(
    assert_scores: Dict[str, ScoreCase],
    no_assert_scores: Dict[str, ScoreCase],
    expected_ids: Dict[str, EvalCase],
) -> None:
    score_assert_ids = set(assert_scores)
    score_no_ids = set(no_assert_scores)
    case_ids = set(expected_ids)

    missing_in_assert = sorted(case_ids - score_assert_ids)
    missing_in_no = sorted(case_ids - score_no_ids)
    if missing_in_assert:
        LOGGER.debug("Missing assert score entries for: %s", ", ".join(missing_in_assert[:10]))
    if missing_in_no:
        LOGGER.debug("Missing no-assert score entries for: %s", ", ".join(missing_in_no[:10]))

def _pick_single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = list(directory.glob(pattern))
    if not matches:
        raise SystemExit(f"No files matching {pattern} found in {directory} for {label}")
    if len(matches) > 1:
        raise SystemExit(f"Multiple files matching {pattern} found in {directory} for {label}")
    return matches[0]

def _derive_score_path(result_path: Path) -> Optional[Path]:
    candidate = result_path.with_name(result_path.name.replace("_result", "_score"))
    if candidate.exists():
        return candidate
    return None

def _resolve_run_files(
    result_arg: Optional[str],
    score_arg: Optional[str],
    dir_arg: Optional[str],
    label: str,
) -> Tuple[Path, Path]:
    # Identify the paired result and score artefacts for a condition; 
    result_path: Optional[Path] = None
    score_path: Optional[Path] = None

    if result_arg:
        result_path = Path(result_arg)
    elif dir_arg:
        directory = Path(dir_arg)
        result_path = _pick_single_file(directory, "*_result.json", label)
    else:
        raise SystemExit(f"No {label} result path or directory provided")

    if score_arg:
        score_path = Path(score_arg)
    elif dir_arg:
        directory = Path(dir_arg)
        score_path = _pick_single_file(directory, "*_score.json", label)
    else:
        derived = _derive_score_path(result_path)
        if not derived:
            raise SystemExit(
                f"Unable to locate a *_score.json file for {label}; place one alongside the result file or use --{label.replace('-', '_')}-dir"
            )
        score_path = derived

    if not result_path.exists():
        raise SystemExit(f"Result file not found: {result_path}")
    if not score_path.exists():
        raise SystemExit(f"Score file not found: {score_path}")

    return result_path, score_path

def _write_json(path: Path, payload) -> None:
    # Serialize structured outputs for later analysis; 
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Wrote JSON output to %s", path)

def _write_report(path: Path, bundle: MetricsBundle) -> None:
    # Produce a concise human-readable summary; 
    lines = [
        "Assertion Effect Metrics",
        "=========================",
        "",
        f"Cases analysed: {bundle.compliance.considered_cases}/{bundle.compliance.total_cases}",
        "",
        "Compliance",
        f"  - Compliance Rate: {bundle.compliance.compliance_rate:.3f}",
        f"  - Non-Compliance Rate: {bundle.compliance.non_compliance_rate:.3f}",
    ]

    if bundle.outcome_groups:
        lines.extend(["", "Outcome-Conditioned Metrics"])
        for group in bundle.outcome_groups:
            lines.append(
                f"  - {group.label} (n={group.count}): "
                f"Compliance={group.compliance.compliance_rate:.3f}, "
                f"Non-compliance={group.compliance.non_compliance_rate:.3f}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote text report to %s", path)

def _serialise_bundle(bundle: MetricsBundle) -> Dict[str, object]:
    # Shape aggregate metrics for machine consumption; 
    return {
        "compliance": asdict(bundle.compliance),
        "case_counts": {
            "compliant": len(bundle.compliant_cases),
            "non_compliant": len(bundle.non_compliant_cases),
            "unknown": len(bundle.unknown_cases),
        },
        "outcome_groups": [
            {
                "category": group.category,
                "label": group.label,
                "count": group.count,
                "compliance": asdict(group.compliance),
            }
            for group in bundle.outcome_groups
        ],
    }


def _serialise_cases(
    compliance_cases: List[CaseCompliance],
    outcomes: List[CaseOutcome],
) -> Dict[str, object]:
    # Flatten per-case diagnostics for offline analysis; 
    return {
        "compliance": [asdict(case) for case in compliance_cases],
        "outcomes": [asdict(entry) for entry in outcomes],
    }

# Collapse multiple attempts for a model into a single median metric row;
def _aggregate_attempts(attempt_results: List[AttemptResult]) -> AggregatedModelMetrics:
    if not attempt_results:
        raise ValueError("No attempt results to aggregate")

    attempts = [result.attempt for result in attempt_results]

    def _median(values: List[float]) -> float:
        return statistics.median(values) if values else 0.0

    compliance_values = [res.metrics.compliance_rate for res in attempt_results]
    non_compliance_values = [res.metrics.non_compliance_rate for res in attempt_results]
    considered_values = [res.metrics.considered_cases for res in attempt_results]
    total_values = [res.metrics.total_cases for res in attempt_results]

    median_compliance = _median(compliance_values)
    median_index = compliance_values.index(median_compliance)

    aggregated = AggregatedModelMetrics(
        compliance_rate=median_compliance,
        non_compliance_rate=_median(non_compliance_values),
        considered_cases=_median(considered_values),
        total_cases=_median(total_values),
        attempts=attempts,
        attempt_metrics=[(result.attempt, result.metrics) for result in attempt_results],
        median_attempt=attempt_results[median_index],
    )

    return aggregated

# Aggregate attempts for a condition into median metrics;
def _collect_category_metrics(
    category_dir: Path,
    metadata_map: Dict[str, AssertionInfo],
    baseline_attempts: List[AttemptInfo],
    store_details: bool,
    excluded_ids: Optional[set[str]] = None,
    compliance_mode: str = "u-sa",
) -> Dict[str, AggregatedModelMetrics]:
    category = category_dir.name
    category_attempts = _discover_attempts(category_dir)
    if not category_attempts:
        LOGGER.warning("No attempts discovered for %s", category)
        return {}
    aligned, models = _align_attempts(category, category_attempts, baseline_attempts)
    if not aligned:
        LOGGER.warning("No aligned attempts available for %s", category)
        return {}
    results: Dict[str, AggregatedModelMetrics] = {}

    for model in models:
        attempt_results: List[AttemptResult] = []
        for cat_attempt, baseline_attempt in aligned:
            assert_dir = cat_attempt.path / model
            baseline_dir = baseline_attempt.path / model
            if not assert_dir.exists() or not baseline_dir.exists():
                LOGGER.warning(
                    "Skipping %s attempt %s model %s due to missing directories",
                    category,
                    cat_attempt.name,
                    model,
                )
                continue
            attempt_result = _evaluate_attempt(
                attempt_name=cat_attempt.name,
                model=model,
                assert_dir=assert_dir,
                no_assert_dir=baseline_dir,
                metadata_map=metadata_map,
                store_details=store_details,
                compliance_mode=compliance_mode,
            )
            if attempt_result is None:
                continue
            attempt_results.append(attempt_result)

        if not attempt_results:
            LOGGER.warning("No attempt metrics collected for %s/%s", category, model)
            continue

        if len(attempt_results) % 2 == 0:
            removed = attempt_results.pop()
            LOGGER.warning(
                "Dropping attempt %s for %s/%s after filtering to maintain odd attempt count",
                removed.attempt,
                category,
                model,
            )
            if not attempt_results:
                continue

        try:
            aggregated = _aggregate_attempts(attempt_results)
        except ValueError as exc:
            LOGGER.warning("Skipping aggregation for %s/%s: %s", category, model, exc)
            continue

        results[model] = aggregated

    return results

def _scalar_metrics_to_dict(metrics: ScalarMetrics) -> Dict[str, float]:
    return {
        "compliance_rate": metrics.compliance_rate,
        "non_compliance_rate": metrics.non_compliance_rate,
        "considered_cases": metrics.considered_cases,
        "total_cases": metrics.total_cases,
    }
def _aggregated_metrics_to_dict(metrics: AggregatedModelMetrics) -> Dict[str, object]:
    return {
        "compliance_rate": metrics.compliance_rate,
        "non_compliance_rate": metrics.non_compliance_rate,
        "considered_cases": int(round(metrics.considered_cases)),
        "total_cases": int(round(metrics.total_cases)),
        "attempts": metrics.attempts,
        "attempt_metrics": [
            {"attempt": attempt, **_scalar_metrics_to_dict(scalar)}
            for attempt, scalar in metrics.attempt_metrics
        ],
    }

def _write_detail_outputs(
    output_root: Path,
    category: str,
    model: str,
    aggregated: AggregatedModelMetrics,
    write_files: bool,
    save_plots: bool,
) -> None:
    if not (write_files or save_plots):
        return
    detail_dir = output_root / category / model
    detail_dir.mkdir(parents=True, exist_ok=True)
    if write_files:
        aggregated_payload = _aggregated_metrics_to_dict(aggregated)
        _write_json(detail_dir / AGGREGATED_MODEL_FILENAME, aggregated_payload)

    median_attempt = aggregated.median_attempt
    if (
        median_attempt is None
        or median_attempt.bundle is None
        or median_attempt.compliance_cases is None
    ):
        LOGGER.warning(
            "Skipping median detail outputs for %s/%s; detailed artefacts unavailable",
            category,
            model,
        )
        return

    if write_files:
        summary_payload = _serialise_bundle(median_attempt.bundle)
        cases_payload = _serialise_cases(
            median_attempt.compliance_cases,
            median_attempt.outcome_cases or [],
        )
        _write_json(detail_dir / DETAIL_SUMMARY_FILENAME, summary_payload)
        _write_json(detail_dir / DETAIL_CASES_FILENAME, cases_payload)
        _write_report(detail_dir / DETAIL_REPORT_FILENAME, median_attempt.bundle)

    if save_plots:
        plots_dir = detail_dir / PLOTS_SUBDIR
        plot_compliance(median_attempt.bundle.compliance, plots_dir)

# Emit consolidated CSV view for easy spreadsheet analysis;
def _write_summary_csv(
    output_path: Path,
    table: Dict[str, Dict[str, AggregatedModelMetrics]],
    mode: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "model",
        "compliance_rate",
        "non_compliance_rate",
        "considered_cases",
        "total_cases",
        "attempts_used",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for condition in sorted(table.keys()):
            for model in sorted(table[condition].keys()):
                metrics = table[condition][model]
                row = [
                    condition,
                    model,
                    f"{metrics.compliance_rate * 100:.2f}",
                    f"{metrics.non_compliance_rate * 100:.2f}",
                    int(round(metrics.considered_cases)),
                    int(round(metrics.total_cases)),
                    ";".join(metrics.attempts),
                ]
                writer.writerow(row)

    LOGGER.info("Wrote summary CSV to %s", output_path)


# Persist the aggregate table as structured JSON for downstream tooling;
def _write_summary_json(
    output_path: Path,
    table: Dict[str, Dict[str, AggregatedModelMetrics]],
    mode: str,
) -> None:
    payload: Dict[str, Dict[str, object]] = {"conditions": {}}
    for condition, models in table.items():
        condition_entry: Dict[str, object] = {"models": {}}
        for model, metrics in models.items():
            entry = {
                "compliance_rate": metrics.compliance_rate,
                "non_compliance_rate": metrics.non_compliance_rate,
                "considered_cases": int(round(metrics.considered_cases)),
                "total_cases": int(round(metrics.total_cases)),
                "attempts": metrics.attempts,
            }
            condition_entry["models"][model] = entry
        payload["conditions"][condition] = condition_entry

    _write_json(output_path, payload)


_CR_METRIC_ORDER = [
    ("overall", "CR"),
    ("success_to_success", "CR (success→success)"),
    ("success_to_failure", "CR (success→failure)"),
    ("failure_to_success", "CR (failure→success)"),
    ("failure_to_failure", "CR (failure→failure)"),
]


def _format_condition_label(condition: str) -> str:
    suffix = ""
    label = condition
    if condition.endswith("-usa"):
        suffix = " (U-SA)"
        label = condition[:-4]
    elif condition.endswith("-fsa"):
        suffix = " (F-SA)"
        label = condition[:-4]
    if label.startswith("assert_"):
        label = label[7:]
    return label + suffix


def _extract_compliance_rates(
    aggregated: AggregatedModelMetrics,
) -> Dict[str, Tuple[Optional[float], Optional[int]]]:
    rates: Dict[str, Tuple[Optional[float], Optional[int]]] = {
        "overall": (
            aggregated.compliance_rate,
            int(round(aggregated.considered_cases)),
        )
    }

    median_attempt = aggregated.median_attempt
    outcome_map: Dict[str, Tuple[float, int]] = {}
    if median_attempt and median_attempt.outcome_groups:
        for group in median_attempt.outcome_groups:
            outcome_map[group.category] = (
                group.compliance.compliance_rate,
                group.count,
            )

    for category, _ in _CR_METRIC_ORDER[1:]:
        rates[category] = outcome_map.get(category, (None, None))

    return rates


def _write_cr_summary_csv(
    output_path: Path,
    table: Dict[str, Dict[str, AggregatedModelMetrics]],
    mode: str = "u-sa",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = sorted(table.keys())
    models = sorted({model for models in table.values() for model in models.keys()})

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)

        if mode == "u-sa":
            header = ["model", "condition"]
            header.extend(label for _, label in _CR_METRIC_ORDER)
            writer.writerow(header)

            for model in models:
                for condition in conditions:
                    aggregated = table.get(condition, {}).get(model)
                    if not aggregated:
                        continue
                    rates = _extract_compliance_rates(aggregated)
                    row = [model, _format_condition_label(condition)]
                    for key, _label in _CR_METRIC_ORDER:
                        value, count = rates.get(key, (None, None))
                        row.append(
                            f"{value * 100:.1f}% (n={count})"
                            if value is not None and count is not None
                            else ""
                        )
                    writer.writerow(row)

        elif mode == "f-sa":
            header_top = ["model"]
            for condition in conditions:
                label = _format_condition_label(condition)
                header_top.extend([label] * len(_CR_METRIC_ORDER))
            header_second = ["model"]
            for _ in conditions:
                header_second.extend(label for _, label in _CR_METRIC_ORDER)
            writer.writerow(header_top)
            writer.writerow(header_second)

            for model in models:
                row: List[str] = [model]
                for condition in conditions:
                    aggregated = table.get(condition, {}).get(model)
                    if not aggregated:
                        row.extend([""] * len(_CR_METRIC_ORDER))
                        continue
                    rates = _extract_compliance_rates(aggregated)
                    for key, _label in _CR_METRIC_ORDER:
                        value, count = rates.get(key, (None, None))
                        row.append(
                            f"{value * 100:.1f}% (n={count})"
                            if value is not None and count is not None
                            else ""
                        )
                writer.writerow(row)

        else:  # interaction mode
            header = ["model", "condition"]
            header.extend(label for _, label in _CR_METRIC_ORDER)
            writer.writerow(header)

            for model in models:
                for condition in conditions:
                    aggregated = table.get(condition, {}).get(model)
                    if not aggregated:
                        continue
                    rates = _extract_compliance_rates(aggregated)
                    row = [model, _format_condition_label(condition)]
                    for key, _label in _CR_METRIC_ORDER:
                        value, count = rates.get(key, (None, None))
                        row.append(
                            f"{value * 100:.1f}% (n={count})"
                            if value is not None and count is not None
                            else ""
                        )
                    writer.writerow(row)

    LOGGER.info("Wrote compliance-rate summary CSV to %s", output_path)


def _write_cr_summary_json(
    output_path: Path,
    table: Dict[str, Dict[str, AggregatedModelMetrics]],
    mode: str = "u-sa",
) -> None:
    conditions = sorted(table.keys())
    models = sorted({model for models in table.values() for model in models.keys()})

    payload: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {"models": {}}

    for model in models:
        model_entry: Dict[str, Dict[str, Optional[float]]] = {}
        for condition in conditions:
            aggregated = table.get(condition, {}).get(model)
            if not aggregated:
                continue
            rates = _extract_compliance_rates(aggregated)
            condition_entry: Dict[str, Optional[float]] = {}
            for key, label in _CR_METRIC_ORDER:
                value, count = rates.get(key, (None, None))
                if value is None or count is None:
                    condition_entry[label] = None
                else:
                    condition_entry[label] = {
                        "percent": round(value * 100, 1),
                        "count": count,
                    }
            model_entry[_format_condition_label(condition)] = condition_entry
        payload["models"][model] = model_entry

    _write_json(output_path, payload)

# Run the full metric pipeline for a single attempt/model pairing;
def _evaluate_attempt(
    attempt_name: str,
    model: str,
    assert_dir: Path,
    no_assert_dir: Path,
    metadata_map: Dict[str, AssertionInfo],
    store_details: bool,
    compliance_mode: str = "u-sa",
) -> Optional[AttemptResult]:
    label = f"{attempt_name}-{model}"
    try:
        assert_result_path, assert_score_path = _resolve_run_files(
            result_arg=None,
            score_arg=None,
            dir_arg=str(assert_dir),
            label=f"assert {label}",
        )
    except SystemExit as exc:
        LOGGER.warning("Skipping %s: %s", label, exc)
        return None

    try:
        no_result_path, no_score_path = _resolve_run_files(
            result_arg=None,
            score_arg=None,
            dir_arg=str(no_assert_dir),
            label=f"no-assert {label}",
        )
    except SystemExit as exc:
        LOGGER.warning("Skipping %s baseline: %s", label, exc)
        return None

    assert_cases = parse_eval_file(assert_result_path, excluded_ids=SPECIAL_CASE_IDS)
    no_assert_cases = parse_eval_file(no_result_path, excluded_ids=SPECIAL_CASE_IDS)
    assert_scores = load_score_file(assert_score_path)
    no_assert_scores = load_score_file(no_score_path)

    _validate_case_sets(assert_cases, no_assert_cases)
    _validate_score_sets(assert_scores, no_assert_scores, assert_cases)

    if compliance_mode == "f-sa":
        compliance_cases, compliance_agg = compute_followup_compliance_metrics(
            assert_cases,
            metadata_map,
            excluded_ids=SPECIAL_CASE_IDS,
        )
    else:
        compliance_cases, compliance_agg = compute_compliance_metrics(
            assert_cases,
            metadata_map,
            excluded_ids=SPECIAL_CASE_IDS,
        )
    outcome_groups, outcome_cases_full = _compute_outcome_groups(
        assert_cases,
        no_assert_cases,
        assert_scores,
        no_assert_scores,
        metadata_map,
        compliance_mode=compliance_mode,
    )

    bundle: Optional[MetricsBundle] = None
    outcome_cases: Optional[List[CaseOutcome]] = None

    if store_details:
        bundle = bundle_metrics(
            compliance_cases,
            compliance_agg,
            outcome_groups,
        )
        outcome_cases = outcome_cases_full

    metrics = ScalarMetrics(
        compliance_rate=compliance_agg.compliance_rate,
        non_compliance_rate=compliance_agg.non_compliance_rate,
        considered_cases=compliance_agg.considered_cases,
        total_cases=compliance_agg.total_cases,
    )

    result = AttemptResult(
        attempt=attempt_name,
        model=model,
        metrics=metrics,
        bundle=bundle,
        compliance_cases=compliance_cases if store_details else None,
        outcome_groups=outcome_groups,
        outcome_cases=outcome_cases,
        mode=compliance_mode,
    )

    return result

def _compute_outcome_groups(
    assert_cases: Dict[str, EvalCase],
    no_assert_cases: Dict[str, EvalCase],
    assert_scores: Dict[str, ScoreCase],
    no_assert_scores: Dict[str, ScoreCase],
    metadata: Dict[str, AssertionInfo],
    compliance_mode: str = "u-sa",
) -> Tuple[List[OutcomeGroupMetrics], List[CaseOutcome]]:
    # Partition cases by outcome shifts and recompute metrics per bucket; 
    category_to_ids: Dict[str, List[str]] = defaultdict(list)
    case_outcomes: List[CaseOutcome] = []

    def _score_lookup(scores: Dict[str, ScoreCase], cid: str) -> Tuple[bool, Optional[str]]:
        entry = scores.get(cid)
        if entry:
            return entry.success, entry.error_type
        return True, None

    for case_id in assert_cases:
        no_success, no_error = _score_lookup(no_assert_scores, case_id)
        assert_success, assert_error = _score_lookup(assert_scores, case_id)
        key, _ = OUTCOME_DEFINITIONS[(no_success, assert_success)]
        category_to_ids[key].append(case_id)
        case_outcomes.append(
            CaseOutcome(
                case_id=case_id,
                category=key,
                no_assert_success=no_success,
                assert_success=assert_success,
                no_assert_error=no_error,
                assert_error=assert_error,
            )
        )

    outcome_groups: List[OutcomeGroupMetrics] = []

    for key in OUTCOME_ORDER:
        ids = category_to_ids.get(key, [])
        if not ids:
            continue
        group_label = next(lbl for (cat, lbl) in OUTCOME_DEFINITIONS.values() if cat == key)
        subset_assert = {cid: assert_cases[cid] for cid in ids}
        subset_metadata = {cid: metadata[cid] for cid in ids if cid in metadata}

        if compliance_mode == "f-sa":
            _, compliance_agg_subset = compute_followup_compliance_metrics(
                subset_assert,
                subset_metadata,
            )
        else:
            _, compliance_agg_subset = compute_compliance_metrics(
                subset_assert,
                subset_metadata,
            )

        outcome_groups.append(
            OutcomeGroupMetrics(
                category=key,
                label=group_label,
                count=len(ids),
                compliance=compliance_agg_subset,
            )
        )

    case_outcomes.sort(key=lambda entry: entry.case_id)
    return outcome_groups, case_outcomes

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate non-accuracy metrics across assertion conditions"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing per-condition results (default: data/results)",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
        help="Root directory containing assertion metadata files (default: data/assertions)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where aggregated outputs will be written (default: eval/output/non_accuracy)",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional override for the summary CSV output path",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional override for the summary JSON output path",
    )
    parser.add_argument(
        "--emit-details",
        action="store_true",
        help="Write median-attempt reports, JSON artefacts, and per-model directories",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate plots for the median attempt of each model/condition",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODES),
        default="u-sa",
        help="Evaluation mode: u-sa (default), f-sa, or inter",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    results_root: Path = args.results_root
    metadata_root: Path = args.metadata_root
    output_dir: Path = args.output_dir
    mode: str = args.mode

    if not results_root.exists():
        raise SystemExit(f"Results root not found: {results_root}")
    if not metadata_root.exists():
        raise SystemExit(f"Metadata root not found: {metadata_root}")

    baseline_dir = results_root / "noassert"
    if not baseline_dir.exists():
        raise SystemExit("Baseline no-assert results directory not found")

    baseline_attempts = _discover_attempts(baseline_dir)
    if not baseline_attempts:
        raise SystemExit("No attempts discovered under the no-assert condition")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = args.summary_csv or (output_dir / SUMMARY_CSV_NAME)
    summary_json_path = args.summary_json or (output_dir / SUMMARY_JSON_NAME)

    store_details = args.emit_details or args.plots

    aggregated_table: Dict[str, Dict[str, AggregatedModelMetrics]] = {}

    for category_dir in sorted(results_root.iterdir(), key=lambda path: path.name):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        if category == "noassert":
            continue

        if mode == "u-sa" and category.startswith("assert_f-sa"):
            continue
        if mode == "f-sa" and (not category.startswith("assert_f-sa") or "interaction" in category):
            continue
        if mode == "inter" and not category.startswith("assert_f-sa-interaction"):
            continue
        if mode == "inter":
            usa_metadata = _load_metadata(metadata_root, category, mode="inter")
            fsa_metadata = _load_metadata(metadata_root, category, mode="f-sa")
            if not usa_metadata and not fsa_metadata:
                continue

            if usa_metadata:
                label = f"{category}-usa"
                category_metrics = _collect_category_metrics(
                    category_dir,
                    usa_metadata,
                    baseline_attempts,
                    store_details=store_details,
                    excluded_ids=SPECIAL_CASE_IDS,
                    compliance_mode="u-sa",
                )
                if category_metrics:
                    aggregated_table[label] = category_metrics
                    if args.emit_details or args.plots:
                        for model, metrics in category_metrics.items():
                            _write_detail_outputs(
                                output_root=output_dir,
                                category=label,
                                model=model,
                                aggregated=metrics,
                                write_files=args.emit_details,
                                save_plots=args.plots,
                            )

            if fsa_metadata:
                label = f"{category}-fsa"
                category_metrics = _collect_category_metrics(
                    category_dir,
                    fsa_metadata,
                    baseline_attempts,
                    store_details=store_details,
                    excluded_ids=SPECIAL_CASE_IDS,
                    compliance_mode="f-sa",
                )
                if category_metrics:
                    aggregated_table[label] = category_metrics
                    if args.emit_details or args.plots:
                        for model, metrics in category_metrics.items():
                            _write_detail_outputs(
                                output_root=output_dir,
                                category=label,
                                model=model,
                                aggregated=metrics,
                                write_files=args.emit_details,
                                save_plots=args.plots,
                            )
            continue

        metadata_map = _load_metadata(metadata_root, category, mode)
        if not metadata_map:
            continue

        compliance_mode = "f-sa" if mode == "f-sa" else "u-sa"
        category_metrics = _collect_category_metrics(
            category_dir,
            metadata_map,
            baseline_attempts,
            store_details=store_details,
            excluded_ids=SPECIAL_CASE_IDS,
            compliance_mode=compliance_mode,
        )

        if not category_metrics:
            continue

        aggregated_table[category] = category_metrics

        if args.emit_details or args.plots:
            for model, metrics in category_metrics.items():
                _write_detail_outputs(
                    output_root=output_dir,
                    category=category,
                    model=model,
                    aggregated=metrics,
                    write_files=args.emit_details,
                    save_plots=args.plots,
                )

    if not aggregated_table:
        raise SystemExit("No non-accuracy metrics could be aggregated; check inputs")
    _write_summary_csv(summary_csv_path, aggregated_table, mode)
    _write_summary_json(summary_json_path, aggregated_table, mode)
    cr_csv_path = output_dir / "cr_summary.csv"
    cr_json_path = output_dir / "cr_summary.json"
    _write_cr_summary_csv(cr_csv_path, aggregated_table, mode)
    _write_cr_summary_json(cr_json_path, aggregated_table, mode)

if __name__ == "__main__":
    main()
