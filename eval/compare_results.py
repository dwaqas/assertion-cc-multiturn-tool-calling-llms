from __future__ import annotations
import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from eval.dependencies.loader import load_assertion_metadata, load_score_file, parse_eval_file  # type: ignore
from eval.dependencies.metrics import bundle_metrics, compute_compliance_metrics, compute_step_metrics, compute_tool_error_metrics  # type: ignore
from eval.dependencies.plots import plot_compliance, plot_error_rates, plot_step_ecdf
from eval.dependencies.structures import AssertionInfo, CaseOutcome, EvalCase, MetricsBundle, OutcomeGroupMetrics, ScoreCase
LOGGER = logging.getLogger(__name__)
OUTCOME_DEFINITIONS = {(True, True): ("success_to_success", "Resilient (success→success)"), (True, False): ("success_to_failure", "Regressed (success→failure)"), (False, True): ("failure_to_success", "Improved (failure→success)"), (False, False): ("failure_to_failure", "Persistent failure (failure→failure)")}
OUTCOME_ORDER = ["success_to_success", "success_to_failure", "failure_to_success", "failure_to_failure"]

def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

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
        f"  - Transient Compliance Rate (TCR): {bundle.compliance.transient_rate:.3f}",
        f"  - Persistent Compliance Rate (PCR): {bundle.compliance.persistent_rate:.3f}",
        f"  - Non-Compliance Rate (NCR): {bundle.compliance.non_compliance_rate:.3f}",
        f"  - Correction Rate (CR): {bundle.compliance.correction_rate:.3f}",
        "",
        "Step Overhead",
        f"  - Mean Δsteps: {bundle.steps.mean_delta:.2f}",
        f"  - Median Δsteps: {bundle.steps.median_delta:.2f}",
        f"  - P90 Δsteps: {bundle.steps.p90_delta:.2f}",
        "",
        "Tool Error Delta",
        f"  - No-assert error rate: {bundle.tool_errors.no_assert_error_rate:.3f}",
        f"  - Assert error rate: {bundle.tool_errors.assert_error_rate:.3f}",
        f"  - ΔTE: {bundle.tool_errors.delta_error_rate:.3f}",
    ]

    if bundle.tool_errors.top_error_types:
        lines.append("  - Top error type shifts:")
        for shift in bundle.tool_errors.top_error_types:
            lines.append(
                f"      • {shift.error_type}: assert={shift.assert_count}, "
                f"no-assert={shift.no_assert_count}, Δ={shift.delta}"
            )

    if bundle.outcome_groups:
        lines.extend(["", "Outcome-Conditioned Metrics"])
        for group in bundle.outcome_groups:
            lines.append(
                f"  - {group.label} (n={group.count}): "
                f"TCR={group.compliance.transient_rate:.3f}, "
                f"PCR={group.compliance.persistent_rate:.3f}, "
                f"NCR={group.compliance.non_compliance_rate:.3f}, "
                f"Δsteps_mean={group.steps.mean_delta:.2f}, ΔTE={group.tool_errors.delta_error_rate:.3f}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote text report to %s", path)

def _serialise_bundle(
    bundle,
    compliance_cases,
    step_metrics,
    tool_metrics,
):
    # Shape aggregate metrics for machine consumption; 
    return {
        "compliance": asdict(bundle.compliance),
        "steps": {
            "mean_delta": bundle.steps.mean_delta,
            "median_delta": bundle.steps.median_delta,
            "p90_delta": bundle.steps.p90_delta,
        },
        "tool_errors": {
            "assert_error_rate": bundle.tool_errors.assert_error_rate,
            "no_assert_error_rate": bundle.tool_errors.no_assert_error_rate,
            "delta_error_rate": bundle.tool_errors.delta_error_rate,
            "top_error_types": [asdict(shift) for shift in bundle.tool_errors.top_error_types],
        },
        "case_counts": {
            "persistent": len(bundle.persistent_cases),
            "transient": len(bundle.transient_cases),
            "non_compliant": len(bundle.non_compliant_cases),
        },
        "outcome_groups": [
            {
                "category": group.category,
                "label": group.label,
                "count": group.count,
                "compliance": asdict(group.compliance),
                "steps": {
                    "mean_delta": group.steps.mean_delta,
                    "median_delta": group.steps.median_delta,
                    "p90_delta": group.steps.p90_delta,
                },
                "tool_errors": {
                    "assert_error_rate": group.tool_errors.assert_error_rate,
                    "no_assert_error_rate": group.tool_errors.no_assert_error_rate,
                    "delta_error_rate": group.tool_errors.delta_error_rate,
                },
            }
            for group in bundle.outcome_groups
        ],
    }

def _serialise_cases(compliance_cases, step_metrics, tool_metrics, outcomes):
    # Flatten per-case diagnostics for offline analysis; 
    return {
        "compliance": [asdict(case) for case in compliance_cases],
        "steps": [asdict(delta) for delta in step_metrics.per_case],
        "tool_usage": [asdict(entry) for entry in tool_metrics.per_case],
        "outcomes": [asdict(entry) for entry in outcomes],
    }

def _compute_outcome_groups(
    assert_cases: Dict[str, EvalCase],
    no_assert_cases: Dict[str, EvalCase],
    assert_scores: Dict[str, ScoreCase],
    no_assert_scores: Dict[str, ScoreCase],
    metadata: Dict[str, AssertionInfo],
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
        subset_no = {cid: no_assert_cases[cid] for cid in ids}
        subset_metadata = {cid: metadata[cid] for cid in ids if cid in metadata}

        compliance_subset, compliance_agg_subset = compute_compliance_metrics(subset_assert, subset_metadata)
        steps_subset = compute_step_metrics(subset_assert, subset_no)
        tool_subset = compute_tool_error_metrics(subset_assert, subset_no)

        outcome_groups.append(
            OutcomeGroupMetrics(
                category=key,
                label=group_label,
                count=len(ids),
                compliance=compliance_agg_subset,
                steps=steps_subset,
                tool_errors=tool_subset,
            )
        )

    case_outcomes.sort(key=lambda entry: entry.case_id)
    return outcome_groups, case_outcomes

def main() -> None:
    # CLI entrypoint for assertion effect comparison; 
    parser = argparse.ArgumentParser(description="Compare assert vs no-assert evaluation results")
    parser.add_argument(
        "--no-assert-dir",
        required=True,
        help="Directory containing exactly one *_result.json and one *_score.json for the no-assert run",
    )
    parser.add_argument(
        "--assert-dir",
        required=True,
        help="Directory containing exactly one *_result.json and one *_score.json for the assert run",
    )
    parser.add_argument(
        "--assert-metadata",
        required=True,
        help="Path to assertion metadata JSON/JSONL used to identify target functions",
    )
    parser.add_argument("--summary-json", default="eval/output/assertion_metrics_summary.json")
    parser.add_argument("--case-json", default="eval/output/assertion_metrics_cases.json")
    parser.add_argument("--report", default="eval/output/assertion_metrics.txt")
    parser.add_argument("--plots", action="store_true", help="Generate plots in eval/plots/ or custom directory")
    parser.add_argument(
        "--focal-turn",
        choices=("none", "first", "last"),
        default="none",
        help="Restrict analysis to the first turn, last turn, or the entire run (none)",
    )
    parser.add_argument("--plots-dir", default="eval/plots")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    no_assert_path, no_assert_score_path = _resolve_run_files(
        result_arg=None,
        score_arg=None,
        dir_arg=args.no_assert_dir,
        label="no-assert",
    )
    assert_path, assert_score_path = _resolve_run_files(
        result_arg=None,
        score_arg=None,
        dir_arg=args.assert_dir,
        label="assert",
    )
    metadata_path = Path(args.assert_metadata)

    LOGGER.info("Loading runs...")
    no_assert_cases = parse_eval_file(no_assert_path, focal_turn=args.focal_turn)
    assert_cases = parse_eval_file(assert_path, focal_turn=args.focal_turn)
    no_assert_scores = load_score_file(no_assert_score_path)
    assert_scores = load_score_file(assert_score_path)
    metadata = load_assertion_metadata(metadata_path)

    LOGGER.info(
        "Loaded %d assert cases and %d no-assert cases", len(assert_cases), len(no_assert_cases)
    )

    _validate_case_sets(assert_cases, no_assert_cases)
    _validate_score_sets(assert_scores, no_assert_scores, assert_cases)

    LOGGER.info("Computing metrics...")
    compliance_cases, compliance_agg = compute_compliance_metrics(assert_cases, metadata)
    step_metrics = compute_step_metrics(assert_cases, no_assert_cases)
    tool_metrics = compute_tool_error_metrics(assert_cases, no_assert_cases)
    outcome_groups, outcome_per_case = _compute_outcome_groups(
        assert_cases,
        no_assert_cases,
        assert_scores,
        no_assert_scores,
        metadata,
    )
    bundle = bundle_metrics(
        compliance_cases,
        compliance_agg,
        step_metrics,
        tool_metrics,
        outcome_groups,
    )

    summary_payload = _serialise_bundle(bundle, compliance_cases, step_metrics, tool_metrics)
    cases_payload = _serialise_cases(compliance_cases, step_metrics, tool_metrics, outcome_per_case)

    _write_json(Path(args.summary_json), summary_payload)
    _write_json(Path(args.case_json), cases_payload)
    _write_report(Path(args.report), bundle)

    LOGGER.info(
        "TCR=%.3f, PCR=%.3f, NCR=%.3f, CR=%.3f",
        bundle.compliance.transient_rate,
        bundle.compliance.persistent_rate,
        bundle.compliance.non_compliance_rate,
        bundle.compliance.correction_rate,
    )
    LOGGER.info(
        "Δsteps mean=%.2f median=%.2f p90=%.2f",
        bundle.steps.mean_delta,
        bundle.steps.median_delta,
        bundle.steps.p90_delta,
    )
    LOGGER.info(
        "ΔTE=%.3f (assert=%.3f, no-assert=%.3f)",
        bundle.tool_errors.delta_error_rate,
        bundle.tool_errors.assert_error_rate,
        bundle.tool_errors.no_assert_error_rate,
    )

    if args.plots:
        plots_dir = Path(args.plots_dir)
        plot_compliance(bundle.compliance, plots_dir)
        plot_step_ecdf(bundle.steps, plots_dir)
        plot_error_rates(bundle.tool_errors, plots_dir)
        LOGGER.info("Plots written to %s", plots_dir)

if __name__ == "__main__":  # pragma: no cover
    main()
