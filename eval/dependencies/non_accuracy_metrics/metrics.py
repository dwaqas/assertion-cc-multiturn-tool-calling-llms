from __future__ import annotations
import logging
import math
import statistics
from collections import Counter
from typing import Dict, Iterable, List, Tuple
from .structures import (
    AssertionInfo,
    CaseCompliance,
    CaseStepDelta,
    CaseToolUsage,
    ComplianceAggregate,
    EvalCase,
    MetricsBundle,
    OutcomeGroupMetrics,
    StepMetrics,
    ToolErrorMetrics,
    TurnSummary,
    ErrorTypeShift,
)

LOGGER = logging.getLogger(__name__)

CORRECTION_PHRASES = (
    "actually",
    "correction",
    "i was mistaken",
    "sorry",
    "apolog",
    "let me correct",
    "retry without",
    "that was wrong",
)

def _collect_text_after_turn(case: EvalCase, turn_index: int) -> str:
    fragments: List[str] = []
    for turn in case.turns[turn_index + 1 :]:
        fragments.extend(turn.text_messages)
    return " ".join(fragments).lower()

def _final_text(case: EvalCase) -> str:
    for turn in reversed(case.turns):
        if turn.text_messages:
            return " ".join(turn.text_messages).lower()
    return case.all_text.lower()

def _classify_compliance(case: EvalCase, assertion: AssertionInfo) -> CaseCompliance:

    target = assertion.target_function
    if not target:
        # No function to evaluate → unknown case
        return CaseCompliance(
            case_id=case.case_id,
            status="unknown",
            initial_compliance=False,
            corrected=False,
            target_function=None,
        )

    if not case.tool_call_sequence:
        # No recorded tool usage → unknown, not non-compliant
        return CaseCompliance(
            case_id=case.case_id,
            status="unknown",
            initial_compliance=False,
            corrected=False,
            target_function=target,
        )

    target_lower = target.lower()
    call_sequence = [name.lower() for name in case.tool_call_sequence]

    # Compliance simply means: did the model ever call the function?
    if target_lower in call_sequence:
        status = "compliant"
        initial_compliance = True
    else:
        status = "non_compliant"
        initial_compliance = False

    # We are intentionally NOT tracking "corrected" anymore
    return CaseCompliance(
        case_id=case.case_id,
        status=status,
        initial_compliance=initial_compliance,
        corrected=False,
        target_function=target,
    )


def compute_compliance_rate(compliance_cases: List[CaseCompliance]) -> dict:
    total = len(compliance_cases)
    compliant = sum(1 for c in compliance_cases if c.status == "compliant")
    non_compliant = sum(1 for c in compliance_cases if c.status == "non_compliant")
    unknown = sum(1 for c in compliance_cases if c.status == "unknown")

    return {
        "total_cases": total,
        "compliance_rate": compliant / total if total > 0 else 0.0,
        "non_compliance_rate": non_compliant / total if total > 0 else 0.0,
        "unknown_rate": unknown / total if total > 0 else 0.0,
        "counts": {
            "compliant": compliant,
            "non_compliant": non_compliant,
            "unknown": unknown
        }
    }


def _quantile(data: List[float], quantile: float) -> float:
    if not data:
        return 0.0
    if len(data) == 1:
        return float(data[0])
    ordered = sorted(float(x) for x in data)
    k = (len(ordered) - 1) * quantile
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)

def compute_step_metrics(
    assert_cases: Dict[str, EvalCase],
    no_assert_cases: Dict[str, EvalCase],
) -> StepMetrics:
    per_case: List[CaseStepDelta] = []

    for case_id, assert_case in assert_cases.items():
        other = no_assert_cases.get(case_id)
        if not other:
            LOGGER.warning("No matching no-assert case for %s; skipping step delta", case_id)
            continue
        delta = assert_case.total_steps - other.total_steps
        per_case.append(
            CaseStepDelta(
                case_id=case_id,
                assert_steps=assert_case.total_steps,
                no_assert_steps=other.total_steps,
                delta=delta,
            )
        )

    deltas = [c.delta for c in per_case]
    mean_delta = statistics.fmean(deltas) if deltas else 0.0
    median_delta = float(statistics.median(deltas)) if deltas else 0.0
    p90_delta = _quantile([float(d) for d in deltas], 0.90) if deltas else 0.0

    return StepMetrics(
        mean_delta=mean_delta,
        median_delta=median_delta,
        p90_delta=p90_delta,
        per_case=per_case,
    )

def compute_tool_error_metrics(
    assert_cases: Dict[str, EvalCase],
    no_assert_cases: Dict[str, EvalCase],
) -> ToolErrorMetrics:
    per_case: List[CaseToolUsage] = []
    total_assert_calls = 0
    total_no_calls = 0
    total_assert_errors = 0
    total_no_errors = 0
    assert_error_counter: Counter[str] = Counter()
    no_error_counter: Counter[str] = Counter()

    for case_id, assert_case in assert_cases.items():
        other = no_assert_cases.get(case_id)
        if not other:
            LOGGER.warning("No matching no-assert case for %s; skipping tool error delta", case_id)
            continue

        per_case.append(
            CaseToolUsage(
                case_id=case_id,
                assert_tool_calls=assert_case.total_tool_calls,
                assert_errors=assert_case.total_errors,
                no_assert_tool_calls=other.total_tool_calls,
                no_assert_errors=other.total_errors,
            )
        )

        total_assert_calls += assert_case.total_tool_calls
        total_no_calls += other.total_tool_calls
        total_assert_errors += assert_case.total_errors
        total_no_errors += other.total_errors
        assert_error_counter.update(assert_case.error_types)
        no_error_counter.update(other.error_types)

    assert_rate = (total_assert_errors / total_assert_calls) if total_assert_calls else 0.0
    no_rate = (total_no_errors / total_no_calls) if total_no_calls else 0.0
    delta_rate = assert_rate - no_rate

    # Determine top error type shifts
    combined_types = set(assert_error_counter) | set(no_error_counter)
    shifts: List[ErrorTypeShift] = []
    for error_type in combined_types:
        a_count = assert_error_counter.get(error_type, 0)
        n_count = no_error_counter.get(error_type, 0)
        shifts.append(
            ErrorTypeShift(
                error_type=error_type,
                assert_count=a_count,
                no_assert_count=n_count,
                delta=a_count - n_count,
            )
        )

    shifts.sort(key=lambda item: abs(item.delta), reverse=True)
    top_shifts = shifts[:5]

    return ToolErrorMetrics(
        assert_error_rate=assert_rate,
        no_assert_error_rate=no_rate,
        delta_error_rate=delta_rate,
        per_case=per_case,
        top_error_types=top_shifts,
    )

def bundle_metrics(
    compliance_cases: List[CaseCompliance],
    compliance_agg: ComplianceAggregate,
    step_metrics: StepMetrics,
    tool_error_metrics: ToolErrorMetrics,
    outcome_groups: List[OutcomeGroupMetrics],
) -> MetricsBundle:

    compliant_cases = [c for c in compliance_cases if c.status == "compliant"]
    non_compliant_cases = [c for c in compliance_cases if c.status == "non_compliant"]
    unknown_cases = [c for c in compliance_cases if c.status == "unknown"]

    return MetricsBundle(
        compliance=compliance_agg,
        steps=step_metrics,
        tool_errors=tool_error_metrics,
        compliant_cases=compliant_cases,
        non_compliant_cases=non_compliant_cases,
        unknown_cases=unknown_cases,
        outcome_groups=outcome_groups,
    )


