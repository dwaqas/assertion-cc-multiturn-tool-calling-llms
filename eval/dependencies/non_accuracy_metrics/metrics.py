from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
from .structures import (
    AssertionInfo,
    CaseCompliance,
    ComplianceAggregate,
    EvalCase,
    MetricsBundle,
    OutcomeGroupMetrics,
    TurnSummary,
)

LOGGER = logging.getLogger(__name__)


def _classify_compliance(case: EvalCase, assertion: AssertionInfo) -> CaseCompliance:
    target = assertion.target_function
    if not target:
        # No function to evaluate → unknown case
        return CaseCompliance(
            case_id=case.case_id,
            status="unknown",
            initial_compliance=False,
            target_function=None,
            evaluated_turn=None,
        )

    target_lower = target.lower()
    evaluated_turn = assertion.turn_idx
    evaluated_turn_index: Optional[int] = None

    if evaluated_turn is None:
        scoped_calls = [name.lower() for name in case.tool_call_sequence]
    else:
        target_turn: Optional[TurnSummary]
        if evaluated_turn < 0:
            target_turn = case.turns[-1] if case.turns else None
        else:
            target_turn = next(
                (turn for turn in case.turns if turn.index == evaluated_turn), None
            )

        if target_turn is None:
            if case.turns:
                LOGGER.debug(
                    "Missing turn %s for case %s; defaulting to final observed turn",
                    evaluated_turn,
                    case.case_id,
                )
                target_turn = case.turns[-1]
            else:
                return CaseCompliance(
                    case_id=case.case_id,
                    status="unknown",
                    initial_compliance=False,
                    target_function=target,
                    evaluated_turn=evaluated_turn,
                )

        evaluated_turn_index = target_turn.index
        scoped_calls = [name.lower() for name in target_turn.tool_calls]

    if target_lower in scoped_calls:
        status = "compliant"
        initial_compliance = True
    else:
        status = "non_compliant"
        initial_compliance = False

    if not scoped_calls:
        status = "non_compliant"
        initial_compliance = False

    return CaseCompliance(
        case_id=case.case_id,
        status=status,
        initial_compliance=initial_compliance,
        target_function=target,
        evaluated_turn=evaluated_turn_index if evaluated_turn_index is not None else evaluated_turn,
    )


def _classify_followup_compliance(case: EvalCase, assertion: AssertionInfo) -> CaseCompliance:
    followup = (
        (assertion.followup_function or {}).get("name")
        if assertion.followup_function
        else assertion.followup_function_name
    )
    if not followup:
        return CaseCompliance(
            case_id=case.case_id,
            status="unknown",
            initial_compliance=False,
            target_function=None,
            evaluated_turn=assertion.turn_idx,
        )

    followup_lower = followup.lower()
    target_turn = assertion.turn_idx if assertion.turn_idx is not None else 0
    evaluated_turn_index: Optional[int] = None
    scoped_calls: List[str] = []

    if not case.turns:
        return CaseCompliance(
            case_id=case.case_id,
            status="unknown",
            initial_compliance=False,
            target_function=followup,
            evaluated_turn=target_turn,
        )

    turns_sorted = sorted(case.turns, key=lambda t: t.index)
    for turn in turns_sorted:
        if turn.index < (target_turn if target_turn >= 0 else turns_sorted[-1].index):
            continue
        evaluated_turn_index = turn.index
        scoped_calls.extend([name.lower() for name in turn.tool_calls])
        if followup_lower in scoped_calls:
            break

    status = "compliant" if followup_lower in scoped_calls else "non_compliant"

    return CaseCompliance(
        case_id=case.case_id,
        status=status,
        initial_compliance=(status == "compliant"),
        target_function=followup,
        evaluated_turn=evaluated_turn_index,
    )


def compute_compliance_metrics(
    assert_cases: Dict[str, EvalCase],
    metadata: Dict[str, AssertionInfo],
    excluded_ids: Optional[set[str]] = None,
) -> Tuple[List[CaseCompliance], ComplianceAggregate]:
    compliance_cases: List[CaseCompliance] = []
    considered_cases = 0
    compliant_count = 0
    non_compliant_count = 0
    excluded_ids = excluded_ids or set()

    for case_id, case in assert_cases.items():
        info = metadata.get(case_id)
        if info is None:
            compliance = CaseCompliance(
                case_id=case.case_id,
                status="unknown",
                initial_compliance=False,
                target_function=None,
                evaluated_turn=None,
            )
        else:
            compliance = _classify_compliance(case, info)

        compliance_cases.append(compliance)

        if compliance.status == "unknown" or case_id in excluded_ids:
            continue

        considered_cases += 1
        if compliance.status == "compliant":
            compliant_count += 1
        else:
            non_compliant_count += 1

    denominator = considered_cases if considered_cases else 1
    compliance_rate = compliant_count / denominator
    non_compliance_rate = non_compliant_count / denominator

    aggregate = ComplianceAggregate(
        total_cases=len(assert_cases),
        considered_cases=considered_cases,
        compliance_rate=compliance_rate,
        non_compliance_rate=non_compliance_rate,
    )

    compliance_cases.sort(key=lambda entry: entry.case_id)
    return compliance_cases, aggregate


def compute_followup_compliance_metrics(
    assert_cases: Dict[str, EvalCase],
    metadata: Dict[str, AssertionInfo],
    excluded_ids: Optional[set[str]] = None,
) -> Tuple[List[CaseCompliance], ComplianceAggregate]:
    compliance_cases: List[CaseCompliance] = []
    considered_cases = 0
    compliant_count = 0
    non_compliant_count = 0
    excluded_ids = excluded_ids or set()

    for case_id, case in assert_cases.items():
        info = metadata.get(case_id)
        if info is None:
            compliance = CaseCompliance(
                case_id=case.case_id,
                status="unknown",
                initial_compliance=False,
                target_function=None,
                evaluated_turn=None,
            )
        else:
            compliance = _classify_followup_compliance(case, info)

        compliance_cases.append(compliance)

        if compliance.status == "unknown" or case_id in excluded_ids:
            continue

        considered_cases += 1
        if compliance.status == "compliant":
            compliant_count += 1
        else:
            non_compliant_count += 1

    denominator = considered_cases if considered_cases else 1
    compliance_rate = compliant_count / denominator
    non_compliance_rate = non_compliant_count / denominator

    aggregate = ComplianceAggregate(
        total_cases=len(assert_cases),
        considered_cases=considered_cases,
        compliance_rate=compliance_rate,
        non_compliance_rate=non_compliance_rate,
    )

    compliance_cases.sort(key=lambda entry: entry.case_id)
    return compliance_cases, aggregate


def bundle_metrics(
    compliance_cases: List[CaseCompliance],
    compliance_agg: ComplianceAggregate,
    outcome_groups: List[OutcomeGroupMetrics],
) -> MetricsBundle:
    compliant_cases = [c for c in compliance_cases if c.status == "compliant"]
    non_compliant_cases = [c for c in compliance_cases if c.status == "non_compliant"]
    unknown_cases = [c for c in compliance_cases if c.status == "unknown"]

    return MetricsBundle(
        compliance=compliance_agg,
        compliant_cases=compliant_cases,
        non_compliant_cases=non_compliant_cases,
        unknown_cases=unknown_cases,
        outcome_groups=outcome_groups,
    )
