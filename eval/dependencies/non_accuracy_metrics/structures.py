from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

# Typed containers shared across evaluation stages; 

@dataclass
class TurnSummary:
    index: int
    tool_calls: List[str]
    text_messages: List[str]
    error_types: List[str]
    had_error: bool

@dataclass
class EvalCase:
    case_id: str
    turns: List[TurnSummary]
    total_steps: int
    total_tool_calls: int
    total_errors: int
    error_types: Counter[str]
    all_text: str
    tool_call_sequence: List[str]
    tool_call_turns: List[int]

@dataclass
class AssertionInfo:
    case_id: str
    target_function: Optional[str]
    assertion_text: str
    turn_idx: Optional[int] = None
    injection_position: Optional[str] = None
    followup_function_name: Optional[str] = None
    followup_function: Optional[dict] = None

@dataclass
class CaseCompliance:
    case_id: str
    status: str
    initial_compliance: bool
    target_function: Optional[str]
    evaluated_turn: Optional[int] = None

# Aggregated compliance ratios for quick reporting;
@dataclass
class ComplianceAggregate:
    total_cases: int
    considered_cases: int
    compliance_rate: float
    non_compliance_rate: float

@dataclass
class ScoreCase:
    case_id: str
    success: bool
    error_type: Optional[str]

@dataclass
class CaseOutcome:
    case_id: str
    category: str
    no_assert_success: bool
    assert_success: bool
    no_assert_error: Optional[str]
    assert_error: Optional[str]

@dataclass
class OutcomeGroupMetrics:
    category: str
    label: str
    count: int
    compliance: ComplianceAggregate

# Bundled aggregates plus optional per-case views; 
@dataclass
class MetricsBundle:
    compliance: ComplianceAggregate

    compliant_cases: List[CaseCompliance] = field(default_factory=list)
    non_compliant_cases: List[CaseCompliance] = field(default_factory=list)
    unknown_cases: List[CaseCompliance] = field(default_factory=list)

    outcome_groups: List[OutcomeGroupMetrics] = field(default_factory=list)
