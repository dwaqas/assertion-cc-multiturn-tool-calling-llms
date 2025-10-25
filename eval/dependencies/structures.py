from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

@dataclass
class CaseCompliance:
    case_id: str
    status: str
    initial_compliance: bool
    corrected: bool
    target_function: Optional[str]

@dataclass
class ComplianceAggregate:
    total_cases: int
    considered_cases: int
    persistent_rate: float
    transient_rate: float
    non_compliance_rate: float
    correction_rate: float

@dataclass
class CaseStepDelta:
    case_id: str
    assert_steps: int
    no_assert_steps: int
    delta: int

@dataclass
class StepMetrics:
    mean_delta: float
    median_delta: float
    p90_delta: float
    per_case: List[CaseStepDelta]

@dataclass
class CaseToolUsage:
    case_id: str
    assert_tool_calls: int
    assert_errors: int
    no_assert_tool_calls: int
    no_assert_errors: int

@dataclass
class ErrorTypeShift:
    error_type: str
    assert_count: int
    no_assert_count: int
    delta: int

@dataclass
class ToolErrorMetrics:
    assert_error_rate: float
    no_assert_error_rate: float
    delta_error_rate: float
    per_case: List[CaseToolUsage]
    top_error_types: List[ErrorTypeShift]

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
    steps: StepMetrics
    tool_errors: ToolErrorMetrics

@dataclass
class MetricsBundle:
    compliance: ComplianceAggregate
    steps: StepMetrics
    tool_errors: ToolErrorMetrics
    transient_cases: List[CaseCompliance] = field(default_factory=list)
    persistent_cases: List[CaseCompliance] = field(default_factory=list)
    non_compliant_cases: List[CaseCompliance] = field(default_factory=list)
    outcome_groups: List[OutcomeGroupMetrics] = field(default_factory=list)
