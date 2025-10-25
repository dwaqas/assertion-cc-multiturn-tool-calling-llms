from __future__ import annotations
import logging
from pathlib import Path
from typing import List
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
from .structures import ComplianceAggregate, StepMetrics, ToolErrorMetrics

LOGGER = logging.getLogger(__name__)

def ensure_matplotlib() -> bool:
    if plt is None:
        LOGGER.warning("matplotlib not available; skipping plot generation")
        return False
    return True

def plot_compliance(aggregate: ComplianceAggregate, output_dir: Path) -> None:
    if not ensure_matplotlib():
        return

    labels = ["TCR", "PCR", "NCR", "CR"]
    values = [
        aggregate.transient_rate * 100,
        aggregate.persistent_rate * 100,
        aggregate.non_compliance_rate * 100,
        aggregate.correction_rate * 100,
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#f97316", "#22c55e", "#ef4444", "#6366f1"])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Compliance Rates")
    ax.set_ylim(0, max(values + [10]))
    ax.bar_label(bars, fmt="%.1f%%")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "compliance_rates.svg")
    plt.close(fig)

def plot_step_ecdf(step_metrics: StepMetrics, output_dir: Path) -> None:
    if not ensure_matplotlib():
        return
    deltas = sorted(c.delta for c in step_metrics.per_case)
    if not deltas:
        LOGGER.info("No step deltas available; skipping ECDF plot")
        return
    n = len(deltas)
    y = [i / n for i in range(1, n + 1)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(deltas, y, where="post", color="#0ea5e9")
    ax.set_xlabel("Step Delta (assert - no-assert)")
    ax.set_ylabel("ECDF")
    ax.set_title("Step Delta ECDF")
    ax.grid(True, alpha=0.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "step_delta_ecdf.svg")
    plt.close(fig)

def plot_error_rates(tool_metrics: ToolErrorMetrics, output_dir: Path) -> None:
    if not ensure_matplotlib():
        return

    labels = ["No-assert", "Assert"]
    values = [tool_metrics.no_assert_error_rate * 100, tool_metrics.assert_error_rate * 100]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#10b981", "#dc2626"])
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Tool Error Rates")
    ax.bar_label(bars, fmt="%.2f%%")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "tool_error_rates.svg")
    plt.close(fig)
