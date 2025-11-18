from __future__ import annotations
import logging
from pathlib import Path
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
from .structures import ComplianceAggregate

LOGGER = logging.getLogger(__name__)

# Guard against optional plotting dependency;
def ensure_matplotlib() -> bool:
    if plt is None:
        LOGGER.warning("matplotlib not available; skipping plot generation")
        return False
    return True

def plot_compliance(aggregate: ComplianceAggregate, output_dir: Path) -> None:
    if not ensure_matplotlib():
        return

    labels = ["Compliance", "Non-compliance"]
    values = [
        aggregate.compliance_rate * 100,
        aggregate.non_compliance_rate * 100,
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#22c55e", "#ef4444"])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Compliance Rates")
    ax.set_ylim(0, max(values + [10]))
    ax.bar_label(bars, fmt="%.1f%%")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "compliance_rates.svg")
    plt.close(fig)
