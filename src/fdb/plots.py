"""Risk-coverage plot (T22d): the curves from fdb.curve drawn as a PNG for reports and demos.

Harness side only (matplotlib, no detector code). Uses matplotlib's object API (Figure, no
pyplot), so it needs no screen and keeps no global state; the same curves give the same bytes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from .curve import CurvePoint, aurc, excess_aurc, optimal_aurc

# Colour-blind-safe pair; the line style differs too, so the chart still reads in greyscale.
SPLIT_STYLE = {
    "cal": {"color": "#2a78d6", "linestyle": "-"},
    "test": {"color": "#eb6834", "linestyle": "--"},
}
OTHER_SPLIT = {"color": "#1baf7a", "linestyle": "-"}
REFERENCE_INK = "#52514e"
BUDGET_INK = "#8a8984"
GRID_INK = "#ecebe8"


def optimal_curve(n: int, defective: int) -> tuple[list[float], list[float]]:
    """Coverage and selective risk of a perfect score: every normal frame first, then the defective ones."""
    if not 0 <= defective <= n or n == 0:
        raise ValueError(f"need 0 <= defective <= n and n > 0, got n={n}, defective={defective}")
    normal = n - defective
    ks = range(1, n + 1)
    return [k / n for k in ks], [max(0, k - normal) / k for k in ks]


def plot_risk_coverage(path: Path, curves: Mapping[str, list[CurvePoint]], reference: str,
                       budgets: Sequence[float] = (), title: str = "", note: str = "") -> Path:
    """Draw one staircase per split plus the useless and perfect references; save as PNG."""
    if not curves:
        raise ValueError("nothing to plot: no curves given")
    if reference not in curves:
        raise ValueError(f"reference split {reference!r} is not one of {sorted(curves)}")

    fig = Figure(figsize=(8, 5.5), dpi=150)
    ax = fig.subplots()

    # References, from one split (cal and test are stratified, so their defect rates match closely).
    ref = curves[reference][-1].counts
    rate = ref.defective / ref.n
    ax.axhline(rate, color=REFERENCE_INK, linestyle=":", linewidth=1.5,
               label=f"useless score: risk = defect rate {rate:.1%} ({reference})")
    ox, oy = optimal_curve(ref.n, ref.defective)
    ax.step(ox, oy, where="pre", color=REFERENCE_INK, linestyle="-.", linewidth=1.5,
            label=f"perfect score: AURC {optimal_aurc(ref.n, ref.defective):.4f} ({reference})")

    for i, eps in enumerate(sorted(budgets)):
        ax.axhline(eps, color=BUDGET_INK, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1,
                   label="risk budgets ε: " + ", ".join(f"{e:.1%}" for e in sorted(budgets)) if i == 0 else None)

    # The model's curves. 'pre' steps: the risk at each point covers the coverage it added,
    # which is exactly the area fdb.curve.aurc sums. Coverage 0 (risk undefined) is not drawn.
    for split, points in curves.items():
        shown = [p for p in points if p.counts.accepted > 0]
        ax.step([p.coverage for p in shown], [p.selective_risk for p in shown], where="pre",
                linewidth=2, zorder=3, **SPLIT_STYLE.get(split, OTHER_SPLIT),
                label=f"{split}: AURC {aurc(points):.4f}, E-AURC {excess_aurc(points):.4f}")

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Coverage: share of frames auto-passed")
    ax.set_ylabel("Selective risk: defective among auto-passed")
    ax.grid(color=GRID_INK, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=11, pad=20 if note else 6)
    if note:
        ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=8, color=REFERENCE_INK, style="italic")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=8)

    path = Path(path)
    fig.savefig(path, format="png", bbox_inches="tight", metadata={"Software": None})
    return path