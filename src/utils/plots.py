"""Matplotlib styling helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Headless backend; required for offline runs.

import matplotlib.pyplot as plt


PRIMARY = "#1B3A5C"
ACCENT = "#2E75B6"
WARM = "#C0392B"
GREY = "#7F8C8D"


def apply_style() -> None:
    """Apply a clean, audit-friendly matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 1.7,
            "legend.frameon": False,
        }
    )


def save_both(fig, path_stem: str) -> None:
    """Save a figure as both PNG (300 DPI) and SVG."""
    fig.savefig(f"{path_stem}.png", bbox_inches="tight")
    fig.savefig(f"{path_stem}.svg", bbox_inches="tight")
