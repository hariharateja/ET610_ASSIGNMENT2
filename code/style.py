"""Shared plotting style for all report figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Categorical slots (fixed order)
C1, C2, C3, C4, C5, C6, C7, C8 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                  "#e87ba4", "#008300", "#4a3aa7", "#e34948")

OUTCOME_COLORS = {"Distinction": C1, "Pass": C3, "Fail": C4, "Withdrawn": C8}
OUTCOME_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]


def apply():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.labelsize": 8.5, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "lines.linewidth": 2.0, "axes.titlelocation": "left",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
