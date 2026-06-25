"""
Shared plotting style for the HiMuon benchmark suite.

Conventions: navy protagonist (HiMuon) vs. hatched gray baseline (Muon), 2-3
colors per figure, sans-serif fonts, white background, no grid, top/right spines
removed, value labels on bars.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
_STYLE_FILE = os.path.join(os.path.dirname(__file__), "publication.mplstyle")


def apply_style():
    """Apply the unified publication style. Call once at script start."""
    if os.path.exists(_STYLE_FILE):
        plt.style.use(_STYLE_FILE)
    plt.rcParams.update(
        {
            # Font — sans-serif matches modern ML papers
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            # Sizes — larger for readability at paper scale
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            # Spines — only left and bottom
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            # Grid — off by default
            "axes.grid": False,
            # Figure
            "figure.facecolor": "white",
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            # Legend
            "legend.frameon": False,
            "legend.borderpad": 0.3,
            "legend.handlelength": 1.5,
        }
    )


# ---------------------------------------------------------------------------
# Color palettes — switch by changing PALETTE below
# ---------------------------------------------------------------------------
# Option A: navy + gray (default)
_PALETTE_INDEX = {
    "HIMUON": "#2B5EA7",
    "MUON": "#888888",
    "ADAMW": "#BBBBBB",
    "OPT": "#D4621A",
    "FWD": "#A8CBE0",
    "BWD": "#2B5EA7",
}
# Option B: warm vs cool
_PALETTE_DUAL = {
    "HIMUON": "#E8734A",
    "MUON": "#4AAECC",
    "ADAMW": "#BBBBBB",
    "OPT": "#E8734A",
    "FWD": "#A8CBE0",
    "BWD": "#4AAECC",
}
# Option C: muted academic
_PALETTE_EFFI = {
    "HIMUON": "#2B4590",
    "MUON": "#4A9060",
    "ADAMW": "#D4A020",
    "OPT": "#D4621A",
    "FWD": "#A8CBE0",
    "BWD": "#2B4590",
}

PALETTE = _PALETTE_INDEX  # <-- change this to switch palette


class COLORS:
    # Optimizer identity (driven by PALETTE)
    HIMUON = PALETTE["HIMUON"]
    MUON = PALETTE["MUON"]
    ADAMW = PALETTE["ADAMW"]

    # Training step phases (driven by PALETTE)
    FWD = PALETTE["FWD"]
    BWD = PALETTE["BWD"]
    OPT = PALETTE["OPT"]

    # Status
    OK = "#2B7A4B"  # muted green
    WARN = "#D4A020"  # golden yellow
    FAIL = "#C4541A"  # muted vermillion

    # Neutral
    GRAY = "#888888"
    LIGHT_GRAY = "#CCCCCC"
    BLACK = "#333333"

    @classmethod
    def for_optimizer(cls, name):
        return {
            "adamw": cls.ADAMW,
            "muon": cls.MUON,
            "himuon": cls.HIMUON,
        }.get(name.lower(), cls.GRAY)


# ---------------------------------------------------------------------------
# Dtype colors for numerical stability plots
# ---------------------------------------------------------------------------
DTYPE_COLORS = {
    "bf16": "#2B5EA7",  # navy
    "fp16": "#2B7A4B",  # muted green
    "fp8e5m2": "#D4A020",  # golden
    "fp8e4m3": "#C4541A",  # vermillion
}

DTYPE_MARKERS = {
    "bf16": "o",
    "fp16": "s",
    "fp8e5m2": "D",
    "fp8e4m3": "^",
}

# ---------------------------------------------------------------------------
# Scale mode colors
# ---------------------------------------------------------------------------
SCALE_MODE_COLORS = {
    "none": "#888888",
    "tile_row": "#2B5EA7",
    "tile_col": "#2B7A4B",
    "tile": "#D4A020",
    "row": "#C4541A",
    "col": "#6B4C9A",
}

# ---------------------------------------------------------------------------
# Model colors (for real_shapes, throughput)
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "Qwen3-0.6B": "#D4A020",  # golden
    "Qwen3-1.7B": "#2B5EA7",  # navy
    "Qwen3-4B": "#2B7A4B",  # green
}

MODEL_MARKERS = {
    "Qwen3-0.6B": "o",
    "Qwen3-1.7B": "s",
    "Qwen3-4B": "D",
}

# ---------------------------------------------------------------------------
# Hatching patterns
# ---------------------------------------------------------------------------
HATCH_BASELINE = "////"  # for Muon / baseline bars
HATCH_NONE = ""  # for HiMuon / protagonist bars

# ---------------------------------------------------------------------------
# Standard figure sizes (inches)
# ---------------------------------------------------------------------------
FIGSIZE_SINGLE = (5, 4)
FIGSIZE_WIDE = (10, 4.5)
FIGSIZE_TRIPLE = (14, 4.5)
FIGSIZE_HEATMAP = (8, 6)

# Standardized line/marker sizes for cross-figure consistency
LINE_WIDTH = 2.0  # primary lines
LINE_WIDTH_SECONDARY = 1.2  # secondary / dashed lines
MARKER_SIZE = 8  # primary markers
MARKER_SIZE_SMALL = 6  # secondary markers


# ---------------------------------------------------------------------------
# Spine cleanup helper
# ---------------------------------------------------------------------------
def clean_spines(ax):
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Bar styling
# ---------------------------------------------------------------------------
BAR_DEFAULTS = dict(
    alpha=0.9,
    edgecolor="#333333",
    linewidth=0.5,
    width=0.32,
    zorder=3,
)

ERRORBAR_KW = dict(
    elinewidth=0.7,
    ecolor="#444444",
    capsize=2.5,
    capthick=0.6,
)


def styled_bar(
    ax,
    x,
    values,
    width=None,
    bottom=None,
    yerr=None,
    color=None,
    label=None,
    hatch="",
    **kwargs,
):
    """Draw a bar with publication defaults."""
    kw = dict(BAR_DEFAULTS)
    if width is not None:
        kw["width"] = width
    if color is not None:
        kw["color"] = color
    if label is not None:
        kw["label"] = label
    if bottom is not None:
        kw["bottom"] = bottom
    if hatch:
        kw["hatch"] = hatch
    if yerr is not None:
        kw["yerr"] = yerr
        kw["error_kw"] = dict(ERRORBAR_KW)
    kw.update(kwargs)
    return ax.bar(x, values, **kw)


def bar_value_labels(ax, x, values, fmt="{:.1f}", offset=0.02, **kwargs):
    """Add value labels above bars."""
    finite = [v for v in values if np.isfinite(v) and v > 0]
    ymax = max(finite) if finite else 1
    defaults = dict(ha="center", va="bottom", fontsize=8, color="#333333")
    defaults.update(kwargs)
    for xi, v in zip(x, values):
        if np.isfinite(v) and v > 0:
            ax.text(xi, v + ymax * offset, fmt.format(v), **defaults)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_plots_dir():
    """Return the plots/ directory next to this file, creating it if needed."""
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def save_fig(fig, name, out_dir=None, dpi=300):
    """Save figure to out_dir/name.png. Defaults to plots/ directory."""
    if out_dir is None:
        out_dir = get_plots_dir()
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Plot saved to {path}")
    return path


def panel_label(ax, label):
    """Set panel label as part of axes title, e.g. '(a) Title'.

    Usage: ax.set_title("Title"); panel_label(ax, "a")
    This prepends '(a) ' to the existing title.
    """
    current = ax.get_title()
    ax.set_title(f"({label}) {current}")


def annotate_speedup(
    ax, x, ref_vals, vals, fmt="{:.1f}×", offset_frac=0.03, color="#333333", **kwargs
):
    """Add speedup annotations (ref/val) above bars."""
    finite = [v for v in vals if np.isfinite(v)]
    ymax = max(finite) if finite else 1
    defaults = dict(ha="center", va="bottom", fontsize=8, color=color)
    defaults.update(kwargs)
    for xi, (r, v) in zip(x, zip(ref_vals, vals)):
        if r > 0 and v > 0 and np.isfinite(r) and np.isfinite(v):
            ax.text(xi, v + ymax * offset_frac, fmt.format(r / v), **defaults)
