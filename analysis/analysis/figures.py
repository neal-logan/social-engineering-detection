"""
Figure generation. Each public function:
  - returns the underlying object (matplotlib Figure or pandas DataFrame),
  - displays inline (via plt.show() / IPython.display) when called in a
    notebook,
  - saves to `analysis_output/` as PNG and SVG (figures) and as CSV
    (tables) under the supplied artifact name.

Usage from notebook:
    from analysis.figures import (
        save_table, save_figure, hist_first_violation_turn, ...
    )
    set_output_dir("analysis_output")  # done once

The output directory is configured per-process via `set_output_dir`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import schema as S


# ---------------------------------------------------------------------------
# UNC Charlotte color palette
# ---------------------------------------------------------------------------
# Official primary brand colors (https://brand.charlotte.edu/visual-identity/color-palette/):
#   Charlotte Green  PMS 7484   #005035
#   Niner Gold       PMS 7503   #A49665
# Athletic-mark variants (slightly brighter):
#   Athletic Green   PMS 349    #046A38
#   Athletic Gold    PMS 465    #B9975B
# We expose a curated palette anchored on the official primaries with
# supporting tones for series differentiation.

UNCC_GREEN_DARK   = "#005035"   # Charlotte Green (primary)
UNCC_GREEN_BRIGHT = "#046A38"   # Athletic green (lighter, more saturated)
UNCC_GOLD         = "#A49665"   # Niner Gold (primary)
UNCC_GOLD_BRIGHT  = "#B9975B"   # Athletic gold (lighter, more saturated)
UNCC_BLACK        = "#27251F"   # Athletic black
UNCC_GREY_DARK    = "#5A5A5A"
UNCC_GREY         = "#8C8C8C"
UNCC_GREY_LIGHT   = "#CCCCCC"

# Default series palette: green-and-gold leading, neutrals filling out.
# Used by ROC curves and any other multi-series plots.
UNCC_PALETTE: tuple[str, ...] = (
    UNCC_GREEN_DARK,
    UNCC_GOLD,
    UNCC_GREEN_BRIGHT,
    UNCC_GOLD_BRIGHT,
    UNCC_BLACK,
    UNCC_GREY_DARK,
    UNCC_GREY,
    UNCC_GREY_LIGHT,
)

# Single-color anchors for specific roles
COLOR_PRIMARY = UNCC_GREEN_DARK    # primary signal (e.g., "predicted positive")
COLOR_SECONDARY = UNCC_GOLD        # secondary signal (e.g., "actual violation")
COLOR_NEUTRAL = UNCC_GREY          # neutral / "before"
COLOR_HIGHLIGHT = UNCC_GREEN_BRIGHT # highlighted / "after"
COLOR_DIVIDER = UNCC_BLACK         # axis lines, dividers

# Heatmap colormaps. Both are tuned so that even the most saturated end
# of the gradient stays light enough that black cell-text reads cleanly.
# This is more important for legibility than maximum dynamic range; if
# you need a high-contrast variant for a specific plot, pass `cmap=` to
# heatmap_from_dataframe explicitly.
import matplotlib.colors as _mcolors

# Sequential: pale cream → Niner Gold (warm, monotone, never goes dark)
UNCC_GREEN_CMAP = _mcolors.LinearSegmentedColormap.from_list(
    "uncc_sequential",
    ["#FFFDF5", "#F4ECD2", "#E5D7A1", "#C9B777", UNCC_GOLD],
)

# Diverging: muted gold ← cream → muted Charlotte Green. Both poles are
# tinted but light; the midpoint is near-white so the direction of
# departure from 0.5 is obvious without forcing dark text on dark cells.
UNCC_DIVERGING_CMAP = _mcolors.LinearSegmentedColormap.from_list(
    "uncc_diverging",
    [UNCC_GOLD_BRIGHT, "#F0E5C4", "#FFFDF5", "#C9DDD0", "#5C9479"],
)


# ---------------------------------------------------------------------------
# Output directory management
# ---------------------------------------------------------------------------

_OUTPUT_DIR: Path = Path("analysis_output")


def set_output_dir(path: str | Path) -> None:
    """Set the directory where save_table / save_figure write."""
    global _OUTPUT_DIR
    _OUTPUT_DIR = Path(path)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_output_dir() -> Path:
    return _OUTPUT_DIR


# ---------------------------------------------------------------------------
# Title / label helpers — translate internal variable names into plain English
# ---------------------------------------------------------------------------

_OBJECTIVE_TITLES = {
    "social_engineering": "social engineering attack",
    "policy_violation":   "policy violation",
    "combined":           "combined SE-or-PV",
}

_SCENARIO_TITLES = {
    "se_detection":  "Social engineering attack detection",
    "se_prevention": "Social engineering attack prevention",
    "pv_detection":  "Policy violation detection",
    "pv_on_time":    "Policy violation on-time prediction",
}

_STANCE_TITLES = {
    "high_precision": "high-precision",
    "balanced":       "balanced",
    "high_recall":    "high-recall",
}

_FILTER_TITLES = {
    "threat":  "threat conversations",
    "benign":  "benign conversations",
    None:      "all conversations",
}


def _pretty_objective(obj: str) -> str:
    return _OBJECTIVE_TITLES.get(obj, obj.replace("_", " "))


def _pretty_scenario(scen: str) -> str:
    return _SCENARIO_TITLES.get(scen, scen.replace("_", " "))


def _pretty_stance(stance: str) -> str:
    return _STANCE_TITLES.get(stance, stance.replace("_", " "))


def _pretty_filter(f: Optional[str]) -> str:
    return _FILTER_TITLES.get(f, str(f))


def _pretty_violation(v: str) -> str:
    """improper_authentication -> 'Improper authentication'."""
    return v.replace("_", " ").capitalize()


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_table(name: str, df: pd.DataFrame, *, index: bool = True) -> Path:
    """Write a DataFrame to <output_dir>/<name>.xlsx and return the path.

    Uses xlsx instead of csv to sidestep character-encoding issues
    (e.g. when slice labels carry non-ASCII characters that round-trip
    awkwardly through CSV without explicit encoding).
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUTPUT_DIR / f"{name}.xlsx"
    sheet = name[:31] if name else "Sheet1"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=index, sheet_name=sheet)
    return path


def save_figure(
    name: str,
    fig: plt.Figure,
    *,
    close: bool = False,
) -> tuple[Path, Path]:
    """Write a Figure to <output_dir>/<name>.{png,svg} and return both paths.

    Set `close=True` to release the figure from matplotlib's pyplot
    registry after saving. Default keeps it open for notebook display;
    pass `close=True` from non-interactive scripts producing many figures.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = _OUTPUT_DIR / f"{name}.png"
    svg = _OUTPUT_DIR / f"{name}.svg"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    if close:
        plt.close(fig)
    return png, svg


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

def hist_first_violation_turn_by_type(
    conv_df: pd.DataFrame,
    *,
    conversation_filter: Optional[str] = None,  # "threat", "benign", or None
    name: str = "hist_first_violation_turn_by_type",
) -> plt.Figure:
    """One panel per violation type, distribution of first-violation turn."""
    if conversation_filter is not None:
        df = conv_df[conv_df["conversation_type"] == conversation_filter]
        title_suffix = f" ({conversation_filter})"
    else:
        df = conv_df
        title_suffix = " (combined)"

    fig, axes = plt.subplots(
        1, len(S.POLICY_VIOLATION_TYPES),
        figsize=(4 * len(S.POLICY_VIOLATION_TYPES), 3.2),
        sharey=True,
    )
    for ax, vtype in zip(axes, S.POLICY_VIOLATION_TYPES):
        col = f"first_violation_turn__{vtype}"
        vals = df[col].dropna().to_numpy()
        n_with = len(vals)
        n_without = int(df[col].isna().sum())
        if n_with > 0:
            max_t = int(np.max(vals))
            bins = np.arange(0, max_t + 2) - 0.5
            ax.hist(vals, bins=bins, color=COLOR_PRIMARY, edgecolor="white")
        ax.set_title(
            f"{_pretty_violation(vtype)}\n({n_with} with, {n_without} without)"
        )
        ax.set_xlabel("Turn index")
        ax.set_ylabel("Conversations")
    fig.suptitle(
        f"Turn of first policy violation by violation type — "
        f"{_pretty_filter(conversation_filter)}"
    )
    fig.tight_layout()
    suffix = "" if conversation_filter is None else f"__{conversation_filter}"
    save_figure(name + suffix, fig)
    return fig


def hist_violations_by_type_x_representative(
    conv_df: pd.DataFrame,
    *,
    conversation_filter: Optional[str] = None,  # "threat", "benign", or None
    name: str = "hist_violations_by_type_x_representative",
) -> plt.Figure:
    """Grouped bar chart: total violations of each type, by representative type.

    For the chosen conversation filter (threat / benign / combined), this
    produces one panel per violation type. Within each panel, one bar per
    representative type (the generation dimension `representative`,
    e.g. "by_book", "tired"), height = total count of that violation type
    summed across conversations of that representative type.
    """
    if conversation_filter is not None:
        df = conv_df[conv_df["conversation_type"] == conversation_filter]
        title_suffix = f" ({conversation_filter})"
    else:
        df = conv_df
        title_suffix = " (combined)"

    # Prefer the short-label column built by preliminaries; fall back
    # to the raw prompt-text column with the substring-match labeler.
    if "short_representative" in df.columns:
        rep_col = "short_representative"
    elif "representative" in df.columns:
        df = df.copy()
        df["__rep_short"] = df["representative"].apply(S.representative_short_label)
        rep_col = "__rep_short"
    else:
        raise ValueError(
            "build_conversation_table is missing both 'short_representative' "
            "and 'representative'; cannot break violations down by representative."
        )

    rep_values = list(df[rep_col].dropna().unique())
    # Use canonical order if all observed values are in the canonical list;
    # otherwise sort alphabetically as a fallback so plots remain stable.
    canonical_order = [r for r in S.REPRESENTATIVE_SHORT_LABELS if r in rep_values]
    extras = sorted([r for r in rep_values if r not in canonical_order], key=str)
    rep_values = canonical_order + extras
    table_rows = []
    for rep in rep_values:
        sub = df[df[rep_col] == rep]
        n_conv = len(sub)
        for v in S.POLICY_VIOLATION_TYPES:
            table_rows.append({
                "representative": rep,
                "violation_type": v,
                "total_count": int(sub[f"total_{v}"].sum()),
                "n_conversations": n_conv,
                "mean_per_conversation": (
                    float(sub[f"total_{v}"].sum() / n_conv) if n_conv else float("nan")
                ),
            })
    table = pd.DataFrame(table_rows)
    suffix = "" if conversation_filter is None else f"__{conversation_filter}"
    save_table(name + suffix, table, index=False)

    fig, axes = plt.subplots(
        1, len(S.POLICY_VIOLATION_TYPES),
        figsize=(max(4 * len(S.POLICY_VIOLATION_TYPES), 9), 3.6),
        sharey=False,
    )
    if len(S.POLICY_VIOLATION_TYPES) == 1:
        axes = [axes]
    for ax, vtype in zip(axes, S.POLICY_VIOLATION_TYPES):
        sub = table[table["violation_type"] == vtype].set_index("representative")
        sub = sub.reindex(rep_values)
        x = np.arange(len(rep_values))
        ax.bar(x, sub["total_count"], color=COLOR_PRIMARY, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(rep_values, rotation=20, ha="right", fontsize=8)
        ax.set_title(_pretty_violation(vtype))
        ax.set_ylabel("Total violations")
        # Annotate each bar with count + mean per conversation
        for i, rep in enumerate(rep_values):
            row = sub.loc[rep]
            tot = int(row["total_count"])
            mean = row["mean_per_conversation"]
            ax.annotate(
                f"{tot}\n({mean:.2f}/conv)",
                xy=(i, tot), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color=UNCC_BLACK,
            )
    fig.suptitle(
        f"Policy violations by type and representative behavior{title_suffix}"
    )
    fig.tight_layout()
    save_figure(name + suffix, fig)
    return fig


def hist_first_prediction_turn_by_stance(
    conv_df: pd.DataFrame,
    *,
    objective: str,    # "social_engineering" | "policy_violation" | "combined"
    conversation_filter: Optional[str] = None,
    name: Optional[str] = None,
) -> plt.Figure:
    """One panel per stance, distribution of first-prediction turn.

    `objective` may be "social_engineering", "policy_violation", or
    "combined" (the earliest of either detector firing for a given stance).
    """
    if conversation_filter is not None:
        df = conv_df[conv_df["conversation_type"] == conversation_filter]
        title_suffix = f" ({conversation_filter})"
    else:
        df = conv_df
        title_suffix = " (combined)"

    fig, axes = plt.subplots(
        1, len(S.STANCES),
        figsize=(4 * len(S.STANCES), 3.2), sharey=True,
    )
    for ax, stance in zip(axes, S.STANCES):
        col = f"first_prediction_turn__{objective}__{stance}"
        vals = df[col].dropna().to_numpy()
        n_with = len(vals)
        n_without = int(df[col].isna().sum())
        if n_with > 0:
            max_t = int(np.max(vals))
            bins = np.arange(0, max_t + 2) - 0.5
            ax.hist(vals, bins=bins, color=COLOR_SECONDARY, edgecolor="white")
        ax.set_title(
            f"{_pretty_stance(stance).capitalize()}\n"
            f"({n_with} fired, {n_without} never)"
        )
        ax.set_xlabel("Turn index")
        ax.set_ylabel("Conversations")
    fig.suptitle(
        f"Turn of first {_pretty_objective(objective)} prediction — "
        f"{_pretty_filter(conversation_filter)}"
    )
    fig.tight_layout()

    out_name = name or (
        f"hist_first_prediction_turn__{objective}"
        + ("" if conversation_filter is None else f"__{conversation_filter}")
    )
    save_figure(out_name, fig)
    return fig


def hist_violation_minus_pred_diff(
    conv_df: pd.DataFrame,
    *,
    objective: str,    # "social_engineering" | "policy_violation" | "combined"
    conversation_filter: Optional[str] = None,
    name: Optional[str] = None,
) -> plt.Figure:
    """For each stance: histogram of (first_violation_turn − first_prediction_turn).

    Positive values (right of zero) mean the prediction came BEFORE the
    first actual violation — these are conversations the system would
    have prevented. Negative values (left of zero) mean the prediction
    came AFTER the first actual violation — those are misses. Zero means
    the prediction landed on the same turn as the first violation.

    NaN diffs (no first prediction OR no actual violation) are excluded
    from the histogram; their counts are shown in the panel title.
    """
    if conversation_filter is not None:
        df = conv_df[conv_df["conversation_type"] == conversation_filter]
    else:
        df = conv_df

    fig, axes = plt.subplots(
        1, len(S.STANCES),
        figsize=(4 * len(S.STANCES), 3.2), sharey=True,
    )
    for ax, stance in zip(axes, S.STANCES):
        col = f"first_violation_minus_first_pred__{objective}__{stance}"
        vals = df[col].dropna().to_numpy()
        n_excl = int(df[col].isna().sum())
        if vals.size > 0:
            lo = int(np.floor(vals.min()))
            hi = int(np.ceil(vals.max()))
            bins = np.arange(lo, hi + 2) - 0.5
            ax.hist(vals, bins=bins, color=COLOR_HIGHLIGHT, edgecolor="white")
            ax.axvline(0, color=UNCC_BLACK, linestyle="--", linewidth=1)
        ax.set_title(
            f"{_pretty_stance(stance).capitalize()}\n(N={vals.size}, excluded={n_excl})"
        )
        ax.set_xlabel("First violation turn − first prediction turn\n← missed       prevented →")
        ax.set_ylabel("Conversations")
    fig.suptitle(
        f"First actual violation relative to first {_pretty_objective(objective)} "
        f"prediction — {_pretty_filter(conversation_filter)}"
    )
    fig.tight_layout()

    out_name = name or (
        f"hist_violation_minus_pred__{objective}"
        + ("" if conversation_filter is None else f"__{conversation_filter}")
    )
    save_figure(out_name, fig)
    return fig


def hist_violations_pre_at_post(
    conv_df: pd.DataFrame,
    *,
    objective: str,
    conversation_filter: Optional[str] = None,
    name: Optional[str] = None,
) -> plt.Figure:
    """Stacked bar per (stance, violation type): mean count pre/at/post first prediction."""
    if conversation_filter is not None:
        df = conv_df[conv_df["conversation_type"] == conversation_filter]
        title_suffix = f" ({conversation_filter})"
    else:
        df = conv_df
        title_suffix = " (combined)"

    rows = []
    for stance in S.STANCES:
        key = f"{objective}__{stance}"
        for v in S.POLICY_VIOLATION_TYPES:
            sub = df[[
                f"violations_pre__{v}__{key}",
                f"violations_at__{v}__{key}",
                f"violations_post__{v}__{key}",
            ]].dropna()
            n_used = len(sub)
            if n_used == 0:
                rows.append({
                    "stance": stance, "violation_type": v,
                    "pre": 0, "at": 0, "post": 0, "n_conversations": 0,
                })
                continue
            rows.append({
                "stance": stance,
                "violation_type": v,
                "pre": float(sub[f"violations_pre__{v}__{key}"].mean()),
                "at": float(sub[f"violations_at__{v}__{key}"].mean()),
                "post": float(sub[f"violations_post__{v}__{key}"].mean()),
                "n_conversations": n_used,
            })
    table = pd.DataFrame(rows)
    suffix = "" if conversation_filter is None else f"__{conversation_filter}"
    save_table(
        (name or f"violations_pre_at_post__{objective}") + suffix,
        table, index=False,
    )

    fig, axes = plt.subplots(
        1, len(S.STANCES),
        figsize=(4.2 * len(S.STANCES), 3.6), sharey=True,
    )
    width = 0.7
    # For SE: "at" is irrelevant — the "first violation turn" is a
    # representative-response turn (even index), and the SE detector
    # only fires on caller turns (odd indices), so SE first-prediction
    # turn can never equal first-violation turn. We omit the "at" bar
    # for SE; PV keeps all three.
    show_at = (objective != "social_engineering")
    for ax, stance in zip(axes, S.STANCES):
        sub = table[table["stance"] == stance].set_index("violation_type")
        sub = sub.reindex(list(S.POLICY_VIOLATION_TYPES))
        x = np.arange(len(sub))
        ax.bar(x, sub["pre"], width, label="pre", color=COLOR_NEUTRAL)
        if show_at:
            ax.bar(x, sub["at"], width, bottom=sub["pre"], label="at",
                   color=COLOR_SECONDARY)
            post_bottom = sub["pre"] + sub["at"]
        else:
            post_bottom = sub["pre"]
        ax.bar(x, sub["post"], width,
               bottom=post_bottom, label="post", color=COLOR_PRIMARY)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_pretty_violation(v) for v in sub.index],
            rotation=20, ha="right",
        )
        ax.set_title(_pretty_stance(stance).capitalize())
        ax.set_ylabel("Mean violations per conversation")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Actual violations relative to first {_pretty_objective(objective)} "
        f"prediction (mean per conversation) — {_pretty_filter(conversation_filter)}"
    )
    fig.tight_layout()

    out_name = (name or f"hist_violations_pre_at_post__{objective}") + suffix
    save_figure(out_name, fig)
    return fig


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def heatmap_from_dataframe(
    df: pd.DataFrame,
    *,
    title: str,
    name: str,
    fmt: str = "{:.2f}",
    cmap=None,
) -> plt.Figure:
    """Render an arbitrary numeric DataFrame as a heatmap with cell labels.

    Index becomes y-tick labels; columns become x-tick labels. Saves
    PNG+SVG and the underlying xlsx. Default colormap is the UNCC
    white-to-green sequential map; pass `cmap=UNCC_DIVERGING_CMAP` for
    AUC-style heatmaps where 0.5 is the meaningful midpoint.
    """
    if cmap is None:
        cmap = UNCC_GREEN_CMAP
    save_table(name, df)
    arr = df.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(0.7 * len(df.columns) + 2.5, 0.55 * len(df) + 2),
    )
    im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    # Both UNCC_GREEN_CMAP and UNCC_DIVERGING_CMAP stay light enough at
    # all gradient stops that black text is legible everywhere, so we
    # use uniform black rather than thresholded white/black.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j, i, fmt.format(v),
                ha="center", va="center",
                color=UNCC_BLACK,
                fontsize=8,
            )
    fig.tight_layout()
    save_figure(name, fig)
    return fig


# ---------------------------------------------------------------------------
# Sankey
# ---------------------------------------------------------------------------

def sankey_threat_to_outcomes(
    conv_df: pd.DataFrame,
    *,
    se_stance: str = "balanced",
    pv_stance: str = "balanced",
    name: str = "sankey_threat_to_outcomes",
) -> "object":
    """Conversation flow:
        type (threat|benign)
        -> violation occurred (yes|no)
        -> SE predicted (yes|no, by se_stance)
        -> PV predicted (yes|no, by pv_stance)

    Saves a static SVG via plotly's image export when kaleido is available;
    otherwise saves the underlying counts as CSV and returns the figure.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        raise RuntimeError(
            "Sankey requires plotly. Install with `pip install plotly`."
        )

    se_col = f"first_prediction_turn__social_engineering__{se_stance}"
    pv_col = f"first_prediction_turn__policy_violation__{pv_stance}"

    df = conv_df.copy()
    df["__type"] = df["conversation_type"]
    df["__violation"] = np.where(
        df["total_any_violation"] > 0, "violation", "no_violation"
    )
    df["__se"] = np.where(df[se_col].notna(), "se_predicted", "se_not_predicted")
    df["__pv"] = np.where(df[pv_col].notna(), "pv_predicted", "pv_not_predicted")

    type_nodes = list(df["__type"].unique())
    viol_nodes = ["violation", "no_violation"]
    se_nodes = ["se_predicted", "se_not_predicted"]
    pv_nodes = ["pv_predicted", "pv_not_predicted"]
    all_nodes = type_nodes + viol_nodes + se_nodes + pv_nodes
    idx = {n: i for i, n in enumerate(all_nodes)}

    def edges(df_, a_col, b_col):
        return df_.groupby([a_col, b_col]).size().reset_index(name="count")

    e1 = edges(df, "__type", "__violation")
    e2 = edges(df, "__violation", "__se")
    e3 = edges(df, "__se", "__pv")

    sources = (
        list(e1["__type"].map(idx))
        + list(e2["__violation"].map(idx))
        + list(e3["__se"].map(idx))
    )
    targets = (
        list(e1["__violation"].map(idx))
        + list(e2["__se"].map(idx))
        + list(e3["__pv"].map(idx))
    )
    values = list(e1["count"]) + list(e2["count"]) + list(e3["count"])

    fig = go.Figure(go.Sankey(
        node=dict(label=all_nodes, pad=18, thickness=14),
        link=dict(source=sources, target=targets, value=values),
    ))
    fig.update_layout(
        title_text=(
            f"Conversation flow: type → violation → SE predicted ({se_stance}) "
            f"→ PV predicted ({pv_stance})"
        ),
        font_size=11,
    )

    edges_combined = pd.concat([
        e1.rename(columns={"__type": "from", "__violation": "to"}),
        e2.rename(columns={"__violation": "from", "__se": "to"}),
        e3.rename(columns={"__se": "from", "__pv": "to"}),
    ], ignore_index=True)
    save_table(name, edges_combined, index=False)

    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{name}.html"
    fig.write_html(html_path)
    try:
        fig.write_image(str(out_dir / f"{name}.png"), scale=2)
        fig.write_image(str(out_dir / f"{name}.svg"))
    except Exception:
        pass  # kaleido not installed; HTML is sufficient
    return fig


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: list,
    *,
    title: str,
    name: str,
    close: bool = False,
) -> plt.Figure:
    """Plot one or more ROC curves on a single set of axes.

    Each `ROCResult` from `analysis.roc` becomes one curve. Markers at
    each operating point (the three stances) are drawn on the curve.
    AUC and sample sizes are reported in the legend. Saves PNG + SVG
    plus an xlsx with all (slice, stance, fpr, tpr, auc) rows.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=COLOR_NEUTRAL, linestyle=":", linewidth=1)

    csv_rows = []
    palette = UNCC_PALETTE

    for i, r in enumerate(results):
        color = palette[i % len(palette)]
        label = (
            f"{r.slice_label}  (AUC={r.auc:.3f}, "
            f"n+={r.n_pos}, n-={r.n_neg})"
        )
        ax.plot(r.roc_x, r.roc_y, marker="o", color=color,
                label=label, linewidth=1.5)
        for op in r.operating_points:
            csv_rows.append({
                "scenario": r.scenario,
                "slice_label": r.slice_label,
                "stance": op.stance,
                "fpr": op.fpr, "tpr": op.tpr,
                "tp": op.tp, "fp": op.fp,
                "n_positive": op.n_pos, "n_negative": op.n_neg,
                "auc": r.auc,
            })

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    # Remove grid and the top/right spines for a cleaner look
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    save_figure(name, fig, close=close)
    if csv_rows:
        save_table(name, pd.DataFrame(csv_rows), index=False)
    return fig
