"""
ROC analysis at the conversation level.

The three sensitivity stances (high_precision, balanced, high_recall) act
as three operating points on a ROC curve. Together with the trivial
endpoints (0,0) and (1,1) they yield a 5-point empirical ROC. AUC is
computed by trapezoidal rule.

Four scenarios are supported. Note carefully which ground truth defines
the positive class in each case:

    SE_DETECTION    Positive = conversation labeled as threat (regardless
                    of whether the threat scenario actually realized as a
                    policy violation).
                    Score per stance = SE prediction fires anywhere in
                    the conversation.

    SE_PREVENTION   Positive = conversation labeled as threat. Same
                    positive class as SE_DETECTION; different scoring
                    rule:
                    - If the conversation has an actual violation: score
                      is 1 iff the SE prediction fires *strictly before*
                      the first actual violation.
                    - If the conversation has no actual violation: score
                      is 1 iff the SE prediction fires anywhere (this
                      means a benign conversation with an SE fire counts
                      as a false positive in the same way as for
                      SE_DETECTION; only threat conversations differ).

    PV_DETECTION    Positive = conversation has at least one actual
                    policy violation. THIS IS NOT TIED TO THREAT/BENIGN
                    LABEL: a benign conversation with an actual violation
                    counts as positive, and a threat conversation with no
                    actual violation counts as negative.
                    Score per stance = PV prediction fires at any turn
                    at-or-after the first actual violation. For
                    conversations with no actual violation, "at-or-after
                    first violation" is undefined, so any PV firing is
                    counted toward score=1 (which becomes a false
                    positive since the label is 0).

    PV_ON_TIME      Positive = conversation has at least one actual
                    policy violation (same as PV_DETECTION).
                    Score per stance = PV prediction fires on the same
                    turn as the first actual violation. For conversations
                    with no actual violation, any PV firing counts as
                    score=1 (i.e., false positive).

For each scenario we compute (overall) and slice analyses:
    - by Cialdini emphasis
    - by benign context level
    - heatmap: benign_context x cialdini_emphasis -> AUC
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import schema as S


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: tuple[str, ...] = (
    "se_detection",
    "se_prevention",
    "pv_detection",
    "pv_on_time",
)

_SCENARIO_OBJECTIVE = {
    "se_detection": "social_engineering",
    "se_prevention": "social_engineering",
    "pv_detection": "policy_violation",
    "pv_on_time": "policy_violation",
}


def _y_true_for_scenario(conv_df: pd.DataFrame, scenario: str) -> np.ndarray:
    """Per-conversation positive-class label (1/0)."""
    if scenario in ("se_detection", "se_prevention"):
        return (conv_df["conversation_type"] == "threat").astype(int).to_numpy()
    if scenario in ("pv_detection", "pv_on_time"):
        return (conv_df["total_any_violation"] > 0).astype(int).to_numpy()
    raise ValueError(f"unknown scenario: {scenario!r}")


def _y_score_for_scenario_stance(
    conv_df: pd.DataFrame, scenario: str, stance: str
) -> np.ndarray:
    """Per-conversation 0/1 score for one stance under one scenario.

    Each of the three stances is an independent classifier; the resulting
    binary scores form one operating point on the ROC.
    """
    if scenario == "se_detection":
        col = f"first_prediction_turn__social_engineering__{stance}"
        return conv_df[col].notna().astype(int).to_numpy()

    if scenario == "se_prevention":
        col_se = f"first_prediction_turn__social_engineering__{stance}"
        first_v = conv_df["first_violation_turn__any"]
        first_se = conv_df[col_se]
        # For each conversation:
        #   - if no actual violation: positive iff SE ever fired
        #   - if violation at V: positive iff SE first prediction turn < V
        no_viol = first_v.isna()
        score = np.zeros(len(conv_df), dtype=int)
        # Conversations with no violation: any SE fire counts as positive
        score = np.where(no_viol, first_se.notna().astype(int), score)
        # Conversations with violation: SE fire strictly before V
        viol_idx = ~no_viol
        if viol_idx.any():
            sub = (first_se[viol_idx].notna() &
                   (first_se[viol_idx] < first_v[viol_idx]))
            score[viol_idx.to_numpy()] = sub.astype(int).to_numpy()
        return score

    if scenario == "pv_detection":
        col_pv = f"first_prediction_turn__policy_violation__{stance}"
        first_v = conv_df["first_violation_turn__any"]
        first_pv = conv_df[col_pv]
        # Per-conversation: did PV ever fire at or after first violation?
        # Implementation: walk the per-stance prediction list and check for
        # any 1 at index >= first_v. We approximate that quickly using
        # conversation-level info: if first_pv is NaN, score is 0.
        # If first_v is NaN (no violation), any PV fire is FP -> score 1
        # if first_pv is not NaN.
        # If first_v is not NaN and first_pv >= first_v -> score 1.
        # If first_v is not NaN and first_pv < first_v, the detector still
        # might have fired a SECOND time at-or-after first_v. We need to
        # consult the per-turn list to be exact; if the conversation table
        # doesn't include that, we use the persistence + first_pv signals
        # as a coarse approximation, which is correct except for the rare
        # "fired early then didn't fire again" pattern.
        # For exactness, the caller should pass the long-form turn table
        # through a separate function (`_pv_fired_at_or_after_per_conv`).
        raise NotImplementedError(
            "pv_detection score requires per-turn prediction lists; use "
            "`compute_roc_curve` which has access to the turn table."
        )

    if scenario == "pv_on_time":
        col_pv = f"first_prediction_turn__policy_violation__{stance}"
        first_v = conv_df["first_violation_turn__any"]
        first_pv = conv_df[col_pv]
        # Positive iff (first_v not NaN AND first_pv == first_v).
        # For benign / threat-no-violation: any PV fire is FP, so score
        # equals "first_pv not NaN".
        no_viol = first_v.isna()
        score = np.zeros(len(conv_df), dtype=int)
        score = np.where(no_viol, first_pv.notna().astype(int), score)
        viol_idx = ~no_viol
        if viol_idx.any():
            sub = (first_pv[viol_idx].notna() &
                   (first_pv[viol_idx] == first_v[viol_idx]))
            score[viol_idx.to_numpy()] = sub.astype(int).to_numpy()
        return score

    raise ValueError(f"unknown scenario: {scenario!r}")


def _pv_fired_at_or_after_per_conv(
    conv_df: pd.DataFrame, turn_df: pd.DataFrame, stance: str
) -> np.ndarray:
    """For each conversation: 1 iff PV detector fired at any turn at-or-
    after `first_violation_turn__any`. For conversations with no actual
    violation, returns 1 iff the PV detector ever fired (any FP).
    """
    pred_col = f"policy_violation__{stance}__prediction"
    n = len(conv_df)
    out = np.zeros(n, dtype=int)
    # Map request_id -> list of (turn_index, prediction) on rep-eligible turns
    fires_by_rid: dict[str, list[int]] = {}
    sub = turn_df[turn_df[pred_col] == 1][["request_id", "turn_index"]]
    for rid, t in zip(sub["request_id"].tolist(), sub["turn_index"].tolist()):
        fires_by_rid.setdefault(rid, []).append(int(t))
    for i, (rid, fv) in enumerate(zip(
        conv_df["request_id"].tolist(),
        conv_df["first_violation_turn__any"].tolist(),
    )):
        fires = fires_by_rid.get(rid, [])
        if not fires:
            out[i] = 0
            continue
        if pd.isna(fv):
            # no actual violation: any fire = false positive
            out[i] = 1
        else:
            out[i] = 1 if any(f >= int(fv) for f in fires) else 0
    return out


# ---------------------------------------------------------------------------
# Operating-point and ROC computation
# ---------------------------------------------------------------------------

@dataclass
class OperatingPoint:
    stance: str
    fpr: float
    tpr: float
    n_pos: int
    n_neg: int
    tp: int
    fp: int


@dataclass
class ROCResult:
    scenario: str
    slice_label: str
    n_pos: int
    n_neg: int
    operating_points: list[OperatingPoint]
    roc_x: list[float]   # FPR
    roc_y: list[float]   # TPR
    auc: float


def _operating_points(y_true: np.ndarray,
                      scores_by_stance: dict[str, np.ndarray]) -> list[OperatingPoint]:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    pts: list[OperatingPoint] = []
    for stance in S.STANCES:
        s = scores_by_stance[stance]
        tp = int(((y_true == 1) & (s == 1)).sum())
        fp = int(((y_true == 0) & (s == 1)).sum())
        tpr = tp / n_pos if n_pos else float("nan")
        fpr = fp / n_neg if n_neg else float("nan")
        pts.append(OperatingPoint(stance=stance, fpr=fpr, tpr=tpr,
                                  n_pos=n_pos, n_neg=n_neg, tp=tp, fp=fp))
    return pts


def _roc_from_operating_points(
    pts: list[OperatingPoint],
) -> tuple[list[float], list[float], float]:
    """Build a 5-point ROC: (0,0), the three operating points sorted by
    FPR (and ties broken by TPR), and (1,1). Compute AUC by trapezoidal
    rule on the resulting curve. NaN points are excluded.
    """
    valid = [p for p in pts if not (np.isnan(p.fpr) or np.isnan(p.tpr))]
    pts_sorted = sorted(valid, key=lambda p: (p.fpr, p.tpr))
    xs = [0.0] + [p.fpr for p in pts_sorted] + [1.0]
    ys = [0.0] + [p.tpr for p in pts_sorted] + [1.0]
    # Make monotone in x by collapsing equal-x points to max y, then
    # ensuring y is non-decreasing along x.
    auc = 0.0
    for i in range(1, len(xs)):
        auc += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return xs, ys, auc


def compute_roc(
    conv_df: pd.DataFrame,
    turn_df: pd.DataFrame,
    scenario: str,
    *,
    slice_label: str = "overall",
) -> ROCResult:
    """ROC for one scenario over the three stances on the given conversation slice.

    `conv_df` should already be sliced (e.g. by Cialdini emphasis); pass
    the full table for the overall ROC. `turn_df` is the long-format turn
    table (only consulted for `pv_detection`, which requires the per-turn
    prediction lists to evaluate "at-or-after" correctly).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")

    y_true = _y_true_for_scenario(conv_df, scenario)
    scores_by_stance: dict[str, np.ndarray] = {}
    for stance in S.STANCES:
        if scenario == "pv_detection":
            # Restrict the turn table to this slice's request_ids
            rids = set(conv_df["request_id"].tolist())
            sub_turn = turn_df[turn_df["request_id"].isin(rids)]
            scores_by_stance[stance] = _pv_fired_at_or_after_per_conv(
                conv_df, sub_turn, stance
            )
        else:
            scores_by_stance[stance] = _y_score_for_scenario_stance(
                conv_df, scenario, stance
            )

    pts = _operating_points(y_true, scores_by_stance)
    xs, ys, auc = _roc_from_operating_points(pts)
    return ROCResult(
        scenario=scenario,
        slice_label=slice_label,
        n_pos=int((y_true == 1).sum()),
        n_neg=int((y_true == 0).sum()),
        operating_points=pts,
        roc_x=xs,
        roc_y=ys,
        auc=auc,
    )


def compute_roc_by_slice(
    conv_df: pd.DataFrame,
    turn_df: pd.DataFrame,
    scenario: str,
    slice_column: str,
) -> list[ROCResult]:
    """One ROC per unique value of `slice_column`.

    Accepts short-label columns (`short_cialdini_emphasis`,
    `short_benign_context`, `short_representative`) directly, or raw
    prompt-text columns (`cialdini_emphasis`, `benign_context`,
    `representative`) which are normalized at slice time.

    For Cialdini and benign-context slices, the returned list is
    ordered to match the canonical ordering from
    `schema.CIALDINI_PRINCIPLES` /
    `schema.BENIGN_CONTEXT_SHORT_LABELS` so that figure legends list
    slices in a predictable order.
    """
    # Normalize slice labels for display
    if slice_column in ("cialdini_emphasis", "short_cialdini_emphasis"):
        normalize = S.cialdini_principle_label
    elif slice_column in ("benign_context", "short_benign_context"):
        normalize = S.benign_context_short_label
    elif slice_column in ("representative", "short_representative"):
        normalize = S.representative_short_label
    else:
        normalize = str

    # Group on the raw column values; collapse to display labels
    by_label: dict[str, pd.DataFrame] = {}
    for val, grp in conv_df.groupby(slice_column, dropna=False):
        label = normalize(val)
        if label in by_label:
            by_label[label] = pd.concat([by_label[label], grp])
        else:
            by_label[label] = grp

    # Order labels canonically when applicable
    if slice_column in ("cialdini_emphasis", "short_cialdini_emphasis"):
        canonical = list(S.CIALDINI_PRINCIPLES)
    elif slice_column in ("benign_context", "short_benign_context"):
        canonical = list(S.BENIGN_CONTEXT_SHORT_LABELS)
    elif slice_column in ("representative", "short_representative"):
        canonical = list(S.REPRESENTATIVE_SHORT_LABELS)
    else:
        canonical = []

    observed = list(by_label.keys())
    in_canonical = [v for v in canonical if v in observed]
    extras = sorted([v for v in observed if v not in canonical], key=str)
    ordered_labels = in_canonical + extras

    out: list[ROCResult] = []
    for label in ordered_labels:
        out.append(
            compute_roc(by_label[label], turn_df, scenario, slice_label=label)
        )
    return out


def _is_cialdini_column(col: str) -> bool:
    return col in ("cialdini_emphasis", "short_cialdini_emphasis")


def _is_benign_context_column(col: str) -> bool:
    return col in ("benign_context", "short_benign_context")


def _canonical_order_for(col: str, observed: list[str]) -> list[str]:
    """Return `observed` reordered to match the canonical order for `col`.

    Values in the canonical list and present in `observed` come first in
    canonical order; any extras (unrecognized values) follow,
    alphabetically sorted, so plots remain stable.
    """
    if _is_cialdini_column(col):
        canonical = list(S.CIALDINI_PRINCIPLES)
    elif _is_benign_context_column(col):
        canonical = list(S.BENIGN_CONTEXT_SHORT_LABELS)
    else:
        return sorted(observed, key=str)
    in_canonical = [v for v in canonical if v in observed]
    extras = sorted([v for v in observed if v not in canonical], key=str)
    return in_canonical + extras


def auc_heatmap_table(
    conv_df: pd.DataFrame,
    turn_df: pd.DataFrame,
    scenario: str,
    *,
    row_column: str = "short_benign_context",
    col_column: str = "short_cialdini_emphasis",
) -> pd.DataFrame:
    """Heatmap-shaped DataFrame: row_column on the index, col_column on the
    columns, AUC values in cells. NaN where the slice has fewer than two
    classes (AUC undefined).

    Default row/column args refer to the short-label columns built by
    `preliminaries.build_conversation_table`. If those columns are
    absent, this function falls back to the raw prompt-text columns
    (`benign_context`, `cialdini_emphasis`) and normalizes labels at
    plot time. Pass either set of names explicitly to override.

    When an axis is a Cialdini column, the displayed labels are collapsed
    to the bare principle name and ordered to match
    `schema.CIALDINI_PRINCIPLES`. When an axis is a benign-context
    column, labels are ordered to match
    `schema.BENIGN_CONTEXT_SHORT_LABELS`.
    """
    # Fall back gracefully if the short_* columns aren't in the frame.
    if row_column not in conv_df.columns and row_column.startswith("short_"):
        row_column = row_column.replace("short_", "", 1)
    if col_column not in conv_df.columns and col_column.startswith("short_"):
        col_column = col_column.replace("short_", "", 1)

    # Decide on the labeling function for each axis. Short-label columns
    # already carry clean labels, so we pass them through; raw columns get
    # normalized.
    def _labeler(col: str):
        if col == "cialdini_emphasis":
            return S.cialdini_principle_label
        if col == "benign_context":
            return S.benign_context_short_label
        return str

    raw_rows = list(conv_df[row_column].dropna().unique())
    raw_cols = list(conv_df[col_column].dropna().unique())
    row_label = _labeler(row_column)
    col_label = _labeler(col_column)

    # Build the raw -> display map then collapse duplicate display
    # values that map back to the same principle. (Almost never happens
    # if the upstream pipeline is well-behaved, but harmless.)
    row_map: dict[object, str] = {r: row_label(r) for r in raw_rows}
    col_map: dict[object, str] = {c: col_label(c) for c in raw_cols}

    display_rows_observed = list(dict.fromkeys(row_map.values()))
    display_cols_observed = list(dict.fromkeys(col_map.values()))
    display_rows = _canonical_order_for(row_column, display_rows_observed)
    display_cols = _canonical_order_for(col_column, display_cols_observed)
    out = pd.DataFrame(index=display_rows, columns=display_cols, dtype=float)

    # For each (display_r, display_c) combination, slice the data using
    # all raw values that map to the current display label.
    for disp_r in display_rows:
        rows_for_r = [k for k, v in row_map.items() if v == disp_r]
        for disp_c in display_cols:
            cols_for_c = [k for k, v in col_map.items() if v == disp_c]
            sub = conv_df[
                conv_df[row_column].isin(rows_for_r)
                & conv_df[col_column].isin(cols_for_c)
            ]
            if sub.empty:
                continue
            y_true = _y_true_for_scenario(sub, scenario)
            if len(set(y_true)) < 2:
                out.at[disp_r, disp_c] = np.nan
                continue
            res = compute_roc(sub, turn_df, scenario,
                              slice_label=f"{disp_r}|{disp_c}")
            out.at[disp_r, disp_c] = res.auc
    return out


# ---------------------------------------------------------------------------
# Summary table builders
# ---------------------------------------------------------------------------

def operating_points_to_dataframe(results: Iterable[ROCResult]) -> pd.DataFrame:
    """Flatten a list of ROCResult into a long-format DataFrame of
    operating points (one row per (slice_label, stance))."""
    rows = []
    for r in results:
        for p in r.operating_points:
            rows.append({
                "scenario": r.scenario,
                "slice_label": r.slice_label,
                "stance": p.stance,
                "n_positive_conversations": p.n_pos,
                "n_negative_conversations": p.n_neg,
                "tp": p.tp,
                "fp": p.fp,
                "tpr": p.tpr,
                "fpr": p.fpr,
                "auc": r.auc,
            })
    return pd.DataFrame(rows)


def auc_summary_dataframe(results: Iterable[ROCResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "scenario": r.scenario,
            "slice_label": r.slice_label,
            "n_positive_conversations": r.n_pos,
            "n_negative_conversations": r.n_neg,
            "auc": r.auc,
        })
    return pd.DataFrame(rows)
