"""
Ibis (DuckDB backend) queries on the conversation and turn tables.

Use this module after `preliminaries.build_conversation_table` and
`preliminaries.build_turn_table` have produced pandas DataFrames. Those
frames are registered with an in-memory DuckDB connection and queried
through ibis expressions.

Why ibis here and pandas earlier: the per-conversation derivations
involve list-shaped fields and per-turn loops that are awkward in a
relational model. Aggregate queries (percentiles, group-by counts,
confusion-matrix inputs) are exactly what a SQL-backed engine is built
for, so we pivot to ibis once the data is rectangular.

Conventions:
    - All public functions take an `AnalysisContext` (the ibis tables +
      shared connection) and return either a pandas DataFrame or
      another ibis expression.
    - Function names are descriptive of the question being answered, not
      the SQL used to answer it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import ibis
import ibis.expr.types as it
import pandas as pd

from . import schema as S


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class AnalysisContext:
    """Bundle of ibis tables + the underlying connection."""
    con: ibis.BaseBackend
    conversations: it.Table          # one row per conversation
    turns: it.Table                  # one row per (conversation, turn)


def _prepare_for_duckdb(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe to register with DuckDB.

    Two adjustments:

    1. Drop list-typed columns. DuckDB has list support but it's awkward
       to round-trip through pandas object columns; the analysis module
       keeps list-shaped data on the underlying pandas frame and uses
       ibis only for tabular aggregates.

    2. Replace all-null object/datetime columns with an empty-string
       column. DuckDB can't infer a type from an all-null column and
       raises IbisTypeError. The columns we hit this on in practice
       (detection_last_error, occasional metadata fields) are textual
       in their populated state; coercing all-null values to "" is a
       safe choice. We only do this for object-dtype columns that
       round-trip cleanly to text; numeric all-null columns get
       float-cast (DuckDB accepts a typed all-NaN float column).
    """
    out = df.copy()

    # 1. Drop list-typed columns
    list_cols = [
        c for c in out.columns
        if out[c].apply(lambda v: isinstance(v, list)).any()
    ]
    if list_cols:
        out = out.drop(columns=list_cols)

    # 2. Type-fix all-null columns
    for col in out.columns:
        s = out[col]
        if not s.isna().all():
            continue
        # All-null. Pick a target type by looking at the original dtype.
        if s.dtype == object:
            # Could be string-bearing-when-populated; safe to coerce to ""
            out[col] = ""
        else:
            # Numeric / datetime / bool: cast to float NaN, which DuckDB
            # accepts as DOUBLE
            out[col] = s.astype(float)

    return out


def make_context(
    conversation_df: pd.DataFrame,
    turn_df: pd.DataFrame,
) -> AnalysisContext:
    """Register both DataFrames with a fresh in-memory DuckDB connection.

    See `_prepare_for_duckdb` for the dataframe sanitisation rules
    (list-column dropping, all-null column type-fixing).
    """
    con = ibis.duckdb.connect()

    conv_scalar = _prepare_for_duckdb(conversation_df)
    turn_scalar = _prepare_for_duckdb(turn_df)

    con.create_table("conversations", conv_scalar, overwrite=True)
    con.create_table("turns", turn_scalar, overwrite=True)

    return AnalysisContext(
        con=con,
        conversations=con.table("conversations"),
        turns=con.table("turns"),
    )


# ---------------------------------------------------------------------------
# Latency percentile tables
# ---------------------------------------------------------------------------

def latency_percentiles_table(
    ctx: AnalysisContext,
    percentiles: Iterable[float] = (0.25, 0.50, 0.75, 0.90, 0.99, 0.999),
    conversation_types: Iterable[str] = ("threat", "benign", "combined"),
) -> pd.DataFrame:
    """One row per (detector_key, conversation_type), one percentile per col.

    Default percentiles include Q1, median, Q3, plus the high tails
    (90th, 99th, 99.9th). Latency is in milliseconds, taken from the
    per-turn `__latency_ms` columns on the long-format turn table.
    NaN turns (ineligible) are excluded from the percentile computation.
    """
    pct_list = list(percentiles)
    # DuckDB exposes `quantile_cont` via the ibis percentile method.
    # Compute one row per (detector_key, conv_type) by stacking results.

    def _pct_label(p: float) -> str:
        # Format percentile 0.25 -> 'p25', 0.50 -> 'p50', 0.999 -> 'p999'
        # Avoid float-precision ambiguity by rendering as a clean string.
        if p >= 0.999:
            return f"p{round(p * 1000)}"
        if p >= 0.99:
            return f"p{round(p * 100)}"
        return f"p{round(p * 100)}"

    rows = []
    turns = ctx.turns
    for ctype in conversation_types:
        if ctype == "combined":
            t = turns
        else:
            t = turns.filter(turns["conversation_type"] == ctype)
        for key in S.DETECTION_KEYS:
            col = f"{key}__latency_ms"
            t_valid = t.filter(t[col].notnull())
            row = {"detection_key": key, "conversation_type": ctype,
                   "n_turns_used": int(t_valid.count().to_pandas())}
            agg_exprs = {
                _pct_label(p): t_valid[col].quantile(p)
                for p in pct_list
            }
            agg = t_valid.aggregate(**agg_exprs).to_pandas()
            for c in agg.columns:
                row[c] = float(agg.iloc[0][c]) if not agg.empty else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Confusion matrices (counts as pandas DataFrames)
# ---------------------------------------------------------------------------

def confusion_for_scenario(
    ctx: AnalysisContext,
    scenario: str,
    *,
    stance: str = "balanced",
) -> pd.DataFrame:
    """2x2 confusion matrix using the same positive-class definition and
    scoring rule as the corresponding ROC scenario in `analysis.roc`.

    Rows are the ground-truth label (1/0) and columns are the per-stance
    predicted score (1/0). Counts in cells.

    Scenarios:
      - se_detection:   pos = threat;          score = SE fires anywhere
      - se_prevention:  pos = threat;          score = SE fires before first
                                                violation (or anywhere if no
                                                violation)
      - pv_detection:   pos = any_violation;   score = PV fires at-or-after
                                                first violation (or anywhere
                                                if no violation: counts as FP)
      - pv_on_time:     pos = any_violation;   score = PV fires on the exact
                                                first-violation turn (or
                                                anywhere if no violation:
                                                counts as FP)
    """
    # Local import to avoid a circular dependency: roc imports schema, and
    # query is itself a peer module to roc.
    from . import roc as _roc

    conv_df = ctx.conversations.to_pandas()
    turn_df = ctx.turns.to_pandas()

    y_true = _roc._y_true_for_scenario(conv_df, scenario)
    if scenario == "pv_detection":
        y_score = _roc._pv_fired_at_or_after_per_conv(conv_df, turn_df, stance)
    else:
        y_score = _roc._y_score_for_scenario_stance(conv_df, scenario, stance)

    cm = pd.DataFrame(
        {"score=0": [0, 0], "score=1": [0, 0]},
        index=pd.Index([0, 1], name="actual"),
    )
    for t, s in zip(y_true.tolist(), y_score.tolist()):
        cm.at[int(t), f"score={int(s)}"] += 1
    cm.columns.name = "predicted"
    return cm


# Backwards-compatible aliases (deprecated; new code should use
# `confusion_for_scenario`). Kept so older notebook cells continue to work
# during the transition.
def confusion_threat_vs_se_ever_predicted(
    ctx: AnalysisContext, se_stance: str = "balanced",
) -> pd.DataFrame:
    """Deprecated. Use `confusion_for_scenario(ctx, "se_detection", stance=...)`."""
    return confusion_for_scenario(ctx, "se_detection", stance=se_stance)


def confusion_violation_vs_pv_ever_predicted(
    ctx: AnalysisContext, pv_stance: str = "balanced",
) -> pd.DataFrame:
    """Deprecated. Use `confusion_for_scenario(ctx, "pv_detection", stance=...)`."""
    return confusion_for_scenario(ctx, "pv_detection", stance=pv_stance)


# ---------------------------------------------------------------------------
# Turn-level confusion (cumulative ground truth)
# ---------------------------------------------------------------------------

def confusion_turn_violation_vs_pv_prediction(
    ctx: AnalysisContext,
    pv_stance: str = "balanced",
) -> pd.DataFrame:
    """Turn-level confusion. Restricted to representative response turns.

    The ground-truth label is *cumulative*: positive iff any actual policy
    violation has occurred at or before this turn. This matches what the
    PV detector is asked to assess (it is shown the entire conversation
    up to the queried turn, not just the queried turn in isolation).
    """
    pred_col = f"policy_violation__{pv_stance}__prediction"
    t = ctx.turns.to_pandas()
    elig = t[t[pred_col].notna()].copy()
    elig["had_cumulative_violation"] = elig["cumulative_any_violation"] == 1
    elig["pv_predicted_positive"] = elig[pred_col] == 1
    return (
        elig.groupby(["had_cumulative_violation", "pv_predicted_positive"])
        .size()
        .unstack(fill_value=0)
        .rename_axis(index="had_cumulative_violation",
                     columns="pv_predicted_positive")
    )


# ---------------------------------------------------------------------------
# Detection coverage / data completeness summary
# ---------------------------------------------------------------------------

def detection_coverage_summary(ctx: AnalysisContext) -> pd.DataFrame:
    """Per detection_key: total eligible turns vs how many had prediction in {0,1}.

    Since input is restricted to detection_status='success', this should
    be 100% by construction. Reporting it anyway as a sanity check.
    """
    rows = []
    t = ctx.turns.to_pandas()
    for key in S.DETECTION_KEYS:
        col = f"{key}__prediction"
        elig = t[col].notna()
        n_elig = int(elig.sum())
        n_valid = int(t.loc[elig, col].isin([0, 1]).sum())
        rows.append({
            "detection_key": key,
            "n_eligible_turns": n_elig,
            "n_valid_predictions": n_valid,
            "coverage": (n_valid / n_elig) if n_elig else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation-rule sensitivity at the conversation level
# ---------------------------------------------------------------------------

def conversation_recall_under_aggregation_rules(
    ctx: AnalysisContext,
    objective: str = "social_engineering",
    rules: Iterable[str] = ("any", "two_or_more", "majority"),
) -> pd.DataFrame:
    """Per-stance recall on threat conversations under different
    'conversation flagged?' aggregation rules.

    Recall is computed on threat conversations only; precision/false-
    positive counterparts can be added by analogously slicing benign.

    Rules:
        - any: at least one eligible-turn prediction == 1
        - two_or_more: at least two eligible-turn predictions == 1
        - majority: more than half of eligible turns predicted == 1
    """
    df = ctx.conversations.to_pandas()
    threat = df[df["conversation_type"] == "threat"]

    rows = []
    for stance in S.STANCES:
        key = f"{objective}__{stance}"
        col_first = f"first_prediction_turn__{key}"

        # Re-derive from per-turn lists for two_or_more and majority
        # (the conversation table only carries 'first prediction').
        # Walk the metadata frame for the predictions list.
        meta = ctx.turns.to_pandas()
        per_conv = (
            meta[meta[f"{key}__prediction"].notna()]
            .groupby("request_id")[f"{key}__prediction"]
            .agg(list)
        )

        n_total = len(threat)

        for rule in rules:
            n_flagged = 0
            for rid in threat["request_id"]:
                if rid not in per_conv.index:
                    continue
                preds = per_conv.loc[rid]
                if rule == "any":
                    flagged = any(p == 1 for p in preds)
                elif rule == "two_or_more":
                    flagged = sum(1 for p in preds if p == 1) >= 2
                elif rule == "majority":
                    n = len(preds)
                    flagged = (sum(1 for p in preds if p == 1) > n / 2)
                else:
                    raise ValueError(f"unknown rule: {rule!r}")
                if flagged:
                    n_flagged += 1
            rows.append({
                "objective": objective,
                "stance": stance,
                "rule": rule,
                "n_threat_conversations": n_total,
                "n_flagged": n_flagged,
                "recall": n_flagged / n_total if n_total else float("nan"),
            })
    return pd.DataFrame(rows)
