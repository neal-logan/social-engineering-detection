"""
Conversation-level and turn-level feature derivation.

Two main public functions:

- `build_conversation_table(loaded)` returns one row per conversation with:
    * dimension keys + run-level fields
    * conversation_type ("threat" / "benign")
    * n_turns, n_caller_turns, n_rep_turns
    * total flag counts per Cialdini principle and per policy violation type
    * turn-of-first-violation columns (per type and overall)
    * turn-of-first-prediction columns (per detection key)
    * combined turn-of-first-prediction (min over SE and PV) per stance
    * difference columns (first prediction - first violation), per key
      and for the combined-by-stance value
    * pre/at/post counts of policy violations relative to first prediction,
      per (policy violation type, detection key)
    * persistence rate columns (per detection key)
    * concordance / disagreement counts across stances per objective
    * confidence-margin summaries per detection key
    * cumulative running-total list columns for Cialdini and policy
      violations (lists of length n_turns)

- `build_turn_table(loaded)` returns the long-format DataFrame with one
    row per (conversation, turn). Used for turn-level analyses. Includes
    a `cumulative_any_violation` column populated on rep response turns
    (>= 2): 1 iff at least one actual policy violation has occurred at
    or before this turn. This is the appropriate ground truth for the
    PV detector, which is shown the entire conversation up to the turn
    it is asked about, not just that turn in isolation.

Both functions take a `LoadedDataset` (from `loading.py`) and return
pandas DataFrames. We use pandas + numpy here because the work is
fundamentally per-conversation list manipulation; ibis/duckdb shows up
in `query.py` where we register the resulting tables and run aggregate
queries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import schema as S
from .loading import LoadedDataset, turn_dict_to_ordered_list


# ===========================================================================
# Per-conversation primitive arrays
# ===========================================================================

def _flag_arrays_for_conversation(
    turns: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Extract length-n_turns 0/1 arrays for every Cialdini and violation flag.

    Caller-turn flags are 0 on rep turns; rep-turn flags are 0 on caller
    turns. Turn 0 is always 0 for every flag (it's the rep greeting,
    which carries no flags in the schema).
    """
    n = len(turns)
    out: dict[str, np.ndarray] = {}

    for col in S.CIALDINI_FLAG_COLUMNS:
        arr = np.zeros(n, dtype=np.int8)
        for i, t in enumerate(turns):
            if S.speaker_for_turn(i) == S.SPEAKER_CALLER:
                arr[i] = int(bool(t.get(col, False)))
        out[col] = arr

    for col in S.POLICY_VIOLATION_TYPES:
        arr = np.zeros(n, dtype=np.int8)
        for i, t in enumerate(turns):
            if S.speaker_for_turn(i) == S.SPEAKER_REPRESENTATIVE and i >= 2:
                arr[i] = int(bool(t.get(col, False)))
        out[col] = arr

    # "Any policy violation" combined indicator (logical OR across types)
    any_violation = np.zeros(n, dtype=np.int8)
    for col in S.POLICY_VIOLATION_TYPES:
        any_violation |= out[col]
    out["any_violation"] = any_violation

    return out


def _first_index_or_nan(arr: np.ndarray) -> float:
    """Return the index of the first 1 in arr, or np.nan if none."""
    nz = np.flatnonzero(arr)
    if nz.size == 0:
        return np.nan
    return float(nz[0])


def _running_total(arr: np.ndarray) -> np.ndarray:
    """Cumulative sum, zero-prepended at index 0? No - includes index 0.

    For a length-n array, returns a length-n array of running totals,
    where running_total[i] is the sum from index 0 through index i
    inclusive. Storing as Python ints in a list keeps Excel-friendly
    serialisation simple later.
    """
    return np.cumsum(arr, dtype=np.int32)


# ===========================================================================
# Per-conversation derivation
# ===========================================================================

def _derive_one_conversation(
    metadata_row: pd.Series,
    conversation_record: dict[str, Any],
) -> dict[str, Any]:
    """Compute every conversation-level derived field for a single row."""
    turns = turn_dict_to_ordered_list(conversation_record["conversation"])
    n_turns = len(turns)
    if n_turns != int(metadata_row["n_turns"]):
        # Sanity check (already enforced upstream, but harmless)
        raise ValueError(
            f"n_turns mismatch for {metadata_row['request_id']}: "
            f"conversation={n_turns} metadata={metadata_row['n_turns']}"
        )

    out: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Conversation type & speaker counts
    # ------------------------------------------------------------------
    template_key = str(metadata_row["prompt_template_key"]).strip()
    out["conversation_type"] = template_key  # "threat" or "benign"

    speakers = [S.speaker_for_turn(i) for i in range(n_turns)]
    out["n_caller_turns"] = sum(1 for s in speakers if s == S.SPEAKER_CALLER)
    out["n_rep_turns_total"] = sum(
        1 for s in speakers if s == S.SPEAKER_REPRESENTATIVE
    )
    out["n_rep_response_turns"] = max(out["n_rep_turns_total"] - 1, 0)

    # ------------------------------------------------------------------
    # Flag arrays (the workhorse)
    # ------------------------------------------------------------------
    flags = _flag_arrays_for_conversation(turns)

    # Total counts
    for col in S.CIALDINI_FLAG_COLUMNS:
        out[f"total_{col}"] = int(flags[col].sum())
    for col in S.POLICY_VIOLATION_TYPES:
        out[f"total_{col}"] = int(flags[col].sum())
    out["total_any_violation"] = int(flags["any_violation"].sum())

    # Running totals (lists, length n_turns) populated on all turns
    for col in S.CIALDINI_FLAG_COLUMNS:
        out[f"running_total_{col}"] = _running_total(flags[col]).tolist()
    for col in S.POLICY_VIOLATION_TYPES:
        out[f"running_total_{col}"] = _running_total(flags[col]).tolist()
    out["running_total_any_violation"] = (
        _running_total(flags["any_violation"]).tolist()
    )

    # Turn of first violation by type and overall
    for col in S.POLICY_VIOLATION_TYPES:
        out[f"first_violation_turn__{col}"] = _first_index_or_nan(flags[col])
    out["first_violation_turn__any"] = _first_index_or_nan(
        flags["any_violation"]
    )

    # Cialdini trajectory features (recommended addition #2)
    for col in S.CIALDINI_FLAG_COLUMNS:
        # First occurrence among caller turns
        out[f"first_occurrence_turn__{col}"] = _first_index_or_nan(flags[col])

    # Early/late split among caller turns: count in first half vs second half
    caller_indices = [i for i in range(n_turns)
                      if S.speaker_for_turn(i) == S.SPEAKER_CALLER]
    if caller_indices:
        midpoint = caller_indices[len(caller_indices) // 2]
        for col in S.CIALDINI_FLAG_COLUMNS:
            arr = flags[col]
            out[f"early_count_{col}"] = int(arr[:midpoint].sum())
            out[f"late_count_{col}"] = int(arr[midpoint:].sum())
    else:
        for col in S.CIALDINI_FLAG_COLUMNS:
            out[f"early_count_{col}"] = 0
            out[f"late_count_{col}"] = 0

    # ------------------------------------------------------------------
    # Prediction-derived columns (one set per detection key)
    # ------------------------------------------------------------------
    for key in S.DETECTION_KEYS:
        objective = key.split("__", 1)[0]
        eligible = set(S.detector_eligible_turns(n_turns, objective))
        preds = list(metadata_row[f"{key}__prediction"])

        # Treat None / non-1 as 0 for "first prediction" purposes (per
        # design: None signals an ineligible cell in success-only data).
        binary = np.array(
            [1 if (preds[i] == 1 and i in eligible) else 0
             for i in range(n_turns)],
            dtype=np.int8,
        )
        first_pred_turn = _first_index_or_nan(binary)
        out[f"first_prediction_turn__{key}"] = first_pred_turn

        # Difference column: first actual violation - first prediction.
        # Negative => prediction came AFTER actual violation (we missed it).
        # Zero    => prediction landed on the same turn as actual violation.
        # Positive => prediction came BEFORE actual violation (we prevented it).
        # NaN whenever either is missing.
        first_v = out["first_violation_turn__any"]
        if np.isnan(first_pred_turn) or np.isnan(first_v):
            diff = np.nan
        else:
            diff = first_v - first_pred_turn
        out[f"first_violation_minus_first_pred__{key}"] = diff

        # Pre / at / post counts of actual policy violations relative to
        # the first prediction by this detector. One set of (pre, at,
        # post) triplets per (violation_type, detection_key). NaN if no
        # first prediction exists.
        for v_type in S.POLICY_VIOLATION_TYPES:
            v_arr = flags[v_type]
            if np.isnan(first_pred_turn):
                pre = at = post = np.nan
            else:
                fp = int(first_pred_turn)
                pre = int(v_arr[:fp].sum())
                at = int(v_arr[fp])
                post = int(v_arr[fp + 1:].sum())
            out[f"violations_pre__{v_type}__{key}"] = pre
            out[f"violations_at__{v_type}__{key}"] = at
            out[f"violations_post__{v_type}__{key}"] = post

        # Persistence: of eligible turns at-or-after the first prediction,
        # what fraction also fired? NaN when no first prediction exists.
        if np.isnan(first_pred_turn):
            out[f"persistence_rate__{key}"] = np.nan
        else:
            fp = int(first_pred_turn)
            elig_after = [i for i in eligible if i >= fp]
            if not elig_after:
                out[f"persistence_rate__{key}"] = np.nan
            else:
                fired = sum(1 for i in elig_after if preds[i] == 1)
                out[f"persistence_rate__{key}"] = fired / len(elig_after)

        # Confidence summary: mean confidence margin (p_detected -
        # p_not_detected) on eligible turns. None when there are no
        # eligible turns.
        p_det = list(metadata_row[f"{key}__p_detected"])
        p_nd = list(metadata_row[f"{key}__p_not_detected"])
        margins = [
            (p_det[i] - p_nd[i])
            for i in range(n_turns)
            if i in eligible and p_det[i] is not None and p_nd[i] is not None
        ]
        out[f"mean_confidence_margin__{key}"] = (
            float(np.mean(margins)) if margins else np.nan
        )

    # ------------------------------------------------------------------
    # Combined first-prediction by stance: earliest turn at which EITHER
    # detector (SE or PV) at the same stance fired. NaN if neither ever
    # fires. The diff against the first actual policy violation mirrors
    # the per-key diff column.
    # ------------------------------------------------------------------
    first_v_any = out["first_violation_turn__any"]
    for stance in S.STANCES:
        se_first = out[f"first_prediction_turn__social_engineering__{stance}"]
        pv_first = out[f"first_prediction_turn__policy_violation__{stance}"]
        candidates = [v for v in (se_first, pv_first) if not np.isnan(v)]
        if candidates:
            combined = float(min(candidates))
        else:
            combined = np.nan
        out[f"first_prediction_turn__combined__{stance}"] = combined

        if np.isnan(combined) or np.isnan(first_v_any):
            diff = np.nan
        else:
            diff = first_v_any - combined
        out[f"first_violation_minus_first_pred__combined__{stance}"] = diff

    # ------------------------------------------------------------------
    # Stance concordance per objective: of eligible turns, in how many
    # do all 3 stances agree? (Recommended addition #4.)
    # ------------------------------------------------------------------
    for objective in S.OBJECTIVES:
        eligible = set(S.detector_eligible_turns(n_turns, objective))
        keys_for_obj = [k for k in S.DETECTION_KEYS if k.startswith(objective + "__")]
        agree = 0
        disagree = 0
        for i in range(n_turns):
            if i not in eligible:
                continue
            vals = [int(metadata_row[f"{k}__prediction"][i] == 1)
                    for k in keys_for_obj]
            if len(set(vals)) == 1:
                agree += 1
            else:
                disagree += 1
        out[f"stance_concordant_turns__{objective}"] = agree
        out[f"stance_discordant_turns__{objective}"] = disagree

    return out


def build_conversation_table(loaded: LoadedDataset) -> pd.DataFrame:
    """Build the conversation-level base table.

    The result has one row per conversation in `loaded.metadata`,
    augmented with all derived columns described in this module's
    docstring. Heavy list-shaped columns (running totals, raw per-turn
    detection lists) are kept as Python lists in DataFrame cells.
    """
    base = loaded.metadata.copy()
    derived_rows: list[dict[str, Any]] = []
    for i in range(len(base)):
        row = base.iloc[i]
        rid = str(row["request_id"])
        record = loaded.conversations[rid]
        derived_rows.append(_derive_one_conversation(row, record))

    derived_df = pd.DataFrame(derived_rows)
    out = pd.concat([base.reset_index(drop=True), derived_df], axis=1)

    # Short-label columns. Sources, in order of preference:
    #   1. The `*_key` companion column on the metadata DataFrame, if present.
    #   2. The `selection.{name}_key` field on each conversation JSON record,
    #      if present (the generator writes these).
    #   3. Substring match against the alias maps in `schema.py`.
    #
    # Downstream code should group/plot on the `short_*` columns, not the
    # raw prompt-text columns.

    def _key_from_selection(rid: str, key_field: str) -> object:
        rec = loaded.conversations.get(rid, {})
        sel = rec.get("selection") if isinstance(rec, dict) else None
        if isinstance(sel, dict):
            v = sel.get(key_field)
            if v is not None and str(v).strip():
                return v
        return None

    def _short_column(value_col: str, key_col: str, key_field: str, normalizer):
        out_vals: list[str] = []
        rids = out["request_id"].astype(str).tolist()
        meta_keys = (
            out[key_col].tolist() if key_col in out.columns
            else [None] * len(out)
        )
        prompt_vals = (
            out[value_col].tolist() if value_col in out.columns
            else [None] * len(out)
        )
        for rid, mk, pv in zip(rids, meta_keys, prompt_vals):
            # 1. metadata key column
            if mk is not None and pd.notna(mk) and str(mk).strip():
                out_vals.append(str(mk).strip())
                continue
            # 2. conversation JSON selection.{name}_key
            sk = _key_from_selection(rid, key_field)
            if sk is not None:
                out_vals.append(str(sk).strip())
                continue
            # 3. substring fallback against alias maps
            out_vals.append(normalizer(pv))
        return out_vals

    short_specs = [
        ("representative",    "representative_key",    "representative_key",
         S.representative_short_label,   "short_representative",
         set(S.REPRESENTATIVE_SHORT_LABELS)),
        ("benign_context",    "benign_context_key",    "benign_context_key",
         S.benign_context_short_label,   "short_benign_context",
         set(S.BENIGN_CONTEXT_SHORT_LABELS)),
        ("cialdini_emphasis", "cialdini_emphasis_key", "cialdini_emphasis_key",
         S.cialdini_principle_label,     "short_cialdini_emphasis",
         set(S.CIALDINI_PRINCIPLES)),
    ]
    unrecognized: dict[str, set[str]] = {}
    for (value_col, key_col, key_field, normalizer, target,
         canonical_set) in short_specs:
        col_data = _short_column(value_col, key_col, key_field, normalizer)
        out[target] = col_data
        # Diagnose: anything not in the canonical set means we fell through
        # to the substring fallback AND that fallback also missed.
        leakage = {v for v in col_data if v not in canonical_set}
        if leakage:
            unrecognized[target] = leakage

    if unrecognized:
        import warnings
        msgs = []
        for col, vals in unrecognized.items():
            sample = sorted(vals, key=str)[:3]
            msgs.append(
                f"  {col}: {len(vals)} unrecognized value(s); "
                f"first {len(sample)}: {sample!r}"
            )
        warnings.warn(
            "build_conversation_table: short-label columns contain "
            "values that did not match a canonical short label. "
            "These values were passed through unchanged so they remain "
            "visible in plots; to clean them up, paste a few representative "
            "values into the relevant alias map in analysis/schema.py "
            "(REPRESENTATIVE_ALIASES / BENIGN_CONTEXT_ALIASES / "
            "Cialdini aliases inside cialdini_principle_label).\n"
            + "\n".join(msgs),
            stacklevel=2,
        )

    return out


# ===========================================================================
# Long-format turn table
# ===========================================================================

def build_turn_table(loaded: LoadedDataset) -> pd.DataFrame:
    """One row per (conversation, turn).

    Columns:
        request_id, conversation_type, turn_index, speaker, n_turns
        passage_length (chars)
        cialdini_<principle> (0/1) for caller turns, NaN on rep turns
        improper_<type> (0/1) for rep response turns (>=2), NaN otherwise
        any_violation (0/1) on rep response turns, NaN otherwise
        cumulative_any_violation (0/1) on rep response turns: 1 iff any
            actual policy violation has occurred at or before this turn.
            NaN on caller turns and turn 0. This is the appropriate
            ground truth for the policy-violation detector, which is
            shown the entire conversation up to the queried turn.
        <key>__prediction, <key>__p_detected, <key>__p_not_detected,
            <key>__latency_ms, <key>__input_tokens, <key>__output_tokens
            for each detection key; populated on eligible turns, NaN
            otherwise.
    """
    base = loaded.metadata
    rows: list[dict[str, Any]] = []
    for i in range(len(base)):
        meta = base.iloc[i]
        rid = str(meta["request_id"])
        record = loaded.conversations[rid]
        turns = turn_dict_to_ordered_list(record["conversation"])
        n_turns = len(turns)
        ctype = str(meta["prompt_template_key"]).strip()

        # Pre-compute cumulative any-violation per turn for this conversation.
        # cum[t] = 1 iff any rep response turn in [2..t] had a violation.
        cum_any = [0] * n_turns
        running = 0
        for t in range(n_turns):
            if S.speaker_for_turn(t) == S.SPEAKER_REPRESENTATIVE and t >= 2:
                if any(bool(turns[t].get(c, False)) for c in S.POLICY_VIOLATION_TYPES):
                    running = 1
            cum_any[t] = running

        for t in range(n_turns):
            speaker = S.speaker_for_turn(t)
            row: dict[str, Any] = {
                "request_id": rid,
                "conversation_type": ctype,
                "turn_index": t,
                "speaker": speaker,
                "n_turns": n_turns,
                "passage_length": len(turns[t].get("passage", "")),
            }

            # Caller flags
            for col in S.CIALDINI_FLAG_COLUMNS:
                if speaker == S.SPEAKER_CALLER:
                    row[col] = int(bool(turns[t].get(col, False)))
                else:
                    row[col] = np.nan
            # Rep flags
            for col in S.POLICY_VIOLATION_TYPES:
                if speaker == S.SPEAKER_REPRESENTATIVE and t >= 2:
                    row[col] = int(bool(turns[t].get(col, False)))
                else:
                    row[col] = np.nan
            # any_violation: did a violation occur ON THIS TURN
            if speaker == S.SPEAKER_REPRESENTATIVE and t >= 2:
                row["any_violation"] = int(any(
                    bool(turns[t].get(c, False)) for c in S.POLICY_VIOLATION_TYPES
                ))
            else:
                row["any_violation"] = np.nan
            # cumulative_any_violation: has a violation occurred AT OR BEFORE this turn
            if speaker == S.SPEAKER_REPRESENTATIVE and t >= 2:
                row["cumulative_any_violation"] = int(cum_any[t])
            else:
                row["cumulative_any_violation"] = np.nan

            # Per-turn prediction fields, only populated on eligible turns
            for key in S.DETECTION_KEYS:
                objective = key.split("__", 1)[0]
                eligible = t in set(S.detector_eligible_turns(n_turns, objective))
                for field in S.PER_TURN_FIELDS:
                    src = meta[f"{key}__{field}"]
                    val = src[t] if eligible else np.nan
                    row[f"{key}__{field}"] = val
            rows.append(row)

    return pd.DataFrame(rows)
