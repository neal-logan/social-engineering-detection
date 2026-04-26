"""
Loading and validation for detection metadata and the corresponding
conversation JSON.

Public entry point is `load_dataset(detection_metadata_path, conversations_path)`,
which returns a `LoadedDataset` containing:
    - `metadata`: pandas DataFrame, one row per successful conversation,
      with all dimension columns, run-level detection columns, and
      JSON-list per-turn fields parsed into Python lists.
    - `conversations`: dict mapping request_id -> parsed conversation
      record (the inner `conversation` dict from the generation JSON).

Strict validation rules (raise on violation):
    1. Every retained metadata row has detection_status == "success".
    2. For each retained row and each detection_key, every list-field
       has length n_turns.
    3. For each retained row, the `prediction` list has 0/1 only at
       *eligible* (key, turn) cells and a sentinel non-numeric value
       (None) at ineligible cells. Eligibility is defined by
       `schema.detector_eligible_turns`. ANY None at an eligible cell
       raises.
    4. Every metadata row has a corresponding conversation in the
       conversations JSON, and the conversation has exactly n_turns
       turn_N keys.

Non-success rows are dropped before validation. The number dropped is
recorded on the returned object.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import schema as S


_TURN_KEY_RE = re.compile(r"^turn_(\d+)$")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class LoadedDataset:
    """Validated detection metadata + matching conversation records."""
    metadata: pd.DataFrame
    conversations: dict[str, dict[str, Any]]
    n_dropped_non_success: int
    detection_metadata_path: Path
    conversations_path: Path

    @property
    def n_conversations(self) -> int:
        return len(self.metadata)


# ---------------------------------------------------------------------------
# Conversation JSON loading
# ---------------------------------------------------------------------------

def load_conversations_json(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the aggregated generation-phase conversations file.

    Returns a dict keyed by request_id. The value is the full record
    written by the generation pipeline; the conversation-turns dict is
    accessible via record["conversation"].
    """
    with open(path, "r", encoding="utf-8") as f:
        store = json.load(f)
    if "conversations" not in store:
        raise ValueError(
            f"{path}: expected top-level 'conversations' key, "
            f"got keys: {sorted(store.keys())}"
        )
    return store["conversations"]


def turn_dict_to_ordered_list(
    conversation: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort turn_N entries by N and return them as a list.

    The generation pipeline stores conversation as `{turn_0: {...}, turn_1:
    {...}, ...}`. Modern Python dicts preserve insertion order so this is
    usually already sorted, but we sort defensively.
    """
    pairs: list[tuple[int, dict[str, Any]]] = []
    for k, v in conversation.items():
        m = _TURN_KEY_RE.match(k)
        if not m:
            raise ValueError(f"unexpected conversation key: {k!r}")
        pairs.append((int(m.group(1)), v))
    pairs.sort(key=lambda pair: pair[0])
    expected_indices = list(range(len(pairs)))
    actual_indices = [i for i, _ in pairs]
    if actual_indices != expected_indices:
        raise ValueError(
            f"conversation turn indices not contiguous from 0: got {actual_indices}"
        )
    return [v for _, v in pairs]


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def _parse_json_list_cell(val: Any, column: str, request_id: str) -> list[Any]:
    """Parse a metadata cell that should hold a JSON-serialised list."""
    if isinstance(val, list):
        return val
    if pd.isna(val) or val is None:
        raise ValueError(
            f"row request_id={request_id}: column {column!r} is null; "
            f"expected JSON list"
        )
    if not isinstance(val, str):
        raise ValueError(
            f"row request_id={request_id}: column {column!r} has type "
            f"{type(val).__name__}; expected JSON-string list"
        )
    try:
        out = json.loads(val)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"row request_id={request_id}: column {column!r} is not valid JSON: {e}"
        ) from e
    if not isinstance(out, list):
        raise ValueError(
            f"row request_id={request_id}: column {column!r} parsed to "
            f"{type(out).__name__}; expected list"
        )
    return out


def load_detection_metadata(path: str | Path) -> pd.DataFrame:
    """Read the detection metadata.xlsx as a DataFrame.

    JSON-list columns are NOT yet parsed; that happens in `load_dataset`
    after the success filter so we don't waste work on dropped rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_excel(path)
    if "detection_status" not in df.columns:
        raise ValueError(
            f"{path}: missing required column 'detection_status'"
        )
    if "request_id" not in df.columns:
        raise ValueError(f"{path}: missing required column 'request_id'")
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_per_turn_lists(
    row_idx: int,
    request_id: str,
    n_turns: int,
    parsed: dict[str, list[Any]],
) -> None:
    """All per-(key, field) lists must have length n_turns. Raises otherwise."""
    for key in S.DETECTION_KEYS:
        for field in S.PER_TURN_FIELDS:
            col = f"{key}__{field}"
            if col not in parsed:
                raise ValueError(
                    f"row {row_idx} request_id={request_id}: missing column {col!r}"
                )
            lst = parsed[col]
            if len(lst) != n_turns:
                raise ValueError(
                    f"row {row_idx} request_id={request_id}: column {col!r} "
                    f"has length {len(lst)}, expected n_turns={n_turns}"
                )


def _normalize_prediction_token(val: Any) -> int | None:
    """Map a single prediction-cell value to 0, 1, or None.

    The detection pipeline writes 'Y' (positive) or 'N' (negative) on
    eligible turns. Ineligible turns hold None (or sometimes an empty
    string after Excel round-tripping). The pipeline also defines two
    error sentinels — 'E' (no valid Y/N in top-10) and 'X' (call failed
    after retries) — but those should not appear in detection_status =
    'success' rows; we still treat them as None here and let
    _validate_predictions raise if they show up at an eligible cell.

    We accept already-numeric 0/1 (and 0.0/1.0) defensively in case the
    upstream format changes.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        if val == 1:
            return 1
        if val == 0:
            return 0
        return None
    if isinstance(val, str):
        s = val.strip()
        if s == "Y":
            return 1
        if s == "N":
            return 0
        # 'E', 'X', '', and anything else become None; the validator will
        # raise if any of these appear on eligible turns in a success-only
        # input.
        return None
    return None


def _normalize_prediction_list(
    parsed: dict[str, list[Any]],
) -> None:
    """Rewrite prediction lists in-place from Y/N strings to 0/1/None.

    Operates on every `<key>__prediction` list in `parsed`.
    """
    for key in S.DETECTION_KEYS:
        col = f"{key}__prediction"
        parsed[col] = [_normalize_prediction_token(v) for v in parsed[col]]


def _validate_predictions(
    row_idx: int,
    request_id: str,
    n_turns: int,
    parsed: dict[str, list[Any]],
) -> None:
    """For each detection key, every *eligible* turn must hold 0 or 1.

    Run AFTER `_normalize_prediction_list`, so by this point predictions
    are integers (0/1) at eligible cells and None at ineligible cells.

    Any None at an eligible cell raises, because we have stipulated that
    the input dataset is restricted to detection_status='success' (so
    'E', 'X', and any other non-Y/N tokens that normalised to None are
    treated as schema violations on eligible cells).

    Any non-None value at an ineligible cell also raises (the upstream
    pipeline is supposed to leave those slots unset).
    """
    for key in S.DETECTION_KEYS:
        objective = key.split("__", 1)[0]
        eligible = set(S.detector_eligible_turns(n_turns, objective))
        preds = parsed[f"{key}__prediction"]
        for t, val in enumerate(preds):
            if t in eligible:
                if val not in (0, 1):
                    raise ValueError(
                        f"row {row_idx} request_id={request_id}: "
                        f"{key}__prediction[{t}] = {val!r} on eligible turn; "
                        f"expected 0 or 1 (success-only inputs). If the "
                        f"upstream pipeline emits non-Y/N tokens (e.g. E, X) "
                        f"on success rows, that's a schema violation; "
                        f"otherwise check that 'Y'/'N' are reaching this "
                        f"normalizer correctly."
                    )
            else:
                if val is not None:
                    raise ValueError(
                        f"row {row_idx} request_id={request_id}: "
                        f"{key}__prediction[{t}] = {val!r} on ineligible turn; "
                        f"expected None"
                    )


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def _conversation_n_turns(record: dict[str, Any]) -> int:
    if "conversation" not in record:
        raise ValueError(
            f"conversation record missing 'conversation' key; "
            f"got keys {sorted(record.keys())}"
        )
    return len(turn_dict_to_ordered_list(record["conversation"]))


def load_dataset(
    detection_metadata_path: str | Path,
    conversations_path: str | Path,
) -> LoadedDataset:
    """Load + validate detection metadata and conversation JSON together.

    Drops non-success rows. Raises on schema violations among the retained
    rows or among required fields.
    """
    detection_metadata_path = Path(detection_metadata_path)
    conversations_path = Path(conversations_path)

    df_all = load_detection_metadata(detection_metadata_path)
    n_total = len(df_all)
    df = df_all[df_all["detection_status"] == S.DETECTION_STATUS_SUCCESS].copy()
    n_dropped = n_total - len(df)
    df.reset_index(drop=True, inplace=True)

    # Parse the per-turn JSON-list columns in place
    list_columns = [
        f"{key}__{field}"
        for key in S.DETECTION_KEYS
        for field in S.PER_TURN_FIELDS
    ]
    missing = [c for c in list_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{detection_metadata_path}: missing required per-turn list columns: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    for col in list_columns:
        df[col] = [
            _parse_json_list_cell(v, col, str(rid))
            for v, rid in zip(df[col].tolist(), df["request_id"].tolist())
        ]

    # Load conversations and cross-check
    conversations = load_conversations_json(conversations_path)

    # Build a column of n_turns from the conversation (preferred) and
    # cross-check against per-turn list length
    n_turns_list: list[int] = []
    for row_idx, rid in enumerate(df["request_id"].tolist()):
        rid_str = str(rid)
        if rid_str not in conversations:
            raise ValueError(
                f"row {row_idx} request_id={rid_str}: no matching record "
                f"in {conversations_path}"
            )
        n_turns_list.append(_conversation_n_turns(conversations[rid_str]))
    df["n_turns"] = n_turns_list

    # Validate per-row. Normalize prediction tokens (Y/N -> 1/0) BEFORE
    # validation so the validator can rely on numeric inputs. The
    # normalized lists are written back to the DataFrame so downstream
    # consumers (preliminaries, query, roc) see numeric predictions.
    for row_idx in range(len(df)):
        rid = str(df.at[row_idx, "request_id"])
        nt = int(df.at[row_idx, "n_turns"])
        parsed = {col: df.at[row_idx, col] for col in list_columns}
        _validate_per_turn_lists(row_idx, rid, nt, parsed)
        _normalize_prediction_list(parsed)
        _validate_predictions(row_idx, rid, nt, parsed)
        # Write normalized prediction lists back to the DataFrame
        for key in S.DETECTION_KEYS:
            col = f"{key}__prediction"
            df.at[row_idx, col] = parsed[col]

    return LoadedDataset(
        metadata=df,
        conversations=conversations,
        n_dropped_non_success=n_dropped,
        detection_metadata_path=detection_metadata_path,
        conversations_path=conversations_path,
    )


def load_combined(
    threat_metadata_path: str | Path,
    threat_conversations_path: str | Path,
    benign_metadata_path: str | Path,
    benign_conversations_path: str | Path,
) -> tuple[LoadedDataset, LoadedDataset]:
    """Convenience: load both threat and benign datasets in one call."""
    threat = load_dataset(threat_metadata_path, threat_conversations_path)
    benign = load_dataset(benign_metadata_path, benign_conversations_path)
    return threat, benign
