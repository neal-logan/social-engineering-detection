"""
Schema constants for the analysis module.

Centralises every column / key name used across the conversation generation
metadata, the conversation JSON, and the detection metadata. Keeping these in
one place lets every other module reference them by name and means a schema
change in upstream pipelines requires only one edit here.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Cialdini principles (caller-turn flags)
# ---------------------------------------------------------------------------

CIALDINI_PRINCIPLES: tuple[str, ...] = (
    "reciprocity",
    "commitment_consistency",
    "social_proof",
    "authority",
    "liking",
    "scarcity",
    "unity",
)

CIALDINI_FLAG_COLUMNS: tuple[str, ...] = tuple(
    f"cialdini_{p}" for p in CIALDINI_PRINCIPLES
)


def cialdini_principle_label(value: object) -> str:
    """Return a short, human-readable Cialdini principle name.

    The `cialdini_emphasis` column may contain either the principle key
    ("authority", "liking", ...) or the full prompt text the generator
    used (e.g. a long paragraph emphasizing authority). Both should
    display as the bare principle name. We do a substring match against
    the canonical principle list and return the first hit.

    For values that contain no recognizable principle, returns the
    original value as a string (so unexpected slices remain visible
    rather than silently collapsing).
    """
    if value is None:
        return "unknown"
    text = str(value).lower()
    # Prefer longer principle names first so e.g. 'commitment_consistency'
    # wins over a shorter principle that happens to be a substring.
    for principle in sorted(CIALDINI_PRINCIPLES, key=len, reverse=True):
        # Match on either underscore form or space form
        if principle in text or principle.replace("_", " ") in text:
            return principle
    return str(value).strip()

# ---------------------------------------------------------------------------
# Policy violation types (representative-turn flags)
# ---------------------------------------------------------------------------

POLICY_VIOLATION_TYPES: tuple[str, ...] = (
    "improper_authentication",
    "improper_disclosure",
    "improper_action",
)

# ---------------------------------------------------------------------------
# Detection objectives, stances, keys
# ---------------------------------------------------------------------------

OBJECTIVES: tuple[str, ...] = ("policy_violation", "social_engineering")
STANCES: tuple[str, ...] = ("high_precision", "balanced", "high_recall")

# (objective, stance) keys used throughout detection.py and metadata column
# names. Order matches detection pipeline conventions.
DETECTION_KEYS: tuple[str, ...] = tuple(
    f"{obj}__{st}" for obj in OBJECTIVES for st in STANCES
)

# Per-(key, turn) fields stored as JSON-serialized lists in the detection
# metadata.xlsx. Each list has length == n_turns of that conversation.
PER_TURN_FIELDS: tuple[str, ...] = (
    "prediction",
    "p_detected",
    "p_not_detected",
    "latency_ms",
    "input_tokens",
    "output_tokens",
)

# ---------------------------------------------------------------------------
# Speaker model
# ---------------------------------------------------------------------------

SPEAKER_REPRESENTATIVE = "representative"
SPEAKER_CALLER = "caller"


def speaker_for_turn(turn_index: int) -> str:
    """Speaker assignment by parity. Turn 0 is the rep greeting."""
    if turn_index == 0:
        return SPEAKER_REPRESENTATIVE
    return SPEAKER_CALLER if (turn_index % 2 == 1) else SPEAKER_REPRESENTATIVE


def detector_eligible_turns(n_turns: int, objective: str) -> list[int]:
    """Return the turn indices the given detector objective should fire on.

    Turn 0 is never eligible (it's the rep greeting; the detector is not run on it).
    SE detector runs on caller turns (odd indices >= 1).
    Policy detector runs on representative responses (even indices >= 2).
    """
    if objective == "social_engineering":
        return [i for i in range(1, n_turns) if i % 2 == 1]
    if objective == "policy_violation":
        return [i for i in range(2, n_turns) if i % 2 == 0]
    raise ValueError(f"unknown detection objective: {objective!r}")


# ---------------------------------------------------------------------------
# Generation-phase metadata columns kept on the detection metadata file
# ---------------------------------------------------------------------------

# Subset that we expect to find on detection metadata rows. Some have been
# prefixed `generation_` by the detection pipeline because they were
# ambiguous (e.g. `model` is the *generation* model in this file, the
# detection model lives in `detection_model`). We keep both sets here so
# loaders can be tolerant of either naming.
GENERATION_DIMENSION_COLUMNS: tuple[str, ...] = (
    "request_id",
    "replicate_index",
    "prompt_template_key",
    "scenario",
    "representative",
    "caller",
    "benign_context",
    "cialdini_emphasis",
    "turn_count_value",
    "flavor",
)

# Columns that the detection pipeline renames with a `generation_` prefix.
GENERATION_PREFIXED_COLUMNS: tuple[str, ...] = (
    "generation_model",
    "generation_temperature",
    "generation_top_p",
    "generation_status",
    "generation_attempts",
    "generation_last_error",
    "generation_generated_at_utc",
    "generation_input_tokens",
    "generation_output_tokens",
    "generation_total_tokens",
)

# Run-level detection columns we care about for analysis.
DETECTION_RUN_COLUMNS: tuple[str, ...] = (
    "detection_model",
    "detection_status",
    "detection_total_latency_ms",
    "detection_total_input_tokens",
    "detection_total_output_tokens",
    "detection_started_at_utc",
    "detection_finished_at_utc",
)

# ---------------------------------------------------------------------------
# Status values
# ---------------------------------------------------------------------------

DETECTION_STATUS_SUCCESS = "success"
DETECTION_STATUS_PARTIAL = "partial"
DETECTION_STATUS_ERROR = "error"

# ---------------------------------------------------------------------------
# Conversation type labels (derived from prompt_template_key)
# ---------------------------------------------------------------------------

CONVERSATION_TYPE_THREAT = "threat"
CONVERSATION_TYPE_BENIGN = "benign"
