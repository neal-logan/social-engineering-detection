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
    used. Both should display as the bare principle name. We do a
    substring match against the canonical principle list, with explicit
    aliases for principles whose prompt phrasing differs from the
    underscore-form key.

    For values that contain no recognizable principle, returns the
    original value as a string (so unexpected slices remain visible
    rather than silently collapsing).
    """
    if value is None:
        return "unknown"
    text = str(value).lower()

    # Aliases: each principle maps to the key plus any phrasings the
    # prompt text might use. Order does not matter; the loop below
    # picks the first principle for which any alias matches.
    aliases: dict[str, tuple[str, ...]] = {
        "commitment_consistency": (
            "commitment_consistency",
            "commitment consistency",
            "commitment and consistency",
            "commitment/consistency",
            "commitment-consistency",
        ),
        "social_proof": (
            "social_proof",
            "social proof",
        ),
        "reciprocity": ("reciprocity",),
        "authority": ("authority",),
        "liking": ("liking",),
        "scarcity": ("scarcity",),
        "unity": ("unity",),
    }

    # Match longest aliases first so e.g. "commitment and consistency"
    # is preferred over a shorter principle that happens to be a
    # substring.
    flat = sorted(
        ((principle, alias) for principle, al in aliases.items() for alias in al),
        key=lambda pa: -len(pa[1]),
    )
    for principle, alias in flat:
        if alias in text:
            return principle
    return str(value).strip()


# ---------------------------------------------------------------------------
# Short-label helpers for other prompt-text dimensions
# ---------------------------------------------------------------------------

# These columns may carry either the dimension *key* (e.g. "by_book") or
# the rendered prompt *value* (e.g. a paragraph describing a by-the-book
# representative). The analysis pipeline prefers the key when available
# (via the `*_key` companion column) and falls back to substring matches
# against known short labels.

REPRESENTATIVE_SHORT_LABELS: tuple[str, ...] = (
    "by_book",
    "tired",
    "helpful",
    "distracted",
)

BENIGN_CONTEXT_SHORT_LABELS: tuple[str, ...] = (
    "minimal",
    "moderate",
    "heavy",
)


# Aliases: each canonical short label maps to a tuple of phrasings the
# prompt text might use. Order does not matter; the matcher picks the
# canonical label whose longest matched alias appears in the value.
# Add new phrasings here whenever a real-data value falls through.

REPRESENTATIVE_ALIASES: dict[str, tuple[str, ...]] = {
    "by_book": (
        "by_book",
        "by the book",
        "by-the-book",
        "by-book",
        "follows policy",
        "follows procedure",
        "strict",
        "rule-following",
        "rule following",
        "policy-bound",
        "procedural",
    ),
    "tired": (
        "tired",
        "fatigued",
        "exhausted",
        "end of shift",
        "sleepy",
        "weary",
        "burnt out",
        "burned out",
    ),
    "helpful": (
        "helpful",
        "eager to help",
        "friendly",
        "accommodating",
        "obliging",
    ),
    "distracted": (
        "distracted",
        "multitasking",
        "rushed",
        "busy",
        "overwhelmed",
        "scattered",
    ),
}

BENIGN_CONTEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "minimal": (
        "minimal",
        "little benign",
        "brief",
        "few benign",
        "sparse",
        "low context",
        "no benign",
        # Phrasings observed in real generator prompts:
        "mostly stays on topic",
        "just a little extraneous",
        "businesslike",
    ),
    "moderate": (
        "moderate",
        "some benign",
        "balanced",
        "medium",
        "intermediate",
        # Phrasings observed in real generator prompts:
        "very substantial amount of benign",
        "two or three extraneous sentences",
        "obscure the attack shape",
    ),
    "heavy": (
        "heavy",
        "extensive benign",
        "lots of benign",
        "extensive",
        "verbose",
        "high context",
        "abundant benign",
        # Phrasings observed in real generator prompts:
        "absolutely unhinged",
        "rambling",
        "several extraneous sentences",
        "unreasonable manner",
    ),
}


def _alias_match(
    value: object,
    aliases: dict[str, tuple[str, ...]],
) -> str | None:
    """Return the canonical key whose longest matched alias is in `value`.

    Each canonical key (e.g. "by_book") maps to phrasings that may appear
    in the rendered prompt text (e.g. "by-the-book", "follows policy").
    We search across ALL aliases, sorted longest-first, and return the
    canonical key associated with the first hit. This guarantees that
    "by-the-book" wins over a shorter accidental substring like "book".
    """
    if value is None:
        return None
    text = str(value).lower()
    flat = sorted(
        ((canonical, alias)
         for canonical, al in aliases.items()
         for alias in al),
        key=lambda pa: -len(pa[1]),
    )
    for canonical, alias in flat:
        if alias in text:
            return canonical
    return None


def representative_short_label(value: object) -> str:
    """Return a short representative label.

    Tries the alias map; returns the canonical key on hit, otherwise
    returns the value unchanged so unrecognized phrasings remain
    visible (and patchable in REPRESENTATIVE_ALIASES) rather than
    silently collapsing.
    """
    if value is None:
        return "unknown"
    matched = _alias_match(value, REPRESENTATIVE_ALIASES)
    return matched if matched is not None else str(value).strip()


def benign_context_short_label(value: object) -> str:
    """Return a short benign-context label (minimal / moderate / heavy)."""
    if value is None:
        return "unknown"
    matched = _alias_match(value, BENIGN_CONTEXT_ALIASES)
    return matched if matched is not None else str(value).strip()


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
