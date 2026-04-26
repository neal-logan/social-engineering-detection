"""
Configuration: paths, pacing, and the UNC Charlotte palette.

Everything tunable lives here so the rest of the app stays clean.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env. Prefer the project-root .env (the same one your uv venv uses
# for conversation_generation and the detection pipeline). Fall back to a
# web_demo/.env if you'd rather keep demo-specific keys separate.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_THIS_DIR / ".env", override=False)  # demo-local takes second pass


# ---------------------------------------------------------------------------
# Paths (relative to web_demo/, overridable via .env)
# ---------------------------------------------------------------------------

PROJECT_ROOT = _THIS_DIR.parent

DIMENSIONS_PATH = Path(os.environ.get(
    "DIMENSIONS_PATH",
    PROJECT_ROOT / "conversation_generation" / "prompt_dimensions.json",
))

CONVERSATIONS_PATH = Path(os.environ.get(
    "CONVERSATIONS_PATH",
    PROJECT_ROOT / "conversations" / "conversations.json",
))

DETECTION_METADATA_PATH = Path(os.environ.get(
    "DETECTION_METADATA_PATH",
    PROJECT_ROOT / "detection_results" / "detection_metadata.xlsx",
))


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------

# Static mode: artificial delay between displaying a turn and showing its
# detection scores, so the demo visually separates the two phases.
STATIC_DETECTION_DELAY_S = 0.6

# Dynamic mode: delay per word during turn "transcription". 50 ms/word is
# fast enough to feel snappy in a demo while still being readable.
DYNAMIC_WORD_DELAY_S = 0.05

# Tiny delay between turns in dynamic mode (after detection scoring) so
# transitions are visible.
DYNAMIC_INTER_TURN_DELAY_S = 0.3


# ---------------------------------------------------------------------------
# Detection model defaults (used in dynamic mode)
# ---------------------------------------------------------------------------

# Generation defaults — the dynamic-mode UI exposes these as dropdowns
# but these provide sensible starting values.
DEFAULT_GENERATION_MODEL = "gpt-5.4"
DEFAULT_GENERATION_TEMPERATURE = 1.0
DEFAULT_GENERATION_TOP_P = 1.0

# Detection model — matches the project's detection_inference notebook.
DEFAULT_DETECTION_MODEL = "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# UNC Charlotte palette
# ---------------------------------------------------------------------------
# Source: https://brand.charlotte.edu/visual-identity/color-palette/
# Primaries:
CHARLOTTE_GREEN = "#005035"      # PMS 7484
NINER_GOLD      = "#A49665"      # PMS 7503
# Athletic-mark variants:
ATHLETIC_GREEN  = "#046A38"      # PMS 349 (slightly brighter)
ATHLETIC_GOLD   = "#B9975B"      # PMS 465
# Neutrals:
WHITE           = "#FFFFFF"
NEAR_BLACK      = "#1A1A1A"
GREY_DARK       = "#3F3F3F"
GREY            = "#7A7A7A"
GREY_LIGHT      = "#D9D9D9"
BG_TINT         = "#F5F5F0"      # very faint warm off-white

# Semantic uses
SPEAKER_REP_COLOR    = CHARLOTTE_GREEN   # representative
SPEAKER_CALLER_COLOR = ATHLETIC_GOLD     # caller
FLAG_THREAT_COLOR    = "#A33A3A"         # actual threat / violation
FLAG_PRED_COLOR      = "#7A5C26"         # predicted / suspected (golden-brown)
FLAG_NEUTRAL_COLOR   = GREY
