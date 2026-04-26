"""
HTML rendering helpers.

The chat window is a single HTML component that we re-render on every state
change. Building HTML strings here keeps the UI logic in `ui.py` clean.
"""

from __future__ import annotations

import html
import time
from typing import Any


# Cialdini flag keys we expect on caller turns.
CIALDINI_KEYS = (
    "cialdini_reciprocity",
    "cialdini_commitment_consistency",
    "cialdini_social_proof",
    "cialdini_authority",
    "cialdini_liking",
    "cialdini_scarcity",
    "cialdini_unity",
)

# Improper-action flag keys we expect on rep turns.
IMPROPER_KEYS = (
    "improper_authentication",
    "improper_disclosure",
    "improper_action",
)

# Stances and objectives used by the detection module.
OBJECTIVES = ("policy_violation", "social_engineering")
STANCES = ("high_precision", "balanced", "high_recall")


def _short(name: str) -> str:
    """Display-friendly short labels."""
    s = name.replace("cialdini_", "").replace("improper_", "").replace("_", " ")
    return s


def render_turn_actual_flags(turn: dict[str, Any]) -> str:
    """Chips for actual ground-truth flags on the turn."""
    chips: list[str] = []
    speaker = turn.get("speaker", "")
    if speaker == "caller":
        for k in CIALDINI_KEYS:
            if turn.get(k):
                chips.append(f'<span class="flag flag-actual">{html.escape(_short(k))}</span>')
    elif speaker == "representative":
        for k in IMPROPER_KEYS:
            if turn.get(k):
                chips.append(f'<span class="flag flag-actual">{html.escape(_short(k))}</span>')
    if not chips:
        return ""
    return f'<div class="flags-row">{"".join(chips)}</div>'


def render_detection_panel(
    detection_for_turn: dict[str, dict[str, Any]] | None,
    pending: bool = False,
    pending_objective: str | None = None,
    pending_stances: list[str] | None = None,
) -> str:
    """
    Render the detection result(s) for one turn as a stack of large
    status-driven boxes — one per (objective, stance) the detector ran.

    Layout per box:
      [PENDING / DETECTED / NOT-SUSPECTED / FAILED headline]
      <small subtext: confidence, latency, stance>

    Args:
        detection_for_turn: keyed by `<objective>__<stance>`; each value
            has `prediction`, `p_detected`, `p_not_detected`, `latency_ms`.
            Empty/None means "no results yet" — combined with `pending=True`
            triggers the pending boxes.
        pending: if True and no results yet, render gray "Detection pending"
            boxes whose latency counts up from 0.0 to 3.0 then flips to
            "failed" via JS.
        pending_objective: which objective ('policy_violation' or
            'social_engineering') is being scored — controls the headline
            label of pending boxes.
        pending_stances: which stances are being scored. Controls how many
            pending boxes are rendered.
    """
    if pending and not detection_for_turn:
        return _render_pending_boxes(pending_objective, pending_stances or [])

    if not detection_for_turn:
        return ""

    parts: list[str] = ['<div class="detection-panel-v2">']
    for key, res in detection_for_turn.items():
        if not res:
            continue
        obj, _, stance = key.partition("__")
        pred = (res.get("prediction") or "").upper()
        parts.append(_render_one_detection_box(obj, stance, pred,
                                               res.get("p_detected"),
                                               res.get("p_not_detected"),
                                               res.get("latency_ms")))
    parts.append("</div>")
    return "".join(parts)


def _render_pending_boxes(objective: str | None, stances: list[str]) -> str:
    """Render gray 'Detection pending' boxes for in-flight detection calls."""
    if not objective:
        return ""
    if not stances:
        # If we don't know which stances will run, show one generic pending box.
        stances = ["balanced"]
    parts: list[str] = ['<div class="detection-panel-v2">']
    # A unique time-base id lets the JS animator find these specific boxes.
    base_id = f"detbox-{int(time.time()*1000)%10**8}"
    for i, stance in enumerate(stances):
        bid = f"{base_id}-{i}"
        parts.append(
            f'<div class="det-box det-box-pending" id="{bid}" '
            f'data-pending="1" data-start-ts="">'
            f'<div class="det-box-stance">{html.escape(stance.replace("_", " "))}</div>'
            f'<div class="det-box-headline det-box-headline-pending">'
            f'<span class="det-pending-spinner"></span>Security detection pending'
            '</div>'
            f'<div class="det-box-sub" data-latency-display>0.00s</div>'
            '</div>'
        )
    # Inline JS animator: walks the box, counts up from now, flips to
    # "failed" at 3s. Idempotent across multiple injections.
    parts.append(_PENDING_TIMER_JS)
    parts.append("</div>")
    return "".join(parts)


def _render_one_detection_box(
    obj: str,
    stance: str,
    pred: str,
    p_det: Any,
    p_not_det: Any,
    lat_ms: Any,
) -> str:
    """One concrete detection result box (Y / N / E / X)."""
    obj_pretty = "Social Engineering" if obj == "social_engineering" else "Policy Violation"
    stance_pretty = stance.replace("_", " ")

    # Headline + class based on prediction
    if pred == "Y":
        headline = f"{obj_pretty} Detected"
        cls = "det-box-detected"
        head_cls = "det-box-headline-detected"
    elif pred in ("N",):
        if obj == "social_engineering":
            headline = "SE attack not suspected"
        else:
            headline = "Policy compliance detected"
        cls = "det-box-clear"
        head_cls = "det-box-headline-clear"
    elif pred in ("X", "E"):
        headline = "Security detection failed"
        cls = "det-box-failed"
        head_cls = "det-box-headline-failed"
    else:
        # "S" or anything unexpected: treat as a positive-leaning detection
        headline = f"{obj_pretty} Suspected"
        cls = "det-box-detected"
        head_cls = "det-box-headline-detected"

    # Subtext
    sub_parts: list[str] = []
    if isinstance(p_det, (int, float)):
        sub_parts.append(f"confidence {p_det:.2f}")
    if isinstance(lat_ms, (int, float)):
        sub_parts.append(f"{int(lat_ms)} ms")
    sub = " · ".join(sub_parts)

    return (
        f'<div class="det-box {cls}">'
        f'<div class="det-box-stance">{html.escape(stance_pretty)}</div>'
        f'<div class="det-box-headline {head_cls}">{html.escape(headline)}</div>'
        + (f'<div class="det-box-sub">{html.escape(sub)}</div>' if sub else "")
        + "</div>"
    )


# Inline JS animator for pending boxes. Re-evaluated every render. We mark
# each pending box with data-pending="1" and stamp data-start-ts on first
# visit. setInterval keeps refreshing the latency display until 3s, then
# flips to "failed".
_PENDING_TIMER_JS = """
<script>
(function() {
  if (window.__detPendingTimerInstalled) return;
  window.__detPendingTimerInstalled = true;
  function tick() {
    var boxes = document.querySelectorAll('.det-box-pending[data-pending="1"]');
    var now = Date.now();
    boxes.forEach(function(box) {
      if (!box.dataset.startTs) {
        box.dataset.startTs = String(now);
      }
      var elapsed = (now - parseInt(box.dataset.startTs, 10)) / 1000;
      var disp = box.querySelector('[data-latency-display]');
      if (elapsed >= 3.0) {
        // Flip to "failed" — same gray styling, different headline
        box.dataset.pending = "0";
        var hl = box.querySelector('.det-box-headline');
        if (hl) {
          hl.classList.remove('det-box-headline-pending');
          hl.classList.add('det-box-headline-failed');
          hl.innerHTML = 'Security detection failed';
        }
        if (disp) disp.textContent = '> 3.00s — timed out';
      } else if (disp) {
        disp.textContent = elapsed.toFixed(2) + 's';
      }
    });
  }
  setInterval(tick, 100);
})();
</script>
"""


def render_turn(
    turn: dict[str, Any],
    turn_index: int,
    detection_for_turn: dict[str, dict[str, Any]] | None = None,
    transcribing: bool = False,
    pending_objective: str | None = None,
    pending_stances: list[str] | None = None,
) -> str:
    """
    Render a single turn card.

    Args:
        turn: the turn dict (must have 'text' or partial_text and the actual
            flags fields).
        turn_index: 0-based turn index.
        detection_for_turn: results dict (rendered into status boxes) or
            None (suppress).
        transcribing: if True, render a blinking cursor at the end.
        pending_objective: if set and detection_for_turn is empty, render
            pending status boxes for this objective.
        pending_stances: which stances should have pending boxes rendered.
    """
    speaker = turn.get("speaker", "?")
    speaker_label = "Representative" if speaker == "representative" else (
        "Caller" if speaker == "caller" else speaker.title()
    )
    speaker_class = (
        "speaker-rep" if speaker == "representative"
        else "speaker-caller" if speaker == "caller"
        else "speaker-other"
    )
    text = html.escape(turn.get("text", ""))
    if transcribing:
        text += '<span class="transcribing-cursor"></span>'
    actual_flags = render_turn_actual_flags(turn)
    if detection_for_turn:
        det_html = render_detection_panel(detection_for_turn)
    elif pending_objective:
        det_html = render_detection_panel(
            None, pending=True,
            pending_objective=pending_objective,
            pending_stances=pending_stances,
        )
    else:
        det_html = ""

    return (
        f'<div class="turn">'
        f'  <div class="turn-header">'
        f'    <span class="{speaker_class}">{speaker_label}</span>'
        f'    <span class="turn-index">turn {turn_index}</span>'
        f'  </div>'
        f'  <div class="turn-text">{text}</div>'
        f'  {actual_flags}'
        f'  {det_html}'
        f'</div>'
    )


def render_generating(message: str = "Generating conversation...") -> str:
    """
    Big friendly placeholder shown in the chat area while the dynamic-mode
    generator's first API call is in flight (before any turns exist).
    """
    return (
        '<div class="chat-window">'
        '<div class="generating-indicator">'
        '<div class="generating-spinner"></div>'
        f'<div class="generating-text">{html.escape(message)}</div>'
        '</div></div>'
    )


def render_chat(
    turns: list[dict[str, Any]],
    visible_count: int,
    detection_results: dict[int, dict[str, dict[str, Any]]] | None = None,
    transcribing_index: int | None = None,
    transcribing_partial_text: str | None = None,
    pending_for_turn: int | None = None,
    pending_objective: str | None = None,
    pending_stances: list[str] | None = None,
) -> str:
    """
    Render the full chat window.

    - `visible_count`: how many turns to show (0..len(turns)).
    - `detection_results`: keyed by turn index. Only shown for turns where it
      exists.
    - `transcribing_index` + `transcribing_partial_text`: when set, the turn
      at that index renders with a partial text body and a blinking cursor.
    - `pending_for_turn`: index of the turn currently awaiting detection.
      The pending boxes are rendered into that turn until detection_results
      contains an entry for it.
    - `pending_objective` / `pending_stances`: control which pending
      indicator(s) appear on the pending turn.
    """
    detection_results = detection_results or {}
    parts: list[str] = ['<div class="chat-window">']
    if visible_count == 0 and transcribing_index is None:
        parts.append(
            '<div style="color:var(--uncc-grey);text-align:center;'
            'padding:60px 20px;font-style:italic;">No conversation loaded yet.</div>'
        )
    for i in range(visible_count):
        turn = turns[i]
        det = detection_results.get(i)
        if det is None and i == pending_for_turn:
            parts.append(render_turn(
                turn, i, None,
                pending_objective=pending_objective,
                pending_stances=pending_stances,
            ))
        else:
            parts.append(render_turn(turn, i, det))
    if transcribing_index is not None and transcribing_index < len(turns):
        partial_turn = dict(turns[transcribing_index])
        partial_turn["text"] = transcribing_partial_text or ""
        # Don't show actual flags or detection while transcribing.
        partial_turn = {k: v for k, v in partial_turn.items()
                        if k not in CIALDINI_KEYS + IMPROPER_KEYS}
        parts.append(render_turn(partial_turn, transcribing_index, None, transcribing=True))
    parts.append("</div>")
    return "".join(parts)


def render_status(state: str, message: str = "") -> str:
    """Render a small status pill (idle / running / done / error)."""
    cls_map = {
        "idle": "status-idle",
        "running": "status-running",
        "done": "status-done",
        "error": "status-error",
    }
    cls = cls_map.get(state, "status-idle")
    return f'<span class="status-pill {cls}">{html.escape(message or state)}</span>'
