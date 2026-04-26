"""
UI construction and event handlers.

Two tabs: Static and Dynamic.

Static mode:
    - Conversation dropdown -> Prev/Next buttons step through turns.
    - Detection appears after a small delay each step.

Dynamic mode:
    - Template dropdown drives which other dropdowns are visible.
    - Each dimension dropdown has Random + Custom options in addition to the
      values from prompt_dimensions.json.
    - "Generate & Run" -> generate full conversation, then transcribe each turn
      word-by-word, then call detection after each turn.
"""

from __future__ import annotations

import html
import random
import time
import traceback
from typing import Any

import gradio as gr

import config
import adapters
import render


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SENTINEL = "🎲 Random"
CUSTOM_SENTINEL = "✏️ Custom..."


def html_escape_for_pre(s: str) -> str:
    """Escape so a string can be dropped into HTML preformatted-style content."""
    return html.escape(s or "")


def render_error_in_chat(title: str, exc: BaseException) -> str:
    """
    Render a full error report (type, message, traceback) inside the chat
    area so the user can see what went wrong without checking the terminal.
    """
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return (
        '<div class="chat-window">'
        '<div style="padding:14px;color:var(--uncc-grey-dark);font-family:monospace;'
        'font-size:12px;white-space:pre-wrap;line-height:1.5;">'
        f'<strong style="color:var(--flag-threat);">{html.escape(title)}</strong>\n\n'
        f'<strong>{html.escape(type(exc).__name__)}:</strong> '
        f'{html.escape(str(exc))}\n\n'
        f'{html.escape(tb)}'
        '</div></div>'
    )


# ---------------------------------------------------------------------------
# Dimension introspection
# ---------------------------------------------------------------------------

def _template_dim_map(dimensions: dict, template_key: str) -> dict[str, str]:
    """
    Return the {field_name: dimension_name} mapping for a given template, e.g.
        {"scenario_key": "scenario", "caller_key": "threat_caller_profile", ...}

    The current generation module stores this under
    dimensions["prompt_templates"][template_key]["dimensions"].
    """
    tpl_block = dimensions["prompt_templates"][template_key]
    if isinstance(tpl_block, dict) and "dimensions" in tpl_block:
        return dict(tpl_block["dimensions"])
    return {}


def _dimension_options(dimensions: dict, dim_name: str) -> list[tuple[str, str]]:
    """Return [(label, key), ...] for a dimension."""
    block = dimensions.get(dim_name, {})
    return [(f"{key}: {str(value)[:60]}", key) for key, value in block.items()]


# ---------------------------------------------------------------------------
# Static mode handlers
# ---------------------------------------------------------------------------

def _conversation_label(record: dict[str, Any]) -> str:
    sel = record.get("selection", {})
    src = record.get("_source_label", "")
    rid = record.get("request_id", "")
    bits: list[str] = []
    if src:
        bits.append(src)
    for k in ("scenario_key", "caller_key", "cialdini_emphasis_key", "benign_context_key"):
        if k in sel and sel[k]:
            bits.append(str(sel[k]))
    short_id = rid[:8] if rid else "??"
    return f"[{short_id}] " + " · ".join(bits) if bits else f"[{short_id}]"


def static_load_conversations() -> tuple[list[tuple[str, str]], dict[str, dict]]:
    """Return (dropdown choices, lookup by id)."""
    try:
        records = adapters.load_conversations()
    except FileNotFoundError:
        return [], {}
    by_id: dict[str, dict] = {}
    choices: list[tuple[str, str]] = []
    for rec in records:
        rid = rec.get("request_id", str(id(rec)))
        by_id[rid] = rec
        choices.append((_conversation_label(rec), rid))
    return choices, by_id


# ---------------------------------------------------------------------------
# Static mode: dimension-by-dimension filtering (mirrors dynamic mode)
# ---------------------------------------------------------------------------

# Mapping from dropdown field name (e.g. 'scenario_key') to the corresponding
# key in record['selection']. They're identical in current generation output,
# so this is just an alias map.
_STATIC_FIELDS = (
    "scenario_key",
    "representative_key",
    "caller_key",
    "benign_context_key",
    "cialdini_emphasis_key",
    "turn_count_key",
)


def static_template_changed(
    template_key: str | None,
    dimensions: dict,
    by_id: dict[str, dict],
):
    """
    When the user picks a (or clears the) template, rebuild every
    dimension dropdown so it shows only the values that actually appear in
    matching conversations. Each dropdown gets RANDOM_SENTINEL plus the keys
    that exist in the conversation pool for that template.

    Returns 6 dropdown updates (one per dimension) followed by the
    matching-count + state-reset payload (count_html, chat, state,
    pick_btn, prev_btn, next_btn).
    """
    if not template_key or template_key not in dimensions.get("prompt_templates", {}):
        # Nothing selected: hide all dim dropdowns, clear chat.
        hidden = gr.update(visible=False)
        n_match_html = render.render_status("idle", "select a template")
        return [hidden] * 6 + [
            n_match_html,
            render.render_chat([], 0),
            {"turns": [], "step": 0, "detection": {}, "id": None},
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        ]

    dim_map = _template_dim_map(dimensions, template_key)

    # Find every conversation that uses this template; collect the set of
    # keys that actually appear in the pool for each dimension.
    matching_recs = [
        rec for rec in by_id.values()
        if rec.get("prompt_template_key") == template_key
    ]
    pool_keys: dict[str, set[str]] = {f: set() for f in _STATIC_FIELDS}
    for rec in matching_recs:
        sel = rec.get("selection", {})
        for f in _STATIC_FIELDS:
            if sel.get(f):
                pool_keys[f].add(str(sel[f]))

    updates: list[Any] = []
    for field in _STATIC_FIELDS:
        if field not in dim_map:
            updates.append(gr.update(visible=False))
            continue
        dim_name = dim_map[field]
        dim_block = dimensions.get(dim_name, {})
        # Show every key from the dim file in the order it appears, even if
        # not present in the pool — user asked for "show all values; if no
        # match, display 'no matching conversation'".
        labels = [
            f"{key}: {str(value)[:60]}"
            for key, value in dim_block.items()
        ]
        choices = [RANDOM_SENTINEL] + labels
        updates.append(gr.update(
            visible=True,
            choices=choices,
            value=RANDOM_SENTINEL,
            label=f"{field.replace('_key','').replace('_',' ').title()}  ({dim_name})",
        ))

    n = len(matching_recs)
    if not by_id:
        # No conversations loaded at all — bubble up the load diagnostic so
        # the user can see why.
        diag = adapters.LAST_LOAD_DIAGNOSTIC or "(no diagnostic available)"
        count_msg = "no conversations loaded — see terminal for details"
        # Also dump the diagnostic to the chat area so it's visible.
        chat_html = (
            '<div class="chat-window">'
            '<div style="padding:14px;color:var(--uncc-grey-dark);font-family:monospace;'
            'font-size:12px;white-space:pre-wrap;line-height:1.5;">'
            '<strong style="color:var(--flag-threat);">'
            'No conversations loaded.</strong>\n\n'
            'Tried the following:\n\n'
            f'{html_escape_for_pre(diag)}\n\n'
            'Set <code>CONVERSATIONS_PATH</code> in your project-root '
            '<code>.env</code> file to point at the directory containing '
            '<code>threat_conversations.json</code> / '
            '<code>benign_conversations.json</code>, or at one of those '
            'files directly. Then restart the app.'
            '</div></div>'
        )
    else:
        count_msg = (
            f"{n} matching conversation" + ("s" if n != 1 else "")
            if n else "no matching conversation"
        )
        chat_html = render.render_chat([], 0)
    return updates + [
        render.render_status("idle", count_msg),
        chat_html,
        {"turns": [], "step": 0, "detection": {}, "id": None},
        gr.update(interactive=n > 0),  # pick button
        gr.update(interactive=False),  # prev
        gr.update(interactive=False),  # next
    ]


def _matching_conversations(
    template_key: str | None,
    dim_map: dict[str, str],
    field_labels: dict[str, str],
    by_id: dict[str, dict],
    stance_filter: str | None = None,
    stance_index: dict[str, set[str]] | None = None,
) -> list[dict]:
    """
    Return all conversations matching the current filter set. Dropdown
    labels in `field_labels` are dropdown labels (e.g. "polished: ...") or
    RANDOM_SENTINEL; we resolve them to keys and require equality on
    record['selection'][field].

    If `stance_filter` is a non-Random stance key and `stance_index` is
    provided, also require that the conversation's id appears in
    stance_index with that stance present (i.e. at least one Y/S
    prediction was logged for that stance).
    """
    if not template_key:
        return []

    # Resolve each field's chosen label to a key (or None if unfiltered).
    chosen_keys: dict[str, str] = {}
    for field, label in field_labels.items():
        if field not in dim_map:
            continue
        if not label or label == RANDOM_SENTINEL:
            continue
        chosen_keys[field] = _label_to_key(label)

    use_stance = (
        stance_filter and stance_filter != RANDOM_SENTINEL and stance_index is not None
    )
    stance_key = _label_to_key(stance_filter) if use_stance else None

    out = []
    for rec in by_id.values():
        if rec.get("prompt_template_key") != template_key:
            continue
        sel = rec.get("selection", {})
        if not all(str(sel.get(f, "")) == k for f, k in chosen_keys.items()):
            continue
        if use_stance:
            cid = rec.get("request_id", "")
            stances_with_hits = stance_index.get(cid, set())
            if stance_key not in stances_with_hits:
                continue
        out.append(rec)
    return out


def static_count_matches(
    template_key: str | None,
    scenario_label: str,
    rep_label: str,
    caller_label: str,
    benign_label: str,
    cialdini_label: str,
    turn_count_label: str,
    stance_label: str,
    dimensions: dict,
    by_id: dict[str, dict],
    stance_index: dict[str, set[str]],
):
    """
    Recompute the matching-count display when any filter changes.
    Doesn't touch the chat — that only updates when the user clicks
    'Pick random match'. Also (de)activates the pick button.
    """
    if not template_key:
        return (
            render.render_status("idle", "select a template"),
            gr.update(interactive=False),
        )
    dim_map = _template_dim_map(dimensions, template_key)
    field_labels = {
        "scenario_key":          scenario_label,
        "representative_key":    rep_label,
        "caller_key":            caller_label,
        "benign_context_key":    benign_label,
        "cialdini_emphasis_key": cialdini_label,
        "turn_count_key":        turn_count_label,
    }
    matches = _matching_conversations(
        template_key, dim_map, field_labels, by_id,
        stance_filter=stance_label, stance_index=stance_index,
    )
    n = len(matches)
    if n == 0:
        return (
            render.render_status("idle", "no matching conversation"),
            gr.update(interactive=False),
        )
    msg = f"{n} matching conversation" + ("s" if n != 1 else "")
    return (
        render.render_status("idle", msg),
        gr.update(interactive=True),
    )


def static_pick_random(
    template_key: str | None,
    scenario_label: str,
    rep_label: str,
    caller_label: str,
    benign_label: str,
    cialdini_label: str,
    turn_count_label: str,
    stance_label: str,
    dimensions: dict,
    by_id: dict[str, dict],
    stance_index: dict[str, set[str]],
):
    """
    Pick a random conversation matching the current filter set and render
    its first turn. Returns updates for (chat, state, status, prev_btn, next_btn).
    """
    if not template_key:
        return (
            render.render_chat([], 0),
            {"turns": [], "step": 0, "detection": {}, "id": None},
            render.render_status("idle", "select a template"),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    dim_map = _template_dim_map(dimensions, template_key)
    field_labels = {
        "scenario_key":          scenario_label,
        "representative_key":    rep_label,
        "caller_key":            caller_label,
        "benign_context_key":    benign_label,
        "cialdini_emphasis_key": cialdini_label,
        "turn_count_key":        turn_count_label,
    }
    matches = _matching_conversations(
        template_key, dim_map, field_labels, by_id,
        stance_filter=stance_label, stance_index=stance_index,
    )
    if not matches:
        return (
            render.render_chat([], 0),
            {"turns": [], "step": 0, "detection": {}, "id": None},
            render.render_status("idle", "no matching conversation"),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    rec = random.choice(matches)
    return static_select_conversation(rec.get("request_id", ""), by_id)


def static_select_conversation(
    conv_id: str,
    by_id: dict[str, dict],
):
    """Selecting a conversation resets the step counter and renders turn 0."""
    if not conv_id or conv_id not in by_id:
        return (
            render.render_chat([], 0),
            {"turns": [], "step": 0, "detection": {}, "id": None},
            render.render_status("idle", "no conversation"),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    rec = by_id[conv_id]
    turns = rec.get("conversation", {}).get("turns", []) or []
    detection = adapters.load_detection_results(conv_id) or {}
    state = {
        "turns": turns,
        "step": 1 if turns else 0,
        "detection": detection,
        "id": conv_id,
    }
    return (
        render.render_chat(turns, state["step"], None),  # detection added by stepper
        state,
        render.render_status("idle", f"{len(turns)} turns loaded"),
        gr.update(interactive=False),                    # prev disabled at step 1
        gr.update(interactive=len(turns) > 1),           # next enabled if >1 turn
    )


def static_step(state: dict, direction: int):
    """
    Generator: yields chat html in two phases per step --
        1. immediately render through state['step'] turns with PENDING
           detection boxes on the latest turn
        2. after a small delay, render again WITH detection (where available)
    """
    turns = state.get("turns", [])
    if not turns:
        yield (render.render_chat([], 0), state,
               render.render_status("idle"), gr.update(), gr.update())
        return

    step = max(1, min(len(turns), state.get("step", 1) + direction))
    state["step"] = step

    # Determine objective for pending boxes on the latest turn (step-1).
    latest_turn_idx = step - 1
    speaker = adapters._speaker_for_turn(latest_turn_idx)
    if speaker == "representative" and latest_turn_idx > 0:
        pending_obj = "policy_violation"
    elif speaker == "caller":
        pending_obj = "social_engineering"
    else:
        pending_obj = None

    detection = state.get("detection") or {}

    # Phase 1: pending boxes on the latest turn (only if we don't already
    # have results for it from the precomputed metadata).
    if pending_obj and latest_turn_idx not in detection:
        phase1_chat = render.render_chat(
            turns, step, detection,
            pending_for_turn=latest_turn_idx,
            pending_objective=pending_obj,
            pending_stances=["high_precision", "balanced", "high_recall"],
        )
    else:
        phase1_chat = render.render_chat(turns, step, detection)

    yield (
        phase1_chat,
        state,
        render.render_status("running", f"turn {step-1}"),
        gr.update(interactive=step > 1),
        gr.update(interactive=step < len(turns)),
    )

    # Phase 2: detection appears after delay.
    time.sleep(config.STATIC_DETECTION_DELAY_S)
    yield (
        render.render_chat(turns, step, detection),
        state,
        render.render_status("done", f"turn {step-1} of {len(turns)-1}"),
        gr.update(interactive=step > 1),
        gr.update(interactive=step < len(turns)),
    )


# ---------------------------------------------------------------------------
# Dynamic mode handlers
# ---------------------------------------------------------------------------

def _build_dropdown_choices(
    dimensions: dict,
    dim_name: str,
) -> list[str]:
    """
    Choices shown in a dropdown, in display-label form. The corresponding key
    is recovered from the label by stripping the prefix before ': '.
    """
    pairs = _dimension_options(dimensions, dim_name)
    labels = [label for label, _key in pairs]
    return [RANDOM_SENTINEL, CUSTOM_SENTINEL] + labels


def _label_to_key(label: str) -> str:
    """Recover the JSON key from a dropdown label."""
    if label in (RANDOM_SENTINEL, CUSTOM_SENTINEL):
        return label
    return label.split(":", 1)[0].strip() if ":" in label else label


def _resolve_dim_choice(
    dimensions: dict,
    dim_name: str,
    label: str,
    custom_text: str,
) -> tuple[str, str]:
    """
    Resolve a dropdown choice into (key, value).
      - Random: pick a random key from the dim, return (chosen_key, value).
      - Custom: return ("__custom__", custom_text).
      - Otherwise: look up the value from the dim block.
    """
    if label == RANDOM_SENTINEL:
        block = dimensions.get(dim_name, {})
        if not block:
            return ("", "")
        chosen_key = random.choice(list(block.keys()))
        return (chosen_key, str(block[chosen_key]))
    if label == CUSTOM_SENTINEL:
        return ("__custom__", custom_text or "")
    key = _label_to_key(label)
    val = dimensions.get(dim_name, {}).get(key, "")
    return (key, str(val))


def dynamic_template_changed(template_key: str, dimensions: dict):
    """
    When the user picks a template, show only the dropdowns the template
    consumes, and populate them with that template's dimensions.

    Returns updates for each of the dropdowns + their custom text inputs.
    """
    if not template_key or template_key not in dimensions.get("prompt_templates", {}):
        # Hide everything.
        hidden = gr.update(visible=False)
        return [hidden] * 12  # 6 dropdowns × 2 (dropdown + custom textbox)

    dim_map = _template_dim_map(dimensions, template_key)

    # Map field_name -> (dropdown_component_index, dim_name)
    # Field order must match the order we return updates in (see app.py wiring).
    field_order = [
        "scenario_key",
        "representative_key",
        "caller_key",
        "benign_context_key",
        "cialdini_emphasis_key",
        "turn_count_key",
    ]

    updates: list[Any] = []
    for field in field_order:
        if field in dim_map:
            dim_name = dim_map[field]
            choices = _build_dropdown_choices(dimensions, dim_name)
            updates.append(gr.update(
                visible=True,
                choices=choices,
                value=choices[0] if choices else None,
                label=f"{field.replace('_key','').replace('_',' ').title()}  ({dim_name})",
            ))
            # Custom textbox starts hidden until user picks Custom.
            updates.append(gr.update(visible=False, value=""))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(visible=False))
    return updates


def custom_visibility(dropdown_value: str):
    """Show the custom-text box only when the dropdown is set to Custom."""
    return gr.update(visible=(dropdown_value == CUSTOM_SENTINEL))


def dynamic_generate_and_run(
    dimensions: dict,
    template_key: str,
    scenario_label: str, scenario_custom: str,
    rep_label: str, rep_custom: str,
    caller_label: str, caller_custom: str,
    benign_label: str, benign_custom: str,
    cialdini_label: str, cialdini_custom: str,
    turn_count_label: str, turn_count_custom: str,
    flavor_label: str,
    model: str, temperature: float, top_p: float,
    detection_model: str,
    stance_label: str,
    stance_custom: str,
):
    """
    Generator yielding UI updates as the demo progresses through:
      1. building the selection,
      2. generating the conversation (one API call up front),
      3. transcribing each turn word-by-word,
      4. running detection after each completed turn.

    Detection args:
      detection_model: model used for the detection calls.
      stance_label: a stance key (e.g. "balanced"), RANDOM_SENTINEL (run all
        preset stances), or CUSTOM_SENTINEL (run only the custom stance).
      stance_custom: free-text stance instruction; used when
        stance_label == CUSTOM_SENTINEL.
    """
    # ---- Build selection
    dim_map = _template_dim_map(dimensions, template_key)

    selection: dict[str, Any] = {
        "prompt_template_key": template_key,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "caller_dim_name": dim_map.get("caller_key"),
    }

    field_inputs = [
        ("scenario_key",          scenario_label,    scenario_custom),
        ("representative_key",    rep_label,         rep_custom),
        ("caller_key",            caller_label,      caller_custom),
        ("benign_context_key",    benign_label,      benign_custom),
        ("cialdini_emphasis_key", cialdini_label,    cialdini_custom),
        ("turn_count_key",        turn_count_label,  turn_count_custom),
    ]
    for field, label, custom in field_inputs:
        if field in dim_map and label:
            dim_name = dim_map[field]
            key, value = _resolve_dim_choice(dimensions, dim_name, label, custom)
            selection[field] = key
            selection[field.replace("_key", "_value")] = value
            selection[field.replace("_key", "_custom")] = custom

    # Flavor: dropdown was populated from dimensions['flavors']; allow random.
    flavors = dimensions.get("flavors", []) or [""]
    if flavor_label == RANDOM_SENTINEL:
        selection["flavor"] = random.choice(flavors)
    else:
        selection["flavor"] = flavor_label or random.choice(flavors)

    state = {
        "turns": [],
        "step": 0,
        "detection": {},
        "id": None,
    }

    # ---- Phase 1: generation
    yield (
        render.render_generating("Generating conversation..."),
        state,
        render.render_status("running", "generating..."),
    )
    try:
        record = adapters.generate_conversation(selection)
    except NotImplementedError as e:
        yield (
            render_error_in_chat("Adapter not wired", e),
            state,
            render.render_status("error", f"adapter not wired: {e}"),
        )
        return
    except Exception as e:  # noqa: BLE001
        yield (
            render_error_in_chat("Conversation generation failed", e),
            state,
            render.render_status("error", f"generation failed: {type(e).__name__}"),
        )
        return

    turns = record.get("conversation", {}).get("turns", []) or []
    state["turns"] = turns
    state["id"] = record.get("request_id")

    if not turns:
        yield (
            render.render_chat([], 0),
            state,
            render.render_status("error", "generator returned 0 turns"),
        )
        return

    # ---- Resolve detection stance settings
    if stance_label == RANDOM_SENTINEL or not stance_label:
        stances_to_run: list[str] = []  # empty -> run all preset stances
        stance_text = ""
    elif stance_label == CUSTOM_SENTINEL:
        stances_to_run = ["__custom__"]
        stance_text = stance_custom or ""
    else:
        stances_to_run = [_label_to_key(stance_label)]
        stance_text = ""

    # ---- Phase 2: transcribe + score per turn
    detection: dict[int, dict[str, dict[str, Any]]] = {}
    for i, turn in enumerate(turns):
        words = (turn.get("text") or "").split(" ")
        partial = ""
        for j, w in enumerate(words):
            partial = w if j == 0 else partial + " " + w
            yield (
                render.render_chat(
                    turns,
                    visible_count=i,
                    detection_results=detection,
                    transcribing_index=i,
                    transcribing_partial_text=partial,
                ),
                state,
                render.render_status("running", f"transcribing turn {i}"),
            )
            time.sleep(config.DYNAMIC_WORD_DELAY_S)

        # Reveal complete turn (with actual flags) — no detection yet.
        # Then yield again with PENDING boxes so the user sees activity
        # while we wait for the detection API.
        speaker = adapters._speaker_for_turn(i)
        if speaker == "representative":
            pending_obj = "policy_violation"
        elif speaker == "caller":
            pending_obj = "social_engineering"
        else:
            pending_obj = None

        # Resolve which stances will actually run, for the pending boxes.
        if stances_to_run:
            pending_stance_list = list(stances_to_run)
        else:
            pending_stance_list = ["high_precision", "balanced", "high_recall"]

        if i > 0 and pending_obj:
            yield (
                render.render_chat(
                    turns,
                    visible_count=i + 1,
                    detection_results=detection,
                    pending_for_turn=i,
                    pending_objective=pending_obj,
                    pending_stances=pending_stance_list,
                ),
                state,
                render.render_status("running", f"scoring turn {i}..."),
            )
        else:
            yield (
                render.render_chat(turns, visible_count=i + 1, detection_results=detection),
                state,
                render.render_status("running", f"scoring turn {i}..."),
            )

        # Detection call against truncated conversation.
        try:
            det = adapters.score_turns_through(
                record,
                target_turn_index=i,
                stances=stances_to_run,
                custom_stance_text=stance_text,
                model=detection_model,
            )
        except NotImplementedError as e:
            yield (
                render_error_in_chat(f"Detection adapter not wired (turn {i})", e),
                state,
                render.render_status("error", f"detection adapter not wired: {e}"),
            )
            return
        except Exception as e:  # noqa: BLE001
            yield (
                render_error_in_chat(f"Detection failed at turn {i}", e),
                state,
                render.render_status("error", f"detection failed at turn {i}: {type(e).__name__}"),
            )
            return

        if det:
            detection[i] = det

        state["detection"] = detection
        yield (
            render.render_chat(turns, visible_count=i + 1, detection_results=detection),
            state,
            render.render_status("running", f"turn {i} of {len(turns)-1} complete"),
        )
        time.sleep(config.DYNAMIC_INTER_TURN_DELAY_S)

    # ---- Done.
    yield (
        render.render_chat(turns, visible_count=len(turns), detection_results=detection),
        state,
        render.render_status("done", f"complete — {len(turns)} turns"),
    )
