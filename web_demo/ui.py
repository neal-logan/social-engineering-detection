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
    pick_btn, play_btn).
    """
    empty_state = {"record": None, "turns": [], "detection": {},
                   "stored_detection": {}, "id": None,
                   "pv_stance": None, "se_stance": None}
    if not template_key or template_key not in dimensions.get("prompt_templates", {}):
        # Nothing selected: hide all dim dropdowns, clear chat.
        hidden = gr.update(visible=False)
        n_match_html = render.render_status("idle", "select a template")
        return [hidden] * 6 + [
            n_match_html,
            render.render_chat([], 0),
            empty_state,
            gr.update(interactive=False),  # pick button
            gr.update(interactive=False),  # play button
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
        empty_state,
        gr.update(interactive=n > 0),  # pick button
        gr.update(interactive=False),  # play button (no conversation loaded yet)
    ]


def _matching_conversations(
    template_key: str | None,
    dim_map: dict[str, str],
    field_labels: dict[str, str],
    by_id: dict[str, dict],
    pv_stance: str | None = None,
    se_stance: str | None = None,
    stance_index: dict[str, dict[str, set[str]]] | None = None,
    dimensions: dict | None = None,
) -> list[dict]:
    """
    Return all conversations matching the current filter set.

    Dimension matching: each `field_labels` entry is a dropdown label
    (e.g. "credit_union: <description...>") or RANDOM_SENTINEL. The
    generator stores the rendered VALUE in `selection['scenario']` (etc.)
    rather than the key, so we look up the chosen key in `dimensions` to
    get the value and compare against that.

    Stance matching is per-objective:
      - pv_stance / se_stance: a stance key, RANDOM_SENTINEL, or empty.
      - RANDOM_SENTINEL means "any stance fired for this objective"
        (i.e., the conversation has at least one Y/S in any stance for
        that objective).
      - Empty/None means "don't filter by this objective".
      - A specific stance key means "this stance fired Y/S for this
        objective".
    """
    if not template_key:
        return []

    dimensions = dimensions or {}

    # Resolve each filter into the actual record-side value to compare against.
    # For most dims: dropdown key -> dim value (textual description).
    # For turn_count: dropdown key -> dim value (a number, stored in
    # selection['turn_count_value']).
    expected: dict[str, Any] = {}  # selection_field_name -> expected value
    for field, label in field_labels.items():
        if field not in dim_map:
            continue
        if not label or label == RANDOM_SENTINEL:
            continue
        chosen_key = _label_to_key(label)
        dim_name = dim_map[field]
        dim_block = dimensions.get(dim_name, {})
        if not isinstance(dim_block, dict):
            continue
        # Map filter field to the corresponding selection field name.
        if field == "turn_count_key":
            sel_field = "turn_count_value"
        else:
            sel_field = field[:-len("_key")] if field.endswith("_key") else field
        # Look up the rendered value the record would have stored.
        expected_value = dim_block.get(chosen_key)
        if expected_value is None:
            continue
        expected[sel_field] = expected_value

    # Stance filters. Per-task semantics:
    #   - empty/None or RANDOM_SENTINEL => DON'T filter by stance for this task
    #   - a specific stance key => require that stance to have fired Y/S
    #     for that objective on this conversation.
    def _filter_stance(arg: str | None) -> str | None:
        """Normalize a stance filter input to a key, or None (no filter)."""
        if not arg or arg == RANDOM_SENTINEL:
            return None
        return _label_to_key(arg)

    pv_filter = _filter_stance(pv_stance) if stance_index is not None else None
    se_filter = _filter_stance(se_stance) if stance_index is not None else None

    out = []
    for rec in by_id.values():
        if rec.get("prompt_template_key") != template_key:
            continue
        sel = rec.get("selection", {})

        # Dimension filters
        ok = True
        for sel_field, expected_value in expected.items():
            if str(sel.get(sel_field, "")) != str(expected_value):
                ok = False
                break
        if not ok:
            continue

        # Stance filters (per-objective): only apply when a specific stance
        # was chosen, not on Random.
        if pv_filter is not None or se_filter is not None:
            cid = rec.get("request_id", "")
            stances_per_obj = (stance_index or {}).get(cid, {})

            if pv_filter is not None:
                pv_stances = stances_per_obj.get("policy_violation", set())
                if pv_filter not in pv_stances:
                    continue

            if se_filter is not None:
                se_stances = stances_per_obj.get("social_engineering", set())
                if se_filter not in se_stances:
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
    pv_stance_label: str,
    se_stance_label: str,
    dimensions: dict,
    by_id: dict[str, dict],
    stance_index: dict[str, dict[str, set[str]]],
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
        pv_stance=pv_stance_label, se_stance=se_stance_label,
        stance_index=stance_index, dimensions=dimensions,
    )
    n = len(matches)
    if n == 0:
        # Diagnose: print to terminal what the user picked and one example
        # record's selection values for that template — most useful when
        # the user filter never matches and we can't tell why from the UI.
        active_filters = {f: lab for f, lab in field_labels.items()
                          if lab and lab != RANDOM_SENTINEL}
        if active_filters:
            print(f"[static] 0 matches for template={template_key!r}, "
                  f"filters={active_filters}, "
                  f"pv_stance={pv_stance_label!r}, se_stance={se_stance_label!r}")
            for rec in by_id.values():
                if rec.get("prompt_template_key") == template_key:
                    sel = rec.get("selection", {})
                    print(f"[static] sample record selection: {sel}")
                    break
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
    pv_stance_label: str,
    se_stance_label: str,
    dimensions: dict,
    by_id: dict[str, dict],
    stance_index: dict[str, dict[str, set[str]]],
):
    """
    Pick a random conversation matching the current filter set and load it.
    Returns updates for (chat, state, status, play_btn).
    """
    empty_state = {"record": None, "turns": [], "detection": {},
                   "stored_detection": {}, "id": None,
                   "pv_stance": None, "se_stance": None}
    if not template_key:
        return (
            render.render_chat([], 0),
            empty_state,
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
        pv_stance=pv_stance_label, se_stance=se_stance_label,
        stance_index=stance_index, dimensions=dimensions,
    )
    if not matches:
        return (
            render.render_chat([], 0),
            empty_state,
            render.render_status("idle", "no matching conversation"),
            gr.update(interactive=False),
        )
    rec = random.choice(matches)
    return static_select_conversation(
        rec.get("request_id", ""), by_id,
        pv_stance_label=pv_stance_label,
        se_stance_label=se_stance_label,
        stance_index=stance_index,
    )


def static_select_conversation(
    conv_id: str,
    by_id: dict[str, dict],
    *,
    pv_stance_label: str = RANDOM_SENTINEL,
    se_stance_label: str = RANDOM_SENTINEL,
    stance_index: dict[str, dict[str, set[str]]] | None = None,
):
    """
    Load a conversation into state for static play. Returns (chat, state,
    status, play_btn). The chat shows nothing until Play is clicked.

    `stored_detection` is filtered to keep only the chosen stance per
    objective. RANDOM means pick one of the stances that fired Y/S for
    that objective on this conversation; if none fired, pick any preset
    stance to display its (presumably all-N) results.
    """
    empty_state = {"record": None, "turns": [], "detection": {},
                   "stored_detection": {}, "id": None,
                   "pv_stance": None, "se_stance": None}
    if not conv_id or conv_id not in by_id:
        return (
            render.render_chat([], 0),
            empty_state,
            render.render_status("idle", "no conversation"),
            gr.update(interactive=False),
        )
    rec = by_id[conv_id]
    turns = rec.get("conversation", {}).get("turns", []) or []
    raw_stored = adapters.load_detection_results(conv_id) or {}

    # Resolve the chosen stance per objective. Random => pick one of the
    # stances that actually fired Y/S for that conversation; if none did,
    # fall back to a random preset stance.
    presets = ("high_precision", "balanced", "high_recall")
    rec_stances_per_obj = (stance_index or {}).get(conv_id, {}) if stance_index else {}

    def _resolve(label: str, objective: str) -> str | None:
        if not label or label == RANDOM_SENTINEL:
            fired = list(rec_stances_per_obj.get(objective, set()))
            if fired:
                return random.choice(fired)
            return random.choice(presets)
        return _label_to_key(label)

    pv_chosen = _resolve(pv_stance_label, "policy_violation")
    se_chosen = _resolve(se_stance_label, "social_engineering")

    # Filter raw_stored: keep only the chosen stance per objective.
    keep_keys = set()
    if pv_chosen:
        keep_keys.add(f"policy_violation__{pv_chosen}")
    if se_chosen:
        keep_keys.add(f"social_engineering__{se_chosen}")

    stored_detection: dict[int, dict[str, dict[str, Any]]] = {}
    for turn_idx, det in raw_stored.items():
        filt = {k: v for k, v in det.items() if k in keep_keys}
        if filt:
            stored_detection[turn_idx] = filt

    state = {
        "record": rec,
        "turns": turns,
        "detection": {},
        "stored_detection": stored_detection,
        "id": conv_id,
        "pv_stance": pv_chosen,
        "se_stance": se_chosen,
    }
    return (
        render.render_chat([], 0),
        state,
        render.render_status(
            "idle",
            f"{len(turns)} turns loaded — PV={pv_chosen}, SE={se_chosen} — click Play"
        ),
        gr.update(interactive=len(turns) > 0),
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


def dynamic_clear():
    """Reset the dynamic-mode chat to its empty state."""
    return (
        render.render_chat([], 0),
        {"turns": [], "step": 0, "detection": {}, "id": None},
        render.render_status("idle"),
    )


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
    model: str,
    detection_model: str,
    pv_stance_label: str,
    pv_stance_custom: str,
    se_stance_label: str,
    se_stance_custom: str,
):
    """
    Generator yielding UI updates as the demo progresses through:
      1. building the selection,
      2. generating the conversation (one API call up front),
      3. transcribing each turn word-by-word,
      4. running detection after each completed turn.

    Detection args:
      detection_model: model used for the detection calls.
      pv_stance_label / se_stance_label: per-task stance choice — a stance
        key, RANDOM_SENTINEL (pick one preset stance at random), or
        CUSTOM_SENTINEL (use the custom instruction).
      pv_stance_custom / se_stance_custom: free-text stance instructions
        used when the corresponding label == CUSTOM_SENTINEL.
    """
    # ---- Build selection
    dim_map = _template_dim_map(dimensions, template_key)

    selection: dict[str, Any] = {
        "prompt_template_key": template_key,
        "model": model,
        "temperature": config.DEFAULT_GENERATION_TEMPERATURE,
        "top_p": config.DEFAULT_GENERATION_TOP_P,
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

    # ---- Resolve per-task stance settings.
    # For each task: Random => pick one preset stance at random; Custom =>
    # use __custom__ with the supplied text; otherwise => the chosen preset.
    presets = ("high_precision", "balanced", "high_recall")

    def _resolve_stance(label: str, custom_text: str) -> tuple[str, str]:
        """Return (stance_key, custom_text). custom_text is only non-empty
        when stance_key is '__custom__'."""
        if label == CUSTOM_SENTINEL:
            return ("__custom__", custom_text or "")
        if label == RANDOM_SENTINEL or not label:
            return (random.choice(presets), "")
        return (_label_to_key(label), "")

    pv_chosen_stance, pv_custom_text = _resolve_stance(pv_stance_label, pv_stance_custom)
    se_chosen_stance, se_custom_text = _resolve_stance(se_stance_label, se_stance_custom)

    def score_turn(i: int):
        # Pick stance + custom-text by speaker on this turn.
        speaker = adapters._speaker_for_turn(i)
        if speaker == "representative":
            stance, custom = pv_chosen_stance, pv_custom_text
        elif speaker == "caller":
            stance, custom = se_chosen_stance, se_custom_text
        else:
            return {}
        return adapters.score_turns_through(
            record,
            target_turn_index=i,
            stances=[stance],
            custom_stance_text=custom,
            model=detection_model,
        )

    yield from _play_conversation(
        record=record,
        state=state,
        score_turn=score_turn,
        pv_stance=pv_chosen_stance,
        se_stance=se_chosen_stance,
        pending_delay_s=0.0,
    )


def _play_conversation(
    record: dict,
    state: dict,
    score_turn,
    pv_stance: str | None,
    se_stance: str | None,
    pending_delay_s: float = 0.0,
):
    """
    Shared transcribe-and-score loop used by both modes.

    Args:
        record: a normalised conversation record (must have
            record["conversation"]["turns"]).
        state: mutable Gradio state dict; we set state["detection"].
        score_turn: a callable `score_turn(turn_index) -> dict | exception`.
        pv_stance: the stance key to render pending boxes for on
            representative turns. None => no pending box rendered.
        se_stance: the stance key to render pending boxes for on
            caller turns. None => no pending box rendered.
        pending_delay_s: extra sleep AFTER yielding pending boxes but
            BEFORE calling score_turn — used by static mode to make the
            pending boxes visible for ~600ms even though the lookup is
            instant.

    Yields (chat_html, state, status_html) tuples.
    """
    turns = record["conversation"]["turns"]
    detection: dict[int, dict[str, dict[str, Any]]] = state.get("detection") or {}
    state["detection"] = detection

    for i, turn in enumerate(turns):
        # ---- Word-by-word transcription
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

        # ---- Pending-detection state
        speaker = adapters._speaker_for_turn(i)
        if speaker == "representative":
            pending_obj = "policy_violation"
            pending_stance = pv_stance
        elif speaker == "caller":
            pending_obj = "social_engineering"
            pending_stance = se_stance
        else:
            pending_obj = None
            pending_stance = None

        if i > 0 and pending_obj and pending_stance:
            yield (
                render.render_chat(
                    turns,
                    visible_count=i + 1,
                    detection_results=detection,
                    pending_for_turn=i,
                    pending_objective=pending_obj,
                    pending_stances=[pending_stance],
                ),
                state,
                render.render_status("running", f"scoring turn {i}..."),
            )
            if pending_delay_s > 0:
                time.sleep(pending_delay_s)
        else:
            yield (
                render.render_chat(turns, visible_count=i + 1, detection_results=detection),
                state,
                render.render_status("running", f"scoring turn {i}..."),
            )

        # ---- Score the turn
        try:
            det = score_turn(i)
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

    yield (
        render.render_chat(turns, visible_count=len(turns), detection_results=detection),
        state,
        render.render_status("done", f"complete — {len(turns)} turns"),
    )


def static_clear(state: dict):
    """
    Reset the chat to empty but keep the loaded conversation in state so
    Play still works. To pick a fresh conversation, use Pick random match.
    """
    state = state or {}
    # Preserve record + stored_detection so re-play still works.
    state["detection"] = {}
    return (
        render.render_chat([], 0),
        state,
        render.render_status("idle", "cleared — click Play to replay"),
    )


def static_play(state: dict):
    """
    Static-mode generator. Replays a pre-loaded conversation with the same
    word-by-word transcription as dynamic mode, using stored detection
    results from xlsx instead of live API calls. Only one stance per
    objective is displayed (the one chosen at conversation-load time).
    """
    record = state.get("record")
    if not record:
        yield (
            render.render_chat([], 0),
            state,
            render.render_status("idle", "no conversation loaded"),
        )
        return

    stored = state.get("stored_detection") or {}
    pv_stance = state.get("pv_stance")
    se_stance = state.get("se_stance")

    def score_turn(i: int):
        return stored.get(i, {})

    # Reset accumulated detection so a re-play starts fresh.
    state["detection"] = {}

    yield from _play_conversation(
        record=record,
        state=state,
        score_turn=score_turn,
        pv_stance=pv_stance,
        se_stance=se_stance,
        pending_delay_s=0.6,
    )
