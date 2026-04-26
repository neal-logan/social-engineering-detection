"""
Main Gradio app.

Run with:
    python app.py

Then open http://127.0.0.1:7860 in a browser.
"""

from __future__ import annotations

import gradio as gr

import adapters
import config
import style
import ui


# ---------------------------------------------------------------------------
# Load dimensions once at startup.
# ---------------------------------------------------------------------------

print("=" * 70)
print("SE Detection Demo — startup")
print("=" * 70)
print(f"Project root:     {config.PROJECT_ROOT}")
print(f"Dimensions path:  {config.DIMENSIONS_PATH}  "
      f"(exists={config.DIMENSIONS_PATH.exists()})")
print(f"Conversations:    {config.CONVERSATIONS_PATH}  "
      f"(exists={config.CONVERSATIONS_PATH.exists()})")
print(f"Detection xlsx:   {config.DETECTION_METADATA_PATH}  "
      f"(exists={config.DETECTION_METADATA_PATH.exists()})")
print()

DIMENSIONS = adapters.load_dimensions()
TEMPLATE_KEYS = list(DIMENSIONS.get("prompt_templates", {}).keys())
FLAVORS = list(DIMENSIONS.get("flavors", []) or [])
DETECTION_STANCES = adapters.list_detection_stances()
print(f"Loaded {len(TEMPLATE_KEYS)} prompt template(s): {TEMPLATE_KEYS}")
print(f"Loaded {len(FLAVORS)} flavor(s)")
print(f"Detection stances available: {DETECTION_STANCES}")

# Static-mode conversation pool (loaded once; if the file changes between
# runs of the app, restart it).
STATIC_CHOICES, STATIC_BY_ID = ui.static_load_conversations()
print(f"Loaded {len(STATIC_BY_ID)} conversation record(s) for static mode")
if adapters.LAST_LOAD_DIAGNOSTIC.startswith("dropped"):
    print(f"  ({adapters.LAST_LOAD_DIAGNOSTIC})")

# Diagnostic: what prompt_template_key values are actually seen in the
# loaded records? These should match the template keys from the dim file.
if STATIC_BY_ID:
    from collections import Counter
    seen_template_keys = Counter(
        r.get("prompt_template_key", "<missing>") for r in STATIC_BY_ID.values()
    )
    print(f"Template keys in loaded records: {dict(seen_template_keys)}")
    if not seen_template_keys.keys() & set(TEMPLATE_KEYS):
        print()
        print("!! WARNING: no overlap between the templates declared in the")
        print(f"!! dimensions file ({TEMPLATE_KEYS}) and the prompt_template_key")
        print(f"!! values seen in conversation records ({list(seen_template_keys.keys())}).")
        print("!! Static-mode filtering will not find any matches.")
        print()
        # Show keys of one record so the user can see what fields ARE present
        sample_rec = next(iter(STATIC_BY_ID.values()))
        print(f"!! Sample record fields: {sorted(sample_rec.keys())}")
        sel = sample_rec.get("selection", {})
        if sel:
            print(f"!! Sample record selection keys: {sorted(sel.keys())}")
        print()

# Detection-stance index: which conversations have at least one Y/S
# prediction logged for which stance(s). Empty if no detection xlsx files.
STANCE_INDEX = adapters.index_stance_detections()
print(f"Stance index: {len(STANCE_INDEX)} conversation(s) have detection records")
if not STATIC_BY_ID:
    print()
    print("!! No conversations loaded. Diagnostic:")
    diag = adapters.LAST_LOAD_DIAGNOSTIC or "(no diagnostic available)"
    for line in diag.splitlines():
        print(f"   {line}")
    print()
    print("Set CONVERSATIONS_PATH in your project-root .env to override the")
    print("default. It can point to either a directory containing")
    print("threat_conversations.json / benign_conversations.json, or to a")
    print("single one of those files.")
print("=" * 70)
print()


# ---------------------------------------------------------------------------
# Build the UI.
# ---------------------------------------------------------------------------

with gr.Blocks(title="SE Detection Demo — UNC Charlotte") as demo:

    # ---- Header banner
    gr.HTML(
        '<div class="header-banner">'
        '<h1>Social-Engineering Detection — Live Demo</h1>'
        '<div class="subtitle">'
        'Real-time detection of LLM-enabled social engineering in customer-service '
        'conversations &nbsp;·&nbsp; UNC Charlotte research project'
        '</div>'
        '</div>'
    )

    # ---- Cached dimensions for handlers
    dimensions_state = gr.State(DIMENSIONS)
    static_lookup_state = gr.State(STATIC_BY_ID)
    static_stance_index_state = gr.State(STANCE_INDEX)

    with gr.Tabs():

        # ===================================================================
        # STATIC MODE
        # ===================================================================
        with gr.Tab("Static — pre-generated conversations"):

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">Prompt template</div>')
                    static_template = gr.Dropdown(
                        choices=TEMPLATE_KEYS,
                        label="Template",
                        value=TEMPLATE_KEYS[0] if TEMPLATE_KEYS else None,
                        interactive=True,
                    )

                    gr.HTML('<div class="section-label">Filters</div>')
                    static_scenario = gr.Dropdown(label="Scenario", choices=[], visible=False, interactive=True)
                    static_rep      = gr.Dropdown(label="Representative", choices=[], visible=False, interactive=True)
                    static_caller   = gr.Dropdown(label="Caller", choices=[], visible=False, interactive=True)
                    static_benign   = gr.Dropdown(label="Benign context", choices=[], visible=False, interactive=True)
                    static_cial     = gr.Dropdown(label="Cialdini emphasis", choices=[], visible=False, interactive=True)
                    static_turns    = gr.Dropdown(label="Turn count", choices=[], visible=False, interactive=True)

                    gr.HTML('<div class="section-label">Detection filter</div>')
                    static_stance = gr.Dropdown(
                        label="Stance (require detection by)",
                        choices=[ui.RANDOM_SENTINEL] + DETECTION_STANCES,
                        value=ui.RANDOM_SENTINEL,
                        interactive=True,
                    )
                    if not STANCE_INDEX:
                        gr.Markdown(
                            "_No detection metadata found — the stance filter is a no-op._",
                            elem_classes="section-help",
                        )

                    static_status = gr.HTML(ui.render.render_status("idle", "select a template"))
                    static_pick_btn = gr.Button("🎲 Pick random match", interactive=False, elem_classes="primary-btn")

                    with gr.Row():
                        static_prev_btn = gr.Button("◀ Prev", interactive=False, elem_classes="secondary-btn")
                        static_next_btn = gr.Button("Next ▶", interactive=False, elem_classes="secondary-btn")

                    gr.Markdown(
                        "_🎲 Random in any dropdown leaves that dimension unfiltered. "
                        f"After picking, step through turn-by-turn — detection scores appear "
                        f"~{int(config.STATIC_DETECTION_DELAY_S * 1000)} ms after each turn._",
                        elem_classes="section-help",
                    )

                with gr.Column(scale=2):
                    static_chat = gr.HTML(ui.render.render_chat([], 0))

            static_state = gr.State({"turns": [], "step": 0, "detection": {}, "id": None})

            static_dim_dropdowns = [
                static_scenario, static_rep, static_caller,
                static_benign, static_cial, static_turns,
            ]

            # Template change: rebuild every dim dropdown's choices, reset chat,
            # update count + pick button.
            static_template.change(
                fn=ui.static_template_changed,
                inputs=[static_template, dimensions_state, static_lookup_state],
                outputs=(
                    static_dim_dropdowns
                    + [static_status, static_chat, static_state,
                       static_pick_btn, static_prev_btn, static_next_btn]
                ),
            )

            # Any dim or stance dropdown change: recompute matching count + pick button.
            for dd in static_dim_dropdowns + [static_stance]:
                dd.change(
                    fn=ui.static_count_matches,
                    inputs=[
                        static_template,
                        static_scenario, static_rep, static_caller,
                        static_benign, static_cial, static_turns,
                        static_stance,
                        dimensions_state, static_lookup_state,
                        static_stance_index_state,
                    ],
                    outputs=[static_status, static_pick_btn],
                )

            # Pick a random match and load it.
            static_pick_btn.click(
                fn=ui.static_pick_random,
                inputs=[
                    static_template,
                    static_scenario, static_rep, static_caller,
                    static_benign, static_cial, static_turns,
                    static_stance,
                    dimensions_state, static_lookup_state,
                    static_stance_index_state,
                ],
                outputs=[static_chat, static_state, static_status,
                         static_prev_btn, static_next_btn],
            )

            static_prev_btn.click(
                fn=lambda s: ui.static_step(s, -1),
                inputs=[static_state],
                outputs=[static_chat, static_state, static_status, static_prev_btn, static_next_btn],
            )
            static_next_btn.click(
                fn=lambda s: ui.static_step(s, +1),
                inputs=[static_state],
                outputs=[static_chat, static_state, static_status, static_prev_btn, static_next_btn],
            )

            # Initial population: trigger the template-change handler once on
            # load so the first template's filters appear.
            demo.load(
                fn=ui.static_template_changed,
                inputs=[static_template, dimensions_state, static_lookup_state],
                outputs=(
                    static_dim_dropdowns
                    + [static_status, static_chat, static_state,
                       static_pick_btn, static_prev_btn, static_next_btn]
                ),
            )

        # ===================================================================
        # DYNAMIC MODE
        # ===================================================================
        with gr.Tab("Dynamic — generate live"):

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">Prompt template</div>')
                    dyn_template = gr.Dropdown(
                        choices=TEMPLATE_KEYS,
                        label="Template",
                        value=TEMPLATE_KEYS[0] if TEMPLATE_KEYS else None,
                        interactive=True,
                    )

                    gr.HTML('<div class="section-label">Dimensions</div>')

                    # Each dimension is a (Dropdown, Textbox) pair. The dropdown
                    # offers Random/Custom + the values from prompt_dimensions.
                    # The textbox is hidden unless Custom is picked.
                    dyn_scenario      = gr.Dropdown(label="Scenario", choices=[], visible=False, interactive=True)
                    dyn_scenario_cust = gr.Textbox(label="Custom scenario", visible=False, lines=2)

                    dyn_rep      = gr.Dropdown(label="Representative", choices=[], visible=False, interactive=True)
                    dyn_rep_cust = gr.Textbox(label="Custom representative", visible=False, lines=2)

                    dyn_caller      = gr.Dropdown(label="Caller", choices=[], visible=False, interactive=True)
                    dyn_caller_cust = gr.Textbox(label="Custom caller", visible=False, lines=2)

                    dyn_benign      = gr.Dropdown(label="Benign context", choices=[], visible=False, interactive=True)
                    dyn_benign_cust = gr.Textbox(label="Custom benign context", visible=False, lines=2)

                    dyn_cial      = gr.Dropdown(label="Cialdini emphasis", choices=[], visible=False, interactive=True)
                    dyn_cial_cust = gr.Textbox(label="Custom Cialdini emphasis", visible=False, lines=2)

                    dyn_turns      = gr.Dropdown(label="Turn count", choices=[], visible=False, interactive=True)
                    dyn_turns_cust = gr.Textbox(label="Custom turn count", visible=False, lines=1)

                    gr.HTML('<div class="section-label">Flavor</div>')
                    dyn_flavor = gr.Dropdown(
                        label="Flavor",
                        choices=[ui.RANDOM_SENTINEL] + FLAVORS,
                        value=ui.RANDOM_SENTINEL,
                        interactive=True,
                    )

                    gr.HTML('<div class="section-label">Generation settings</div>')
                    with gr.Row():
                        dyn_model = gr.Textbox(
                            label="Model",
                            value=config.DEFAULT_GENERATION_MODEL,
                            interactive=True,
                        )
                    with gr.Row():
                        dyn_temp = gr.Number(
                            label="Temperature",
                            value=config.DEFAULT_GENERATION_TEMPERATURE,
                            precision=2,
                            interactive=True,
                        )
                        dyn_top_p = gr.Number(
                            label="Top-p",
                            value=config.DEFAULT_GENERATION_TOP_P,
                            precision=2,
                            interactive=True,
                        )

                    gr.HTML('<div class="section-label">Detection settings</div>')
                    dyn_detection_model = gr.Textbox(
                        label="Detection model",
                        value=config.DEFAULT_DETECTION_MODEL,
                        interactive=True,
                    )
                    dyn_stance = gr.Dropdown(
                        label="Stance",
                        choices=[ui.RANDOM_SENTINEL, ui.CUSTOM_SENTINEL] + DETECTION_STANCES,
                        value=ui.RANDOM_SENTINEL,
                        interactive=True,
                    )
                    dyn_stance_custom = gr.Textbox(
                        label="Custom stance instruction",
                        visible=False,
                        lines=3,
                        placeholder="Free-text instructions to give the detection model. "
                                    "E.g., 'Bias strongly toward flagging anything that even mildly "
                                    "resembles social engineering, including borderline cases.'",
                    )

                    dyn_run_btn = gr.Button("Generate & Run", elem_classes="primary-btn")
                    dyn_status = gr.HTML(ui.render.render_status("idle"))

                with gr.Column(scale=2):
                    dyn_chat = gr.HTML(ui.render.render_chat([], 0))

            dyn_state = gr.State({"turns": [], "step": 0, "detection": {}, "id": None})

            # ---- Conditional dropdown visibility based on template
            dyn_template.change(
                fn=ui.dynamic_template_changed,
                inputs=[dyn_template, dimensions_state],
                outputs=[
                    dyn_scenario, dyn_scenario_cust,
                    dyn_rep, dyn_rep_cust,
                    dyn_caller, dyn_caller_cust,
                    dyn_benign, dyn_benign_cust,
                    dyn_cial, dyn_cial_cust,
                    dyn_turns, dyn_turns_cust,
                ],
            )

            # ---- Show/hide custom textboxes based on dropdown selection
            for dd, txt in [
                (dyn_scenario, dyn_scenario_cust),
                (dyn_rep, dyn_rep_cust),
                (dyn_caller, dyn_caller_cust),
                (dyn_benign, dyn_benign_cust),
                (dyn_cial, dyn_cial_cust),
                (dyn_turns, dyn_turns_cust),
            ]:
                dd.change(fn=ui.custom_visibility, inputs=[dd], outputs=[txt])

            # Show stance custom textbox only when "Custom..." picked
            dyn_stance.change(fn=ui.custom_visibility, inputs=[dyn_stance], outputs=[dyn_stance_custom])

            # ---- Run
            dyn_run_btn.click(
                fn=ui.dynamic_generate_and_run,
                inputs=[
                    dimensions_state, dyn_template,
                    dyn_scenario, dyn_scenario_cust,
                    dyn_rep, dyn_rep_cust,
                    dyn_caller, dyn_caller_cust,
                    dyn_benign, dyn_benign_cust,
                    dyn_cial, dyn_cial_cust,
                    dyn_turns, dyn_turns_cust,
                    dyn_flavor,
                    dyn_model, dyn_temp, dyn_top_p,
                    dyn_detection_model,
                    dyn_stance, dyn_stance_custom,
                ],
                outputs=[dyn_chat, dyn_state, dyn_status],
            )

            # Initial population: trigger the template-change handler so the
            # appropriate dropdowns render on first paint.
            demo.load(
                fn=ui.dynamic_template_changed,
                inputs=[dyn_template, dimensions_state],
                outputs=[
                    dyn_scenario, dyn_scenario_cust,
                    dyn_rep, dyn_rep_cust,
                    dyn_caller, dyn_caller_cust,
                    dyn_benign, dyn_benign_cust,
                    dyn_cial, dyn_cial_cust,
                    dyn_turns, dyn_turns_cust,
                ],
            )


# ---------------------------------------------------------------------------
# Launch.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.queue()  # enables generator (yield-based) handlers
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        css=style.CSS,
    )
