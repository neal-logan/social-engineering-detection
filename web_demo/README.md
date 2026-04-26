# Web Demo

A small Gradio app that demonstrates the social-engineering detection pipeline
in real time on either pre-generated conversations (static mode) or freshly
generated conversations (dynamic mode).

It is a **thin UI layer** over the project's existing modules — it imports
generation and detection logic rather than duplicating it. The only
project-specific glue lives in `adapters.py`; everything else (`app.py`,
`ui.py`, `style.py`) is generic.

## Folder layout assumed

```
project_root/
├── conversation_generation/
│   ├── conversation_generation.py
│   └── prompt_dimensions.json
├── detection_pipeline/
│   └── detection.py                 (or whatever you've named it)
├── conversations/
│   └── conversations.json
├── detection_results/
│   └── detection_metadata.xlsx      (optional, used for static mode)
└── web_demo/                        ← this folder
    ├── app.py
    ├── adapters.py
    ├── ui.py
    ├── style.py
    ├── config.py
    ├── requirements.txt
    ├── .env                         (you create: OPENAI_API_KEY=...)
    └── README.md
```

## Setup

The demo runs on the same `uv` venv as the rest of the project. From the
project root:

```powershell
uv add gradio                    # one-time, if not already added
uv run python web_demo\app.py
```

That's it. The demo reuses the project-root `.env` for `OPENAI_API_KEY`, and
imports `conversation_generation` and the detection module as siblings.

The app prints `Running on local URL: http://127.0.0.1:7860` — open that in
a browser. `Ctrl+C` in the terminal to stop.

## Wiring `adapters.py`

`adapters.py` is the single place where the demo touches your project code.
It is fully wired against the current `conversation_generation` and
`detection_pipeline` modules — no TODOs to fill in. If a function signature
in either of those modules changes later, the only file that needs to
change is `adapters.py`.

The five things `adapters.py` provides:

1. `load_dimensions()` — return the parsed `prompt_dimensions.json`.
2. `load_conversations()` — return the union of records from
   `threat_conversations.json` and `benign_conversations.json`.
3. `load_detection_results(conv_id)` — pivot per-conversation rows from
   `*_detection_metadata.xlsx` into a per-turn dict for the UI overlay.
4. `generate_conversation(selection)` — dynamic mode: render prompts via
   `cg.render_one`, call `cg.call_openai`, return a normalised record
   matching the static-mode schema.
5. `score_turns_through(conv, k)` — dynamic mode: build the truncated
   transcript via `det.transcript_through_turn`, run the 3 stance variants
   for the appropriate objective, return predictions + probabilities + latency.

## Schema notes

The on-disk conversation schema stores turns as
`{"turn_0": {"passage": ...}, "turn_1": {...}, ...}` with speaker derived
by index parity. The adapter normalises every record so the UI can read
`record["conversation"]["turns"]` as a flat list with `speaker` and `text`
already populated. Original `turn_N` keys remain on the record so the
detection module (which expects them) still works untouched.

Detection codes follow the pipeline's convention: `Y`/`N`/`E`/`X`/`null`,
where `E` means the model returned no Y or N in the top-10 logprobs and
`X` means the API call itself failed after retries.

## Modes

**Static.** Pick a conversation from the dropdown. Use the **◀ Prev** /
**Next ▶** buttons to step through turn-by-turn. Each turn renders
immediately; the detection result for that turn appears with a small
artificial delay (configurable in `config.py`, default 600 ms) so the demo
visibly separates the two phases.

**Dynamic.** Configure the dimension dropdowns. Each dropdown shows three
choices: pick a specific value, pick "🎲 Random", or pick "✏️ Custom" to type
one in. Press **Generate & Run**. The full conversation is generated up front
(matching the existing generation module exactly), then "transcribed" into
the chat window word-by-word at ~50 ms/word. After each completed turn, the
conversation-to-date is sent to the detection module and its scores +
latencies appear above the next turn.

## Conditional dropdown logic

Dropdown visibility is driven entirely by the `dimensions` field of each
prompt template in `prompt_dimensions.json`. If a template's dim_map doesn't
declare `benign_context`, the benign-context dropdown is hidden. Picking a
threat template shows `cialdini_emphasis` + `threat_caller_profile`; picking
a benign template shows `benign_context` + `benign_caller_profile`. No
dimension-specific code in the UI — it all comes from the JSON.

## Colors

UNC Charlotte primary: Charlotte Green `#005035`, Niner Gold `#A49665`.
Same anchors as the analysis module so the demo and the figures match.
