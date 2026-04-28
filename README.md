# Social Engineering Detection in Customer-Service Chat

A research project assessing how the difficulty of LLM-based detection of LLM-enabled social engineering (SE) attempts varies with attack tactic. The system context is a text-based banking customer-service chat. Threat conversations attempt to compromise confidentiality (eliciting non-public personal information) or integrity (improper account actions); benign conversations exercise the same surface forms without malicious intent.

The project is organized as a four-stage pipeline: conversations are generated, scored by detectors, analyzed in aggregate, and demonstrated interactively.

## Pipeline

1. **Generation** (`conversation_generation/`) — produces multi-turn `threat_conversations.json` and `benign_conversations.json` plus a generation metadata workbook, varying prompted Cialdini emphasis against the type and amount of benign context.
2. **Detection** (`detection_pipeline/`) — runs each conversation turn-by-turn through two detectors (social-engineering on caller turns, policy-violation on representative turns), each under three stances (high precision, high recall, balanced). Writes per-turn predictions and timing to `detection_results/`.
3. **Analysis** (`analysis/` + `main_analysis.ipynb`) — computes per-conversation summaries, ROC/AUC across multiple slicings, confusion matrices, and Cialdini emphasis × signal heatmaps. Writes figures and tables to `analysis_output/`.
4. **Demo** (`web_demo/`) — a Gradio app that replays an existing conversation with live detection annotations (static mode) or generates and scores a new one end-to-end (dynamic mode).

## Folder Structure

```
.
├── README.md
├── .env.example
├── pyproject.toml
├── conversation_generation/        # Stage 1 code (.py, .ipynb, .json)
├── conversations/                  # Stage 1 outputs (Stage 2 inputs)
├── detection_pipeline/             # Stage 2 code (.py, .ipynb, .json)
├── detection_results/              # Stage 2 outputs (Stage 3 inputs)
├── analysis/                       # Stage 3 code (package)
├── main_analysis.ipynb             # Stage 3 orchestration
├── analysis_output/                # Stage 3 outputs (PNG, SVG, xlsx)
└── web_demo/                       # Stage 4 code
```

## Modules

### Conversation generation

Generates simulated multi-turn customer-service conversations from a Cartesian product of prompt dimensions defined in `prompt_dimensions.json`. Each conversation is keyed by `request_id` and stored as an ordered map of turn objects (`turn_0`, `turn_1`, …) with `speaker`, `passage`, and per-turn ground-truth annotations: Cialdini-principle flags on caller turns and improper-disclosure / improper-authentication / improper-account-action flags on representative turns. The structured-output schema enforces these annotations at generation time.

The two prompt templates (`threat`, `benign`) declare which dimensions they consume so that selection enumeration, per-template prompt rendering, and downstream filtering all stay in sync. A batched generation loop with concurrent API calls and periodic atomic flushes writes to the conversations JSON store and a metadata workbook; restart picks up where it left off via the `status` column.

### Detection inference

For each conversation, walks turn-by-turn from turn 1 onward and dispatches each turn to one of two detectors based on speaker — social-engineering for caller turns, policy-violation for representative turns (the turn-0 greeting is excluded). Each detected turn is scored under three stances (`high_precision`, `high_recall`, `balanced`) defined in `detection_prompt_dimensions.json`, yielding parallel per-turn lists per conversation. The detector is asked for a single token; the top-N logprobs are read so the loser's probability is recoverable for ROC analysis. Temperature is 0 and the call is restricted to the `Y` / `N` tokens.

Output mirrors the generation grain (one row per conversation in the metadata workbook). Generation-phase columns are preserved with a `generation_` prefix where their meaning is ambiguous between phases, and per-turn detection lists are stored as JSON-serialized strings. Already-completed conversations are skipped on restart.

### Detection analysis

Loads the threat and benign detection metadata and conversations, joins them into a conversation-level base table (with running totals, first-event turns, pre/at/post-event prediction counts, persistence, concordance across stances) and a long-format turn table. Short-label columns are derived for the longer prompt dimensions so plots remain readable. The `analysis/` package separates concerns — `loading`, `preliminaries`, `query` (ibis), `roc`, `figures`, `schema` — and `main_analysis.ipynb` is a thin orchestration layer.

Reports include latency percentiles (Q1, median, Q3, p90, p99, p99.9), confusion matrices aligned with each ROC scenario, requested histograms, a heatmap of prompted Cialdini emphasis against actual Cialdini signals (canonical ordering on both axes), and a Sankey of conversation flow. ROC analysis runs across four scenarios — overall, sliced by Cialdini emphasis, sliced by amount of benign context, and as a Cialdini × benign-context AUC heatmap — with AUC reported to three decimal places. All figures are written to `analysis_output/` in PNG and SVG, with summary tables in xlsx, in the official UNC Charlotte palette.

### Demo web app

A Gradio app (run locally via `uv run python web_demo/app.py`) that demonstrates the pipeline interactively in two modes. **Static mode** loads an existing conversation from `conversations/` and replays it turn-by-turn with the previously computed detection annotations rendered alongside each message. **Dynamic mode** lets the user pick a prompt template and dimensions, generates a new conversation live, and runs detection after every turn with the same incremental contract as the offline pipeline; conversations and results are held in memory only.

The app is a thin UI layer — generation and detection logic are imported from the existing modules through `adapters.py`, so prompt-rendering and API-call behavior stay in lockstep with the offline pipeline. Dropdowns are populated from `prompt_dimensions.json` and filtered per template, so threat-only and benign-only options are shown only when relevant. Generation uses the project's standard generation model; detection uses the standard detection model. Styling lives in `style.py`.

## Setup

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency management and requires Python 3.13+. If you don't already have `uv` installed, follow the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) for your platform.

### 1. Clone the repository

```bash
git clone <repository-url> social-engineering-detection
cd social-engineering-detection
```

### 2. Install dependencies

```bash
uv sync
```

This creates a virtual environment in `.venv/` and installs everything declared in `pyproject.toml`.

### 3. Configure your OpenAI API key

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY=sk-...` to your actual key. The `.env` file is read automatically via `python-dotenv` whenever any pipeline stage or the demo makes an API call. Don't commit `.env` — it should be in `.gitignore`.

## Running the demo

From the project root:

```bash
uv run python web_demo/app.py
```

The console will print the project paths it resolved and the number of records loaded for static mode, then launch a local server (default `http://127.0.0.1:7860`). Open that URL in a browser.

- **Static mode** works as long as `conversations/` and `detection_results/` are populated (this is the default with the data shipped in the repo).
- **Dynamic mode** requires a working `OPENAI_API_KEY` because it makes live generation and detection calls.

To stop the server, press `Ctrl+C` in the terminal.

## Running the offline pipeline (optional)

If you want to regenerate the dataset from scratch rather than using the included outputs, run the three notebooks in order, each via `uv run`:

```bash
uv run jupyter notebook conversation_generation/conversation_generation.ipynb
uv run jupyter notebook detection_pipeline/detection_inference.ipynb
uv run jupyter notebook main_analysis.ipynb
```

Each stage reads from the previous stage's output folder, so the order matters. Generation and detection are both restartable — interrupted runs pick up where they left off.
