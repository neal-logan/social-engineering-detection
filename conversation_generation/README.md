# Conversation Generation

Stage 1 of the social-engineering detection project. Generates synthetic
voice-call conversations — both **threat** (social engineering attacks) and
**benign** (legitimate customer-service calls) — sharing a single output
schema so downstream detection systems can be evaluated for both true
positives and false positives.

Threat and benign generation are separate runs with separate settings and
separate output files.

## Directory layout

```
project_root/
├── conversation_generation/     <- code + prompt dimensions
│   ├── conversation_generation.py
│   ├── prompt_dimensions.json
│   ├── run_generation.ipynb     <- main pipeline (threat + benign runs)
│   ├── dev.ipynb                <- single-prompt / single-call testing
│   ├── repair.ipynb             <- back-fill metric columns into existing runs
│   ├── reconstruct.ipynb        <- rebuild metadata file from conversations JSON
│   ├── requirements.txt
│   ├── README.md
│   └── .env                     <- OPENAI_API_KEY=sk-...  (you create this)
└── conversations/
    ├── threat_metadata.xlsx
    ├── threat_conversations.json
    ├── benign_metadata.xlsx
    └── benign_conversations.json
```

## Setup

```bash
cd conversation_generation
uv sync   # or: pip install -r requirements.txt
echo 'OPENAI_API_KEY=sk-...' > .env
uv run jupyter notebook run_generation.ipynb
```

## Output JSON shape

The model emits a single JSON object with one top-level field per turn
position, named `turn_0` … `turn_{N-1}`:

- `turn_0` — representative greeting; only field is `passage`.
- Odd turns (`turn_1`, `turn_3`, …) — caller turns; have `passage` plus 7
  Cialdini boolean flags (`cialdini_reciprocity`, `cialdini_commitment_consistency`,
  `cialdini_social_proof`, `cialdini_authority`, `cialdini_liking`,
  `cialdini_scarcity`, `cialdini_unity`).
- Even turns ≥ 2 — representative response turns; have `passage` plus 3
  improper-action flags (`improper_authentication`, `improper_disclosure`,
  `improper_action`).

This shape is enforced via OpenAI's structured outputs against a Pydantic
model that's built dynamically based on the request's turn count (see
`conversation_model_for(turn_count)`). Each unique turn count produces its
own cached model class.

## Retries

Each request is retried up to `MAX_ATTEMPTS` (default 10) on any exception —
network errors, refusals, or schema-validation failures. Backoff schedule is
exponential capped at 60s: 2, 4, 8, 16, 32, 60, 60, 60, 60. The metadata
`attempts` column shows total attempts used for each row; if all retries
fail, the row is marked `error` with `last_error` populated.

## Prompt dimensions

All variable prompt inputs live in `prompt_dimensions.json`:

- **prompt_templates** — `threat` and `benign`, each with an inline
  `template` string and a `dimensions` map declaring which dimensions it
  consumes.
- **scenarios** — currently 1 (credit union).
- **representatives** — 3 (overly_helpful, tired_but_professional, by_the_book).
- **threat_caller_profile** — 1 (`opportunistic_attacker`, who pursues
  whichever objective the conversation opens up).
- **benign_caller_profile** — 8 (routine balance check, confused first-time,
  frustrated legitimate, chatty regular, household proxy, automated question,
  recently widowed, expat abroad).
- **benign_context_levels** — 3 (minimal, moderate, heavy).
- **cialdini_emphasis** — all 7 Cialdini principles.
- **turn_counts** — 1 (`standard` = 14).
- **flavors** — 32 entries, a flat list of evocative descriptions (sailing
  and boats, school pickup runs, leaking dishwashers, etc.).

## Flavor as a sampled axis

For each run, flavor is treated as an additional Cartesian axis whose
composition is governed by a strategy:

- **`FLAVOR_DETERMINISTIC`** — one subset of `flavor_count` flavors is
  sampled once (using `flavor_seed`) and reused for every dimension
  combination. Default for the threat run, with `flavor_count = 32`
  (the entire pool).
- **`FLAVOR_RESAMPLED`** — a fresh subset of `flavor_count` flavors is
  drawn for each combination. Default for the benign run, with
  `flavor_count = 5`.

In both cases the flavor sample is reproducible via `flavor_seed`.

## Default enumeration size

With the defaults in the notebook (`replicates = 1`):

- **Threat:** 1 scenario × 3 reps × 1 caller × 3 benign × 7 cialdini ×
  1 turn × 32 flavors = **2,016**
- **Benign:** 1 × 3 × 8 × 3 × 7 × 1 × 5 = **2,520**

## Metadata file

Each run's `*_metadata.xlsx` file holds one row per request, with all
dimension *values as text* (no raw keys), the flavor text, API parameters,
status, attempts used, token usage, and the full system and user prompt for
each row. After a successful generation, the row also carries per-flag
metrics extracted from the conversation:

- `{flag}_sum` — integer count of turns where the flag was True. Cialdini
  flags are counted only on caller turns; improper flags only on
  representative response turns.
- `{flag}_by_turn` — comma-separated string of length `turn_count_value`,
  with `1` at positions where the flag fired and `0` everywhere else
  (including non-applicable turns). E.g. for a 14-turn conversation:
  `cialdini_unity_by_turn = "0,1,0,1,0,0,0,1,0,0,0,0,0,1"`.

There are 10 flag pairs total: 7 cialdini flags × 2 + 3 improper flags × 2 =
20 metric columns.

## Resumability and durability

`run_generation_loop` skips `success` rows and retries `error` rows. Safe to
interrupt and re-run.

The loop runs requests in batches of `BATCH_SIZE` (default 5) using a thread
pool, so 5 OpenAI calls overlap rather than running serially. Each call
still has its own retry loop, so a single slow or failing call doesn't
block the others.

Successful conversations and metadata updates are buffered in memory and
flushed to disk every `FLUSH_EVERY` requests (default 20, rounded up to
the next batch boundary). This reduces per-request disk writes by ~20×
on long runs.

Both files are written atomically (write to a sibling temp file, then
`os.replace`), so an interrupt or crash during a save cannot leave the
file half-written. On unexpected exit (including KeyboardInterrupt), a
`finally` block flushes any buffered work that was already in memory
before the loop unwinds. Anything that was in flight at interrupt
remains in `pending` status and will be regenerated on the next run.

## Repair notebook

`repair.ipynb` back-fills the per-flag metric columns into an existing
metadata file by walking the conversations JSON. Use it when:

- A run was started before the metric columns existed, or
- The metadata file got out of sync with its conversations JSON for any
  reason.

The notebook adds any missing schema columns to the workbook (via
`cg.ensure_metadata_columns`) and then re-runs `cg.extract_flag_metrics` on
each successful conversation, writing the values back. Pending and errored
rows are left untouched. Both the live pipeline and the repair script call
the same extraction function, so values are guaranteed consistent.

## Reconstruct notebook

`reconstruct.ipynb` rebuilds the metadata xlsx from scratch using only the
conversations JSON. Use it when the metadata file has been lost or
corrupted but the conversations JSON is still intact.

The notebook (1) validates every conversation in the JSON against the
Pydantic schema for its turn count, deleting any malformed records and
reporting them; (2) re-enumerates the request grid using the same
`prompt_dimensions.json` and the same flavor seed/count/strategy that
were used originally — re-enumeration is deterministic given those
inputs; (3) matches surviving conversations to enumerated rows by content
tuple (template, scenario, representative, caller, benign_context,
cialdini_emphasis, turn_count_value, flavor, replicate_index); (4) fills
in tokens, prompts, metrics, status, and timestamp from each matching
conversation, leaving unmatched rows in `pending` so they can be regenerated.

Caveats: per-row retry counts cannot be recovered (the JSON doesn't
record them; all reconstructed successes are marked `attempts = 1`).
Conversations whose content tuple doesn't match any enumerated row are
reported as orphaned and left in the JSON untouched — this typically means
the dimensions or flavor config has changed since the original run.

## Dev notebook

`dev.ipynb` lets you (1) render a single system prompt from explicit
dimension keys, and (2) make a single API call using the same
`call_openai_with_retries` the main pipeline uses, without writing any
files. Useful for tweaking prompt wording before running a full enumeration.

## Notes

- Default model is `gpt-5.4`. GPT-5 family currently only allows
  `temperature=1` and `top_p=1`; the code omits these parameters from the
  API call unless overridden.
- `max_tokens` is intentionally not set — OpenAI API default applies.
