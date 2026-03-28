# munqib

Minimal agent for pretraining data prep, plus a config-driven text curation CLI for FineWeb-Edu and Nemotron-ClimbMix workflows. Two interactive implementations:

- **`munqib` (Rust)** — compiled binary, zero runtime dependencies
- **`core.py` (Python)** — single-file version, stdlib only (except `score`)

Both run the same agentic loop: send user messages to an LLM, execute tool calls, feed results back, repeat. Pair with `dataprep.py` to orchestrate canonical dataset builds and config-driven curation pipelines from natural language.

# add picture later

## Quick start

```bash
# Python (no build step)
export ANTHROPIC_API_KEY="your-key"
python3 core.py

# Rust (faster startup, no Python needed)
cargo build --release
./target/release/munqib
```

## Providers

The first matching environment variable is used:

| Variable | Provider |
|----------|----------|
| `OPENROUTER_API_KEY` | OpenRouter (access any model) |
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENAI_API_KEY` | OpenAI |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini |

Override the model:

```bash
export MODEL="claude-opus-4-5"
```

## Commands

| Command | Action |
|---------|--------|
| `/c` | Clear conversation history |
| `/q` or `exit` | Quit |

## Tools

Both implementations expose these tools to the LLM:

| Tool | Description | Python | Rust |
|------|-------------|--------|------|
| `bash` | Run a shell command (stdout+stderr merged) | yes | yes |
| `read` | Read a file with line numbers, optional offset/limit | yes | yes |
| `write` | Write content to a file (overwrites) | yes | yes |
| `edit` | Replace a unique string in a file | yes | — |
| `glob` | Find files by pattern, sorted by mtime | yes | — |
| `grep` | Search files for a regex pattern | yes | — |
| `stats` | Corpus statistics for a JSONL or plain-text file | yes | yes |
| `dataprep_*` | Native dataprep tools: workspace profile, recipes, source inspection, background jobs, artifacts | yes | — |

### `stats` tool

Reports doc count, total chars, estimated tokens, and length distribution. Auto-detects JSONL vs plain text; plain-text splits on `<|endoftext|>` or `\n\n`.

```
docs:    12,345
chars:   45,678,901
tokens:  ~11,419,725  (chars / 4)
lengths: min=100  mean=3,701  p50=2,100  p90=8,200  max=32,100
format:  jsonl  field: text
```

Parameters: `path` (required), `format` (`jsonl`|`text`, auto-detected), `field` (JSON key, default `text`).

## dataprep.py

Pretraining data pipeline — importable as a library or run from the CLI. The primary human-facing paths are:

```
python3 dataprep.py build  --source fineweb-edu|nemotron-climbmix --split train --subset NAME --limit N --row-start 0 --row-count N --files file1.parquet,file2.parquet --output raw.jsonl
python3 dataprep.py curate --config configs/fineweb_climbmix.yaml --audit-output audit.jsonl --final-output train.jsonl
python3 dataprep.py profile get
python3 dataprep.py recipe list
python3 dataprep.py job status --job-id JOB_ID
```

`build` downloads a source dataset and emits canonical JSONL:

```json
{"id":"...","text":"...","source":"fineweb-edu","dataset":"HuggingFaceFW/fineweb-edu","split":"train","subset":"sample-10BT","upstream_id":"...","source_url":"...","metadata":{}}
```

`curate` reads either canonical JSONL or inline source definitions from the config and runs:

1. Text cleaning and normalization
2. Language identification
3. Heuristic quality filtering
4. Exact deduplication
5. Optional fuzzy deduplication
6. Quality, domain, and toxicity classification
7. PII redaction
8. Audit JSONL + final training JSONL export

The audit output keeps classification scores and drop reasons, while the final output keeps the minimal training fields: `id`, `text`, `source`, `dataset`, `lang`.

### Agentic dataprep

The Python agent runtime now exposes native structured dataprep tools instead of relying only on `bash`. That gives the LLM a stable JSON tool surface for:

1. Reading and updating a repo-local workspace profile in `.munqib/profile.yaml`
2. Saving and listing named recipes in `.munqib/recipes/`
3. Inspecting source presets and shard selectors
4. Starting background `build` and `curate` jobs
5. Polling job status, logs, and artifact refs from `.munqib/jobs/` and `.munqib/artifacts/manifest.json`

This avoids repeating dataset defaults and lets the agent refer to previous outputs by artifact refs such as `artifact:fineweb_sample.raw` or `artifact:train_latest.final`.

### Data dependencies

Core local dependencies for `build` and richer configs:

```bash
python3 -m pip install -r requirements-dataprep.txt
```

Optional extras:

- `scikit-learn` for the legacy `score` command
- `fasttext-wheel` plus a `lid.176.bin` model for fastText language ID
- `presidio-analyzer` and `presidio-anonymizer` for a richer PII backend

For NeMo Curator execution, target Linux/Ubuntu rather than macOS:

```bash
# CPU text curation modules
python3 -m pip install nemo-curator

# GPU text curation modules
python3 -m pip install --extra-index-url https://pypi.nvidia.com "nemo-curator[cuda12x]"
```

### Legacy local commands

The original small-data commands are still available:

```
python3 dataprep.py filter  [--input FILE] [--output FILE] [--min-len 100] [--max-symbol-ratio 0.1] [--max-rep-ratio 0.3]
python3 dataprep.py dedup   [--input FILE] [--output FILE] [--threshold 0.85] [--num-hashes 128] [--ngram-size 5] [--seed 42]
python3 dataprep.py score   [--input FILE] [--output FILE] [--gold FILE] [--min-score 0.5]
python3 dataprep.py sample  [--sources src1.jsonl:0.4,src2.jsonl:0.6] [--n 10000] [--output FILE] [--seed 42]
python3 dataprep.py stats   [--input FILE] [--field text]
```

| Subcommand | What it does | Dependencies |
|------------|-------------|--------------|
| `build` | Download FineWeb-Edu or Nemotron-ClimbMix and emit canonical JSONL, with shard/file and row selectors | `datasets`, `tiktoken` for ClimbMix |
| `curate` | Config-driven cleaning, filtering, dedup, classification, PII redaction, audit/final export | stdlib; optional `fasttext`, `presidio`, `nemo-curator` |
| `profile` | Inspect or update repo-local dataprep defaults in `.munqib/profile.yaml` | stdlib |
| `recipe` | Save, show, and list reusable dataprep recipes in `.munqib/recipes/` | stdlib |
| `job` | Inspect background build/curate jobs and resolve artifact refs | stdlib |
| `filter` | Heuristic quality filter: length, symbol ratio, line repetition | stdlib |
| `dedup` | MinHash LSH near-deduplication | stdlib |
| `score` | TF-IDF + logistic regression quality scoring against a gold set | scikit-learn |
| `sample` | Proportional mixture sampling from multiple sources | stdlib |
| `stats` | Doc count, chars, tokens, length distribution | stdlib |

### Example pipeline

```bash
# 1. Save a reusable recipe once
python3 dataprep.py recipe save --name fineweb_sample --file configs/fineweb_climbmix.yaml --set-default

# 2. Launch a background curate job
python3 dataprep.py curate --recipe fineweb_sample --artifact-name train_latest --background

# 3. Poll the job and inspect artifacts
python3 dataprep.py job status --job-id JOB_ID
python3 dataprep.py job artifacts --job-id JOB_ID
```

Example config:

```yaml
sources:
  - path: fineweb.raw.jsonl
language:
  min_score: 0.8
classifiers:
  toxicity:
    enabled: true
    gate: true
    max_score: 0.4
dedup:
  fuzzy:
    enabled: false
```

Or keep using the legacy local pipeline:

```
python3 dataprep.py filter --input raw.jsonl --output filtered.jsonl
python3 dataprep.py dedup --input filtered.jsonl --output deduped.jsonl --threshold 0.8
python3 dataprep.py score --input deduped.jsonl --gold gold.jsonl --min-score 0.6 --output scored.jsonl
python3 dataprep.py sample --sources scored.jsonl:0.7,extra.jsonl:0.3 --n 100000 --output train.jsonl
```

## License

MIT
