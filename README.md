# munqib

Minimal agent for pretraining data prep — FineWeb-Edu / ClimbX style workflows. Two implementations:

- **`nanocode` (Rust)** — compiled binary, zero runtime dependencies
- **`core.py` (Python)** — single-file version, stdlib only (except `score`)

Both run the same agentic loop: send user messages to an LLM, execute tool calls, feed results back, repeat. Pair with `dataprep.py` to orchestrate filter → dedup → score → sample pipelines from natural language.

# add picture later

## Quick start

```bash
# Python (no build step)
export ANTHROPIC_API_KEY="your-key"
python3 core.py

# Rust (faster startup, no Python needed)
cargo build --release
./target/release/nanocode
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

Pretraining data pipeline — importable as a library or run from the CLI. When `dataprep.py` is present in the working directory, the agent's system prompt advertises it automatically so the LLM can orchestrate full pipelines via `bash`.

```
python3 dataprep.py filter  [--input FILE] [--output FILE] [--min-len 100] [--max-symbol-ratio 0.1] [--max-rep-ratio 0.3]
python3 dataprep.py dedup   [--input FILE] [--output FILE] [--threshold 0.85] [--num-hashes 128] [--ngram-size 5] [--seed 42]
python3 dataprep.py score   [--input FILE] [--output FILE] [--gold FILE] [--min-score 0.5]
python3 dataprep.py sample  [--sources src1.jsonl:0.4,src2.jsonl:0.6] [--n 10000] [--output FILE] [--seed 42]
python3 dataprep.py stats   [--input FILE] [--field text]
```

Input/output default to stdin/stdout when omitted.

| Subcommand | What it does | Dependencies |
|------------|-------------|--------------|
| `filter` | Heuristic quality filter: length, symbol ratio, line repetition | stdlib |
| `dedup` | MinHash LSH near-deduplication | stdlib |
| `score` | TF-IDF + logistic regression quality scoring against a gold set | scikit-learn |
| `sample` | Proportional mixture sampling from multiple sources | stdlib |
| `stats` | Doc count, chars, tokens, length distribution | stdlib |

### Example pipeline

```bash
# 1. Filter low-quality docs
python3 dataprep.py filter --input raw.jsonl --output filtered.jsonl

# 2. Deduplicate
python3 dataprep.py dedup --input filtered.jsonl --output deduped.jsonl --threshold 0.8

# 3. Score against a reference set and keep top docs
python3 dataprep.py score --input deduped.jsonl --gold gold.jsonl --min-score 0.6 --output scored.jsonl

# 4. Mix sources for final training set
python3 dataprep.py sample --sources scored.jsonl:0.7,extra.jsonl:0.3 --n 100000 --output train.jsonl
```

Or just ask the agent in plain English:

```
❯ filter raw.jsonl, dedup at 0.8, then sample 50k docs with 70/30 mix from scored and extra
```

## License

MIT
