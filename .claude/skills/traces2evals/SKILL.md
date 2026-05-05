# /traces2evals — Generate Evals from Production Traces

## What this does

Runs the traces2evals pipeline: ingests agent sessions from Langfuse, clusters them for behavioral diversity, and outputs structured eval test cases.

## When to use

When the user wants to generate an eval suite from their production agent conversations. They should have:
- A Langfuse instance with logged sessions
- An LLM API key (OpenAI, Anthropic, or any litellm provider)
- A `traces2evals.yaml` config file (or willingness to provide credentials inline)

## Steps

1. Check if `traces2evals.yaml` exists in the working directory. If not, ask the user for:
   - Langfuse host, public key, secret key
   - Which LLM to use (default: gpt-4o-mini)
   - Output format preference (jsonl/braintrust/csv)
   - Date range for sessions (default: last 30 days)

2. Create or validate the config file.

3. Run the pipeline:
   ```bash
   traces2evals run --config traces2evals.yaml
   ```

4. Report results: number of sessions ingested, clusters found, and evals generated.

## Configuration

The pipeline is fully configurable via `traces2evals.yaml`. Key options:
- `source.langfuse.*` — Connection details (supports `${ENV_VAR}` interpolation)
- `llm.model` — Any litellm model string
- `clustering.auto_tune` — Set `true` for Optuna hyperparameter optimization
- `eval_generation.max_evals_per_cluster` — Controls output volume
- `output.format` — jsonl, braintrust, or csv

## Resumability

The pipeline checkpoints after every expensive step. If interrupted, re-running picks up where it left off. Use `traces2evals clean` to start fresh.

## Output

Evals are written to the configured output directory (default: `./output/`). Each eval contains:
- `input` — The user query (privatized, self-contained)
- `ideal`/`expected` — Reference response
- `metadata` — Cluster info, judge remarks, capability tested
