# traces2evals

Generate diverse eval test suites from production agent traces.

`traces2evals` ingests conversation sessions from observability platforms, clusters them for behavioral diversity using the COMPASS algorithm, and outputs structured eval datasets ready for your eval runner.

## Why

You're logging agent conversations in Langfuse (or similar). You want evals that reflect what your users actually ask — not synthetic questions from docs. This tool bridges that gap automatically:

1. **Ingest** sessions from your observability platform
2. **Quality-gate** to filter noise and failed interactions
3. **Extract facets** — privatized behavioral summaries of each conversation
4. **Embed + Cluster** using COMPASS (UMAP → HDBSCAN → merge → prune → hierarchy)
5. **Label** clusters with human-readable titles
6. **Generate evals** — diverse test cases with judge remarks for automated scoring

## Install

```bash
pip install traces2evals
```

Or from source:

```bash
git clone https://github.com/rahulraj-jhawar-devrev/traces2evals.git
cd traces2evals
pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Copy and configure
cp traces2evals.yaml.example traces2evals.yaml
# Edit traces2evals.yaml with your Langfuse credentials and LLM API key

# 2. Run the pipeline
traces2evals run --config traces2evals.yaml

# Or with inline credentials
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
traces2evals run --config traces2evals.yaml --model gpt-4o-mini
```

## CLI

```bash
traces2evals run       # Full pipeline end-to-end
traces2evals status    # Show pipeline progress and cached state
traces2evals clean     # Remove all cached state
```

All commands accept `--config <path>` (defaults to `traces2evals.yaml` in working dir).

CLI flags override YAML config:

```bash
traces2evals run \
  --langfuse-host "https://cloud.langfuse.com" \
  --langfuse-public-key "pk-..." \
  --langfuse-secret-key "sk-..." \
  --model "anthropic/claude-sonnet-4-20250514" \
  --output-format braintrust
```

## Output Formats

| Format | Flag | Description |
|--------|------|-------------|
| JSONL | `--output-format jsonl` | OpenAI evals compatible (`input`, `ideal`, `metadata`) |
| Braintrust | `--output-format braintrust` | Braintrust dataset format (`input`, `expected`, `metadata`) |
| CSV | `--output-format csv` | Simple spreadsheet (`user_message`, `reference_output`, `remarks`) |

### Example output (JSONL)

```json
{
  "input": "How do I set up automated notifications when a ticket SLA is about to breach?",
  "ideal": "You can set up SLA breach notifications by configuring alert rules...",
  "metadata": {
    "cluster_id": 42,
    "cluster_title": "SLA Configuration and Monitoring",
    "judge_remarks": "Tests configuring time-based alerting rules. The assistant should explain the setup flow including thresholds, notification channels, and testing.",
    "source_session_id": "sess_abc123",
    "synthesis_type": "synthesized",
    "capability_tested": "workflow automation configuration"
  }
}
```

## Configuration

See [`traces2evals.yaml.example`](traces2evals.yaml.example) for all options with comments.

Key sections:

- **source** — Which platform to ingest from and connection details
- **llm** — Model for all LLM calls (any [litellm](https://docs.litellm.ai/docs/providers) model string)
- **quality** — Score thresholds and fallback behavior
- **clustering** — COMPASS hyperparameters (defaults work well for 500-10k sessions)
- **eval_generation** — How many evals per cluster, synthesis behavior
- **output** — Format and directory
- **checkpoint** — Resume support (enabled by default)

## How It Works

### COMPASS Clustering

The core algorithm that ensures behavioral diversity:

1. **UMAP** reduces high-dimensional embeddings to 20 dimensions
2. **HDBSCAN** discovers natural clusters without requiring a pre-set count
3. **Merge** combines clusters with >0.83 cosine similarity
4. **Prune** removes clusters below minimum size thresholds
5. **Reassign noise** points to nearest cluster centroid
6. **Ward linkage hierarchy** groups clusters into navigable levels

### Checkpointing

Every expensive step is cached. If the pipeline crashes or you add more sessions, it resumes from where it left off. Change your embedding model? Only downstream steps re-run.

```
.traces2evals_cache/
├── manifest.json       # Session progress tracking
├── raw_sessions.jsonl  # Fetched sessions
├── quality_scores.json # Quality gate results
├── facets.parquet      # Extracted facets
├── embeddings.npy      # Vector embeddings
├── cluster_state.json  # Cluster assignments + labels
└── meta.json           # Config hash for invalidation
```

## Adding Adapters

`traces2evals` is designed to support any observability platform. To add one:

```python
from traces2evals.adapters.base import BaseAdapter
from traces2evals.models.session import NormalizedSession, SessionScore

class MyPlatformAdapter(BaseAdapter):
    def fetch_sessions(self, from_date, to_date, max_sessions):
        # Yield NormalizedSession objects
        ...

    def get_session_scores(self, session_id: str) -> list[SessionScore]:
        # Return any platform-side quality scores
        ...

    def health_check(self) -> bool:
        # Verify connection
        ...
```

Currently supported: **Langfuse**

## Claude Code Integration

`traces2evals` works as a Claude Code skill. Add the skill file to your Claude Code skills directory and invoke it with `/traces2evals` to run the pipeline conversationally.

## Requirements

- Python 3.11+
- An LLM API key (OpenAI, Anthropic, or any litellm-supported provider)
- A Langfuse instance with logged agent sessions

## License

MIT
