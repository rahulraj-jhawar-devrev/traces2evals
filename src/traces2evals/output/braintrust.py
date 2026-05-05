import json
from pathlib import Path

from traces2evals.models.eval_item import EvalItem


def write_braintrust(evals: list[EvalItem], output_path: Path) -> None:
    """Write evals in Braintrust dataset format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in evals:
            row = {
                "input": item.user_message,
                "expected": item.reference_response,
                "metadata": {
                    "cluster_id": item.cluster_id,
                    "cluster_title": item.cluster_title,
                    "judge_remarks": item.judge_remarks,
                    "source_session_id": item.source_session_id,
                    "synthesis_type": item.synthesis_type,
                    "capability_tested": item.capability_tested,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Written {len(evals)} evals to {output_path}")
