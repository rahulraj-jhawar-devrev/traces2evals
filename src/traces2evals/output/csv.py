import csv
from pathlib import Path

from traces2evals.models.eval_item import EvalItem


def write_csv(evals: list[EvalItem], output_path: Path) -> None:
    """Write evals as CSV (user_message, reference_output, remarks)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_message", "reference_output", "remarks"])
        for item in evals:
            writer.writerow([item.user_message, item.reference_response, item.judge_remarks])
    print(f"  Written {len(evals)} evals to {output_path}")
