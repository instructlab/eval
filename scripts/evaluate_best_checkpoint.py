#!/usr/bin/env python3

"""
Example usage:
python scripts/evaluate_best_checkpoint.py \
    /path/to/checkpoint_dir \
    --output-file /path/to/output_file
"""

# Standard
from pathlib import Path
from typing import Optional
import json

# Third Party
import typer

app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Argument(..., help="Input directory to process"),
    output_file: Optional[Path] = typer.Option(None, help="Optional output file path"),
):
    """
    Process files in the input directory and optionally save results to an output file.
    """
    if not input_dir.exists():
        typer.echo(f"Error: Input directory '{input_dir}' does not exist")
        raise typer.Exit(1)

    if not input_dir.is_dir():
        typer.echo(f"Error: '{input_dir}' is not a directory")
        raise typer.Exit(1)

    checkpoint_dirs = list(input_dir.glob("hf_format/samples_*"))
    typer.echo(f"Found {len(checkpoint_dirs)} samples files")

    if not checkpoint_dirs:
        typer.echo(
            f"No checkpoint directories found in the input directory: {input_dir}"
        )
        raise typer.Exit(1)

    typer.echo("importing LeaderboardV2Evaluator, this may take a while...")
    # First Party
    from instructlab.eval.leaderboard import LeaderboardV2Evaluator

    checkpoint_results = {}
    for checkpoint in checkpoint_dirs:
        typer.echo(f"Processing checkpoint: {checkpoint}")
        ckpt_output_file = checkpoint / "leaderboard_results.json"
        evaluator = LeaderboardV2Evaluator(
            model_path=str(checkpoint), output_file=ckpt_output_file, num_gpus=8
        )
        result = evaluator.run()
        checkpoint_results[checkpoint.name] = result
        typer.echo(f"Checkpoint {checkpoint.name} results: {result['overall_score']}")

    # Sort checkpoints by score
    sorted_checkpoints = sorted(
        checkpoint_results.items(), key=lambda x: x[1]["overall_score"], reverse=True
    )
    typer.echo("Sorted checkpoints by score:")
    for checkpoint_name, result in sorted_checkpoints:
        typer.echo(f"{'=' * 100}")
        typer.echo(json.dumps(result, indent=2))

    typer.echo(f"{'=' * 100}")
    typer.echo(f"Best checkpoint: {sorted_checkpoints[0][0]}")

    if output_file:
        typer.echo(f"Output will be saved to: {output_file}")
        with open(output_file, "w") as f:
            json.dump(checkpoint_results, f, indent=2)

    # Add your processing logic here

    typer.echo("Processing complete!")


if __name__ == "__main__":
    app()
