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
from rich import print
from typing_extensions import Annotated
import typer

app = typer.Typer()


def print_metrics(result: dict, checkpoint_name: str = None, prefix: str = ""):
    """
    Print formatted metrics for a checkpoint result.

    Args:
        result: The evaluation result dictionary
        checkpoint_name: Optional checkpoint name to display
        prefix: Optional prefix for each line
    """
    if checkpoint_name:
        print(f"{prefix}[bold]Leaderboard results[/bold]: {checkpoint_name}")
    print(f"{prefix}Overall: {result['overall_score'] * 100:.2f}%")
    if "leaderboard_bbh" in result:
        print(f"{prefix}BBH: {result['leaderboard_bbh']['score'] * 100:.2f}%")
    if "leaderboard_gpqa" in result:
        print(f"{prefix}GPQA: {result['leaderboard_gpqa']['score'] * 100:.2f}%")
    if "leaderboard_ifeval" in result:
        print(f"{prefix}IFEval: {result['leaderboard_ifeval']['score'] * 100:.2f}%")
    if "leaderboard_math_hard" in result:
        print(
            f"{prefix}MATH-Hard: {result['leaderboard_math_hard']['score'] * 100:.2f}%"
        )
    if "leaderboard_mmlu_pro" in result:
        print(f"{prefix}MMLU-Pro: {result['leaderboard_mmlu_pro']['score'] * 100:.2f}%")
    if "leaderboard_musr" in result:
        print(f"{prefix}MUSR: {result['leaderboard_musr']['score'] * 100:.2f}%")


@app.command()
def best_checkpoint(
    input_dir: Path = typer.Argument(..., help="Input directory to process"),
    output_file: Optional[Path] = typer.Option(None, help="Optional output file path"),
    tasks: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Specific tasks to evaluate (e.g., 'leaderboard_bbh', 'leaderboard_gpqa')"
        ),
    ] = None,
    num_gpus: int = typer.Option(8, help="Number of GPUs to use for evaluation"),
):
    """
    Find the best checkpoint by evaluating all checkpoints in the input directory.
    Processes all checkpoint subdirectories and ranks them by overall score.
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
            model_path=str(checkpoint), output_file=ckpt_output_file, num_gpus=num_gpus
        )
        if tasks:
            evaluator.tasks = tasks
        result = evaluator.run()
        checkpoint_results[checkpoint.name] = result
        typer.echo(f"Checkpoint {checkpoint.name} results: {result['overall_score']}")

    # Sort checkpoints by score
    sorted_checkpoints = sorted(
        checkpoint_results.items(), key=lambda x: x[1]["overall_score"], reverse=True
    )
    typer.echo("Sorted checkpoints by score:")
    for i, (checkpoint_name, result) in enumerate(sorted_checkpoints):
        typer.echo(f"{'=' * 100}")
        # Add [BEST CHECKPOINT] label for the first checkpoint
        if i == 0:
            checkpoint_display = (
                f"{checkpoint_name} [bold green][BEST CHECKPOINT][/bold green]"
            )
        else:
            checkpoint_display = checkpoint_name
        print_metrics(result, checkpoint_display)

    typer.echo(f"{'=' * 100}")
    typer.echo(
        f"Best checkpoint: {sorted_checkpoints[0][0]} [bold green][BEST CHECKPOINT][/bold green]"
    )

    if output_file:
        typer.echo(f"Output will be saved to: {output_file}")
        with open(output_file, "w") as f:
            json.dump(checkpoint_results, f, indent=2)

    # Add your processing logic here

    typer.echo("Processing complete!")


@app.command()
def evaluate(
    input_dir: Path = typer.Argument(..., help="Input directory to process"),
    tasks: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Specific tasks to evaluate (e.g., 'leaderboard_bbh', 'leaderboard_gpqa')"
        ),
    ] = None,
    num_gpus: int = typer.Option(8, help="Number of GPUs to use for evaluation"),
    output_file: Optional[Path] = typer.Option(
        None,
        help="Custom output file path (default: input_dir/leaderboard_results.json)",
    ),
):
    """
    Evaluate a single checkpoint directory and save results to JSON file.
    """
    if not input_dir.exists():
        typer.echo(f"Error: Input directory '{input_dir}' does not exist")
        raise typer.Exit(1)

    if not input_dir.is_dir():
        typer.echo(f"Error: '{input_dir}' is not a directory")
        raise typer.Exit(1)

    typer.echo("importing LeaderboardV2Evaluator, this may take a while...")
    # First Party
    from instructlab.eval.leaderboard import LeaderboardV2Evaluator

    typer.echo("done")

    evaluator = LeaderboardV2Evaluator(
        model_path=str(input_dir), num_gpus=num_gpus, eval_config={"batch_size": "auto"}
    )
    if tasks:
        evaluator.tasks = tasks
    result = evaluator.run()

    # now just print out the checkpoint results
    print_metrics(result, str(input_dir))

    # Determine output file path
    if output_file is None:
        output_file = input_dir / "leaderboard_results.json"

    # Check if file exists and warn user
    if output_file.exists():
        typer.echo(
            f"Warning: Output file '{output_file}' already exists and will be overwritten"
        )

    output_file.write_text(json.dumps(result, indent=2))
    typer.echo(f"Results saved to: {output_file}")


@app.command()
def find_best(
    input_dir: Path = typer.Argument(..., help="Input directory to process"),
    show_all: bool = typer.Option(
        False, "--show-all", help="Show scores for all checkpoints"
    ),
):
    """
    Find the best checkpoint by looking through leaderboard_results.json files.
    """
    if not input_dir.exists():
        typer.echo(f"Error: Input directory '{input_dir}' does not exist")
        raise typer.Exit(1)

    if not input_dir.is_dir():
        typer.echo(f"Error: '{input_dir}' is not a directory")
        raise typer.Exit(1)

    # Find all leaderboard_results.json files
    result_files = list(input_dir.glob("**/leaderboard_results.json"))

    if not result_files:
        typer.echo("No leaderboard results found in any subdirectories")
        raise typer.Exit(1)

    # Load and compare results
    best_score = -1
    best_checkpoint = None
    best_results = None
    all_results = []

    for result_file in result_files:
        try:
            results = json.loads(result_file.read_text())
            score = results.get("overall_score", -1)
            all_results.append((result_file.parent, score, results))

            if score > best_score:
                best_score = score
                best_checkpoint = result_file.parent
                best_results = results
        except Exception as e:
            typer.echo(f"Error reading {result_file}: {e}")
            continue

    if best_checkpoint is None:
        typer.echo("No valid results found")
        raise typer.Exit(1)

    # Sort all results by score
    all_results.sort(key=lambda x: x[1], reverse=True)

    # Print all results if requested
    if show_all:
        print("\n[bold]All checkpoint results:[/bold]")
        for checkpoint, score, results in all_results:
            is_best = checkpoint == best_checkpoint
            prefix = "→ " if is_best else "  "
            print(f"\n{prefix}Checkpoint: {checkpoint}")
            print_metrics(results, prefix="  ")
    else:
        # Print only best results
        print(f"\n[bold]Best checkpoint found[/bold]: {best_checkpoint}")
        print_metrics(best_results)


if __name__ == "__main__":
    app()
