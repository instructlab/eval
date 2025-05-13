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
from typing_extensions import Annotated
import json

# Third Party
from rich import print
import typer

app = typer.Typer()


@app.command()
def best_checkpoint(
    input_dir: Path = typer.Argument(..., help="Input directory to process"),
    output_file: Optional[Path] = typer.Option(None, help="Optional output file path"),
    tasks: Annotated[Optional[list[str]], typer.Option()] = None,
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
            typer.echo(
                f"[bold]Leaderboard results[/bold]: {checkpoint_name} [bold green][BEST CHECKPOINT][/bold green]"
            )
        else:
            typer.echo(f"[bold]Leaderboard results[/bold]: {checkpoint_name}")
        typer.echo(f"Overall: {result['overall_score'] * 100:.2f}%")
        if "leaderboard_bbh" in result:
            typer.echo(f"BBH: {result['leaderboard_bbh']['score'] * 100:.2f}%")
        if "leaderboard_gpqa" in result:
            typer.echo(f"GPQA: {result['leaderboard_gpqa']['score'] * 100:.2f}%")
        if "leaderboard_ifeval" in result:
            typer.echo(f"IFEval: {result['leaderboard_ifeval']['score'] * 100:.2f}%")
        if "leaderboard_math_hard" in result:
            typer.echo(
                f"MATH-Hard: {result['leaderboard_math_hard']['score'] * 100:.2f}%"
            )
        if "leaderboard_mmlu_pro" in result:
            typer.echo(
                f"MMLU-Pro: {result['leaderboard_mmlu_pro']['score'] * 100:.2f}%"
            )
        if "leaderboard_musr" in result:
            typer.echo(f"MUSR: {result['leaderboard_musr']['score'] * 100:.2f}%")

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
    tasks: Annotated[Optional[list[str]], typer.Option()] = None,
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

    typer.echo("importing LeaderboardV2Evaluator, this may take a while...")
    # First Party
    from instructlab.eval.leaderboard import LeaderboardV2Evaluator

    typer.echo("done")

    evaluator = LeaderboardV2Evaluator(
        model_path=str(input_dir), num_gpus=8, eval_config={"batch_size": "auto"}
    )
    if tasks:
        evaluator.tasks = tasks
    result = evaluator.run()

    # now just print out the checkpoint results
    print(f"[bold]Leaderboard results[/bold]: {input_dir}")
    print(f"Overall: {result['overall_score'] * 100:.2f}%")
    if "leaderboard_bbh" in result:
        print(f"BBH: {result['leaderboard_bbh']['score'] * 100:.2f}%")
    if "leaderboard_gpqa" in result:
        print(f"GPQA: {result['leaderboard_gpqa']['score'] * 100:.2f}%")
    if "leaderboard_ifeval" in result:
        print(f"IFEval: {result['leaderboard_ifeval']['score'] * 100:.2f}%")
    if "leaderboard_math_hard" in result:
        print(f"MATH-Hard: {result['leaderboard_math_hard']['score'] * 100:.2f}%")
    if "leaderboard_mmlu_pro" in result:
        print(f"MMLU-Pro: {result['leaderboard_mmlu_pro']['score'] * 100:.2f}%")
    if "leaderboard_musr" in result:
        print(f"MUSR: {result['leaderboard_musr']['score'] * 100:.2f}%")

    output_file = input_dir / "leaderboard_results.json"
    output_file.write_text(json.dumps(result, indent=2))


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
            print(f"  Overall score: {score * 100:.2f}%")
            if "leaderboard_bbh" in results:
                print(f"  BBH: {results['leaderboard_bbh']['score'] * 100:.2f}%")
            if "leaderboard_gpqa" in results:
                print(f"  GPQA: {results['leaderboard_gpqa']['score'] * 100:.2f}%")
            if "leaderboard_ifeval" in results:
                print(f"  IFEval: {results['leaderboard_ifeval']['score'] * 100:.2f}%")
            if "leaderboard_math_hard" in results:
                print(
                    f"  MATH-Hard: {results['leaderboard_math_hard']['score'] * 100:.2f}%"
                )
            if "leaderboard_mmlu_pro" in results:
                print(
                    f"  MMLU-Pro: {results['leaderboard_mmlu_pro']['score'] * 100:.2f}%"
                )
            if "leaderboard_musr" in results:
                print(f"  MUSR: {results['leaderboard_musr']['score'] * 100:.2f}%")
    else:
        # Print only best results
        print(f"\n[bold]Best checkpoint found[/bold]: {best_checkpoint}")
        print(f"Overall score: {best_score * 100:.2f}%")
        if "leaderboard_bbh" in best_results:
            print(f"BBH: {best_results['leaderboard_bbh']['score'] * 100:.2f}%")
        if "leaderboard_gpqa" in best_results:
            print(f"GPQA: {best_results['leaderboard_gpqa']['score'] * 100:.2f}%")
        if "leaderboard_ifeval" in best_results:
            print(f"IFEval: {best_results['leaderboard_ifeval']['score'] * 100:.2f}%")
        if "leaderboard_math_hard" in best_results:
            print(
                f"MATH-Hard: {best_results['leaderboard_math_hard']['score'] * 100:.2f}%"
            )
        if "leaderboard_mmlu_pro" in best_results:
            print(
                f"MMLU-Pro: {best_results['leaderboard_mmlu_pro']['score'] * 100:.2f}%"
            )
        if "leaderboard_musr" in best_results:
            print(f"MUSR: {best_results['leaderboard_musr']['score'] * 100:.2f}%")


if __name__ == "__main__":
    app()
