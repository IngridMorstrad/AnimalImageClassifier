"""Command line interface for animal-classifier.

Commands are placeholders at this stage; the pipeline lands in later steps.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    help="Detect, classify and file safari photographs by animal species.",
)

_NOT_IMPLEMENTED = "not implemented yet"


@app.command()
def classify(
    source: Path = typer.Argument(..., help="SD-card directory to walk recursively."),
    output: Path = typer.Option(
        Path("~/animal_pics"), "--output", "-o", help="Destination root for labelled dirs."
    ),
    link: bool = typer.Option(False, "--link", help="Symlink instead of copying."),
) -> None:
    """Classify every image under SOURCE and file it under OUTPUT/<label>/."""
    typer.echo(f"classify: {_NOT_IMPLEMENTED}", err=True)
    raise typer.Exit(1)


@app.command()
def gui(
    output: Path = typer.Option(
        Path("~/animal_pics"), "--output", "-o", help="Labelled output root to browse."
    ),
    port: int = typer.Option(8765, "--port", help="Port to serve the GUI on."),
) -> None:
    """Serve the review GUI for an already-classified output tree."""
    typer.echo(f"gui: {_NOT_IMPLEMENTED}", err=True)
    raise typer.Exit(1)


@app.command()
def train(
    manifest: Path = typer.Argument(..., help="Training dataset manifest."),
) -> None:
    """Train or finetune the species classifier."""
    typer.echo(f"train: {_NOT_IMPLEMENTED}", err=True)
    raise typer.Exit(1)


@app.command()
def eval(
    manifest: Path = typer.Argument(..., help="Evaluation dataset manifest."),
    model: Path = typer.Option(..., "--model", help="Exported model artifact to evaluate."),
) -> None:
    """Evaluate an exported model artifact."""
    typer.echo(f"eval: {_NOT_IMPLEMENTED}", err=True)
    raise typer.Exit(1)


@app.command("export-trainset")
def export_trainset(
    output: Path = typer.Option(
        Path("~/animal_pics"), "--output", "-o", help="Labelled output root to export from."
    ),
    destination: Path = typer.Option(..., "--destination", help="Where to write the manifest."),
) -> None:
    """Export labelled crops from a classified tree as a training manifest."""
    typer.echo(f"export-trainset: {_NOT_IMPLEMENTED}", err=True)
    raise typer.Exit(1)


@app.command()
def verify() -> None:
    """Report which models, providers and network resources are reachable."""
    typer.echo(f"verify: {_NOT_IMPLEMENTED}", err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
