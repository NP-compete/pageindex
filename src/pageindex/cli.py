"""Command-line interface for PageIndex."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree as RichTree

from pageindex import __version__
from pageindex.config import PageIndexConfig

app = typer.Typer(
    name="pageindex",
    help="Vectorless, reasoning-based RAG using hierarchical document indexing with Vertex AI",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"PageIndex version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """PageIndex: Vectorless, reasoning-based RAG."""
    pass


@app.command()
def pdf(
    path: Annotated[Path, typer.Argument(help="Path to PDF file")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON file path"),
    ] = None,
    project_id: Annotated[
        str | None,
        typer.Option(
            "--project-id", "-p", help="Google Cloud project ID", envvar="PAGEINDEX_PROJECT_ID"
        ),
    ] = None,
    location: Annotated[
        str | None,
        typer.Option("--location", "-l", help="Vertex AI location"),
    ] = "us-central1",
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Gemini model to use"),
    ] = "gemini-1.5-flash",
    toc_check_pages: Annotated[
        int,
        typer.Option("--toc-check-pages", help="Pages to check for TOC"),
    ] = 20,
    max_pages_per_node: Annotated[
        int,
        typer.Option("--max-pages-per-node", help="Max pages per node"),
    ] = 10,
    max_tokens_per_node: Annotated[
        int,
        typer.Option("--max-tokens-per-node", help="Max tokens per node"),
    ] = 20000,
    add_node_id: Annotated[
        bool,
        typer.Option("--add-node-id/--no-node-id", help="Add node IDs"),
    ] = True,
    add_summary: Annotated[
        bool,
        typer.Option("--add-summary/--no-summary", help="Generate summaries"),
    ] = True,
    add_description: Annotated[
        bool,
        typer.Option("--add-description/--no-description", help="Generate doc description"),
    ] = False,
    add_text: Annotated[
        bool,
        typer.Option("--add-text/--no-text", help="Include full text in nodes"),
    ] = False,
) -> None:
    """Process a PDF document and generate tree structure."""
    from pageindex.pdf.processor import page_index

    if not path.exists():
        console.print(f"[red]Error: File not found: {path}[/red]")
        raise typer.Exit(1)

    if not path.suffix.lower() == ".pdf":
        console.print(f"[red]Error: File must be a PDF: {path}[/red]")
        raise typer.Exit(1)

    if not project_id:
        console.print("[red]Error: --project-id is required (or set PAGEINDEX_PROJECT_ID)[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"Processing: [bold]{path}[/bold]", title="PageIndex"))

    try:
        result = page_index(
            doc=str(path),
            project_id=project_id,
            location=location,
            model=model,
            toc_check_page_num=toc_check_pages,
            max_page_num_each_node=max_pages_per_node,
            max_token_num_each_node=max_tokens_per_node,
            if_add_node_id="yes" if add_node_id else "no",
            if_add_node_summary="yes" if add_summary else "no",
            if_add_doc_description="yes" if add_description else "no",
            if_add_node_text="yes" if add_text else "no",
        )

        if output is None:
            output = Path(f"./results/{path.stem}_structure.json")

        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]Success![/green] Tree structure saved to: {output}")

        if result.get("doc_description"):
            console.print(f"\n[bold]Description:[/bold] {result['doc_description']}")

        display_tree(result["structure"])

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def md(
    path: Annotated[Path, typer.Argument(help="Path to Markdown file")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON file path"),
    ] = None,
    project_id: Annotated[
        str | None,
        typer.Option(
            "--project-id", "-p", help="Google Cloud project ID", envvar="PAGEINDEX_PROJECT_ID"
        ),
    ] = None,
    location: Annotated[
        str | None,
        typer.Option("--location", "-l", help="Vertex AI location"),
    ] = "us-central1",
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Gemini model to use"),
    ] = "gemini-1.5-flash",
    thinning: Annotated[
        bool,
        typer.Option("--thinning/--no-thinning", help="Apply tree thinning"),
    ] = False,
    thinning_threshold: Annotated[
        int,
        typer.Option("--thinning-threshold", help="Min token threshold for thinning"),
    ] = 5000,
    summary_threshold: Annotated[
        int,
        typer.Option("--summary-threshold", help="Token threshold for summaries"),
    ] = 200,
    add_node_id: Annotated[
        bool,
        typer.Option("--add-node-id/--no-node-id", help="Add node IDs"),
    ] = True,
    add_summary: Annotated[
        bool,
        typer.Option("--add-summary/--no-summary", help="Generate summaries"),
    ] = False,
    add_description: Annotated[
        bool,
        typer.Option("--add-description/--no-description", help="Generate doc description"),
    ] = False,
    add_text: Annotated[
        bool,
        typer.Option("--add-text/--no-text", help="Include full text in nodes"),
    ] = False,
) -> None:
    """Process a Markdown document and generate tree structure."""
    from pageindex.config import PageIndexConfig
    from pageindex.markdown.processor import md_to_tree

    if not path.exists():
        console.print(f"[red]Error: File not found: {path}[/red]")
        raise typer.Exit(1)

    if path.suffix.lower() not in (".md", ".markdown"):
        console.print(f"[red]Error: File must be Markdown: {path}[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"Processing: [bold]{path}[/bold]", title="PageIndex"))

    try:
        config = PageIndexConfig(
            project_id=project_id or "",
            location=location or "us-central1",
            model=model or "gemini-1.5-flash",
        )

        result = asyncio.run(
            md_to_tree(
                md_path=path,
                config=config,
                if_thinning=thinning,
                min_token_threshold=thinning_threshold,
                if_add_node_summary="yes" if add_summary else "no",
                summary_token_threshold=summary_threshold,
                if_add_doc_description="yes" if add_description else "no",
                if_add_node_text="yes" if add_text else "no",
                if_add_node_id="yes" if add_node_id else "no",
            )
        )

        if output is None:
            output = Path(f"./results/{path.stem}_structure.json")

        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]Success![/green] Tree structure saved to: {output}")

        if result.get("doc_description"):
            console.print(f"\n[bold]Description:[/bold] {result['doc_description']}")

        display_tree(result["structure"])

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def folder(
    path: Annotated[Path, typer.Argument(help="Path to folder containing documents")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for JSON files"),
    ] = None,
    project_id: Annotated[
        str | None,
        typer.Option(
            "--project-id", "-p", help="Google Cloud project ID", envvar="PAGEINDEX_PROJECT_ID"
        ),
    ] = None,
    location: Annotated[
        str | None,
        typer.Option("--location", "-l", help="Vertex AI location"),
    ] = "us-central1",
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Gemini model to use"),
    ] = "gemini-1.5-flash",
    max_concurrent: Annotated[
        int,
        typer.Option("--max-concurrent", "-c", help="Max concurrent processing tasks"),
    ] = 5,
    convert_unsupported: Annotated[
        bool,
        typer.Option("--convert/--no-convert", help="Convert unsupported formats with docling"),
    ] = True,
    docling_serve_url: Annotated[
        str | None,
        typer.Option(
            "--docling-serve-url",
            help="URL of docling-serve API (e.g., http://localhost:5001)",
            envvar="PAGEINDEX_DOCLING_SERVE_URL",
        ),
    ] = None,
    docling_serve_timeout: Annotated[
        int,
        typer.Option(
            "--docling-serve-timeout", help="Timeout for docling-serve API calls (seconds)"
        ),
    ] = 300,
    add_node_id: Annotated[
        bool,
        typer.Option("--add-node-id/--no-node-id", help="Add node IDs"),
    ] = True,
    add_summary: Annotated[
        bool,
        typer.Option("--add-summary/--no-summary", help="Generate summaries"),
    ] = True,
    add_description: Annotated[
        bool,
        typer.Option("--add-description/--no-description", help="Generate doc description"),
    ] = False,
    add_text: Annotated[
        bool,
        typer.Option("--add-text/--no-text", help="Include full text in nodes"),
    ] = False,
) -> None:
    """Process all documents in a folder.

    Supports PDF and Markdown natively. Other formats (DOCX, PPTX, HTML, etc.)
    are converted to Markdown using docling if --convert is enabled.

    Conversion priority:
    1. If --docling-serve-url is set, use remote docling-serve API
    2. Else if local docling is installed, use local docling
    3. Else skip unsupported formats

    Install docling support with: pip install pageindex[docling]
    Or run docling-serve: docker run -p 5001:5001 quay.io/docling-project/docling-serve
    """
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
    from rich.table import Table

    from pageindex.batch import get_supported_files, process_folder_sync

    if not path.exists():
        console.print(f"[red]Error: Folder not found: {path}[/red]")
        raise typer.Exit(1)

    if not path.is_dir():
        console.print(f"[red]Error: Not a directory: {path}[/red]")
        raise typer.Exit(1)

    if not project_id:
        console.print("[red]Error: --project-id is required (or set PAGEINDEX_PROJECT_ID)[/red]")
        raise typer.Exit(1)

    files = get_supported_files(path)

    table = Table(title="Files Found")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Files", style="dim")

    table.add_row(
        "PDF",
        str(len(files["pdf"])),
        ", ".join(f.name for f in files["pdf"][:3]) + ("..." if len(files["pdf"]) > 3 else ""),
    )
    table.add_row(
        "Markdown",
        str(len(files["markdown"])),
        ", ".join(f.name for f in files["markdown"][:3])
        + ("..." if len(files["markdown"]) > 3 else ""),
    )
    table.add_row(
        "Convertible",
        str(len(files["docling"])),
        ", ".join(f.name for f in files["docling"][:3])
        + ("..." if len(files["docling"]) > 3 else ""),
    )
    table.add_row(
        "Unsupported",
        str(len(files["unsupported"])),
        ", ".join(f.name for f in files["unsupported"][:3])
        + ("..." if len(files["unsupported"]) > 3 else ""),
    )

    console.print(table)
    console.print()

    total = len(files["pdf"]) + len(files["markdown"])
    if convert_unsupported:
        total += len(files["docling"])

    if total == 0:
        console.print("[yellow]No supported files found in folder.[/yellow]")
        raise typer.Exit(0)

    console.print(
        Panel(
            f"Processing [bold]{total}[/bold] documents from: [bold]{path}[/bold]",
            title="PageIndex Batch",
        )
    )

    config = PageIndexConfig(
        project_id=project_id,
        location=location or "us-central1",
        model=model or "gemini-1.5-flash",
        if_add_node_id="yes" if add_node_id else "no",
        if_add_node_summary="yes" if add_summary else "no",
        if_add_doc_description="yes" if add_description else "no",
        if_add_node_text="yes" if add_text else "no",
        docling_serve_url=docling_serve_url,
        docling_serve_timeout=docling_serve_timeout,
    )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents...", total=total)

            def on_progress(file_name: str, status: str) -> None:
                if status == "done":
                    progress.advance(task)
                progress.update(task, description=f"[cyan]{file_name}[/cyan]: {status}")

            results = process_folder_sync(
                folder=path,
                config=config,
                output_dir=output,
                max_concurrent=max_concurrent,
                convert_unsupported=convert_unsupported,
            )

        console.print()

        stats = results["statistics"]
        result_table = Table(title="Results")
        result_table.add_column("Status", style="bold")
        result_table.add_column("Count", justify="right")

        result_table.add_row("[green]Success[/green]", str(stats["success"]))
        result_table.add_row("[red]Failed[/red]", str(stats["failed"]))
        result_table.add_row("[yellow]Skipped[/yellow]", str(stats["skipped"]))
        result_table.add_row("[bold]Total[/bold]", str(stats["total"]))

        console.print(result_table)

        if results.get("conversion_method"):
            console.print(f"\n[dim]Conversion method: {results['conversion_method']}[/dim]")

        if results["failed"]:
            console.print("\n[red]Failed files:[/red]")
            for item in results["failed"]:
                console.print(f"  - {item['file']}: {item['error']}")

        if results["success"]:
            console.print(f"\n[green]Output saved to:[/green] {output or './results/'}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def repo(
    path: Annotated[Path, typer.Argument(help="Path to repository root")] = Path("."),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON file path"),
    ] = None,
    project_id: Annotated[
        str | None,
        typer.Option(
            "--project-id", "-p", help="Google Cloud project ID", envvar="PAGEINDEX_PROJECT_ID"
        ),
    ] = None,
    location: Annotated[
        str | None,
        typer.Option("--location", "-l", help="Vertex AI location"),
    ] = "us-central1",
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Gemini model to use"),
    ] = "gemini-1.5-flash",
    add_summaries: Annotated[
        bool,
        typer.Option("--summaries/--no-summaries", help="Generate LLM summaries for directories"),
    ] = True,
    max_concurrent: Annotated[
        int,
        typer.Option("--max-concurrent", "-c", help="Max concurrent LLM calls"),
    ] = 5,
    include: Annotated[
        list[str] | None,
        typer.Option("--include", "-i", help="File patterns to include (can specify multiple)"),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", "-e", help="Patterns to exclude (can specify multiple)"),
    ] = None,
    max_depth: Annotated[
        int,
        typer.Option("--max-depth", help="Max depth to display in tree output"),
    ] = 4,
) -> None:
    """Index a Git repository for semantic code search.

    Scans the repository structure and generates summaries for each directory
    using LLM reasoning. The output can be used for semantic navigation and
    understanding of the codebase.

    Default includes: *.py, *.js, *.ts, *.go, *.rs, *.java, *.md, etc.
    Default excludes: .git, node_modules, __pycache__, dist, build, etc.
    """
    from rich.table import Table

    from pageindex.repo import (
        index_repository_sync,
    )

    if not path.exists():
        console.print(f"[red]Error: Path not found: {path}[/red]")
        raise typer.Exit(1)

    if not path.is_dir():
        console.print(f"[red]Error: Not a directory: {path}[/red]")
        raise typer.Exit(1)

    path = path.resolve()

    if add_summaries and not project_id:
        console.print("[yellow]Warning: --project-id not set. Summaries will be skipped.[/yellow]")
        console.print(
            "[dim]Set PAGEINDEX_PROJECT_ID or use --project-id to enable summaries.[/dim]"
        )
        add_summaries = False

    include_patterns = list(include) if include else None
    exclude_patterns = list(exclude) if exclude else None

    console.print(Panel(f"Indexing repository: [bold]{path}[/bold]", title="PageIndex Repo"))

    if include_patterns:
        console.print(f"[dim]Include patterns: {', '.join(include_patterns)}[/dim]")
    if exclude_patterns:
        console.print(f"[dim]Exclude patterns: {', '.join(exclude_patterns)}[/dim]")

    config = PageIndexConfig(
        project_id=project_id or "",
        location=location or "us-central1",
        model=model or "gemini-1.5-flash",
    )

    try:
        with console.status("[cyan]Scanning repository...[/cyan]"):
            result = index_repository_sync(
                repo_path=path,
                config=config,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                add_summaries=add_summaries,
                max_concurrent=max_concurrent,
            )

        stats = result["statistics"]
        stats_table = Table(title="Repository Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", justify="right")
        stats_table.add_row("Directories", str(stats["total_directories"]))
        stats_table.add_row("Files", str(stats["total_files"]))
        console.print(stats_table)

        if output is None:
            output = Path(f"./results/{path.name}_index.json")

        output.parent.mkdir(parents=True, exist_ok=True)

        import json

        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]Index saved to:[/green] {output}")

        display_repo_tree(result["structure"], max_depth=max_depth)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def display_tree(structure: list[dict], max_depth: int = 3) -> None:
    """Display tree structure using rich."""
    console.print("\n[bold]Table of Contents:[/bold]")

    def add_nodes(tree: RichTree, nodes: list[dict], depth: int = 0) -> None:
        for node in nodes:
            title = node.get("title", "Untitled")
            node_id = node.get("node_id", "")
            label = f"[cyan]{node_id}[/cyan] {title}" if node_id else title

            if depth < max_depth:
                branch = tree.add(label)
                if node.get("nodes"):
                    add_nodes(branch, node["nodes"], depth + 1)
            else:
                tree.add(f"{label} [dim](...)[/dim]")

    rich_tree = RichTree("[bold]Document[/bold]")
    add_nodes(rich_tree, structure)
    console.print(rich_tree)


def display_repo_tree(structure: list[dict], max_depth: int = 4) -> None:
    """Display repository tree structure with summaries."""
    console.print("\n[bold]Repository Structure:[/bold]")

    def add_nodes(tree: RichTree, nodes: list[dict], depth: int = 0) -> None:
        for node in nodes:
            title = node.get("title", "Untitled")
            node_id = node.get("node_id", "")
            summary = node.get("summary", "")
            file_count = len(node.get("files", []))

            label_parts = []
            if node_id:
                label_parts.append(f"[cyan]{node_id}[/cyan]")
            label_parts.append(f"[bold]{title}/[/bold]")
            if file_count:
                label_parts.append(f"[dim]({file_count} files)[/dim]")

            label = " ".join(label_parts)

            if depth < max_depth:
                branch = tree.add(label)
                if summary:
                    branch.add(f"[italic dim]{summary}[/italic dim]")
                if node.get("nodes"):
                    add_nodes(branch, node["nodes"], depth + 1)
            else:
                tree.add(f"{label} [dim](...)[/dim]")

    rich_tree = RichTree(f"[bold]{structure[0]['title'] if structure else 'Repository'}[/bold]")
    if structure and structure[0].get("nodes"):
        add_nodes(rich_tree, structure[0]["nodes"], 0)
    console.print(rich_tree)


if __name__ == "__main__":
    app()
