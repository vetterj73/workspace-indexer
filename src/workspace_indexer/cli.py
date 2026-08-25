"""Command line interface."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from workspace_indexer.app_context import AppContext
from workspace_indexer.config import ConfigError
from workspace_indexer.evaluation import EvalHarness, load_cases
from workspace_indexer.models import FileKind, SearchFilters
from workspace_indexer.search import Reprojector, SearchRequest

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Semantic + keyword index over a multi-repo workspace.",
)
console = Console()

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="Path to workspace.yaml")
]


def _context(config: Path | None) -> AppContext:
    try:
        return AppContext.build(config)
    except ConfigError as exc:
        # A config problem is a user problem, not a traceback.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


@app.command()
def index(
    config: ConfigOption = None,
    root: Annotated[str | None, typer.Option(help="Index only this root label")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the chunk plan and token estimate, no API calls")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Ignore mtime and hash shortcuts")
    ] = False,
) -> None:
    """Index the configured roots, reusing everything unchanged."""

    async def run() -> None:
        ctx = _context(config)
        try:
            stats = await ctx.indexer().run(only_root=root, force=force, dry_run=dry_run)
        finally:
            await ctx.close()

        table = Table(title=f"{stats.mode} · {stats.run_id}", show_header=False)
        table.add_row("files seen", str(stats.files_seen))
        table.add_row("unchanged", str(stats.files_skipped))
        table.add_row("changed", str(stats.files_changed))
        table.add_row("chunks written", str(stats.chunks_upserted))
        table.add_row("chunks removed", str(stats.chunks_deleted))
        table.add_row("tokens", f"{stats.tokens_embedded:,}")
        table.add_row("estimated cost", f"${stats.est_cost_usd:.4f}")
        if stats.errors:
            table.add_row("[red]errors[/red]", str(stats.errors))
        console.print(table)
        if stats.mode == "dry-run":
            console.print("[dim]No API calls were made and nothing was stored.[/dim]")

    asyncio.run(run())


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for")],
    config: ConfigOption = None,
    limit: Annotated[int | None, typer.Option("-n", "--limit")] = None,
    unit: Annotated[str | None, typer.Option(help="Only this repo or folder")] = None,
    lang: Annotated[str | None, typer.Option(help="Only this language")] = None,
    kind: Annotated[str | None, typer.Option(help="code, markdown, text...")] = None,
    path: Annotated[str | None, typer.Option(help="Only under this directory")] = None,
    fusion: Annotated[
        str | None, typer.Option(help="rrf, dense_only or sparse_only (debugging)")
    ] = None,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    full: Annotated[bool, typer.Option("--full", help="Print whole chunks")] = False,
) -> None:
    """Search the index."""

    async def run() -> None:
        ctx = _context(config)
        try:
            hits = await ctx.search_service().search(
                SearchRequest(
                    query=query,
                    limit=limit,
                    fusion=fusion,  # type: ignore[arg-type]
                    rerank=rerank,
                    filters=SearchFilters(
                        unit=unit,
                        language=lang,
                        kind=FileKind(kind) if kind else None,
                        path_prefix=path,
                    ),
                )
            )
        finally:
            await ctx.close()

        if not hits:
            console.print("[yellow]No matches.[/yellow]")
            return

        for position, hit in enumerate(hits, start=1):
            score = hit.rerank_score if hit.rerank_score is not None else hit.score
            flag = " [yellow](stale)[/yellow]" if hit.stale else ""
            symbol = f" [dim]{hit.symbol_path}[/dim]" if hit.symbol_path else ""
            console.print(f"[bold]{position:>2}.[/bold] [cyan]{hit.location}[/cyan]{symbol}{flag}")
            console.print(f"    [dim]score {score:.4f} · {hit.unit or hit.root_label}[/dim]")
            body = hit.source_text if full else _preview(hit.source_text)
            for line in body.splitlines():
                console.print(f"    {line}")
            console.print()

    asyncio.run(run())


@app.command()
def status(config: ConfigOption = None) -> None:
    """What is indexed, in which spaces, and what recent runs cost."""

    async def run() -> None:
        ctx = _context(config)
        try:
            roots = Table(title="files by root")
            roots.add_column("root")
            roots.add_column("files", justify="right")
            for label, count in sorted(ctx.manifest.counts_by_root().items()):
                roots.add_row(label, f"{count:,}")
            console.print(roots)

            spaces = Table(title="embedding spaces")
            spaces.add_column("space")
            spaces.add_column("chunks", justify="right")
            spaces.add_column("stored", justify="right")

            prefix = f"{ctx.config.workspace.name}__"
            collections = {
                name.removeprefix(prefix)
                for name in await ctx.store.collection_names()
                if name.startswith(prefix)
            }
            tracked = set(ctx.manifest.spaces())
            warnings: list[str] = []

            for slug in sorted(tracked):
                active = slug == ctx.space.slug()
                recorded = ctx.manifest.chunk_count(slug)
                stored = await ctx.store.count(ctx.space) if active else None
                if stored is not None and stored != recorded:
                    # Divergence has to announce itself. Printing two numbers
                    # side by side and hoping someone notices is how stale
                    # content sat in a live collection unremarked.
                    warnings.append(
                        f"{slug}: manifest has {recorded:,} chunks, store has {stored:,}"
                    )
                spaces.add_row(
                    slug + (" [green](active)[/green]" if active else ""),
                    f"{recorded:,}",
                    "-" if stored is None else f"{stored:,}",
                )
            console.print(spaces)

            for slug in sorted(tracked - collections):
                warnings.append(f"{slug}: recorded in the manifest but no collection exists")
            for slug in sorted(collections - tracked):
                warnings.append(f"{slug}: collection exists but nothing is recorded for it")

            for warning in warnings:
                console.print(f"[yellow]inconsistent[/yellow] {warning}")

            runs = Table(title="recent runs")
            for column in ("run", "mode", "files", "written", "removed", "tokens", "cost"):
                runs.add_column(column)
            for row in ctx.manifest.recent_runs(limit=5):
                runs.add_row(
                    row.run_id,
                    row.mode + (" [red](unfinished)[/red]" if row.unfinished else ""),
                    f"{row.files_seen:,}",
                    f"{row.chunks_upserted:,}",
                    f"{row.chunks_deleted:,}",
                    f"{row.tokens_embedded:,}",
                    f"${row.est_cost_usd:.4f}",
                )
            console.print(runs)
        finally:
            await ctx.close()

    asyncio.run(run())


@app.command()
def explain(
    path: Annotated[Path, typer.Argument(help="File to chunk")],
    config: ConfigOption = None,
) -> None:
    """Show the chunks a single file produces. The chunk-quality debugging tool."""
    from workspace_indexer.chunking import read_source
    from workspace_indexer.discovery import Walker

    ctx = _context(config)
    try:
        target = path.expanduser().resolve()
        candidate = next(
            (c for c in Walker(ctx.config).walk() if c.abs_path.resolve() == target), None
        )
        if candidate is None:
            console.print(f"[red]{target} is not inside any configured root, or is excluded.[/red]")
            raise typer.Exit(code=1)

        source = read_source(candidate)
        if source is None:
            console.print(f"[red]{target} could not be read.[/red]")
            raise typer.Exit(code=1)

        chunker = ctx.registry.resolve(source, ctx.config.chunking)
        chunks = list(chunker.chunk(source, ctx.config.chunking))
        console.print(
            f"[bold]{source.rel_path}[/bold] · {source.kind.value}"
            f" · {source.language or 'no grammar'} · chunker [cyan]{chunker.name}[/cyan]"
        )
        if not chunks:
            console.print("[yellow]No chunks: recorded in the manifest but not embedded.[/yellow]")
            return
        for chunk in chunks:
            meta = chunk.meta
            console.print(
                f"\n[cyan]{meta.start_line}-{meta.end_line}[/cyan]"
                f" · {meta.chunk_index + 1}/{meta.chunk_total}"
                f" · ~{meta.token_estimate} tokens"
                f" · [dim]{meta.symbol_kind or '-'} {meta.symbol_path or ''}[/dim]"
                + (" [yellow](degraded parse)[/yellow]" if meta.parse_degraded else "")
            )
            for line in chunk.source_text.splitlines()[:12]:
                console.print(f"    {line}")
    finally:
        ctx.manifest.close()


@app.command()
def reproject(
    dimensions: Annotated[int, typer.Option("--dimensions", "-d")],
    config: ConfigOption = None,
) -> None:
    """Derive a narrower collection by Matryoshka truncation. No re-embedding."""

    async def run() -> None:
        ctx = _context(config)
        try:
            target = await Reprojector(ctx.store, ctx.manifest).reproject(
                ctx.space, dimensions
            )
            console.print(
                f"Wrote [cyan]{ctx.store.collection_name(target)}[/cyan] "
                f"({await ctx.store.count(target):,} points) from "
                f"[cyan]{ctx.store.collection_name(ctx.space)}[/cyan]."
            )
            console.print("[dim]Compare them with `workspace-indexer eval`.[/dim]")
        finally:
            await ctx.close()

    asyncio.run(run())


@app.command("eval")
def evaluate(
    config: ConfigOption = None,
    dataset: Annotated[Path | None, typer.Option("--dataset")] = None,
    limit: Annotated[int, typer.Option("-n", "--limit")] = 10,
    fusion: Annotated[str | None, typer.Option()] = None,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    dimensions: Annotated[
        int | None, typer.Option("--dimensions", help="Evaluate a reprojected collection")
    ] = None,
) -> None:
    """Score retrieval quality against a dataset of query/expected-file pairs."""

    async def run() -> None:
        ctx = _context(config)
        try:
            cases = load_cases(dataset or ctx.config.eval.dataset)
            # A reprojected collection is a distinct space, so it has to be
            # addressed as one rather than by width alone.
            space = (
                ctx.space.model_copy(
                    update={"dimensions": dimensions, "derived_from": ctx.space.dimensions}
                )
                if dimensions
                else None
            )
            label = f"{(space or ctx.space).slug()} fusion={fusion or ctx.config.search.fusion}"
            report = await EvalHarness(ctx.search_service(space)).run(
                cases,
                limit=limit,
                label=label,
                fusion=fusion,  # type: ignore[arg-type]
                rerank=rerank,
            )
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc
        finally:
            await ctx.close()

        console.print(f"[bold]{report.label}[/bold]")
        console.print(f"  recall@{limit}: [cyan]{report.recall_at_k:.3f}[/cyan]")
        console.print(f"  MRR@{limit}:    [cyan]{report.mrr_at_k:.3f}[/cyan]")
        console.print(f"  cases: {len(report.results)}, misses: {len(report.misses)}")
        for miss in report.misses:
            console.print(f"  [yellow]miss[/yellow] {miss.query}")
            console.print(f"    expected {miss.expected}")
            console.print(f"    got      {miss.found[:3]}")

    asyncio.run(run())


def _preview(text: str, lines: int = 6) -> str:
    body = text.splitlines()
    if len(body) <= lines:
        return "\n".join(body)
    return "\n".join([*body[:lines], f"... ({len(body) - lines} more lines)"])


if __name__ == "__main__":
    app()
