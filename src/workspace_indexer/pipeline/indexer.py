"""Discovery -> read -> chunk -> embed -> store, driven by the manifest.

The manifest decides how much of that chain each file actually walks. Most
files on a rerun stop at the first step for the price of one stat().
"""

from __future__ import annotations

from datetime import UTC, datetime

from workspace_indexer.chunking import ChunkerRegistry, prefetch_languages, read_source
from workspace_indexer.classification import Classification, DocumentClassifier
from workspace_indexer.config import Settings, WorkspaceConfig
from workspace_indexer.discovery import Walker
from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.embedding.sparse_backend import SparseBackend
from workspace_indexer.graph import ImportScanner
from workspace_indexer.models import EmbeddingSpace, RunStats, SourceFile
from workspace_indexer.obs.context import bound, file_context, new_run_id
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.pipeline.pending_file import PendingFile
from workspace_indexer.secrets import SecretWithheldError
from workspace_indexer.state import IndexDecision, Manifest
from workspace_indexer.storage.vector_store import VectorStore

log = get_logger("workspace_indexer.pipeline")

# Chunks buffered before an embedding request goes out. Large enough that a
# small file does not cost its own round trip, small enough that a failure
# loses little work.
DEFAULT_FLUSH_CHUNKS = 256


class Indexer:
    def __init__(
        self,
        *,
        config: WorkspaceConfig,
        settings: Settings,
        manifest: Manifest,
        registry: ChunkerRegistry,
        embeddings: EmbeddingService,
        sparse: SparseBackend,
        store: VectorStore,
        space: EmbeddingSpace,
        classifier: DocumentClassifier,
        flush_chunks: int = DEFAULT_FLUSH_CHUNKS,
    ) -> None:
        self._config = config
        self._settings = settings
        self._manifest = manifest
        self._registry = registry
        self._embeddings = embeddings
        self._sparse = sparse
        self._store = store
        self._space = space
        self._classifier = classifier
        self._flush_chunks = max(1, flush_chunks)
        self._imports = ImportScanner()

    async def run(
        self, *, only_root: str | None = None, force: bool = False, dry_run: bool = False
    ) -> RunStats:
        stats = RunStats(
            run_id=new_run_id(),
            started_at=datetime.now(UTC),
            mode="dry-run" if dry_run else "index",
            config_hash=self._settings.config_hash(self._config),
        )

        with bound(run_id=stats.run_id):
            log.info(
                "run.start",
                mode=stats.mode,
                space=self._space.slug(),
                root=only_root or "all",
                force=force,
            )
            if not dry_run:
                self._manifest.start_run(stats)
                await self._store.ensure_collection(self._space)
                await self._warn_if_store_diverges()

            await self._index(stats, only_root=only_root, force=force, dry_run=dry_run)

            stats.finished_at = datetime.now(UTC)
            if not dry_run:
                # A dry run has already accumulated its own estimate; the
                # embedding service's counter is zero because it never ran.
                embed = self._embeddings.stats
                stats.tokens_embedded = embed.tokens
                stats.est_cost_usd = embed.est_cost_usd
                # Carried, not dropped. EmbeddingStats has always drawn the
                # distinction; RunStats used to discard it here, which is how
                # "unpriced" became indistinguishable from "free".
                stats.unpriced_requests = embed.unpriced_requests
                stats.cost_is_estimate = embed.cost_is_estimate
            if not dry_run:
                self._manifest.finish_run(stats)

            log.info(
                "run.end",
                mode=stats.mode,
                files_seen=stats.files_seen,
                files_skipped=stats.files_skipped,
                files_changed=stats.files_changed,
                chunks_upserted=stats.chunks_upserted,
                chunks_deleted=stats.chunks_deleted,
                tokens=stats.tokens_embedded,
                est_cost_usd=round(stats.est_cost_usd, 4),
                cost_is_estimate=stats.cost_is_estimate,
                unpriced_requests=stats.unpriced_requests,
                errors=stats.errors,
                seconds=round((stats.finished_at - stats.started_at).total_seconds(), 1),
            )
        return stats

    async def _warn_if_store_diverges(self) -> None:
        """Catch a manifest that describes a different store than the one we
        are writing to.

        The manifest records which space a file is complete for, not which
        *store* holds it. Switching QDRANT_MODE from embedded to server, or
        pointing at a different host, leaves it reporting a full index over an
        empty one -- and the decision ladder then skips every file, so the run
        silently does nothing.
        """
        recorded = self._manifest.chunk_count(self._space.slug())
        stored = await self._store.count(self._space)
        if recorded and stored < recorded * 0.9:
            log.warning(
                "run.store_diverges",
                space=self._space.slug(),
                manifest_chunks=recorded,
                stored_chunks=stored,
                detail="the manifest describes a store with more chunks than this one; "
                "run with --force to rebuild, or the run will skip nearly everything",
            )

    async def _index(
        self, stats: RunStats, *, only_root: str | None, force: bool, dry_run: bool
    ) -> None:
        walker = Walker(self._config)
        candidates = list(walker.walk(only_root=only_root))
        stats.skip_reasons = dict(walker.skips)

        # Warm the grammar cache once. Grammars download on demand, so without
        # this the first file of each language pays a network round trip
        # mid-walk and a transient failure silently degrades it to text.
        prefetch_languages({c.language for c in candidates if c.language})

        seen: set[tuple[str, str]] = set()
        purged: list[tuple[str, str]] = []
        pending: list[PendingFile] = []
        buffered = 0

        for candidate in candidates:
            stats.files_seen += 1
            seen.add((candidate.root_label, candidate.rel_path))

            with file_context(candidate.root_label, candidate.rel_path):
                try:
                    prepared = self._prepare(candidate, stats, force=force)
                except SecretWithheldError:
                    # Withholding the file from future runs is not enough: an
                    # earlier run may already have embedded it, and leaving
                    # that copy in the index defeats the point of detecting it.
                    stats.files_skipped += 1
                    purged.append((candidate.root_label, candidate.rel_path))
                    continue
                except Exception as exc:
                    # One unreadable or unparseable file must not abort a run
                    # that has already paid for thousands of embeddings.
                    stats.errors += 1
                    log.exception("error.indexing_file", error=str(exc))
                    continue

                if prepared is None:
                    continue

                pending.append(prepared)
                buffered += len(prepared.to_embed)

            if buffered >= self._flush_chunks:
                await self._flush(pending, stats, dry_run=dry_run)
                pending, buffered = [], 0

        if pending:
            await self._flush(pending, stats, dry_run=dry_run)

        if not dry_run:
            for root_label, rel_path in purged:
                await self._purge(root_label, rel_path, stats)
            await self._remove_orphans(seen, stats, only_root=only_root)

    def _prepare(
        self, candidate: FileCandidate, stats: RunStats, *, force: bool
    ) -> PendingFile | None:
        """Walk the decision ladder for one file, stopping as early as it can."""
        decision = self._manifest.decide_from_stat(
            candidate,
            space_slug=self._space.slug(),
            chunker_version=self._chunker_version(candidate.kind.value),
            force=force,
        )

        if decision is IndexDecision.SKIP_UNCHANGED:
            stats.files_skipped += 1
            log.debug("file.decision", decision=decision.value)
            return None

        source = read_source(candidate, self._config.index.secret_allow)
        if source is None:
            # Vanished between the walk and now: a normal race on a live
            # workspace, not an error.
            stats.files_skipped += 1
            return None

        if decision is IndexDecision.REINDEX:
            decision = self._manifest.decide_from_hash(source)
            if decision is IndexDecision.SKIP_SAME_CONTENT:
                # Identical bytes rewritten: a formatter pass, a checkout.
                self._manifest.touch(source)
                stats.files_skipped += 1
                log.debug("file.decision", decision=decision.value)
                return None

        log.debug("file.decision", decision=decision.value, sha=source.sha256[:12])
        stats.files_changed += 1

        # Classify once per file. Done here rather than inside the chunkers,
        # which have no business knowing what role a document plays -- their
        # concern is how to split it.
        classification = self._classify(source)

        chunker = self._registry.resolve(source, self._config.chunking)
        if self._config.chunking.embed_doc_type:
            # The header is built during chunking, so the verdict has to be on
            # the file before the split rather than stamped on afterwards.
            source = source.model_copy(update={"doc_type": classification.doc_type.value})
        chunks = list(chunker.chunk(source, self._config.chunking))
        chunks = [
            chunk.model_copy(
                update={
                    "meta": chunk.meta.model_copy(
                        update={
                            "doc_type": classification.doc_type,
                            "doc_type_confidence": classification.confidence,
                            "classifier_version": self._classifier.version,
                        }
                    )
                }
            )
            for chunk in chunks
        ]
        produced = [c.chunk_id for c in chunks]
        delta = self._manifest.diff_chunks(
            source.root_label, source.rel_path, self._space.slug(), produced
        )
        if force:
            # Rebuild rather than reconcile: re-embed everything this file
            # produces. Distrusting mtime alone does not need --force, since
            # the content hash already catches that.
            #
            # to_delete still comes from the diff. Dropping it left chunks in
            # the store that the file no longer produces, with nothing able to
            # remove them -- a rebuild that quietly accumulates orphans is
            # worse than no rebuild.
            delta = delta.model_copy(update={"to_upsert": produced, "unchanged": []})
        return PendingFile(
            source=source,
            chunker=chunker.name,
            chunker_version=chunker.version,
            chunks=chunks,
            delta=delta,
            classification=classification,
            imports=self._imports.scan(source.text or "", source.language or ""),
        )

    def _classify(self, source: SourceFile) -> Classification:
        """Reuse a stored verdict when the bytes and the ruleset both match.

        Rules are cheap enough that this barely matters today. It matters a
        great deal once a model-based rung exists, and building the cache in
        from the start is what stops classification drift from rewriting
        payloads on every run.
        """
        cached = self._manifest.cached_classification(source, self._classifier.version)
        if cached is not None:
            return cached
        return self._classifier.classify(source)

    async def _flush(self, pending: list[PendingFile], stats: RunStats, *, dry_run: bool) -> None:
        if not pending:
            return

        to_embed = [chunk for file in pending for chunk in file.to_embed]

        if dry_run:
            # The whole point of --dry-run: see the chunk plan and the token
            # estimate without paying for a single embedding.
            stats.chunks_upserted += len(to_embed)
            stats.chunks_deleted += sum(len(f.delta.to_delete) for f in pending)
            stats.tokens_embedded += sum(c.meta.token_estimate for c in to_embed)
            return

        if to_embed:
            texts = [chunk.embed_text for chunk in to_embed]
            dense = await self._embeddings.embed_documents(texts)
            sparse = self._sparse.encode_documents(texts)
            await self._store.upsert(self._space, to_embed, dense, sparse)
            stats.chunks_upserted += len(to_embed)

        stale_ids = [cid for file in pending for cid in file.delta.to_delete]
        if stale_ids:
            await self._store.delete_by_ids(self._space, stale_ids)
            stats.chunks_deleted += len(stale_ids)

        # Manifest rows go in only after the vectors landed, so a crash leaves
        # work to redo rather than a file marked done with nothing stored.
        self._manifest.begin()
        try:
            for file in pending:
                self._record(file)
            self._manifest.commit()
        except Exception:
            self._manifest.rollback()
            raise

    def _record(self, file: PendingFile) -> None:
        source = file.source
        self._manifest.record_file(
            source,
            chunker=file.chunker,
            chunker_version=file.chunker_version,
            classification=file.classification,
            classifier_version=self._classifier.version,
        )
        self._manifest.forget_chunks(file.delta.to_delete, self._space.slug())
        self._manifest.record_chunks(file.chunks, self._space.slug())
        self._manifest.record_imports(source.root_label, source.rel_path, file.imports)
        self._manifest.record_space(
            source.root_label, source.rel_path, self._space.slug(), len(file.chunks)
        )

    async def _purge(self, root_label: str, rel_path: str, stats: RunStats) -> None:
        """Remove every trace of a file we have decided not to index."""
        with file_context(root_label, rel_path):
            existing = self._manifest.chunk_ids_for(root_label, rel_path, self._space.slug())
            await self._store.delete_by_path(self._space, root_label, rel_path)
            self._manifest.forget_file(root_label, rel_path)
            stats.chunks_deleted += len(existing)
            if existing:
                log.warning("file.purged", chunks=len(existing), reason="secret_withheld")

    async def _remove_orphans(
        self, seen: set[tuple[str, str]], stats: RunStats, *, only_root: str | None
    ) -> None:
        for root_label, rel_path in self._manifest.orphans(seen, root_label=only_root):
            with file_context(root_label, rel_path):
                ids = self._manifest.chunk_ids_for(root_label, rel_path, self._space.slug())
                # By path rather than by id: correct even if the manifest and
                # the store disagree about which chunks a file produced.
                await self._store.delete_by_path(self._space, root_label, rel_path)
                self._manifest.forget_file(root_label, rel_path)
                stats.chunks_deleted += len(ids)
                log.info("file.removed", chunks=len(ids))

    def _chunker_version(self, kind: str) -> int:
        return self._registry.versions().get(_CHUNKER_FOR_KIND.get(kind, "text"), 1)


# Which chunker each kind resolves to, for the manifest's version check. Kept
# beside the registry's own mapping rather than inferred, so a strategy swap
# invalidates the right files.
_CHUNKER_FOR_KIND = {
    "code": "code",
    "markdown": "markdown",
    "text": "text",
    "pdf": "text",
    "image": "opaque",
    "opaque": "opaque",
}
