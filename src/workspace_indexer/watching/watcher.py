"""Keeping the index fresh without a manual run.

A trigger, not a second indexing path. Every change ends up going through the
same `Indexer.run(only_root=...)` the CLI calls, so the watcher cannot develop
its own opinion about what counts as changed -- and the decision ladder, the
orphan pruning and the root scoping are all already tested there.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

# watchfiles ships no py.typed marker, so strict mode cannot see through to
# awatch's signature. Scoped to the one symbol rather than the module.
from watchfiles import Change, awatch  # pyright: ignore[reportUnknownVariableType]

from workspace_indexer.config import WatchMode, WatchSection, WorkspaceConfig
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.watching.change_debouncer import ChangeDebouncer
from workspace_indexer.watching.filesystem_probe import FilesystemProbe
from workspace_indexer.watching.inotify_budget import InotifyBudget

log = get_logger("workspace_indexer.watching")

# Directory names never worth an inotify watch. Not the same list as the index
# excludes: this is about the watch *budget*, and these are the trees that
# exhaust it. A file inside one is still ignored by the index rules anyway.
UNWATCHED_DIRS = frozenset(
    {".git", "node_modules", ".venv", "venv", "__pycache__", "target", "dist", "build"}
)


class Watcher:
    def __init__(
        self,
        config: WorkspaceConfig,
        *,
        reindex: Callable[[str | None], object],
        config_path: Path | None = None,
        probe: FilesystemProbe | None = None,
        budget: InotifyBudget | None = None,
        reload_config: Callable[[], WorkspaceConfig] | None = None,
    ) -> None:
        self._config = config
        self._reindex = reindex
        self._config_path = config_path.resolve() if config_path else None
        self._probe = probe or FilesystemProbe()
        self._budget = budget or InotifyBudget.detect()
        self._reload_config = reload_config
        self._debouncer = ChangeDebouncer(config, self._config_path)

    @property
    def settings(self) -> WatchSection:
        return self._config.watch

    def plan(self) -> dict[str, WatchMode]:
        """Which mode each root will be watched in, decided before starting.

        Computed up front so `watch` can report it, rather than leaving the
        answer to be inferred from whether anything ever happens.
        """
        configured = self._config.watch.mode
        decided: dict[str, WatchMode] = {}
        for root in self._config.workspace.roots:
            path = root.path.expanduser().resolve()
            if configured is not WatchMode.AUTO:
                decided[root.resolved_label] = configured
                continue
            native = self._probe.supports_inotify(path)
            decided[root.resolved_label] = WatchMode.NATIVE if native else WatchMode.POLL
            if not native:
                log.warning(
                    "watch.polling",
                    root=root.resolved_label,
                    path=str(path),
                    filesystem=self._probe.filesystem_for(path),
                    detail="this filesystem delivers no change notifications, so "
                    "inotify would succeed and then never fire. Polling instead.",
                )
        return decided

    def check_budget(self) -> None:
        """Report the inotify headroom for the roots that will use it."""
        plan = self.plan()
        native = [
            root
            for root in self._config.workspace.roots
            if plan.get(root.resolved_label) is WatchMode.NATIVE
        ]
        if not native:
            return
        needed = sum(
            self._budget.count_directories(
                root.path.expanduser().resolve(), excluded=set(UNWATCHED_DIRS)
            )
            for root in native
        )
        self._budget.check(needed)

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Watch until cancelled, reindexing each root whose files changed."""
        plan = self.plan()
        self.check_budget()
        paths = [str(r.path.expanduser().resolve()) for r in self._config.workspace.roots]
        if self._config_path is not None and self._config.watch.reload_config:
            # Watched explicitly: workspace.yaml usually sits outside every
            # root, so nothing else would notice it change.
            paths.append(str(self._config_path))

        # One poll interval covers the whole watch, so any polled root forces
        # polling for all of them. watchfiles offers no per-path mode, and
        # mixing would mean two concurrent watches to keep in step.
        force_polling = any(mode is WatchMode.POLL for mode in plan.values())
        log.info(
            "watch.start",
            roots={label: mode.value for label, mode in plan.items()},
            polling=force_polling,
            debounce_ms=self._config.watch.debounce_ms,
        )

        async for batch in awatch(
            *paths,
            stop_event=stop,
            debounce=self._config.watch.debounce_ms,
            step=min(50, self._config.watch.debounce_ms),
            force_polling=force_polling,
            poll_delay_ms=self._config.watch.poll_interval_ms,
            recursive=True,
        ):
            await self.handle_changes(batch)

    async def handle_changes(self, batch: set[tuple[Change, str]]) -> None:
        """Process one settled batch of events.

        Public because it is the whole behaviour of this class minus the event
        loop, and driving a torn config write through the real watch is a race
        rather than a test.
        """
        for _, raw in batch:
            self._debouncer.add(Path(raw))

        roots, config_changed = self._debouncer.drain()

        if config_changed:
            await self._reload()
            # A reloaded config can add roots, and `awatch` cannot be told
            # about a new path mid-iteration. Say so rather than pretending.
            log.warning(
                "watch.config_reloaded",
                detail="settings and ignore rules are live; a newly added *root* "
                "needs a restart of `watch` to be observed",
            )

        for label in sorted(roots):
            log.info("watch.reindex", root=label)
            result = self._reindex(label)
            if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                await result

    async def _reload(self) -> None:
        if self._reload_config is None:
            return
        try:
            self._config = self._reload_config()
        except Exception as exc:
            # A half-saved YAML file is a normal thing to observe mid-write.
            # Keeping the old config beats dying on a transient parse error.
            log.error("watch.config_invalid", error=str(exc))
            return
        self._debouncer = ChangeDebouncer(self._config, self._config_path)
