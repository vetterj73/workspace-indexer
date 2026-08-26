"""The `watch:` block."""

from __future__ import annotations

from workspace_indexer.config.strict import Strict
from workspace_indexer.config.watch_mode import WatchMode


class WatchSection(Strict):
    # auto decides per root from the filesystem backing it, which is almost
    # always what you want. The override exists because "almost".
    mode: WatchMode = WatchMode.AUTO
    # How long to wait after the last event before reindexing. One editor save
    # is several events, and a formatter run is hundreds; without this every
    # one of them would be its own reindex.
    debounce_ms: int = 1500
    # Polling interval for roots that cannot use inotify. Slower than events
    # and proportional to tree size, so not something to set aggressively.
    poll_interval_ms: int = 5000
    # Reload workspace.yaml when it changes, so adding a root does not need a
    # restart. The file is watched even when it sits outside every root.
    reload_config: bool = True
