"""Which filesystem backs a path, and whether it can deliver change events.

This exists because of a failure that is completely silent. On WSL and similar
setups, `inotify_add_watch()` against a path under `/mnt/c` *succeeds* and then
never delivers an event: the 9P protocol WSL uses to reach the Windows
filesystem carries no change notification, and inotify is a service of the
kernel's own filesystem layer. The same is true of NFS, CIFS and most FUSE
mounts -- a remote write happens on another machine, so the local kernel has
nothing to report.

Nothing errors. The watcher simply sits there looking healthy and indexing
nothing, which is the worst possible way for this to fail.
"""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.watching.probe")

MOUNTS = Path("/proc/mounts")

# Filesystems whose events inotify actually receives. An allowlist rather than
# a blocklist of the bad ones: a filesystem nobody here has heard of is far
# more likely to be an exotic network or FUSE mount than a new local one, and
# guessing "native" wrong costs a watcher that silently never fires. Guessing
# "poll" wrong only costs some CPU.
NATIVE_FILESYSTEMS = frozenset(
    {
        "ext2",
        "ext3",
        "ext4",
        "btrfs",
        "xfs",
        "zfs",
        "f2fs",
        "jfs",
        "reiserfs",
        "overlay",
        "tmpfs",
        "ramfs",
    }
)


class FilesystemProbe:
    """Reads /proc/mounts once and answers questions about paths."""

    def __init__(self, mounts: str | None = None) -> None:
        self._by_mountpoint = _parse(mounts if mounts is not None else _read_mounts())

    def filesystem_for(self, path: Path) -> str | None:
        """The fstype of the longest mountpoint prefixing `path`.

        Longest wins because mounts nest: `/` and `/mnt/shared` both prefix
        `/mnt/shared/repo`, and only the deeper one describes it.
        """
        resolved = path.resolve()
        best: tuple[int, str] | None = None
        for mountpoint, fstype in self._by_mountpoint.items():
            if resolved == mountpoint or mountpoint in resolved.parents:
                depth = len(mountpoint.parts)
                if best is None or depth > best[0]:
                    best = (depth, fstype)
        return best[1] if best else None

    def supports_inotify(self, path: Path) -> bool:
        """Whether a watch on `path` will actually deliver events.

        Unknown filesystems answer False. See NATIVE_FILESYSTEMS for why the
        asymmetry is deliberate.
        """
        fstype = self.filesystem_for(path)
        if fstype is None:
            log.warning(
                "watch.filesystem_unknown",
                path=str(path),
                detail="no /proc/mounts entry covers this path; polling to be safe",
            )
            return False
        return fstype in NATIVE_FILESYSTEMS


def _read_mounts() -> str:
    try:
        return MOUNTS.read_text(encoding="utf-8")
    except OSError:
        # Not Linux, or a container without /proc. Callers fall back to
        # polling, which works everywhere.
        return ""


def _parse(text: str) -> dict[Path, str]:
    parsed: dict[Path, str] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        # Mountpoints are octal-escaped in /proc/mounts: a path containing a
        # space arrives as `\040`.
        mountpoint = fields[1].replace("\\040", " ").replace("\\011", "\t")
        parsed[Path(mountpoint)] = fields[2]
    return parsed
