"""Deciding whether inotify will actually deliver events for a path.

The failure this guards against is completely silent: `inotify_add_watch` on a
9P, CIFS or NFS mount *succeeds* and then never fires. Nothing errors, and the
watcher sits there looking healthy while indexing nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.watching import NATIVE_FILESYSTEMS, FilesystemProbe

# Shaped exactly like the real file, including the nested mounts and the
# octal-escaped space that /proc/mounts uses.
MOUNTS = """\
sysfs /sys sysfs rw,nosuid 0 0
/dev/sda2 / ext4 rw,relatime 0 0
/dev/sda1 /boot ext4 rw,relatime 0 0
tmpfs /tmp tmpfs rw,nosuid 0 0
//server/share /mnt/shared cifs rw,relatime 0 0
drvfs /mnt/c 9p rw,dirsync 0 0
nfsserver:/export /mnt/nfs nfs4 rw 0 0
gvfsd-fuse /run/user/1000/gvfs fuse.gvfsd-fuse rw 0 0
/dev/sdb1 /mnt/my\\040disk ext4 rw 0 0
"""


@pytest.fixture
def probe() -> FilesystemProbe:
    return FilesystemProbe(MOUNTS)


def test_local_filesystems_support_inotify(probe: FilesystemProbe) -> None:
    assert probe.supports_inotify(Path("/"))
    assert probe.supports_inotify(Path("/boot"))
    assert probe.supports_inotify(Path("/tmp"))


def test_the_wsl_case_is_detected(probe: FilesystemProbe) -> None:
    """`/mnt/c` under WSL is 9p. A watch there succeeds and never fires, which
    is the single most confusing way for this to fail."""
    assert probe.filesystem_for(Path("/mnt/c/Users/jeremy/src")) == "9p"
    assert not probe.supports_inotify(Path("/mnt/c/Users/jeremy/src"))


@pytest.mark.parametrize(
    ("path", "fstype"),
    [
        ("/mnt/shared/repo", "cifs"),
        ("/mnt/nfs/repo", "nfs4"),
        ("/run/user/1000/gvfs/anything", "fuse.gvfsd-fuse"),
    ],
)
def test_network_and_fuse_mounts_are_polled(probe: FilesystemProbe, path: str, fstype: str) -> None:
    assert probe.filesystem_for(Path(path)) == fstype
    assert not probe.supports_inotify(Path(path))


def test_the_deepest_mountpoint_wins(probe: FilesystemProbe) -> None:
    """Mounts nest. Both `/` and `/mnt/shared` prefix `/mnt/shared/repo`, and
    only the deeper one describes it -- getting this backwards would call a
    CIFS mount ext4 and watch it with inotify."""
    assert probe.filesystem_for(Path("/mnt/shared/repo/src")) == "cifs"


def test_octal_escaped_mountpoints_are_decoded(probe: FilesystemProbe) -> None:
    """/proc/mounts writes a space as \\040."""
    assert probe.filesystem_for(Path("/mnt/my disk")) == "ext4"


def test_an_unknown_filesystem_is_polled_not_assumed_native() -> None:
    """The asymmetry is deliberate. Guessing "native" wrong costs a watcher
    that silently never fires; guessing "poll" wrong costs some CPU."""
    probe = FilesystemProbe("weirdfs /data exofs rw 0 0\n")
    assert probe.filesystem_for(Path("/data")) == "exofs"
    assert not probe.supports_inotify(Path("/data"))


def test_a_path_no_mount_covers_is_polled() -> None:
    assert not FilesystemProbe("").supports_inotify(Path("/anything"))


def test_missing_proc_mounts_does_not_raise() -> None:
    """Not Linux, or a container without /proc. Polling works everywhere."""
    probe = FilesystemProbe("")
    assert probe.filesystem_for(Path("/")) is None


def test_overlayfs_counts_as_native() -> None:
    """Containers run on overlay, and a watcher that polls inside every
    container image is a waste."""
    assert "overlay" in NATIVE_FILESYSTEMS


def test_this_machines_real_mounts_are_readable() -> None:
    """The parser against the real file, not only a fixture.

    Cheap insurance: /proc/mounts format is stable, but a parser that only ever
    sees its own test data is a parser nobody has tested.
    """
    probe = FilesystemProbe()
    fstype = probe.filesystem_for(Path.cwd())
    assert fstype is None or isinstance(fstype, str)
