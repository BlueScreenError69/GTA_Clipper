"""Storage safety. Two independent limits, both enforced before any heavy stage:
  1. MAX_DATA_GB      - total size of DATA_DIR (jobs, downloads, clips, models)
  2. MIN_FREE_DISK_GB - free space that must remain on the whole volume
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path

GB = 1024 ** 3


class DiskGuardError(RuntimeError):
    pass


def dir_size(path: Path) -> int:
    """Total bytes under path (symlinks not followed, unreadable files ignored)."""
    total = 0
    stack = [str(path)]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        if e.is_symlink():
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def usage(settings) -> dict:
    used = dir_size(settings.data_dir)
    free = shutil.disk_usage(settings.data_dir).free
    return {
        "used_gb": used / GB,
        "cap_gb": settings.max_data_gb,
        "remaining_gb": max(0.0, settings.max_data_gb - used / GB),
        "free_disk_gb": free / GB,
        "min_free_gb": settings.min_free_disk_gb,
    }


def check(settings, estimate_bytes: int = 0, what: str = "stage") -> None:
    """Raise DiskGuardError if writing ~estimate_bytes would break either limit."""
    if estimate_bytes < 0:
        raise ValueError("estimate_bytes must be >= 0")
    used = dir_size(settings.data_dir)
    cap = int(settings.max_data_gb * GB)
    if used + estimate_bytes > cap:
        raise DiskGuardError(
            f"{what}: data dir would reach {(used + estimate_bytes) / GB:.1f} GB, "
            f"over the {settings.max_data_gb:g} GB cap (currently {used / GB:.1f} GB). "
            f"Delete old jobs under {settings.jobs_dir} or raise MAX_DATA_GB."
        )
    free = shutil.disk_usage(settings.data_dir).free
    floor = int(settings.min_free_disk_gb * GB)
    if free - estimate_bytes < floor:
        raise DiskGuardError(
            f"{what}: only {(free - estimate_bytes) / GB:.1f} GB would stay free, "
            f"below MIN_FREE_DISK_GB={settings.min_free_disk_gb:g}."
        )


def estimate_audio_bytes(duration_s: float) -> int:
    """Source audio (~1.5x of 128 kbps, worst case) + 16 kHz mono WAV + chunk copies."""
    wav = duration_s * 16000 * 2
    src = duration_s * 24000
    return int((wav * 2 + src) * 1.2)


def estimate_clip_bytes(total_clip_seconds: float, sections_pad_s: float = 0) -> int:
    """Section download (~1080p, 8 Mbps) + rendered output (~6 Mbps), 25% headroom."""
    secs = total_clip_seconds + sections_pad_s
    return int(secs * (1_000_000 + 750_000) * 1.25)
