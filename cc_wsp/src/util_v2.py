"""Paths and shared helpers for the v2 (crossfaded-cut) pipeline.

v2 produces an intermediate `cut.mp4` (1080p, BT.709) with audio crossfades
already baked in. Subsequent steps (transcribe, place-images, render, captions)
operate on this cut video instead of cutting through MLT.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from cc_wsp.src import util


def cut_path(name: str) -> Path:
    return util.get_video_dir(name) / "cut.mp4"


def cut_downsampled_path(name: str) -> Path:
    return util.get_video_dir(name) / "cut_downsampled.mp4"


def cut_audio_path(name: str) -> Path:
    return util.get_video_dir(name) / "cut_audio.mp3"


def transcription_v2_path(name: str) -> Path:
    return util.get_video_dir(name) / "transcription_v2.json"


def images_v2_path(name: str) -> Path:
    return util.get_video_dir(name) / "images_v2.json"


def preview_v2_path(name: str) -> Path:
    return util.get_video_dir(name) / "preview_v2.mp4"


def final_v2_path(name: str) -> Path:
    return util.get_video_dir(name) / "final_v2.mp4"


def captioned_v2_path(name: str) -> Path:
    return util.get_video_dir(name) / "final_v2_captioned.mp4"


def _ffmpeg_has_filter(binary: str, name: str) -> bool:
    out = subprocess.run(
        [binary, "-hide_banner", "-filters"],
        capture_output=True, text=True, check=False,
    ).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def get_tonemap_ffmpeg() -> str:
    """Return an ffmpeg binary that has zscale (for HDR tone-mapping).

    Stock Homebrew ffmpeg lacks libzimg; the imageio-ffmpeg bundled binary
    on macOS includes it.
    """
    if _ffmpeg_has_filter("ffmpeg", "zscale"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if _ffmpeg_has_filter(candidate, "zscale"):
            return candidate
    except Exception:
        pass
    return "ffmpeg"


def is_hdr(input_path: Path) -> bool:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=color_transfer,color_primaries",
         "-of", "default=nw=1:nk=1", str(input_path)],
        capture_output=True, text=True, check=True,
    )
    tokens = {t.strip().lower() for t in probe.stdout.split()}
    return bool(tokens & {"smpte2084", "arib-std-b67"}) or "bt2020" in tokens


def hdr_tonemap_filter() -> str:
    """zscale+tonemap chain to map HLG/PQ → BT.709 SDR yuv420p."""
    return (
        "zscale=t=linear:npl=100,format=gbrpf32le,"
        "zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
        "zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
    )


_STANDARD_FPS = [
    ("24000/1001", 24000 / 1001),
    ("24/1",       24.0),
    ("25/1",       25.0),
    ("30000/1001", 30000 / 1001),
    ("30/1",       30.0),
    ("50/1",       50.0),
    ("60000/1001", 60000 / 1001),
    ("60/1",       60.0),
    ("120000/1001", 120000 / 1001),
    ("120/1",       120.0),
]


def probe_avg_fps(path: Path) -> str:
    """Return source's average frame rate snapped to the nearest standard.

    iPhone .mov often advertises r_frame_rate=120/1 (slow-mo capable container)
    while the actual recorded frames are 29.97 or 30 fps. We probe avg, then
    snap to a standard rate (23.976/24/25/29.97/30/50/59.94/60/119.88/120) —
    passing a raw oddball fraction like 4308000/143753 to fps= confuses ffmpeg's
    audio chain and produces a cut whose audio is ~1s shorter than its video.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if "/" in out:
        n, d = out.split("/")
        try:
            value = int(n) / int(d) if int(d) > 0 else 0.0
        except ValueError:
            value = 0.0
        if value > 0:
            best = min(_STANDARD_FPS, key=lambda kv: abs(kv[1] - value))
            return best[0]
    return "30/1"


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)
