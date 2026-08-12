"""Build cut.mp4 from adjusted.json in a single ffmpeg pass.

Video and audio are both hard-concatenated (butt-joined per clip). The earlier
audio-crossfade chain introduced ~1s of audio-vs-video drift on this footage
(iPhone HEVC, 30fps wrapped as 120fps) — audio for late clips was playing under
the wrong video frames. Hard concat eliminates the drift entirely; the tiny
clicks at clip boundaries are not noticeable on speech.

HDR sources are tone-mapped to BT.709 in the same pass. iPhone .mov advertises
`r_frame_rate=120/1` (slow-mo capable container) while actual frames are 29.97
fps — we snap to a standard rate via `fps=` to give MLT a sane playback rate
downstream.

Draft mode (`--draft`) uses `downsampled.mp4` (already SDR/240p) as the source
so iterations on `adjusted.json` rebuild in seconds instead of a minute. The
final pass on the original source still goes through this same pipeline.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from cc_wsp.src import util, util_v2
from cc_wsp.src.constants import HD_1080P_HEIGHT, LOW_RES_HEIGHT


def _build_filter_graph(
    sentences: list[dict],
    tonemap: bool,
    scale_height: int | None,
    fps: str | None,
) -> str:
    n = len(sentences)
    parts = []

    pre_chain = []
    if tonemap:
        pre_chain.append(util_v2.hdr_tonemap_filter())
    if scale_height is not None:
        pre_chain.append(f"scale=-2:{scale_height}")
    if fps is not None:
        pre_chain.append(f"fps={fps}")
    if not pre_chain:
        pre_chain.append("null")
    parts.append(f"[0:v]{','.join(pre_chain)}[vpre]")

    parts.append(f"[vpre]split={n}" + "".join(f"[vp{i}]" for i in range(n)))
    parts.append(f"[0:a]asplit={n}" + "".join(f"[ap{i}]" for i in range(n)))

    for i, s in enumerate(sentences):
        st = float(s["adjusted_start"])
        en = float(s["adjusted_end"])
        parts.append(f"[vp{i}]trim=start={st}:end={en},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[ap{i}]atrim=start={st}:end={en},asetpts=PTS-STARTPTS[a{i}]")

    interleaved = "".join(f"[v{i}][a{i}]" for i in range(n))
    concat_out = "[vcat]" if fps is not None else "[vout]"
    parts.append(f"{interleaved}concat=n={n}:v=1:a=1{concat_out}[aout]")

    # Each clip is trimmed at an arbitrary source time, so after concat the frames
    # sit on slightly different sub-frame phases and the muxed stream is VFR.
    # ffprobe then reports an inflated r_frame_rate (e.g. 240/1 for a 30fps cut)
    # and MLT adopts THAT as its profile, rendering final_v2 at 8x the frames.
    # Re-time the concatenated stream onto a uniform grid so the cut is true CFR.
    if fps is not None:
        parts.append(f"[vcat]fps={fps}[vout]")

    return ";".join(parts)


def create_crossfaded_cut(
    name: str,
    *,
    height: int = HD_1080P_HEIGHT,
    crf: int = 18,
    preset: str = "slow",
    draft: bool = False,
    force: bool = False,
) -> Path:
    """Render `cut.mp4` (BT.709, hard-concat A+V) from adjusted.json.

    With draft=True, sources `downsampled.mp4` (SDR/240p) for ~10x faster
    iteration during cut adjustments. Final quality requires draft=False.
    """
    out = util_v2.cut_path(name)
    if out.exists() and not force:
        print(f"cut.mp4 already exists: {out.name} (use --force to regenerate)")
        return out

    video_dir = util.get_video_dir(name)
    if draft:
        src = video_dir / "downsampled.mp4"
        if not src.exists():
            raise FileNotFoundError(
                f"draft requires downsampled.mp4: run preprocess first ({src} missing)")
    else:
        src = util.get_input_video_path(name)
        if not src.exists():
            raise FileNotFoundError(f"source video not found: {src}")

    adjusted = json.loads(util.adjusted_path(name).read_text())
    sentences = adjusted["sentences"]
    if not sentences:
        raise ValueError("no sentences in adjusted.json")

    tonemap = (not draft) and util_v2.is_hdr(src)
    binary = util_v2.get_tonemap_ffmpeg() if tonemap else "ffmpeg"
    fps = util_v2.probe_avg_fps(src)
    scale_height = None if draft else height
    fc = _build_filter_graph(sentences, tonemap=tonemap, scale_height=scale_height, fps=fps)

    if draft:
        v_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28"]
    else:
        v_args = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]

    cmd = [
        binary, "-y", "-i", str(src),
        "-filter_complex", fc,
        "-map", "[vout]", "-map", "[aout]",
        *v_args,
        "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace", "bt709", "-color_range", "tv",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ]
    label = "DRAFT (downsampled)" if draft else f"{height}p"
    print(f"Building cut.mp4: {len(sentences)} clips, hard-concat A+V, "
          f"{label} @ {fps}fps, {'HDR→BT.709' if tonemap else 'SDR'}")
    subprocess.run(cmd, check=True)
    print(f"Done: {out}")
    return out


def downsample_cut(name: str, *, height: int = LOW_RES_HEIGHT, force: bool = False) -> Path:
    """Downsample `cut.mp4` to 540p for faster image-placement iteration."""
    out = util_v2.cut_downsampled_path(name)
    if out.exists() and not force:
        print(f"cut_downsampled.mp4 exists (use --force to regenerate)")
        return out

    src = util_v2.cut_path(name)
    if not src.exists():
        raise FileNotFoundError(f"run crossfade-cut first: {src} missing")

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace", "bt709", "-color_range", "tv",
        "-c:a", "copy",
        str(out),
    ]
    print(f"Downsampling cut.mp4 → {height}p...")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Done: {out}")
    return out


def extract_cut_audio(name: str, *, force: bool = False) -> Path:
    """Extract MP3 from cut.mp4 for v2 transcription."""
    out = util_v2.cut_audio_path(name)
    if out.exists() and not force:
        return out
    src = util_v2.cut_path(name)
    if not src.exists():
        raise FileNotFoundError(f"run crossfade-cut first: {src} missing")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-c:a", "libmp3lame", "-b:a", "192k",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out
