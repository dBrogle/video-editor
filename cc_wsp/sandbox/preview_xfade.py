"""Render a preview with tiny audio crossfades at each sentence boundary.

Video gets butt-joined hard cuts (same as preview.mp4).
Audio gets `acrossfade` chained between adjacent clips so cuts smooth instead of click.

Usage: python cc_wsp/sandbox/preview_xfade.py <video_name> [--xfade 0.05]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--xfade", type=float, default=0.05, help="Crossfade duration in seconds (default 0.05)")
    args = ap.parse_args()

    video_dir = Path(__file__).resolve().parents[1] / "videos" / args.video
    adjusted = json.loads((video_dir / "adjusted.json").read_text())
    sents = adjusted["sentences"]
    source = video_dir / "downsampled.mp4"
    out = video_dir / f"preview_xfade.mp4"

    if not source.exists():
        sys.exit(f"missing {source}")
    if not sents:
        sys.exit("no sentences")

    xf = args.xfade
    half = xf / 2
    n = len(sents)

    # Pad audio bounds so adjacent clips overlap by `xf` seconds.
    # No prepad on first, no postpad on last — keeps total audio == total video.
    parts = []
    for i, s in enumerate(sents):
        st, en = s["adjusted_start"], s["adjusted_end"]
        a_st = st if i == 0 else max(0.0, st - half)
        a_en = en if i == n - 1 else en + half
        parts.append(f"[0:v]trim=start={st}:end={en},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={a_st}:end={a_en},asetpts=PTS-STARTPTS[a{i}]")

    v_in = "".join(f"[v{i}]" for i in range(n))
    parts.append(f"{v_in}concat=n={n}:v=1:a=0[vout]")

    prev = "a0"
    for i in range(1, n):
        label = "aout" if i == n - 1 else f"ax{i}"
        parts.append(f"[{prev}][a{i}]acrossfade=d={xf}[{label}]")
        prev = label

    fc = ";".join(parts)
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-filter_complex", fc,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ]
    print(f"Rendering {out.name} with {xf*1000:.0f}ms audio crossfades across {n} cuts...")
    subprocess.run(cmd, check=True)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
