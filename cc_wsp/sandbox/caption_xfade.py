"""Replace the audio of an already-rendered captioned video with a crossfaded
audio chain extracted from the source.

Same approach as preview_xfade.py but applied to a video where the visuals are
already baked in (MLT cuts + image overlays + captions). Note: MLT's per-clip
frame count doesn't match int((end-start)*fps), so per-clip A/V drift is
expected. apad pads the audio total to match the video. Use to A/B test the
audible benefit of the crossfade vs the lip-sync cost.

Usage: python cc_wsp/sandbox/caption_xfade.py <video_name> [--xfade 0.05]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--xfade", type=float, default=0.05)
    args = ap.parse_args()

    video_dir = Path(__file__).resolve().parents[1] / "videos" / args.video
    adjusted = json.loads((video_dir / "adjusted.json").read_text())
    sents = adjusted["sentences"]
    captioned = video_dir / "final_captioned.mp4"
    source = video_dir / "1080p.mp4"
    out = video_dir / "final_captioned_crossfade.mp4"

    for p in (captioned, source):
        if not p.exists():
            sys.exit(f"missing {p}")
    if not sents:
        sys.exit("no sentences")

    video_dur = float(
        subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(captioned),
            ],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )

    xf = args.xfade
    half = xf / 2
    n = len(sents)

    parts = []
    for i, s in enumerate(sents):
        st, en = s["adjusted_start"], s["adjusted_end"]
        a_st = st if i == 0 else max(0.0, st - half)
        a_en = en if i == n - 1 else en + half
        parts.append(f"[1:a]atrim=start={a_st}:end={a_en},asetpts=PTS-STARTPTS[a{i}]")

    prev = "a0"
    for i in range(1, n):
        tag = f"ax{i}"
        parts.append(f"[{prev}][a{i}]acrossfade=d={xf}[{tag}]")
        prev = tag
    parts.append(f"[{prev}]apad=whole_dur={video_dur}[aout]")

    fc = ";".join(parts)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(captioned),
        "-i", str(source),
        "-filter_complex", fc,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ]
    print(f"Building {out.name} with {xf*1000:.0f}ms crossfades across {n-1} cuts...")
    subprocess.run(cmd, check=True)

    # BT.709 color tag preservation (in case captioned was tagged)
    tmp = out.with_suffix(".raw.mp4")
    out.rename(tmp)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(tmp),
            "-c", "copy",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-color_range", "tv",
            str(out),
        ],
        check=True, capture_output=True,
    )
    tmp.unlink()

    print(f"Done: {out}")


if __name__ == "__main__":
    main()
