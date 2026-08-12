#!/usr/bin/env python3
"""Create a section sub-project from the master automation_obs transcription.

The 75-min OBS walkthrough is edited in sections (build steps). Each section
becomes a standalone normal project sourced from a re-encoded clip of just that
span, with transcript timestamps rebased to local time — so the v2 pipeline only
ever decodes that ~5-10 min segment instead of the full 75 min.

Usage:
  python cc_wsp/sandbox/make_obs_section.py <section_name> <start_idx> <end_idx>

Example:
  python cc_wsp/sandbox/make_obs_section.py s1_signup 0 24
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YT = ROOT / "cc_wsp" / "videos" / "youtube"
MASTER = YT / "automation_obs"
FULL_MOV = MASTER / "automation_obs.mov"  # symlink to the full OBS recording
PAD = 1.5  # seconds of headroom each side of the section


def main():
    name, start_idx, end_idx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    master = json.loads((MASTER / "transcription.json").read_text())
    sents = master["sentences"]
    section = sents[start_idx:end_idx + 1]
    if not section:
        raise SystemExit("empty section range")

    extract_start = max(0.0, float(section[0]["start"]) - PAD)
    extract_end = float(section[-1]["end"]) + PAD

    proj = f"automation_obs_{name}"
    proj_dir = YT / proj
    proj_dir.mkdir(parents=True, exist_ok=True)
    local_mov = proj_dir / f"{proj}.mov"
    local_audio = proj_dir / "audio.mp3"

    # 1) Re-encode just this span (frame-accurate; keep BT.709 tags).
    print(f"Extracting {extract_start:.2f}s .. {extract_end:.2f}s "
          f"({extract_end - extract_start:.1f}s) -> {local_mov.name}")
    subprocess.run([
        "ffmpeg", "-y", "-ss", f"{extract_start:.3f}", "-to", f"{extract_end:.3f}",
        "-i", str(FULL_MOV),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace", "bt709", "-color_range", "tv", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", str(local_mov),
    ], check=True, capture_output=True)

    # 2) Extract local audio for silence-detect.
    subprocess.run([
        "ffmpeg", "-y", "-i", str(local_mov), "-vn",
        "-acodec", "libmp3lame", "-ar", "44100", "-ac", "1", "-b:a", "128k",
        str(local_audio),
    ], check=True, capture_output=True)

    # 3) Rebased + renumbered transcription.json (local time, 1-based indices).
    def rebase(t):
        return round(float(t) - extract_start, 3)

    new_sents = []
    for s in section:
        ws = [{"word": w["word"], "start": rebase(w["start"]), "end": rebase(w["end"])}
              for w in s.get("words", [])]
        new_sents.append({
            "sentence": s["sentence"],
            "start": rebase(s["start"]),
            "end": rebase(s["end"]),
            "words": ws,
        })
    out = {
        "sentences": new_sents,
        "language": master.get("language", "en"),
        "duration": rebase(section[-1]["end"]),
    }
    (proj_dir / "transcription.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {proj}: {len(new_sents)} sentences "
          f"(master idx {start_idx}-{end_idx}), local 0..{out['duration']:.1f}s")
    print(f"Project name for tool.py: youtube/{proj}")


if __name__ == "__main__":
    main()
