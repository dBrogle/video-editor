#!/usr/bin/env python3
"""First-pass base cut for long-form OBS walkthroughs.

Removes only (1) long silence and (2) retakes/abandoned fragments — the user
does the tight final cut on top of this. Clips get generous lead/trail padding
into the surrounding silence (never bleeding into a neighbouring dropped take),
overlapping clips are merged, and the result is rendered at native res with no
crossfades.

Usage:
  python cc_wsp/sandbox/longform_base_cut.py <video_name> <drop_json> [--dry-run]

drop_json: {"drop": [12, 13, 40], "notes": {...}}  — sentence indices (0-based)
           to remove from stream_transcription.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cc_wsp.src import util

# Padding caps (seconds). Deepgram onsets run ~0.15-0.2s late, so always take a
# lead; BUFFER keeps padding from touching a neighbouring (possibly dropped) take.
LEAD_CAP = 0.30
TRAIL_CAP = 0.55
BUFFER = 0.05

# At a retake boundary the discarded take can butt straight against the kept one
# (zero gap), and cutting on the transcript boundary clips the first phoneme.
# Take at least this much even if it bleeds a sliver of the discarded take —
# a hair of the neighbour's tail beats a chopped word.
MIN_LEAD = 0.10
MIN_TRAIL = 0.15

# Intra-sentence trims cut against the discarded attempt rather than silence, so
# the padding floors have to be smaller — the neighbouring word is speech, not
# room tone, and a 0.10s bleed there is an audible syllable.
TRIM_MIN_LEAD = 0.06
TRIM_MIN_TRAIL = 0.10


def build_intervals(sents, keep_idx, duration, trims=None):
    """Pad each kept sentence into its surrounding silence, then merge overlaps.

    `trims` maps a sentence index to {"from_word": k} / {"to_word": k} — Deepgram
    often merges several attempts of a line into ONE sentence (no pause to split
    on), so dropping whole sentences can't remove those retakes. A trim re-anchors
    the clip boundary to a word inside the sentence; padding is then capped against
    the adjacent (discarded) word instead of the neighbouring sentence.
    """
    trims = trims or {}
    intervals = []
    for i in keep_idx:
        s = sents[i]
        trim = trims.get(str(i)) or trims.get(i) or {}
        words = s.get("words") or []

        if "from_word" in trim:
            w = words[trim["from_word"]]
            prev_w_end = words[trim["from_word"] - 1]["end"] if trim["from_word"] > 0 else s["start"]
            lead = max(TRIM_MIN_LEAD, min(LEAD_CAP, (w["start"] - prev_w_end) - BUFFER))
            start = w["start"] - lead
        else:
            prev_end = sents[i - 1]["end"] if i > 0 else 0.0
            lead = max(MIN_LEAD, min(LEAD_CAP, (s["start"] - prev_end) - BUFFER))
            start = s["start"] - lead

        if "to_word" in trim:
            w = words[trim["to_word"]]
            nxt = words[trim["to_word"] + 1]["start"] if trim["to_word"] + 1 < len(words) else s["end"]
            trail = max(TRIM_MIN_TRAIL, min(TRAIL_CAP, (nxt - w["end"]) - BUFFER))
            end = w["end"] + trail
        else:
            next_start = sents[i + 1]["start"] if i + 1 < len(sents) else duration
            trail = max(MIN_TRAIL, min(TRAIL_CAP, (next_start - s["end"]) - BUFFER))
            end = s["end"] + trail

        intervals.append([round(max(0.0, start), 3), round(min(duration, end), 3)])

    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


ENCODE = [
    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
    "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
    "-color_range", "tv", "-r", "30",
    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
]


def _encode_segment(job):
    """Encode one clip. Input -ss + re-encode is frame-accurate (ffmpeg decodes
    from the preceding keyframe and discards). The concat demuxer's inpoint would
    snap to a keyframe instead — on OBS's sparse-keyframe h264 that can shift a
    cut by seconds and drag a discarded retake back in."""
    i, src, start, end, seg_dir = job
    seg = seg_dir / f"seg_{i:04d}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-i", str(src), *ENCODE, str(seg)],
        check=True)
    return seg


def render(src: Path, intervals, out: Path):
    """Encode each kept span frame-accurately, then concat losslessly."""
    from concurrent.futures import ThreadPoolExecutor

    seg_dir = out.parent / "_segments"
    seg_dir.mkdir(exist_ok=True)
    jobs = [(i, src, s, e, seg_dir) for i, (s, e) in enumerate(intervals)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        segs = list(pool.map(_encode_segment, jobs))
    print(f"encoded {len(segs)} segments")

    listfile = seg_dir / "concat.txt"
    listfile.write_text("\n".join(f"file '{s.resolve()}'" for s in segs) + "\n")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(listfile), "-c", "copy", "-movflags", "+faststart", str(out)],
        check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("drop_json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folder = util.get_video_dir(args.video)
    src = util.get_input_video_path(args.video)
    sents = json.loads((folder / "stream_transcription.json").read_text())["sentences"]
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(src)],
        capture_output=True, text=True, check=True).stdout.strip())

    spec = json.loads(Path(args.drop_json).read_text())
    drop = set(spec["drop"])
    trims = spec.get("trim", {})
    keep_idx = [i for i in range(len(sents)) if i not in drop]
    intervals = build_intervals(sents, keep_idx, duration, trims)
    total = sum(e - s for s, e in intervals)

    print(f"source     : {duration / 60:.1f} min")
    print(f"sentences  : {len(sents)} ({len(drop)} dropped, {len(keep_idx)} kept)")
    print(f"clips      : {len(intervals)}")
    print(f"cut length : {total / 60:.1f} min ({100 * total / duration:.0f}% of source)")

    (folder / "base_cut_clips.json").write_text(json.dumps(
        {"lead_cap": LEAD_CAP, "trail_cap": TRAIL_CAP, "intervals": intervals}, indent=1))

    # Proofread artifact: exactly the words that survive, in order, so trims can be
    # sanity-checked without listening to the render.
    lines = []
    for i in keep_idx:
        s = sents[i]
        trim = trims.get(str(i)) or trims.get(i) or {}
        words = s.get("words") or []
        a = trim.get("from_word", 0)
        b = trim.get("to_word", len(words) - 1)
        text = " ".join(w["word"] for w in words[a:b + 1]) if words else s["sentence"]
        lines.append(f"{i:4d}  {text}")
    (folder / "base_cut_script.txt").write_text("\n".join(lines) + "\n")

    if args.dry_run:
        return

    out = folder / f"{Path(args.video).name}_cut.mp4"
    render(src, intervals, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
