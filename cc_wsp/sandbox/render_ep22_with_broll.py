#!/usr/bin/env python3
"""
Render bcai_top_habits episode 22 with portrait b-roll overlay.

B-roll plays full-screen during parts of the video where elite performers
are being discussed, replacing the main video briefly.
"""
import sys
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cc_wsp.src import util
from cc_wsp.src.services.video.mlt_util import (
    get_video_properties, create_mlt_root_and_profile, frames_to_timecode,
    add_black_producer, save_pretty_xml,
)

NAME = "bcai_top_habits_episodes/bcai_top_habits_22"
EP_DIR = util.get_video_dir(NAME)
VIDEO_PATH = util.downsampled_1080p_path(NAME)
BROLL_PATH = EP_DIR / "top_performers_broll.mp4"
OUTPUT_PATH = EP_DIR / "final.mp4"
MLT_PATH = EP_DIR / "final_mlt.mlt"

# Load adjusted sentences and compute output timeline
adj = util.load_adjusted(NAME)
timeline = []
cumulative = 0.0
for s in adj.sentences:
    dur = s.adjusted_end - s.adjusted_start
    timeline.append({
        "index": s.index,
        "text": s.text,
        "out_start": cumulative,
        "out_end": cumulative + dur,
        "src_start": s.adjusted_start,
        "src_end": s.adjusted_end,
    })
    cumulative += dur

total_duration = cumulative

# Define b-roll placements: (output_start, output_end) in the final timeline
# B-roll during: "top habits of 85 elite performers" part of sentence 1
# Sentence 1 runs 0.00-8.55s. The phrase "85 elite performers" is roughly
# in the middle. Let's show b-roll from ~2.5s to ~6.5s (4 seconds).
# Also a quick flash during sentence 3 (Shaq) and 7 (Branson) transitions.
broll_placements = [
    # During "85 elite performers to see which ones were the most common"
    {"out_start": 2.5, "out_end": 7.0, "broll_start": 0.0},
]

print("Output timeline:")
for t in timeline:
    print(f"  [{t['out_start']:.2f}-{t['out_end']:.2f}] idx={t['index']} {t['text'][:60]}...")
print(f"\nTotal: {total_duration:.2f}s")
print(f"\nB-roll placements:")
for bp in broll_placements:
    print(f"  [{bp['out_start']:.2f}-{bp['out_end']:.2f}] (broll from {bp['broll_start']:.1f}s)")

# Build MLT XML
props = get_video_properties(VIDEO_PATH)
fps = props["fps"]
root = create_mlt_root_and_profile(props)

total_frames = int(total_duration * fps)
total_tc = frames_to_timecode(total_frames, fps)

# Black background
add_black_producer(root, total_tc)

# Background playlist wrapping black producer
bg_playlist = ET.SubElement(root, "playlist", {"id": "background"})
ET.SubElement(bg_playlist, "entry", {
    "producer": "black",
    "in": "00:00:00.000",
    "out": total_tc,
})

# Create chains for each video clip
for i, sent in enumerate(adj.sentences):
    clip_dur = sent.adjusted_end - sent.adjusted_start
    in_frame = int(sent.adjusted_start * fps)
    out_frame = int(sent.adjusted_end * fps) - 1

    chain = ET.SubElement(root, "chain", {
        "id": f"chain_clip_{i}",
        "in": frames_to_timecode(in_frame, fps),
        "out": frames_to_timecode(out_frame, fps),
    })
    ET.SubElement(chain, "property", {"name": "length"}).text = frames_to_timecode(int(clip_dur * fps), fps)
    ET.SubElement(chain, "property", {"name": "eof"}).text = "pause"
    ET.SubElement(chain, "property", {"name": "resource"}).text = str(VIDEO_PATH)
    ET.SubElement(chain, "property", {"name": "mlt_service"}).text = "avformat-novalidate"
    ET.SubElement(chain, "property", {"name": "seekable"}).text = "1"
    ET.SubElement(chain, "property", {"name": "audio_index"}).text = "1"
    ET.SubElement(chain, "property", {"name": "video_index"}).text = "0"
    ET.SubElement(chain, "property", {"name": "mute_on_pause"}).text = "0"

# Video playlist (playlist0)
playlist0 = ET.SubElement(root, "playlist", {"id": "playlist0"})
ET.SubElement(playlist0, "property", {"name": "shotcut:video"}).text = "1"
ET.SubElement(playlist0, "property", {"name": "shotcut:name"}).text = "V1"

for i, sent in enumerate(adj.sentences):
    clip_dur = sent.adjusted_end - sent.adjusted_start
    clip_frames = int(clip_dur * fps) - 1
    ET.SubElement(playlist0, "entry", {
        "producer": f"chain_clip_{i}",
        "in": "00:00:00.000",
        "out": frames_to_timecode(clip_frames, fps),
    })

# B-roll producer chain(s)
broll_props = get_video_properties(BROLL_PATH)
for j, bp in enumerate(broll_placements):
    broll_dur = bp["out_end"] - bp["out_start"]
    broll_in_frame = int(bp["broll_start"] * fps)
    broll_out_frame = int((bp["broll_start"] + broll_dur) * fps) - 1

    chain = ET.SubElement(root, "chain", {
        "id": f"chain_broll_{j}",
        "in": frames_to_timecode(broll_in_frame, fps),
        "out": frames_to_timecode(broll_out_frame, fps),
    })
    ET.SubElement(chain, "property", {"name": "length"}).text = frames_to_timecode(int(broll_dur * fps), fps)
    ET.SubElement(chain, "property", {"name": "eof"}).text = "pause"
    ET.SubElement(chain, "property", {"name": "resource"}).text = str(BROLL_PATH)
    ET.SubElement(chain, "property", {"name": "mlt_service"}).text = "avformat-novalidate"
    ET.SubElement(chain, "property", {"name": "seekable"}).text = "1"
    ET.SubElement(chain, "property", {"name": "audio_index"}).text = "-1"  # no audio from b-roll
    ET.SubElement(chain, "property", {"name": "video_index"}).text = "0"
    ET.SubElement(chain, "property", {"name": "mute_on_pause"}).text = "1"

# B-roll overlay playlist (playlist1) - blanks + entries
playlist1 = ET.SubElement(root, "playlist", {"id": "playlist1"})
ET.SubElement(playlist1, "property", {"name": "shotcut:video"}).text = "1"
ET.SubElement(playlist1, "property", {"name": "shotcut:name"}).text = "V2"

cursor = 0.0
for j, bp in enumerate(broll_placements):
    # Blank before this b-roll
    gap = bp["out_start"] - cursor
    if gap > 0:
        gap_frames = int(gap * fps) - 1
        ET.SubElement(playlist1, "blank", {"length": frames_to_timecode(gap_frames, fps)})

    # B-roll entry
    broll_dur = bp["out_end"] - bp["out_start"]
    broll_frames = int(broll_dur * fps) - 1
    ET.SubElement(playlist1, "entry", {
        "producer": f"chain_broll_{j}",
        "in": "00:00:00.000",
        "out": frames_to_timecode(broll_frames, fps),
    })
    cursor = bp["out_end"]

# Tractor - ties it all together
tractor = ET.SubElement(root, "tractor", {
    "id": "tractor0",
    "title": "Shotcut version 22.12.21",
    "in": "00:00:00.000",
    "out": total_tc,
})
ET.SubElement(tractor, "property", {"name": "shotcut"}).text = "1"
ET.SubElement(tractor, "property", {"name": "shotcut:projectAudioChannels"}).text = "2"

# Tracks
ET.SubElement(tractor, "track", {"producer": "background"})
ET.SubElement(tractor, "track", {"producer": "playlist0"})
ET.SubElement(tractor, "track", {"producer": "playlist1"})

# Transitions: background <-> video
mix0 = ET.SubElement(tractor, "transition", {"id": "transition0"})
ET.SubElement(mix0, "property", {"name": "a_track"}).text = "0"
ET.SubElement(mix0, "property", {"name": "b_track"}).text = "1"
ET.SubElement(mix0, "property", {"name": "mlt_service"}).text = "mix"
ET.SubElement(mix0, "property", {"name": "always_active"}).text = "1"
ET.SubElement(mix0, "property", {"name": "sum"}).text = "1"

fader0 = ET.SubElement(tractor, "transition", {"id": "transition1"})
ET.SubElement(fader0, "property", {"name": "a_track"}).text = "0"
ET.SubElement(fader0, "property", {"name": "b_track"}).text = "1"
ET.SubElement(fader0, "property", {"name": "version"}).text = "0.1"
ET.SubElement(fader0, "property", {"name": "mlt_service"}).text = "frei0r.cairoblend"
ET.SubElement(fader0, "property", {"name": "disable"}).text = "0"

# Transitions: b-roll overlay composited on top (full screen, opaque)
mix1 = ET.SubElement(tractor, "transition", {"id": "transition2"})
ET.SubElement(mix1, "property", {"name": "a_track"}).text = "0"
ET.SubElement(mix1, "property", {"name": "b_track"}).text = "2"
ET.SubElement(mix1, "property", {"name": "mlt_service"}).text = "mix"
ET.SubElement(mix1, "property", {"name": "always_active"}).text = "1"
ET.SubElement(mix1, "property", {"name": "sum"}).text = "1"

comp1 = ET.SubElement(tractor, "transition", {"id": "transition3"})
ET.SubElement(comp1, "property", {"name": "a_track"}).text = "1"
ET.SubElement(comp1, "property", {"name": "b_track"}).text = "2"
ET.SubElement(comp1, "property", {"name": "version"}).text = "0.1"
ET.SubElement(comp1, "property", {"name": "mlt_service"}).text = "frei0r.cairoblend"
ET.SubElement(comp1, "property", {"name": "disable"}).text = "0"

# Save MLT XML
save_pretty_xml(root, MLT_PATH)
print(f"\nMLT XML: {MLT_PATH}")

# Render
print("Rendering with melt...")
cmd = [
    "melt", str(MLT_PATH),
    "-consumer", f"avformat:{OUTPUT_PATH}",
    "vcodec=libx264", "acodec=aac", "crf=23",
    "preset=medium", "pix_fmt=yuv420p",
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"STDERR: {result.stderr[-500:]}")
    raise RuntimeError("melt failed")

print(f"Done! Final video: {OUTPUT_PATH}")
