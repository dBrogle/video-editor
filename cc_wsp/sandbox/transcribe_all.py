#!/usr/bin/env python3
"""Transcribe all bcai_top_habits clips via Deepgram."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cc_wsp.src.services.stt.deepgram import DeepgramSTTService

stt = DeepgramSTTService()
base = Path("cc_wsp/videos/bcai_top_habits/transcripts")

clips = ["IMG_4038", "IMG_4039", "IMG_4040", "IMG_4041", "IMG_4042", "IMG_4043", "IMG_4044"]

for clip in clips:
    audio = base / f"{clip}.mp3"
    out = base / f"{clip}.json"
    if out.exists():
        print(f"{clip}: already done")
        continue
    print(f"Transcribing {clip}...")
    transcript = stt.transcribe(audio)
    out.write_text(transcript.model_dump_json(indent=2))
    print(f"  -> {len(transcript.sentences)} sentences")

print("All done!")
