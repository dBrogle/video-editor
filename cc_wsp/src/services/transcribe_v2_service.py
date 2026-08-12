"""Transcribe the v2 cut.mp4 audio (after crossfades) → transcription_v2.json.

Word/sentence timestamps are in the cut-video timeline, so downstream image
placement and captions can use them directly without remapping.
"""
from __future__ import annotations

from pathlib import Path

from cc_wsp.src import util_v2
from cc_wsp.src.services import crossfade_service
from cc_wsp.src.services.stt.deepgram import DeepgramSTTService


def transcribe_cut(name: str, *, force: bool = False) -> Path:
    out = util_v2.transcription_v2_path(name)
    if out.exists() and not force:
        print(f"transcription_v2.json exists (use --force to regenerate)")
        return out

    audio = crossfade_service.extract_cut_audio(name)
    print(f"Transcribing {audio.name}...")
    stt = DeepgramSTTService()
    transcript = stt.transcribe(audio)
    out.write_text(transcript.model_dump_json(indent=2))
    print(f"Saved: {out.name} ({len(transcript.sentences)} sentences)")
    return out
