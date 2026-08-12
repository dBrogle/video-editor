#!/usr/bin/env python3
"""Chunked Deepgram transcription that survives a dropped connection.

`tool.py stream-transcribe` holds every chunk in memory and writes nothing until
the last one lands, so one "connection reset by peer" three chunks into a 2-hour
source throws the whole run away. This caches each chunk's JSON to disk and
retries, so a rerun resumes instead of restarting.

Usage:
  python cc_wsp/sandbox/longform_transcribe.py <video_name> [--chunk-minutes 20]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cc_wsp.src import util
from cc_wsp.src.models import LLMTranscriptSentence, Transcript
from cc_wsp.src.services.stt.deepgram import DeepgramSTTService

RETRIES = 4


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def _transcribe_chunk(stt, audio: Path, cache: Path):
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(1, RETRIES + 1):
        try:
            transcript = stt.transcribe(audio)
            break
        except Exception as e:  # network resets, upload timeouts
            if attempt == RETRIES:
                raise
            wait = 5 * attempt
            print(f"    attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    data = json.loads(transcript.model_dump_json())
    cache.write_text(json.dumps(data))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--chunk-minutes", type=float, default=20.0)
    args = ap.parse_args()

    folder = util.get_video_dir(args.video)
    audio = util.stream_audio_path(args.video)
    out_path = util.stream_transcription_path(args.video)
    total = _duration(audio)
    chunk_sec = args.chunk_minutes * 60
    n = int(total // chunk_sec) + (1 if total % chunk_sec else 0)

    cache_dir = folder / "_chunks"
    cache_dir.mkdir(exist_ok=True)
    stt = DeepgramSTTService()

    sentences: list[LLMTranscriptSentence] = []
    for i in range(n):
        start = i * chunk_sec
        dur = min(chunk_sec, total - start)
        cache = cache_dir / f"chunk_{i:02d}.json"
        print(f"  chunk {i + 1}/{n}: {start / 60:.0f}-{(start + dur) / 60:.0f} min"
              + (" (cached)" if cache.exists() else ""))

        if not cache.exists():
            piece = cache_dir / f"chunk_{i:02d}.mp3"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-ss", str(start),
                 "-t", str(dur), "-acodec", "copy", str(piece)], check=True)
            _transcribe_chunk(stt, piece, cache)
            piece.unlink()

        data = json.loads(cache.read_text())
        for s in data["sentences"]:
            s["start"] += start
            s["end"] += start
            for w in s["words"]:
                w["start"] += start
                w["end"] += start
            sentences.append(LLMTranscriptSentence(**s))

    transcript = Transcript(sentences=sentences, language="en", duration=total)
    util.save_stream_transcription(args.video, transcript)
    speech = sum(s.end - s.start for s in sentences)
    print(f"wrote {out_path.name}: {len(sentences)} sentences, "
          f"{speech / 60:.1f} min speech / {total / 60:.1f} min source")


if __name__ == "__main__":
    main()
