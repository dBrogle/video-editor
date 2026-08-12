"""Captions on the v2 final render.

transcription_v2.json word timestamps are already in the cut-video timeline,
so this skips build_word_timeline (which v1 needs to map original-source word
times to the cut output) and feeds words straight into chunking.

Title-card sentence matching reuses extract_title_instructions by wrapping
transcription_v2 as a pseudo-AdjustedSentences (same trick as in
mlt_overlay_service).
"""
from __future__ import annotations

import json
from pathlib import Path

from cc_wsp.src import util, util_v2
from cc_wsp.src.constants import (
    CAPTION_Y_PERCENT, CAPTION_MAX_WORDS_PER_CHUNK, TITLE_CARD_DURATION,
)
from cc_wsp.src.models import Transcript
from cc_wsp.src.services.video.caption_service import (
    WordTiming, build_word_timeline as _v1_build,  # noqa: F401  (kept for future reuse)
    group_words_into_chunks, burn_captions, detect_title_text,
    extract_title_instructions, TitleCardConfig,
)
from cc_wsp.src.services.place_images_v2_service import _transcript_as_adjusted


def _words_from_v2_transcript(transcript: Transcript) -> list[WordTiming]:
    return [
        WordTiming(word=w.word, start=w.start, end=w.end)
        for s in transcript.sentences for w in s.words
    ]


def caption_v2(
    name: str,
    *,
    title: str | None = None,
    no_title: bool = False,
    caption_y: float | None = None,
    force: bool = False,
) -> Path:
    out = util_v2.captioned_v2_path(name)
    if out.exists() and not force:
        print(f"final_v2_captioned.mp4 exists (use --force to regenerate)")
        return out

    final = util_v2.final_v2_path(name)
    if not final.exists():
        raise FileNotFoundError(f"run render-v2 first: {final} missing")

    transcript = Transcript(**json.loads(util_v2.transcription_v2_path(name).read_text()))

    timeline = _words_from_v2_transcript(transcript)
    print(f"=> {len(timeline)} words in cut timeline")

    if caption_y is not None:
        y_pct = caption_y
    elif util.face_data_path(name).exists():
        face = util.load_face_data(name)
        y_pct = util.compute_caption_y_from_face(face)
        print(f"   Caption Y from face: {y_pct:.3f}")
    else:
        y_pct = CAPTION_Y_PERCENT

    chunks = group_words_into_chunks(timeline, max_words=CAPTION_MAX_WORDS_PER_CHUNK)
    print(f"=> {len(chunks)} caption chunks")

    title_config = None
    if not no_title:
        title_text = title
        if not title_text:
            try:
                script = util.load_google_doc_script(name)
                title_text = detect_title_text(script)
            except FileNotFoundError:
                pass
        if title_text:
            title_config = TitleCardConfig(text=title_text, start=0.0, end=TITLE_CARD_DURATION)
            print(f"   Title card: \"{title_text}\" (0.0-{TITLE_CARD_DURATION:.1f}s)")

    extra_titles = []
    try:
        script = util.load_google_doc_script(name)
        pseudo = _transcript_as_adjusted(transcript)
        extra_titles = extract_title_instructions(script, pseudo)
        for tc in extra_titles:
            print(f"   Title instruction: \"{tc.text[:50]}\" ({tc.start:.1f}-{tc.end:.1f}s)")
    except FileNotFoundError:
        pass

    raw_out = out.with_name(out.stem + "_raw.mp4")
    burn_captions(
        video_path=final,
        output_path=raw_out,
        chunks=chunks,
        title_config=title_config,
        title_cards=extra_titles,
        caption_y_percent=y_pct,
    )

    # burn_captions re-encodes through ffmpeg overlay and drops the bt709 tags;
    # remux with -c copy to set them back.
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw_out),
        "-c", "copy",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace", "bt709", "-color_range", "tv",
        str(out),
    ], check=True, capture_output=True)
    raw_out.unlink()
    print(f"Captioned: {out}")
    return out
