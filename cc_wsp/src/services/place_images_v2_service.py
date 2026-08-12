"""Image placement on the v2 cut timeline.

Reuses the existing GoogleDocImagePlacer LLM agent — just feeds it sentences
built from transcription_v2.json instead of adjusted.json. The agent's output
schema (sentence_index, start_fraction, end_fraction) maps cleanly to the
new sentence indices.
"""
from __future__ import annotations

import json
from pathlib import Path

from cc_wsp.src import util, util_v2
from cc_wsp.src.models import (
    AdjustedSentence, AdjustedSentences, Transcript, WordTimestamp,
)
from cc_wsp.src.services.agents.google_doc_image_placer import GoogleDocImagePlacer


def _transcript_as_adjusted(transcript: Transcript) -> AdjustedSentences:
    """Wrap a Transcript as AdjustedSentences so the existing LLM agent can
    consume it. Sentence index becomes the 1-based position; start/end are
    already in the cut timeline."""
    sents = []
    for i, s in enumerate(transcript.sentences, start=1):
        sents.append(AdjustedSentence(
            original_start=s.start,
            original_end=s.end,
            adjusted_start=s.start,
            adjusted_end=s.end,
            text=s.sentence,
            index=str(i),
            threshold_source="cut",
            words=[WordTimestamp(word=w.word, start=w.start, end=w.end) for w in s.words],
        ))
    return AdjustedSentences(sentences=sents)


def place_images_v2(name: str, *, force: bool = False) -> Path:
    out = util_v2.images_v2_path(name)
    if out.exists() and not force:
        print(f"images_v2.json exists (use --force to regenerate)")
        return out

    transcript = Transcript(**json.loads(util_v2.transcription_v2_path(name).read_text()))
    script = util.load_google_doc_script(name)
    img_dir = util.google_doc_images_dir(name)

    pseudo = _transcript_as_adjusted(transcript)
    agent = GoogleDocImagePlacer()
    placements = agent.place_images(
        google_doc_script=script,
        adjusted_sentences=pseudo,
        google_doc_images_folder=img_dir,
    )
    out.write_text(placements.model_dump_json(indent=2))
    print(f"Saved: {out.name} ({len(placements.placements)} placements)")
    return out
