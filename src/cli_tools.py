"""
CLI tools for Claude Code (or human) interaction with the video editing pipeline.
Provides audio analysis, word timestamps, sentence splitting, and preview generation.
"""

import json
import argparse

import numpy as np
import librosa

from src.constants import ASSETS_DIR
from src.models import AdjustedSentences, AdjustedSentence, WordTimestamp
from src.services.local_saver import LocalSaverService
from src.services.video.video_service import VideoService
from src.util import (
    get_audio_path,
    get_adjusted_sentences_path,
    get_final_adjusted_sentences_path,
    get_adjusted_sentences_video_path,
    validate_file_exists,
)


def _load_adjusted_sentences(base_name: str, source: str = "s5") -> AdjustedSentences:
    if source == "s6":
        path = get_final_adjusted_sentences_path(base_name)
    else:
        path = get_adjusted_sentences_path(base_name)
    if not path.exists():
        raise FileNotFoundError(f"Adjusted sentences not found: {path}")
    data = json.loads(path.read_text())
    return AdjustedSentences(**data)


def _save_adjusted_sentences(base_name: str, sentences: AdjustedSentences, target: str = "s5") -> str:
    if target == "s6":
        path = get_final_adjusted_sentences_path(base_name)
    else:
        path = get_adjusted_sentences_path(base_name)
    path.write_text(sentences.model_dump_json(indent=2))
    return str(path)


def tool_audio_levels(base_name: str, start: float, end: float, resolution: float = 0.01) -> None:
    """Print audio RMS levels for a time range, useful for finding silence boundaries."""
    audio_path = get_audio_path(base_name)
    validate_file_exists(audio_path)

    sr = 22050
    audio_array, sr = librosa.load(
        str(audio_path), sr=sr, mono=True, offset=start, duration=end - start
    )

    frame_length = 512
    hop_length = 256
    time_per_frame = hop_length / sr

    rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Calculate speech level and threshold
    speech_level_db = np.percentile(rms_db, 85)
    silence_threshold = speech_level_db - 15

    print(f"Audio levels for {base_name}: {start:.3f}s - {end:.3f}s")
    print(f"Speech level (85th pct): {speech_level_db:.1f} dB")
    print(f"Silence threshold: {silence_threshold:.1f} dB")
    print(f"Resolution: {resolution}s")
    print()

    # Downsample to requested resolution
    frames_per_step = max(1, int(resolution / time_per_frame))
    print(f"{'Time':>8s}  {'dB':>6s}  {'Bar'}")
    print("-" * 50)

    for i in range(0, len(rms_db), frames_per_step):
        chunk = rms_db[i:i + frames_per_step]
        db_val = float(np.mean(chunk))
        t = start + i * time_per_frame
        # Visual bar: map -80..0 dB to 0..40 chars
        bar_len = max(0, int((db_val + 80) / 2))
        is_speech = db_val > silence_threshold
        marker = "#" if is_speech else "."
        print(f"{t:8.3f}  {db_val:6.1f}  {marker * bar_len}")


def tool_word_timestamps(base_name: str, sentence_index: int) -> None:
    """Print word-level timestamps for a sentence from the transcription."""
    saver = LocalSaverService()
    transcript = saver.load_transcription(base_name)

    if sentence_index < 1 or sentence_index > len(transcript.sentences):
        print(f"Error: sentence index {sentence_index} out of range (1-{len(transcript.sentences)})")
        return

    sentence = transcript.sentences[sentence_index - 1]
    print(f"Sentence {sentence_index}: \"{sentence.sentence}\"")
    print(f"Sentence time: {sentence.start:.3f}s - {sentence.end:.3f}s")
    print()

    if not sentence.words:
        print("No word-level timestamps available for this sentence.")
        return

    print(f"{'Word':<20s}  {'Start':>8s}  {'End':>8s}  {'Duration':>8s}  {'Gap after':>9s}")
    print("-" * 65)

    for i, word in enumerate(sentence.words):
        duration = word.end - word.start
        gap = ""
        if i < len(sentence.words) - 1:
            next_word = sentence.words[i + 1]
            gap_val = next_word.start - word.end
            gap = f"{gap_val:.3f}s"
            if gap_val > 0.3:
                gap += " ***"  # Flag large gaps
        print(f"{word.word:<20s}  {word.start:8.3f}  {word.end:8.3f}  {duration:8.3f}  {gap:>9s}")


def tool_split_sentence(base_name: str, sentence_index: str, split_time: float, target: str = "s5") -> None:
    """Split a sentence in adjusted_sentences at a given timestamp."""
    sentences = _load_adjusted_sentences(base_name, source=target)

    # Find the sentence by index
    target_idx = None
    for i, s in enumerate(sentences.sentences):
        if s.index == sentence_index:
            target_idx = i
            break

    if target_idx is None:
        print(f"Error: sentence with index '{sentence_index}' not found")
        print(f"Available indices: {[s.index for s in sentences.sentences]}")
        return

    s = sentences.sentences[target_idx]

    if split_time <= s.adjusted_start or split_time >= s.adjusted_end:
        print(f"Error: split_time {split_time} is outside sentence range [{s.adjusted_start:.3f}, {s.adjusted_end:.3f}]")
        return

    # Load word timestamps from transcription to split text
    # The adjusted sentence index doesn't always match the transcript index
    # (sentences may have been removed in editing). Match by timestamp overlap.
    saver = LocalSaverService()
    transcript = saver.load_transcription(base_name)

    words = []
    for ts in transcript.sentences:
        if ts.start <= s.original_start + 0.1 and ts.end >= s.original_end - 0.1:
            words = ts.words
            break
    if not words:
        # Fallback: find sentence with most timestamp overlap
        best_overlap = 0
        for ts in transcript.sentences:
            overlap = min(ts.end, s.original_end) - max(ts.start, s.original_start)
            if overlap > best_overlap:
                best_overlap = overlap
                words = ts.words

    # Split words into two groups based on split_time
    words_before = []
    words_after = []
    text_before = ""
    text_after = ""

    if words:
        for w in words:
            mid = (w.start + w.end) / 2
            if mid < split_time:
                words_before.append(w)
            else:
                words_after.append(w)
        text_before = " ".join(w.word for w in words_before) if words_before else s.text
        text_after = " ".join(w.word for w in words_after) if words_after else ""
    else:
        text_before = s.text + " [part 1]"
        text_after = s.text + " [part 2]"

    # Create two new sentences
    s1 = AdjustedSentence(
        original_start=s.original_start,
        original_end=s.original_end,
        adjusted_start=s.adjusted_start,
        adjusted_end=split_time,
        text=text_before,
        index=s.index,
        threshold_source=s.threshold_source,
        words=[WordTimestamp(word=w.word, start=w.start, end=w.end) for w in words_before],
    )
    s2 = AdjustedSentence(
        original_start=s.original_start,
        original_end=s.original_end,
        adjusted_start=split_time,
        adjusted_end=s.adjusted_end,
        text=text_after,
        index=f"{s.index}b",
        threshold_source=s.threshold_source,
        words=[WordTimestamp(word=w.word, start=w.start, end=w.end) for w in words_after],
    )

    # Replace original with the two parts
    sentences.sentences[target_idx:target_idx + 1] = [s1, s2]

    path = _save_adjusted_sentences(base_name, sentences, target=target)
    print(f"Split sentence {sentence_index} at {split_time:.3f}s")
    print(f"  Part 1 ({s1.index}): {s1.adjusted_start:.3f} - {s1.adjusted_end:.3f}  \"{s1.text}\"")
    print(f"  Part 2 ({s2.index}): {s2.adjusted_start:.3f} - {s2.adjusted_end:.3f}  \"{s2.text}\"")
    print(f"Saved to: {path}")


def tool_preview(base_name: str, source: str = "s5") -> None:
    """Generate a preview video from adjusted sentences."""
    sentences = _load_adjusted_sentences(base_name, source=source)
    video_service = VideoService(ASSETS_DIR)
    output_path = get_adjusted_sentences_video_path(base_name)

    video_service.create_edited_video(
        base_name=base_name,
        adjusted_sentences=sentences,
        use_downsampled=True,
        force=True,
        output_path=output_path,
    )
    print(f"Preview video created: {output_path}")


def run_tool(args: argparse.Namespace) -> None:
    """Dispatch CLI tool commands."""
    if args.command == "run":
        steps = [int(s.strip()) for s in args.steps.split(",")]
        if args.pipeline == "shorts":
            from main import run_shorts_pipeline
            run_shorts_pipeline(args.video, steps, skip_silence_removal=args.skip_silence_removal)
        else:
            from main import run_streams_pipeline
            run_streams_pipeline(args.video, steps)

    elif args.command == "audio-levels":
        tool_audio_levels(args.video, args.start, args.end, resolution=args.resolution)

    elif args.command == "word-timestamps":
        tool_word_timestamps(args.video, args.sentence)

    elif args.command == "split-sentence":
        tool_split_sentence(args.video, args.sentence_index, args.split_time, target=args.target)

    elif args.command == "preview":
        tool_preview(args.video, source=args.source)
