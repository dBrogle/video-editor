"""
Stream editing pipeline.
Processes long livestream recordings (~2 hours) by transcribing,
identifying irrelevant sections via chunked LLM analysis, and
producing a cuts-only final video.
"""

import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from src.services.video import VideoService, MLTVideoService
from src.services.stt.whisper import WhisperSTTService
from src.services.llm.openrouter import OpenRouterLLMService
from src.services.local_saver import LocalSaverService
from src.services.agents import SentenceSelectionAgent, TimestampAdjustmentAgent
from src.models import Transcript, EditingResult, AdjustedSentences
from src.constants import (
    ASSETS_DIR,
    STREAM_LOW_RES_HEIGHT,
    STREAM_CHUNK_SIZE,
    STREAM_CHUNK_OVERLAP,
)
from src.util import (
    get_input_video_path,
    get_stream_downsampled_video_path,
    get_stream_audio_path,
    get_stream_transcription_path,
    get_stream_editing_result_path,
    get_stream_sentence_selection_video_path,
    get_stream_final_editing_result_path,
    get_stream_adjusted_sentences_path,
    get_stream_adjusted_sentences_video_path,
    get_stream_final_adjusted_sentences_path,
    get_stream_final_video_path,
    print_progress,
    validate_file_exists,
)


# ============================================================
# FFmpeg with progress bar
# ============================================================

def _get_duration_seconds(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def _run_ffmpeg_with_progress(cmd: list[str], duration_seconds: float, description: str) -> None:
    """
    Run an ffmpeg command while displaying a tqdm progress bar.
    Uses ffmpeg's -progress pipe:1 to stream progress to stdout.
    """
    # Insert -progress pipe:1 before the output file (last arg)
    cmd_with_progress = cmd[:-1] + ["-progress", "pipe:1", cmd[-1]]

    bar = tqdm(
        total=int(duration_seconds),
        desc=description,
        unit="s",
        bar_format="    {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
        file=sys.stderr,
    )

    process = subprocess.Popen(
        cmd_with_progress,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    for line in process.stdout:
        line = line.strip()
        if line.startswith("out_time_us="):
            try:
                microseconds = int(line.split("=")[1])
                seconds = microseconds / 1_000_000
                bar.n = min(int(seconds), int(duration_seconds))
                bar.refresh()
            except (ValueError, IndexError):
                pass
        elif line == "progress=end":
            bar.n = int(duration_seconds)
            bar.refresh()

    process.wait()
    bar.close()

    if process.returncode != 0:
        stderr = process.stderr.read() if process.stderr else ""
        raise subprocess.CalledProcessError(
            process.returncode, cmd_with_progress, stderr=stderr
        )


# ============================================================
# Stream-specific saver helpers (thin wrappers over file I/O)
# ============================================================

class StreamSaver:
    """Handles loading/saving stream pipeline artifacts using stream-prefixed paths."""

    def save_transcription(self, base_name: str, transcript: Transcript) -> Path:
        path = get_stream_transcription_path(base_name)
        path.write_text(transcript.model_dump_json(indent=2))
        return path

    def load_transcription(self, base_name: str) -> Transcript:
        import json
        path = get_stream_transcription_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Stream transcription not found: {path}")
        data = json.loads(path.read_text())
        return Transcript(**data)

    def transcription_exists(self, base_name: str) -> bool:
        return get_stream_transcription_path(base_name).exists()

    def save_editing_result(self, base_name: str, result: EditingResult) -> Path:
        path = get_stream_editing_result_path(base_name)
        path.write_text(result.model_dump_json(indent=2))
        return path

    def load_editing_result(self, base_name: str) -> EditingResult:
        import json
        path = get_stream_editing_result_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Stream editing result not found: {path}")
        data = json.loads(path.read_text())
        return EditingResult(**data)

    def editing_result_exists(self, base_name: str) -> bool:
        return get_stream_editing_result_path(base_name).exists()

    def save_final_editing_result(self, base_name: str, result: EditingResult) -> Path:
        path = get_stream_final_editing_result_path(base_name)
        path.write_text(result.model_dump_json(indent=2))
        return path

    def load_final_editing_result(self, base_name: str) -> EditingResult:
        import json
        path = get_stream_final_editing_result_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Stream final editing result not found: {path}")
        data = json.loads(path.read_text())
        return EditingResult(**data)

    def final_editing_result_exists(self, base_name: str) -> bool:
        return get_stream_final_editing_result_path(base_name).exists()

    def save_adjusted_sentences(self, base_name: str, adjusted: AdjustedSentences) -> Path:
        path = get_stream_adjusted_sentences_path(base_name)
        path.write_text(adjusted.model_dump_json(indent=2))
        return path

    def load_adjusted_sentences(self, base_name: str) -> AdjustedSentences:
        import json
        path = get_stream_adjusted_sentences_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Stream adjusted sentences not found: {path}")
        data = json.loads(path.read_text())
        return AdjustedSentences(**data)

    def adjusted_sentences_exist(self, base_name: str) -> bool:
        return get_stream_adjusted_sentences_path(base_name).exists()

    def save_final_adjusted_sentences(self, base_name: str, adjusted: AdjustedSentences) -> Path:
        path = get_stream_final_adjusted_sentences_path(base_name)
        path.write_text(adjusted.model_dump_json(indent=2))
        return path

    def load_final_adjusted_sentences(self, base_name: str) -> AdjustedSentences:
        import json
        path = get_stream_final_adjusted_sentences_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Stream final adjusted sentences not found: {path}")
        data = json.loads(path.read_text())
        return AdjustedSentences(**data)

    def final_adjusted_sentences_exist(self, base_name: str) -> bool:
        return get_stream_final_adjusted_sentences_path(base_name).exists()

    def load_best_adjusted_sentences(self, base_name: str) -> AdjustedSentences:
        if self.final_adjusted_sentences_exist(base_name):
            return self.load_final_adjusted_sentences(base_name)
        return self.load_adjusted_sentences(base_name)


# ============================================================
# Pipeline stages
# ============================================================

def stream_stage_1_preprocess(base_name: str, force: bool = False) -> None:
    """Preprocess stream: downsample to proxy and extract audio."""
    print_progress(f"Preprocessing stream: {base_name}")

    input_path = get_input_video_path(base_name)
    validate_file_exists(input_path)

    video_service = VideoService(ASSETS_DIR)

    duration = _get_duration_seconds(input_path)

    # Downsample to proxy resolution
    proxy_path = get_stream_downsampled_video_path(base_name)
    if proxy_path.exists() and not force:
        print_progress("  - Stream proxy video already exists, skipping")
    else:
        print_progress(f"  - Downsampling to {STREAM_LOW_RES_HEIGHT}p proxy...")
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-vf", f"scale=-2:{STREAM_LOW_RES_HEIGHT}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-c:a", "aac",
            "-b:a", "128k",
            "-y",
            str(proxy_path),
        ]
        _run_ffmpeg_with_progress(cmd, duration, "Proxy")
        print_progress("    Proxy video created")

    # Extract audio
    audio_path = get_stream_audio_path(base_name)
    if audio_path.exists() and not force:
        print_progress("  - Stream audio already exists, skipping")
    else:
        print_progress("  - Extracting audio...")
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "32k",
            "-y",
            str(audio_path),
        ]
        _run_ffmpeg_with_progress(cmd, duration, "Audio")
        print_progress("    Audio extracted")

    print_progress("Stream preprocessing complete")


def stream_stage_2_transcribe(base_name: str, saver: StreamSaver) -> Transcript:
    """Transcribe the stream audio using Deepgram."""
    if saver.transcription_exists(base_name):
        print_progress("Stream transcription already exists, loading from file")
        return saver.load_transcription(base_name)

    print_progress(f"Transcribing stream audio: {base_name}")
    audio_path = get_stream_audio_path(base_name)
    validate_file_exists(audio_path)

    stt_service = WhisperSTTService()
    transcript = stt_service.transcribe(audio_path)

    saver.save_transcription(base_name, transcript)
    print_progress(f"Stream transcription saved ({len(transcript.sentences)} sentences)")

    return transcript


def stream_stage_3_llm_edit(base_name: str, saver: StreamSaver, force: bool = False) -> None:
    """Send transcript to LLM in chunks for stream editing analysis."""
    if saver.editing_result_exists(base_name) and not force:
        print_progress("Stream editing result already exists, skipping")
        print_progress("Delete stream_s3_editing_result.json to regenerate")
        return

    print_progress("Loading stream transcript")
    transcript = saver.load_transcription(base_name)

    print_progress(f"Sending {len(transcript.sentences)} sentences to LLM (chunked)")
    llm = OpenRouterLLMService()
    editing_result = llm.get_stream_edits(
        transcript,
        chunk_size=STREAM_CHUNK_SIZE,
        overlap=STREAM_CHUNK_OVERLAP,
    )

    result_path = saver.save_editing_result(base_name, editing_result)
    print_progress(f"Stream editing result saved to: {result_path.name}")


def stream_stage_4_iterate_selection(
    base_name: str, saver: StreamSaver, skip_silence_removal: bool = True
) -> None:
    """Interactive iteration on sentence selection for the stream."""
    if saver.final_editing_result_exists(base_name):
        print_progress("Stream final sentence selection already exists, skipping")
        print_progress("Delete stream_s4_final_editing_result.json to re-iterate")
        return

    print("\n" + "=" * 60)
    print("STREAM SENTENCE SELECTION ITERATION")
    print("Review which sections should be kept or removed")
    print("=" * 60)

    transcript = saver.load_transcription(base_name)
    sentence_agent = SentenceSelectionAgent()
    iteration = 1
    max_iterations = 10

    while iteration <= max_iterations:
        print(f"\n--- Iteration {iteration} ---")

        editing_result = saver.load_editing_result(base_name)

        kept_count = sum(1 for sr in editing_result.sentence_results.values() if sr.keep)
        removed_count = sum(1 for sr in editing_result.sentence_results.values() if not sr.keep)
        print(f"\n   Kept: {kept_count} sentences")
        print(f"   Removed: {removed_count} sentences")

        # Generate preview video
        print("\n   Generating preview video with current selection...")
        video_service = VideoService(ASSETS_DIR)
        adjusted_sentences = video_service.generate_adjusted_sentences(
            base_name=base_name,
            transcript=transcript,
            editing_result=editing_result,
            use_downsampled=False,
            skip_silence_removal=skip_silence_removal,
        )

        # Use the proxy video for preview
        preview_path = get_stream_sentence_selection_video_path(base_name)
        video_service.create_edited_video(
            base_name=base_name,
            adjusted_sentences=adjusted_sentences,
            use_downsampled=False,
            force=True,
            output_path=preview_path,
        )

        print(f"\n   Preview video: {preview_path}")
        print("   Review the video to see which sections are included.")

        print("\n   Is the selection good?")
        print("   (Type 'looks good', 'approve', or 'perfect' if satisfied)")
        print("   (Or provide feedback like 'remove sentence 5' or 'keep sentence 3')")
        user_feedback = input("\nYour feedback: ").strip()

        if not user_feedback:
            print("   No feedback provided. Please try again.")
            continue

        try:
            print("\n   Processing feedback...")
            updated_result, is_approved = sentence_agent.process_feedback(
                editing_result=editing_result,
                user_feedback=user_feedback,
            )

            if is_approved:
                print("\n   Sentence selection approved!")
                saver.save_editing_result(base_name, updated_result)
                final_path = saver.save_final_editing_result(base_name, updated_result)
                print(f"   Final selection saved to: {final_path.name}")
                break

            print("\n   Saving updated selection...")
            saver.save_editing_result(base_name, updated_result)
            iteration += 1

        except Exception as e:
            print(f"\n   Error processing feedback: {str(e)}")
            print("   Please try again.")
            continue

    if iteration > max_iterations:
        print(f"\n   Warning: Reached max iterations ({max_iterations})")

    print("\n" + "=" * 60)
    print("Stream sentence selection complete!")
    print("=" * 60)


def stream_stage_5_adjust_timestamps(
    base_name: str, saver: StreamSaver, skip_silence_removal: bool = True
) -> None:
    """Generate adjusted sentences with optional silence removal."""
    if saver.adjusted_sentences_exist(base_name):
        print_progress("Stream adjusted sentences already exist, skipping")
        print_progress("Delete stream_s5_adjusted_sentences.json to regenerate")
        return

    print_progress("Loading stream transcript and editing result")
    transcript = saver.load_transcription(base_name)
    editing_result = saver.load_final_editing_result(base_name)

    video_service = VideoService(ASSETS_DIR)

    if skip_silence_removal:
        print_progress("Generating adjusted sentences (skipping silence removal)")
    else:
        print_progress("Generating adjusted sentences with silence removal")

    adjusted = video_service.generate_adjusted_sentences(
        base_name=base_name,
        transcript=transcript,
        editing_result=editing_result,
        use_downsampled=False,
        skip_silence_removal=skip_silence_removal,
    )

    saver.save_adjusted_sentences(base_name, adjusted)
    print_progress(f"Stream adjusted sentences saved ({len(adjusted.sentences)} sentences)")


def stream_stage_6_iterate_timestamps(
    base_name: str, saver: StreamSaver, skip_silence_removal: bool = True
) -> None:
    """Interactive iteration on timestamp adjustments for the stream."""
    if saver.final_adjusted_sentences_exist(base_name):
        print_progress("Stream final adjusted sentences already exist, skipping")
        print_progress("Delete stream_s6_final_adjusted_sentences.json to re-iterate")
        return

    print("\n" + "=" * 60)
    print("STREAM TIMESTAMP ADJUSTMENT ITERATION")
    print("Fine-tune the timestamps of selected sentences")
    print("=" * 60)

    transcript = saver.load_transcription(base_name)

    if saver.final_editing_result_exists(base_name):
        editing_result = saver.load_final_editing_result(base_name)
    else:
        editing_result = saver.load_editing_result(base_name)

    # Generate adjusted sentences
    print("\n   Generating adjusted sentences from selection...")
    video_service = VideoService(ASSETS_DIR)
    adjusted = video_service.generate_adjusted_sentences(
        base_name=base_name,
        transcript=transcript,
        editing_result=editing_result,
        use_downsampled=False,
        skip_silence_removal=skip_silence_removal,
    )
    saver.save_adjusted_sentences(base_name, adjusted)

    # Generate initial preview
    preview_path = get_stream_adjusted_sentences_video_path(base_name)
    video_service.create_edited_video(
        base_name=base_name,
        adjusted_sentences=adjusted,
        use_downsampled=False,
        force=True,
        output_path=preview_path,
    )

    print(f"\n   Preview video: {preview_path}")
    print("   Review the video to check timestamps and pacing.")

    timestamp_agent = TimestampAdjustmentAgent()
    iteration = 1
    max_iterations = 10

    while iteration <= max_iterations:
        print(f"\n--- Iteration {iteration} ---")

        print("\n   How do the timestamps look?")
        print("   (Type 'looks good', 'approve', or 'perfect' if satisfied)")
        print("   (Or provide feedback like 'cut 2 seconds from the beginning')")
        user_feedback = input("\nYour feedback: ").strip()

        if not user_feedback:
            print("   No feedback provided. Please try again.")
            continue

        adjusted = saver.load_adjusted_sentences(base_name)

        try:
            print("\n   Processing feedback...")
            updated, is_approved = timestamp_agent.process_feedback(
                adjusted_sentences=adjusted,
                user_feedback=user_feedback,
            )

            if is_approved:
                print("\n   Timestamps approved!")
                saver.save_adjusted_sentences(base_name, updated)
                final_path = saver.save_final_adjusted_sentences(base_name, updated)
                print(f"   Final timestamps saved to: {final_path.name}")
                break

            print("\n   Saving updated timestamps...")
            saver.save_adjusted_sentences(base_name, updated)

            print("\n   Regenerating preview video...")
            video_service = VideoService(ASSETS_DIR)
            video_service.create_edited_video(
                base_name=base_name,
                adjusted_sentences=updated,
                use_downsampled=False,
                force=True,
                output_path=preview_path,
            )
            print(f"   Updated preview: {preview_path}")

            iteration += 1

        except Exception as e:
            print(f"\n   Error: {str(e)}")
            print("   Please try again.")
            continue

    if iteration > max_iterations:
        print(f"\n   Warning: Reached max iterations ({max_iterations})")

    print("\n" + "=" * 60)
    print("Stream timestamp adjustment complete!")
    print("=" * 60)


def stream_stage_7_final_render(base_name: str, saver: StreamSaver, force: bool = False) -> Path:
    """Render the final stream video with cuts applied at native resolution."""
    output_path = get_stream_final_video_path(base_name)

    if output_path.exists() and not force:
        print_progress(f"Stream final video already exists: {output_path}")
        return output_path

    print_progress("Loading best adjusted sentences for final render")
    adjusted = saver.load_best_adjusted_sentences(base_name)

    print_progress("Creating final stream video (cuts only, native resolution)")
    mlt_service = MLTVideoService()
    video_path = mlt_service.create_stream_final_video(
        base_name=base_name,
        adjusted_sentences=adjusted,
        force=force,
    )

    print_progress(f"Stream final video created: {video_path.name}")
    return video_path
