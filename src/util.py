"""
Pure utility functions with no side effects.
Helper functions for file operations and path management.
"""

import subprocess
from pathlib import Path

from src.constants import (
    ASSETS_DIR,
    STAGE_1_DOWNSAMPLED_SUFFIX,
    STAGE_2_AUDIO_SUFFIX,
    STAGE_3_TRANSCRIPTION_SUFFIX,
    STAGE_4_EDITING_DECISION_SUFFIX,
    STAGE_4_EDITING_RESULT_SUFFIX,
    STAGE_5_ADJUSTED_SENTENCES_SUFFIX,
    STAGE_6_EDITED_VIDEO_SUFFIX,
    STAGE_6_DOWNSAMPLED_EDITED_SUFFIX,
    STAGE_7_FINAL_CUT_SUFFIX,
    STAGE_7_AUDIO_SUFFIX,
    STAGE_7_FINAL_CUT_DOWNSAMPLED_SUFFIX,
    STAGE_7_FINAL_CUT_TRANSCRIPTION_SUFFIX,
)


def extract_filename_without_extension(filepath: str | Path) -> str:
    """
    Extract filename without extension from a file path.

    Args:
        filepath: Path to the file

    Returns:
        Filename without extension
    """
    return Path(filepath).stem


def prepare_transcript_for_prompt(
    transcript: "Transcript",
) -> list["LLMTranscriptSentence"]:
    """
    Convert a transcript into a list of sentences for LLM prompts.

    If the transcript already has sentences (from transcription), use those.
    Otherwise, form sentences by collecting words until punctuation (., ?, !) is encountered.

    Args:
        transcript: Transcript object containing segments with word-level timestamps

    Returns:
        List of LLMTranscriptSentence objects with sentence text and timing info
    """
    from src.models import LLMTranscriptSentence

    # If transcript already has sentences, return them
    if transcript.sentences:
        return transcript.sentences

    # Otherwise, generate sentences from segments
    sentences: list[LLMTranscriptSentence] = []
    current_words: list[str] = []
    current_start: float | None = None
    current_end: float | None = None

    # Iterate through all segments and their words
    for segment in transcript.segments:
        for word_obj in segment.words:
            # Set start time if this is the first word in the sentence
            if current_start is None:
                current_start = word_obj.start

            # Update end time with each word
            current_end = word_obj.end

            # Add the word to current sentence
            current_words.append(word_obj.word)

            # Check if the word ends with sentence-ending punctuation
            if word_obj.word.rstrip().endswith((".", "?", "!")):
                # Complete the current sentence
                if (
                    current_words
                    and current_start is not None
                    and current_end is not None
                ):
                    sentence_text = " ".join(current_words)
                    sentences.append(
                        LLMTranscriptSentence(
                            sentence=sentence_text, start=current_start, end=current_end
                        )
                    )

                # Reset for next sentence
                current_words = []
                current_start = None
                current_end = None

    # Handle any remaining words that didn't end with punctuation
    if current_words and current_start is not None and current_end is not None:
        sentence_text = " ".join(current_words)
        sentences.append(
            LLMTranscriptSentence(
                sentence=sentence_text, start=current_start, end=current_end
            )
        )

    return sentences


def _build_asset_path(base_filename: str, suffix: str, extension: str) -> Path:
    """
    Build a path for a derived asset (internal use only).

    Args:
        base_filename: Base filename without extension
        suffix: Suffix to append (e.g., '_downsampled', '_audio')
        extension: File extension (without dot)

    Returns:
        Full path to the asset
    """
    return ASSETS_DIR / f"{base_filename}{suffix}.{extension}"


def validate_file_exists(filepath: Path | str) -> None:
    """
    Validate that a file exists.

    Args:
        filepath: Path to validate

    Raises:
        FileNotFoundError: If file does not exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


def run_command(
    cmd: list[str], capture_output: bool = True, check: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a shell command safely.

    Args:
        cmd: Command and arguments as a list
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise on non-zero exit code

    Returns:
        CompletedProcess instance

    Raises:
        subprocess.CalledProcessError: If command fails and check=True
    """
    try:
        result = subprocess.run(
            cmd, capture_output=capture_output, text=True, check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"Exit code: {e.returncode}\n"
            f"stderr: {e.stderr}"
        ) from e


def print_progress(message: str, prefix: str = "=>") -> None:
    """
    Print a progress message to terminal.

    Args:
        message: Message to print
        prefix: Prefix for the message
    """
    print(f"{prefix} {message}")


def ensure_directory_exists(directory: Path | str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path to ensure exists
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_input_video_path(base_name: str) -> Path:
    """
    Get path to input video file.

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to input video
    """
    # Try common video extensions
    for ext in [".MOV", ".mov", ".mp4", ".MP4"]:
        path = ASSETS_DIR / f"{base_name}{ext}"
        if path.exists():
            return path
    # If none found, return with .MOV as default
    return ASSETS_DIR / f"{base_name}.MOV"


def get_downsampled_video_path(base_name: str) -> Path:
    """
    Get path to downsampled video file (Stage 1).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to downsampled video
    """
    return _build_asset_path(base_name, STAGE_1_DOWNSAMPLED_SUFFIX, "mp4")


def get_audio_path(base_name: str) -> Path:
    """
    Get path to extracted audio file (Stage 2).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to audio file
    """
    return _build_asset_path(base_name, STAGE_2_AUDIO_SUFFIX, "wav")


def get_transcription_path(base_name: str) -> Path:
    """
    Get path to transcription JSON file (Stage 3).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to transcription file
    """
    return _build_asset_path(base_name, STAGE_3_TRANSCRIPTION_SUFFIX, "json")


def get_editing_decision_path(base_name: str) -> Path:
    """
    Get path to editing decision JSON file (LLM response) (Stage 4).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to editing decision file
    """
    return _build_asset_path(base_name, STAGE_4_EDITING_DECISION_SUFFIX, "json")


def get_editing_result_path(base_name: str) -> Path:
    """
    Get path to editing result JSON file (human-editable format) (Stage 4).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to editing result file
    """
    return _build_asset_path(base_name, STAGE_4_EDITING_RESULT_SUFFIX, "json")


def get_edited_video_path(base_name: str, use_downsampled: bool = True) -> Path:
    """
    Get path to edited video file (Stage 6).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')
        use_downsampled: If True, returns path for downsampled edited video

    Returns:
        Full path to edited video file
    """
    suffix = (
        STAGE_6_DOWNSAMPLED_EDITED_SUFFIX
        if use_downsampled
        else STAGE_6_EDITED_VIDEO_SUFFIX
    )
    return _build_asset_path(base_name, suffix, "mp4")


def get_adjusted_sentences_path(base_name: str) -> Path:
    """
    Get path to adjusted sentences JSON file (silence-trimmed timestamps) (Stage 5).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to adjusted sentences file
    """
    return _build_asset_path(base_name, STAGE_5_ADJUSTED_SENTENCES_SUFFIX, "json")


def get_final_cut_path(base_name: str) -> Path:
    """
    Get path to final cut video file (high-res edited video) (Stage 7).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to final cut video file
    """
    return _build_asset_path(base_name, STAGE_7_FINAL_CUT_SUFFIX, "mp4")


def get_final_cut_audio_path(base_name: str) -> Path:
    """
    Get path to final cut audio file (Stage 7).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to final cut audio file
    """
    return _build_asset_path(base_name, STAGE_7_AUDIO_SUFFIX, "wav")


def get_final_cut_downsampled_path(base_name: str) -> Path:
    """
    Get path to downsampled final cut video file (Stage 7).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to downsampled final cut video file
    """
    return _build_asset_path(base_name, STAGE_7_FINAL_CUT_DOWNSAMPLED_SUFFIX, "mp4")


def get_final_cut_transcription_path(base_name: str) -> Path:
    """
    Get path to final cut transcription JSON file (Stage 7).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to final cut transcription file
    """
    return _build_asset_path(base_name, STAGE_7_FINAL_CUT_TRANSCRIPTION_SUFFIX, "json")


def convert_editing_decision_to_result(
    decision: "EditingDecision", transcript: "Transcript"
) -> "EditingResult":
    """
    Convert an EditingDecision (LLM response) to EditingResult (human-editable format).

    Args:
        decision: EditingDecision with sentences_to_remove
        transcript: Transcript with sentences

    Returns:
        EditingResult with sentence_results mapping
    """
    from src.models import EditingResult, SentenceResult

    sentences_to_remove = set(decision.sentences_to_remove)
    sentence_results = {}

    for i, sentence in enumerate(transcript.sentences, 1):
        sentence_results[str(i)] = SentenceResult(
            text=sentence.sentence, keep=(i not in sentences_to_remove)
        )

    return EditingResult(sentence_results=sentence_results)
