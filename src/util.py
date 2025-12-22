"""
Pure utility functions with no side effects.
Helper functions for file operations and path management.
"""

import subprocess
from pathlib import Path

from src.constants import (
    ASSETS_DIR,
    STAGE_1_DOWNSAMPLED_NAME,
    STAGE_2_AUDIO_NAME,
    STAGE_3_TRANSCRIPTION_NAME,
    STAGE_4_EDITING_DECISION_NAME,
    STAGE_4_EDITING_RESULT_NAME,
    STAGE_5_ADJUSTED_SENTENCES_NAME,
    STAGE_6_EDITED_VIDEO_NAME,
    STAGE_6_DOWNSAMPLED_EDITED_NAME,
    STAGE_7_WITH_IMAGES_DOWNSAMPLED_NAME,
    STAGE_7_IMAGES_FOLDER_NAME,
    STAGE_7_IMAGES_METADATA_NAME,
    STAGE_7_MLT_XML_NAME,
    STAGE_8_GOOGLE_DOC_SCRIPT_NAME,
    STAGE_9_GOOGLE_DOC_IMAGE_PLACEMENTS_NAME,
    STAGE_10_WITH_GOOGLE_DOC_IMAGES_NAME,
    STAGE_10_MLT_XML_NAME,
    STAGE_11_FULL_RES_CUT_NAME,
    STAGE_11_FULL_RES_CUT_MLT_NAME,
    STAGE_12_FULL_RES_WITH_IMAGES_NAME,
    STAGE_12_FULL_RES_WITH_IMAGES_MLT_NAME,
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
    from src.models import WordTimestamp

    sentences: list[LLMTranscriptSentence] = []
    current_words: list[str] = []
    current_word_timestamps: list[WordTimestamp] = []
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
            current_word_timestamps.append(word_obj)

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
                            sentence=sentence_text,
                            start=current_start,
                            end=current_end,
                            words=current_word_timestamps,
                        )
                    )

                # Reset for next sentence
                current_words = []
                current_word_timestamps = []
                current_start = None
                current_end = None

    # Handle any remaining words that didn't end with punctuation
    if current_words and current_start is not None and current_end is not None:
        sentence_text = " ".join(current_words)
        sentences.append(
            LLMTranscriptSentence(
                sentence=sentence_text,
                start=current_start,
                end=current_end,
                words=current_word_timestamps,
            )
        )

    return sentences


def _build_asset_path(base_filename: str, stage_name: str, extension: str) -> Path:
    """
    Build a path for a derived asset in folder structure (internal use only).

    New structure: assets/{base_filename}/{stage_name}.{extension}
    Example: assets/IMG_2362/s1_downsampled.mp4

    Args:
        base_filename: Base filename without extension (e.g., 'IMG_2362')
        stage_name: Stage filename (e.g., 's1_downsampled', 's2_audio')
        extension: File extension (without dot)

    Returns:
        Full path to the asset
    """
    folder = ASSETS_DIR / base_filename
    return folder / f"{stage_name}.{extension}"


def _ensure_asset_folder(base_filename: str) -> Path:
    """
    Ensure the asset folder exists for a given base filename.

    Args:
        base_filename: Base filename without extension

    Returns:
        Path to the asset folder
    """
    folder = ASSETS_DIR / base_filename
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_base_folder(base_name: str) -> Path:
    """
    Get the base folder path for a video project.

    Args:
        base_name: Base filename without extension

    Returns:
        Path to the base folder (assets/{base_name})
    """
    return ASSETS_DIR / base_name


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

    New structure: assets/{base_name}/{base_name}.mp4 (or .MOV, .mov, .MP4)
    Example: assets/d1/d1.mp4

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to input video
    """
    folder = ASSETS_DIR / base_name
    # Try common video extensions (prioritize .mp4 for rotated/processed videos)
    for ext in [".mp4", ".MP4", ".MOV", ".mov"]:
        path = folder / f"{base_name}{ext}"
        if path.exists():
            return path
    # If none found, return with .mp4 as default
    return folder / f"{base_name}.mp4"


def get_downsampled_video_path(base_name: str) -> Path:
    """
    Get path to downsampled video file (Stage 1).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to downsampled video
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_1_DOWNSAMPLED_NAME, "mp4")


def get_audio_path(base_name: str) -> Path:
    """
    Get path to extracted audio file (Stage 2).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to audio file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_2_AUDIO_NAME, "wav")


def get_transcription_path(base_name: str) -> Path:
    """
    Get path to transcription JSON file (Stage 3).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to transcription file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_3_TRANSCRIPTION_NAME, "json")


def get_editing_decision_path(base_name: str) -> Path:
    """
    Get path to editing decision JSON file (LLM response) (Stage 4).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to editing decision file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_4_EDITING_DECISION_NAME, "json")


def get_editing_result_path(base_name: str) -> Path:
    """
    Get path to editing result JSON file (human-editable format) (Stage 4).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to editing result file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_4_EDITING_RESULT_NAME, "json")


def get_edited_video_path(base_name: str, use_downsampled: bool = True) -> Path:
    """
    Get path to edited video file (Stage 6).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')
        use_downsampled: If True, returns path for downsampled edited video

    Returns:
        Full path to edited video file
    """
    _ensure_asset_folder(base_name)
    stage_name = (
        STAGE_6_DOWNSAMPLED_EDITED_NAME
        if use_downsampled
        else STAGE_6_EDITED_VIDEO_NAME
    )
    return _build_asset_path(base_name, stage_name, "mp4")


def get_adjusted_sentences_path(base_name: str) -> Path:
    """
    Get path to adjusted sentences JSON file (silence-trimmed timestamps) (Stage 5).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to adjusted sentences file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_5_ADJUSTED_SENTENCES_NAME, "json")


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


# ============================================================================
# Stage 7: Image Management Functions
# ============================================================================


def get_images_folder(base_name: str) -> Path:
    """
    Get path to images folder for a video.

    Args:
        base_name: Base filename without extension

    Returns:
        Path to images folder
    """
    base_folder = get_base_folder(base_name)
    return base_folder / STAGE_7_IMAGES_FOLDER_NAME


def get_images_metadata_path(base_name: str) -> Path:
    """
    Get path to images metadata JSON file.

    Args:
        base_name: Base filename without extension

    Returns:
        Path to images_metadata.json file
    """
    images_folder = get_images_folder(base_name)
    return images_folder / f"{STAGE_7_IMAGES_METADATA_NAME}.json"


def get_stage_7_with_images_path(base_name: str) -> Path:
    """
    Get path to stage 7 video with images (downsampled).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s7_with_images_downsampled.mp4
    """
    base_folder = get_base_folder(base_name)
    return base_folder / f"{STAGE_7_WITH_IMAGES_DOWNSAMPLED_NAME}.mp4"


def create_images_folder(base_name: str) -> Path:
    """
    Create images folder if it doesn't exist.

    Args:
        base_name: Base filename without extension

    Returns:
        Path to created images folder
    """
    images_folder = get_images_folder(base_name)
    images_folder.mkdir(parents=True, exist_ok=True)
    print_progress(f"Images folder ready: {images_folder}")
    return images_folder


def save_images_metadata(base_name: str, metadata: "ImagesMetadataFile") -> Path:
    """
    Save images metadata to JSON file.

    Args:
        base_name: Base filename without extension
        metadata: ImagesMetadataFile object

    Returns:
        Path to saved metadata file
    """
    import json

    metadata_path = get_images_metadata_path(base_name)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    print_progress(f"Images metadata saved: {metadata_path}")
    return metadata_path


def load_images_metadata(base_name: str) -> "ImagesMetadataFile":
    """
    Load images metadata from JSON file.

    Args:
        base_name: Base filename without extension

    Returns:
        ImagesMetadataFile object

    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    import json
    from src.models import ImagesMetadataFile

    metadata_path = get_images_metadata_path(base_name)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Images metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ImagesMetadataFile(**data)


def get_stage_7_mlt_xml_path(base_name: str) -> Path:
    """
    Get path to stage 7 MLT XML file (with images).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s7_with_images_mlt.mlt
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_7_MLT_XML_NAME, "mlt")


# ============================================================================
# Google Doc HTML Management Functions
# ============================================================================


def get_google_doc_folder(base_name: str) -> Path:
    """
    Get path to Google Doc folder for a video project.

    Args:
        base_name: Base filename without extension

    Returns:
        Path to google_doc folder (assets/{base_name}/google_doc)
    """
    base_folder = get_base_folder(base_name)
    return base_folder / "google_doc"


def get_google_doc_html_path(base_name: str) -> Path:
    """
    Get path to Google Doc HTML file.

    Args:
        base_name: Base filename without extension

    Returns:
        Path to {base_name}.html inside google_doc folder
    """
    google_doc_folder = get_google_doc_folder(base_name)
    return google_doc_folder / f"{base_name}.html"


def get_google_doc_images_folder(base_name: str) -> Path:
    """
    Get path to Google Doc images folder.

    Args:
        base_name: Base filename without extension

    Returns:
        Path to images folder inside google_doc folder
    """
    google_doc_folder = get_google_doc_folder(base_name)
    return google_doc_folder / "images"


def get_google_doc_script_path(base_name: str) -> Path:
    """
    Get path to Google Doc script JSON file (Step 8).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s8_google_doc_script.json file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_8_GOOGLE_DOC_SCRIPT_NAME, "json")


def get_google_doc_image_placements_path(base_name: str) -> Path:
    """
    Get path to Google Doc image placements file (Step 9).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s9_google_doc_image_placements.json file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(
        base_name, STAGE_9_GOOGLE_DOC_IMAGE_PLACEMENTS_NAME, "json"
    )


def get_stage_11_with_google_doc_images_path(base_name: str) -> Path:
    """
    Get path to step 10 video with Google Doc images (downsampled).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s10_with_google_doc_images.mp4
    """
    base_folder = get_base_folder(base_name)
    return base_folder / f"{STAGE_10_WITH_GOOGLE_DOC_IMAGES_NAME}.mp4"


def get_stage_11_mlt_xml_path(base_name: str) -> Path:
    """
    Get path to step 10 MLT XML file (with Google Doc images).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s10_with_google_doc_images_mlt.mlt
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_10_MLT_XML_NAME, "mlt")


def get_full_res_cut_video_path(base_name: str) -> Path:
    """
    Get path to full resolution cut video (Stage 11).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s11_full_res_cut.mp4
    """
    base_folder = get_base_folder(base_name)
    return base_folder / f"{STAGE_11_FULL_RES_CUT_NAME}.mp4"


def get_full_res_cut_mlt_path(base_name: str) -> Path:
    """
    Get path to full resolution cut MLT XML file (Stage 11).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s11_full_res_cut_mlt.mlt
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_11_FULL_RES_CUT_MLT_NAME, "mlt")


def get_full_res_with_images_video_path(base_name: str) -> Path:
    """
    Get path to full resolution video with images (Stage 12).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s12_full_res_with_images.mp4
    """
    base_folder = get_base_folder(base_name)
    return base_folder / f"{STAGE_12_FULL_RES_WITH_IMAGES_NAME}.mp4"


def get_full_res_with_images_mlt_path(base_name: str) -> Path:
    """
    Get path to full resolution with images MLT XML file (Stage 12).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s12_full_res_with_images_mlt.mlt
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_12_FULL_RES_WITH_IMAGES_MLT_NAME, "mlt")
