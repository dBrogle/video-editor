"""
Pure utility functions with no side effects.
Helper functions for file operations and path management.
"""

import subprocess
from pathlib import Path

from src.constants import (
    ASSETS_DIR,
    STAGE_1_DOWNSAMPLED_NAME,
    STAGE_1_AUDIO_NAME,
    STAGE_2_TRANSCRIPTION_NAME,
    STAGE_3_EDITING_DECISION_NAME,
    STAGE_3_EDITING_RESULT_NAME,
    STAGE_4_SENTENCE_SELECTION_VIDEO_NAME,
    STAGE_4_FINAL_EDITING_RESULT_NAME,
    STAGE_5_ADJUSTED_SENTENCES_NAME,
    STAGE_6_ADJUSTED_SENTENCES_VIDEO_NAME,
    STAGE_6_FINAL_ADJUSTED_SENTENCES_NAME,
    STAGE_6_EDITED_VIDEO_NAME,
    STAGE_6_DOWNSAMPLED_EDITED_NAME,
    STAGE_7_GOOGLE_DOC_SCRIPT_NAME,
    STAGE_7_IMAGES_FOLDER_NAME,
    STAGE_7_IMAGES_METADATA_NAME,
    STAGE_7_MLT_XML_NAME,
    STAGE_8_GOOGLE_DOC_IMAGE_PLACEMENTS_NAME,
    STAGE_9_WITH_GOOGLE_DOC_IMAGES_NAME,
    STAGE_9_MLT_XML_NAME,
    STAGE_10_1080P_DOWNSAMPLE_NAME,
    STAGE_11_1080P_WITH_IMAGES_NAME,
    STAGE_11_1080P_WITH_IMAGES_MLT_NAME,
    STAGE_12_FULL_RES_WITH_IMAGES_NAME,
    STAGE_12_FULL_RES_WITH_IMAGES_MLT_NAME,
    STAGE_13_FULL_RES_CUT_NAME,
    STAGE_13_FULL_RES_CUT_MLT_NAME,
    STAGE_14_FULL_RES_WITH_IMAGES_NAME,
    STAGE_14_FULL_RES_WITH_IMAGES_MLT_NAME,
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
    Get sentences from a transcript for LLM prompts.

    The transcript should already have sentences generated during transcription.
    This function simply returns them.

    Args:
        transcript: Transcript object containing sentences with word-level timestamps

    Returns:
        List of LLMTranscriptSentence objects with sentence text and timing info
    """
    # Transcript should already have sentences from transcription
    return transcript.sentences


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
    Get path to extracted audio file (Step 1 - Preprocess).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to audio file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_1_AUDIO_NAME, "mp3")


def get_transcription_path(base_name: str) -> Path:
    """
    Get path to transcription JSON file (Step 2).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to transcription file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_2_TRANSCRIPTION_NAME, "json")


def get_editing_decision_path(base_name: str) -> Path:
    """
    Get path to editing decision JSON file (LLM response) (Step 3).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to editing decision file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_3_EDITING_DECISION_NAME, "json")


def get_editing_result_path(base_name: str) -> Path:
    """
    Get path to editing result JSON file (human-editable format) (Step 3 initial).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to editing result file (s3_editing_result.json)
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_3_EDITING_RESULT_NAME, "json")


def get_sentence_selection_video_path(base_name: str) -> Path:
    """
    Get path to sentence selection iteration video file (Step 4 preview).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to sentence selection video file (s4_sentence_selection_video.mp4)
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_4_SENTENCE_SELECTION_VIDEO_NAME, "mp4")


def get_final_editing_result_path(base_name: str) -> Path:
    """
    Get path to final editing result JSON file (Step 4 approved).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to final editing result file (s4_final_editing_result.json)
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_4_FINAL_EDITING_RESULT_NAME, "json")


def get_adjusted_sentences_path(base_name: str) -> Path:
    """
    Get path to adjusted sentences JSON file (Step 5 initial).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to adjusted sentences file (s5_adjusted_sentences.json)
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_5_ADJUSTED_SENTENCES_NAME, "json")


def get_adjusted_sentences_video_path(base_name: str) -> Path:
    """
    Get path to adjusted sentences iteration video file (Step 6 preview).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to adjusted sentences video file (s6_adjusted_sentences_video.mp4)
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_6_ADJUSTED_SENTENCES_VIDEO_NAME, "mp4")


def get_final_adjusted_sentences_path(base_name: str) -> Path:
    """
    Get path to final adjusted sentences JSON file (Step 6 approved).

    Args:
        base_name: Base filename without extension (e.g., 'IMG_0901')

    Returns:
        Full path to final adjusted sentences file (s6_final_adjusted_sentences.json)
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_6_FINAL_ADJUSTED_SENTENCES_NAME, "json")


def get_edited_video_path(base_name: str, use_downsampled: bool = True) -> Path:
    """
    Get path to edited video file (legacy/generic method).

    This is kept for backward compatibility with video_service.py.
    For iteration-specific videos, use:
    - get_sentence_selection_video_path() for step 4
    - get_adjusted_sentences_video_path() for step 6

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


def get_best_edited_video_path(base_name: str) -> Path:
    """
    Get path to the final edited video file from step 6.

    This returns the path to s6_adjusted_sentences_video.mp4, which is the
    edited video output from step 6 (iterate adjusted sentences).

    This should be used by downstream stages (7+) that need the final edited video.

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s6_adjusted_sentences_video.mp4

    Raises:
        FileNotFoundError: If edited video doesn't exist
    """
    adjusted_video_path = get_adjusted_sentences_video_path(base_name)

    if not adjusted_video_path.exists():
        raise FileNotFoundError(
            f"Edited video not found: {adjusted_video_path}\n"
            f"Please run step 6 (iterate adjusted sentences) first to generate the edited video."
        )

    return adjusted_video_path


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
    Get path to Google Doc script JSON file (Step 7).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s7_google_doc_script.json file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_7_GOOGLE_DOC_SCRIPT_NAME, "json")


def get_google_doc_image_placements_path(base_name: str) -> Path:
    """
    Get path to Google Doc image placements file (Step 8).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s8_google_doc_image_placements.json file
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(
        base_name, STAGE_8_GOOGLE_DOC_IMAGE_PLACEMENTS_NAME, "json"
    )


def get_stage_11_with_google_doc_images_path(base_name: str) -> Path:
    """
    Get path to step 9 video with Google Doc images (downsampled).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s9_with_google_doc_images.mp4
    """
    base_folder = get_base_folder(base_name)
    return base_folder / f"{STAGE_9_WITH_GOOGLE_DOC_IMAGES_NAME}.mp4"


def get_stage_11_mlt_xml_path(base_name: str) -> Path:
    """
    Get path to step 9 MLT XML file (with Google Doc images).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s9_with_google_doc_images_mlt.mlt
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_9_MLT_XML_NAME, "mlt")


def get_full_res_cut_video_path(base_name: str) -> Path:
    """
    Get path to full resolution cut video (Stage 13).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s13_full_res_cut.mp4
    """
    base_folder = get_base_folder(base_name)
    return base_folder / f"{STAGE_13_FULL_RES_CUT_NAME}.mp4"


def get_full_res_cut_mlt_path(base_name: str) -> Path:
    """
    Get path to full resolution cut MLT XML file (Stage 13).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s13_full_res_cut_mlt.mlt
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_13_FULL_RES_CUT_MLT_NAME, "mlt")


def get_1080p_downsample_video_path(base_name: str) -> Path:
    """
    Get path to 1080p downsampled video (Step 10).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s10_1080p_downsample.mp4
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_10_1080P_DOWNSAMPLE_NAME, "mp4")


def get_1080p_with_images_video_path(base_name: str) -> Path:
    """
    Get path to 1080p video with images (Step 11).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s11_1080p_with_images.mp4
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_11_1080P_WITH_IMAGES_NAME, "mp4")


def get_1080p_with_images_mlt_path(base_name: str) -> Path:
    """
    Get path to 1080p with images MLT XML file (Step 11).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s11_1080p_with_images_mlt.mlt
    """
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_11_1080P_WITH_IMAGES_MLT_NAME, "mlt")


def get_full_res_with_images_video_path(base_name: str) -> Path:
    """
    Get path to full resolution video with images (Step 12 or Advanced Step 14).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s12_full_res_with_images.mp4 (Step 12) or s14_full_res_with_images.mp4 (Advanced Step 14)
    """
    base_folder = get_base_folder(base_name)
    # Check if Step 12 file exists first (single-pass approach)
    step_12_path = base_folder / f"{STAGE_12_FULL_RES_WITH_IMAGES_NAME}.mp4"
    if step_12_path.exists():
        return step_12_path
    # Otherwise return Step 14 path (two-step approach)
    return base_folder / f"{STAGE_14_FULL_RES_WITH_IMAGES_NAME}.mp4"


def get_full_res_with_images_mlt_path(base_name: str) -> Path:
    """
    Get path to full resolution with images MLT XML file (Step 12 or Advanced Step 14).

    Args:
        base_name: Base filename without extension

    Returns:
        Path to s12_full_res_with_images_mlt.mlt (Step 12) or s14_full_res_with_images_mlt.mlt (Advanced Step 14)
    """
    _ensure_asset_folder(base_name)
    # Check if Step 12 file exists first (single-pass approach)
    step_12_path = _build_asset_path(
        base_name, STAGE_12_FULL_RES_WITH_IMAGES_MLT_NAME, "mlt"
    )
    if step_12_path.exists():
        return step_12_path
    # Otherwise return Step 14 path (two-step approach)
    return _build_asset_path(base_name, STAGE_14_FULL_RES_WITH_IMAGES_MLT_NAME, "mlt")


def reset_pipeline(base_name: str) -> None:
    """
    Reset the pipeline by deleting all generated files (files starting with 's').
    This keeps the original video file but removes all intermediate and output files.

    Args:
        base_name: Base filename without extension

    Raises:
        FileNotFoundError: If the base folder doesn't exist
    """
    import glob
    import os

    base_folder = get_base_folder(base_name)

    if not base_folder.exists():
        raise FileNotFoundError(f"Base folder not found: {base_folder}")

    # Find all files starting with 's' in the base folder
    pattern = str(base_folder / "s*")
    files_to_delete = glob.glob(pattern)

    if not files_to_delete:
        print_progress("No pipeline files found to delete")
        return

    print_progress(f"Found {len(files_to_delete)} pipeline files to delete:")
    for file_path in sorted(files_to_delete):
        file_name = Path(file_path).name
        print(f"  - {file_name}")

    # Delete each file
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            file_path_obj = Path(file_path)
            if file_path_obj.is_file():
                os.remove(file_path)
                deleted_count += 1
            elif file_path_obj.is_dir():
                # Skip directories (like google_doc folder might have subdirs)
                print_progress(f"Skipping directory: {file_path_obj.name}")
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")

    print_progress(f"✓ Deleted {deleted_count} pipeline files")
    print_progress("Pipeline reset complete. Original video file preserved.")
