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
    return Path(filepath).stem


def prepare_transcript_for_prompt(
    transcript: "Transcript",
) -> list["LLMTranscriptSentence"]:
    return transcript.sentences


def _build_asset_path(base_filename: str, stage_name: str, extension: str) -> Path:
    return ASSETS_DIR / base_filename / f"{stage_name}.{extension}"


def _ensure_asset_folder(base_filename: str) -> Path:
    folder = ASSETS_DIR / base_filename
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_base_folder(base_name: str) -> Path:
    return ASSETS_DIR / base_name


def validate_file_exists(filepath: Path | str) -> None:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


def run_command(
    cmd: list[str], capture_output: bool = True, check: bool = True
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=capture_output, text=True, check=check)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"Exit code: {e.returncode}\n"
            f"stderr: {e.stderr}"
        ) from e


def print_progress(message: str, prefix: str = "=>") -> None:
    print(f"{prefix} {message}")


def ensure_directory_exists(directory: Path | str) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_input_video_path(base_name: str) -> Path:
    folder = ASSETS_DIR / base_name
    for ext in [".mp4", ".MP4", ".MOV", ".mov"]:
        path = folder / f"{base_name}{ext}"
        if path.exists():
            return path
    return folder / f"{base_name}.mp4"


def get_downsampled_video_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_1_DOWNSAMPLED_NAME, "mp4")


def get_audio_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_1_AUDIO_NAME, "mp3")


def get_transcription_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_2_TRANSCRIPTION_NAME, "json")


def get_editing_decision_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_3_EDITING_DECISION_NAME, "json")


def get_editing_result_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_3_EDITING_RESULT_NAME, "json")


def get_sentence_selection_video_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_4_SENTENCE_SELECTION_VIDEO_NAME, "mp4")


def get_final_editing_result_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_4_FINAL_EDITING_RESULT_NAME, "json")


def get_adjusted_sentences_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_5_ADJUSTED_SENTENCES_NAME, "json")


def get_adjusted_sentences_video_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_6_ADJUSTED_SENTENCES_VIDEO_NAME, "mp4")


def get_final_adjusted_sentences_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_6_FINAL_ADJUSTED_SENTENCES_NAME, "json")


def get_edited_video_path(base_name: str, use_downsampled: bool = True) -> Path:
    _ensure_asset_folder(base_name)
    stage_name = STAGE_6_DOWNSAMPLED_EDITED_NAME if use_downsampled else STAGE_6_EDITED_VIDEO_NAME
    return _build_asset_path(base_name, stage_name, "mp4")


def get_best_edited_video_path(base_name: str) -> Path:
    adjusted_video_path = get_adjusted_sentences_video_path(base_name)
    if not adjusted_video_path.exists():
        raise FileNotFoundError(
            f"Edited video not found: {adjusted_video_path}\n"
            f"Please run step 6 first to generate the edited video."
        )
    return adjusted_video_path


def convert_editing_decision_to_result(
    decision: "EditingDecision", transcript: "Transcript"
) -> "EditingResult":
    from src.models import EditingResult, SentenceResult

    sentences_to_remove = set(decision.sentences_to_remove)
    sentence_results = {}

    for i, sentence in enumerate(transcript.sentences, 1):
        sentence_results[str(i)] = SentenceResult(
            text=sentence.sentence, keep=(i not in sentences_to_remove)
        )

    return EditingResult(sentence_results=sentence_results)


def get_images_folder(base_name: str) -> Path:
    return get_base_folder(base_name) / STAGE_7_IMAGES_FOLDER_NAME


def get_images_metadata_path(base_name: str) -> Path:
    return get_images_folder(base_name) / f"{STAGE_7_IMAGES_METADATA_NAME}.json"


def create_images_folder(base_name: str) -> Path:
    images_folder = get_images_folder(base_name)
    images_folder.mkdir(parents=True, exist_ok=True)
    print_progress(f"Images folder ready: {images_folder}")
    return images_folder


def save_images_metadata(base_name: str, metadata: "ImagesMetadataFile") -> Path:
    import json

    metadata_path = get_images_metadata_path(base_name)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    print_progress(f"Images metadata saved: {metadata_path}")
    return metadata_path


def load_images_metadata(base_name: str) -> "ImagesMetadataFile":
    import json
    from src.models import ImagesMetadataFile

    metadata_path = get_images_metadata_path(base_name)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Images metadata not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ImagesMetadataFile(**data)


def get_google_doc_folder(base_name: str) -> Path:
    return get_base_folder(base_name) / "google_doc"


def get_google_doc_html_path(base_name: str) -> Path:
    return get_google_doc_folder(base_name) / f"{base_name}.html"


def get_google_doc_images_folder(base_name: str) -> Path:
    return get_google_doc_folder(base_name) / "images"


def get_google_doc_script_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_7_GOOGLE_DOC_SCRIPT_NAME, "json")


def get_google_doc_image_placements_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_8_GOOGLE_DOC_IMAGE_PLACEMENTS_NAME, "json")


def get_stage_11_with_google_doc_images_path(base_name: str) -> Path:
    return get_base_folder(base_name) / f"{STAGE_9_WITH_GOOGLE_DOC_IMAGES_NAME}.mp4"


def get_stage_11_mlt_xml_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_9_MLT_XML_NAME, "mlt")


def get_full_res_cut_video_path(base_name: str) -> Path:
    return get_base_folder(base_name) / f"{STAGE_13_FULL_RES_CUT_NAME}.mp4"


def get_full_res_cut_mlt_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_13_FULL_RES_CUT_MLT_NAME, "mlt")


def get_1080p_downsample_video_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_10_1080P_DOWNSAMPLE_NAME, "mp4")


def get_1080p_with_images_video_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_11_1080P_WITH_IMAGES_NAME, "mp4")


def get_1080p_with_images_mlt_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    return _build_asset_path(base_name, STAGE_11_1080P_WITH_IMAGES_MLT_NAME, "mlt")


def get_full_res_with_images_video_path(base_name: str) -> Path:
    base_folder = get_base_folder(base_name)
    step_12_path = base_folder / f"{STAGE_12_FULL_RES_WITH_IMAGES_NAME}.mp4"
    if step_12_path.exists():
        return step_12_path
    return base_folder / f"{STAGE_14_FULL_RES_WITH_IMAGES_NAME}.mp4"


def get_full_res_with_images_mlt_path(base_name: str) -> Path:
    _ensure_asset_folder(base_name)
    step_12_path = _build_asset_path(base_name, STAGE_12_FULL_RES_WITH_IMAGES_MLT_NAME, "mlt")
    if step_12_path.exists():
        return step_12_path
    return _build_asset_path(base_name, STAGE_14_FULL_RES_WITH_IMAGES_MLT_NAME, "mlt")


def reset_pipeline(base_name: str) -> None:
    import glob
    import os

    base_folder = get_base_folder(base_name)
    if not base_folder.exists():
        raise FileNotFoundError(f"Base folder not found: {base_folder}")

    pattern = str(base_folder / "s*")
    files_to_delete = glob.glob(pattern)

    if not files_to_delete:
        print_progress("No pipeline files found to delete")
        return

    print_progress(f"Found {len(files_to_delete)} pipeline files to delete:")
    for file_path in sorted(files_to_delete):
        print(f"  - {Path(file_path).name}")

    deleted_count = 0
    for file_path in files_to_delete:
        try:
            file_path_obj = Path(file_path)
            if file_path_obj.is_file():
                os.remove(file_path)
                deleted_count += 1
            elif file_path_obj.is_dir():
                print_progress(f"Skipping directory: {file_path_obj.name}")
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")

    print_progress(f"✓ Deleted {deleted_count} pipeline files")
    print_progress("Pipeline reset complete. Original video file preserved.")
