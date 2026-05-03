"""
Utility functions for the cc_wsp video editing workspace.
All paths are relative to cc_wsp/videos/{video_name}/.
"""

import json
import subprocess
from pathlib import Path

from cc_wsp.src.constants import (
    VIDEOS_DIR,
    IMAGE_SAFE_ZONE_TOP_PERCENT,
    IMAGE_SAFE_ZONE_BOTTOM_PERCENT,
    IMAGE_SAFE_ZONE_LEFT_PERCENT,
    IMAGE_SAFE_ZONE_RIGHT_PERCENT,
)
from cc_wsp.src.models import (
    Transcript,
    EditingResult,
    SentenceResult,
    EditingDecision,
    AdjustedSentences,
    GoogleDocScript,
    GoogleDocImagePlacements,
    ZoomFilters,
    FaceData,
)


def get_video_dir(name: str) -> Path:
    d = VIDEOS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_input_video_path(name: str) -> Path:
    d = get_video_dir(name)
    leaf = Path(name).name
    for ext in [".mp4", ".MP4", ".MOV", ".mov"]:
        p = d / f"{leaf}{ext}"
        if p.exists():
            return p
    return d / f"{leaf}.mp4"


# Step file paths
def _step_path(name: str, step: str, ext: str) -> Path:
    return get_video_dir(name) / f"{step}.{ext}"


def audio_path(name: str) -> Path:
    return _step_path(name, "audio", "mp3")


def downsampled_path(name: str) -> Path:
    return _step_path(name, "downsampled", "mp4")


def transcription_path(name: str) -> Path:
    return _step_path(name, "transcription", "json")


def sentences_path(name: str) -> Path:
    return _step_path(name, "sentences", "json")


def adjusted_path(name: str) -> Path:
    return _step_path(name, "adjusted", "json")


def images_path(name: str) -> Path:
    return _step_path(name, "images", "json")


def zooms_path(name: str) -> Path:
    return _step_path(name, "zooms", "json")


def preview_path(name: str) -> Path:
    return _step_path(name, "preview", "mp4")


def final_path(name: str) -> Path:
    return _step_path(name, "final", "mp4")


def captioned_path(name: str) -> Path:
    return _step_path(name, "final_captioned", "mp4")


def final_mlt_path(name: str) -> Path:
    return _step_path(name, "final_mlt", "mlt")


def downsampled_1080p_path(name: str) -> Path:
    return _step_path(name, "1080p", "mp4")


def google_doc_dir(name: str) -> Path:
    return get_video_dir(name) / "google_doc"


def google_doc_html_path(name: str) -> Path:
    return google_doc_dir(name) / f"{name}.html"


def google_doc_images_dir(name: str) -> Path:
    return google_doc_dir(name) / "images"


# JSON read/write helpers
def load_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, data):
    if hasattr(data, "model_dump_json"):
        path.write_text(data.model_dump_json(indent=2))
    else:
        path.write_text(json.dumps(data, indent=2))


def load_transcription(name: str) -> Transcript:
    return Transcript(**load_json(transcription_path(name)))


def save_transcription(name: str, transcript: Transcript):
    save_json(transcription_path(name), transcript)


def load_sentences(name: str) -> EditingResult:
    return EditingResult(**load_json(sentences_path(name)))


def save_sentences(name: str, result: EditingResult):
    save_json(sentences_path(name), result)


def load_adjusted(name: str) -> AdjustedSentences:
    return AdjustedSentences(**load_json(adjusted_path(name)))


def save_adjusted(name: str, adj: AdjustedSentences):
    save_json(adjusted_path(name), adj)


def load_images(name: str) -> GoogleDocImagePlacements:
    return GoogleDocImagePlacements(**load_json(images_path(name)))


def save_images(name: str, placements: GoogleDocImagePlacements):
    save_json(images_path(name), placements)


def load_zooms(name: str) -> ZoomFilters:
    return ZoomFilters(**load_json(zooms_path(name)))


def save_zooms(name: str, zooms: ZoomFilters):
    save_json(zooms_path(name), zooms)


def load_google_doc_script(name: str) -> GoogleDocScript:
    p = get_video_dir(name) / "google_doc_script.json"
    return GoogleDocScript(**load_json(p))


def save_google_doc_script(name: str, script: GoogleDocScript):
    save_json(get_video_dir(name) / "google_doc_script.json", script)


def face_data_path(name: str) -> Path:
    return get_video_dir(name) / "face_data.json"


def load_face_data(name: str) -> FaceData:
    return FaceData(**load_json(face_data_path(name)))


def save_face_data(name: str, data: FaceData):
    save_json(face_data_path(name), data)


FACE_IMAGE_GAP = -0.02  # gap between face edge and image safe zone (negative = slight overlap with forehead)
FACE_CAPTION_GAP = 0.04  # gap between face bottom and caption baseline


MIN_IMAGE_ZONE_HEIGHT = 0.20  # require at least 20% of frame height for image area


def _face_derived_safe_zone(face: FaceData) -> dict:
    """Compute image safe zone.
    Prefer the space above the head (more common framing); fall back below
    the face if there is not enough headroom. The larger of the two wins.
    """
    above_top, above_bottom = 0.10, max(0.0, face.top_frac - FACE_IMAGE_GAP)
    below_top, below_bottom = min(1.0, face.bottom_frac + FACE_IMAGE_GAP), 0.98

    above_h = above_bottom - above_top
    below_h = below_bottom - below_top

    if above_h >= MIN_IMAGE_ZONE_HEIGHT:
        top, bottom = above_top, above_bottom
    elif below_h >= MIN_IMAGE_ZONE_HEIGHT:
        top, bottom = below_top, below_bottom
    else:
        top, bottom = above_top, above_bottom
    return {"image_safe_zone_top": top, "image_safe_zone_bottom": bottom}


def load_config(name: str) -> dict:
    """Load project-level config from config.json in the video directory.
    Falls back to face_data.json for safe zone, then to global constants."""
    defaults = {
        "image_safe_zone_top": IMAGE_SAFE_ZONE_TOP_PERCENT,
        "image_safe_zone_bottom": IMAGE_SAFE_ZONE_BOTTOM_PERCENT,
        "image_safe_zone_left": IMAGE_SAFE_ZONE_LEFT_PERCENT,
        "image_safe_zone_right": IMAGE_SAFE_ZONE_RIGHT_PERCENT,
    }
    fd_path = face_data_path(name)
    if fd_path.exists():
        try:
            defaults.update(_face_derived_safe_zone(load_face_data(name)))
        except Exception:
            pass
    config_path = get_video_dir(name) / "config.json"
    if config_path.exists():
        overrides = load_json(config_path)
        defaults.update(overrides)
    return defaults


def compute_caption_y_from_face(face: FaceData) -> float:
    """Caption baseline = just below the face; clamped into a sensible range."""
    y = face.bottom_frac + FACE_CAPTION_GAP
    return max(0.50, min(0.92, y))


def convert_decision_to_result(decision: EditingDecision, transcript: Transcript) -> EditingResult:
    to_remove = set(decision.sentences_to_remove)
    results = {}
    for i, s in enumerate(transcript.sentences, 1):
        results[str(i)] = SentenceResult(text=s.sentence, keep=(i not in to_remove))
    return EditingResult(sentence_results=results)


def print_progress(msg: str):
    print(f"=> {msg}")


# Compatibility functions for copied services that expect the old util API
def validate_file_exists(filepath):
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")


def prepare_transcript_for_prompt(transcript):
    return transcript.sentences


def extract_filename_without_extension(filepath):
    return Path(filepath).stem


def ensure_directory_exists(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_input_video_path(name):
    """Compatibility wrapper — searches for video file matching folder name."""
    d = VIDEOS_DIR / name
    leaf = Path(name).name
    for ext in [".mp4", ".MP4", ".MOV", ".mov"]:
        p = d / f"{leaf}{ext}"
        if p.exists():
            return p
    return d / f"{leaf}.mp4"


def get_downsampled_video_path(name):
    return VIDEOS_DIR / name / "downsampled.mp4"


def get_audio_path(name):
    return VIDEOS_DIR / name / "audio.mp3"


def get_edited_video_path(name, use_downsampled=True):
    return VIDEOS_DIR / name / "preview.mp4"


def get_1080p_downsample_video_path(name):
    return VIDEOS_DIR / name / "1080p.mp4"


def get_1080p_with_images_video_path(name):
    return VIDEOS_DIR / name / "final.mp4"


def get_1080p_with_images_mlt_path(name):
    return VIDEOS_DIR / name / "final_mlt.mlt"


def run_command(cmd, capture_output=True, check=True):
    import subprocess
    return subprocess.run(cmd, capture_output=capture_output, text=True, check=check)


# MLT video service compatibility stubs
def get_stage_11_with_google_doc_images_path(name):
    return VIDEOS_DIR / name / "with_images.mp4"

def get_stage_11_mlt_xml_path(name):
    return VIDEOS_DIR / name / "with_images_mlt.mlt"

def get_best_edited_video_path(name):
    return VIDEOS_DIR / name / "preview.mp4"

def get_full_res_with_images_video_path(name):
    return VIDEOS_DIR / name / "final.mp4"

def get_full_res_with_images_mlt_path(name):
    return VIDEOS_DIR / name / "final_mlt.mlt"

def get_full_res_cut_video_path(name):
    return VIDEOS_DIR / name / "full_res_cut.mp4"

def get_full_res_cut_mlt_path(name):
    return VIDEOS_DIR / name / "full_res_cut_mlt.mlt"

def get_stream_final_video_path(name):
    return VIDEOS_DIR / name / "stream_final.mp4"

def get_stream_final_mlt_path(name):
    return VIDEOS_DIR / name / "stream_final_mlt.mlt"


# Stream-specific step file paths
def stream_audio_path(name: str) -> Path:
    return _step_path(name, "stream_audio", "mp3")


def stream_downsampled_path(name: str) -> Path:
    return _step_path(name, "stream_downsampled", "mp4")


def stream_transcription_path(name: str) -> Path:
    return _step_path(name, "stream_transcription", "json")


def stream_sentences_path(name: str) -> Path:
    return _step_path(name, "stream_sentences", "json")


def stream_adjusted_path(name: str) -> Path:
    return _step_path(name, "stream_adjusted", "json")


def stream_preview_path(name: str) -> Path:
    return _step_path(name, "stream_preview", "mp4")


def stream_final_path(name: str) -> Path:
    return _step_path(name, "stream_final", "mp4")


# Stream load/save helpers
def load_stream_transcription(name: str) -> Transcript:
    return Transcript(**load_json(stream_transcription_path(name)))


def save_stream_transcription(name: str, transcript: Transcript):
    save_json(stream_transcription_path(name), transcript)


def load_stream_sentences(name: str) -> EditingResult:
    return EditingResult(**load_json(stream_sentences_path(name)))


def save_stream_sentences(name: str, result: EditingResult):
    save_json(stream_sentences_path(name), result)


def load_stream_adjusted(name: str) -> AdjustedSentences:
    return AdjustedSentences(**load_json(stream_adjusted_path(name)))


def save_stream_adjusted(name: str, adj: AdjustedSentences):
    save_json(stream_adjusted_path(name), adj)
