"""
Pipeline orchestration functions.
Each function represents a step in the video editing pipeline.
"""

from src.services.video import VideoService, MLTVideoService
from src.services.stt.elevenlabs import ElevenLabsSTTService
from src.services.llm.openrouter import OpenRouterLLMService
from src.services.local_saver import LocalSaverService
from src.models import Transcript
from src.constants import ASSETS_DIR
from src.util import (
    get_input_video_path,
    get_audio_path,
    print_progress,
    convert_editing_decision_to_result,
    get_final_cut_path,
    get_final_cut_audio_path,
    get_final_cut_downsampled_path,
    get_final_cut_transcription_path,
)


def downsample_video(base_name: str, saver: LocalSaverService) -> None:
    """
    Step 1: Downsample video to low resolution.

    Args:
        base_name: Base filename without extension
        saver: Local saver service for checking existence
    """
    if saver.downsampled_video_exists(base_name):
        print_progress("Downsampled video already exists, skipping")
        return

    print_progress(f"Downsampling video: {base_name}")
    input_path = get_input_video_path(base_name)

    video_service = VideoService(ASSETS_DIR)
    video_service.generate_proxy_video(input_path, force=False)

    print_progress("Downsampled video created")


def extract_audio(base_name: str, saver: LocalSaverService) -> None:
    """
    Step 2: Extract audio from video.

    Args:
        base_name: Base filename without extension
        saver: Local saver service for checking existence
    """
    if saver.audio_exists(base_name):
        print_progress("Audio file already exists, skipping")
        return

    print_progress(f"Extracting audio: {base_name}")
    input_path = get_input_video_path(base_name)

    video_service = VideoService(ASSETS_DIR)
    video_service.extract_audio(input_path, force=False)

    print_progress("Audio extracted")


def get_transcription(base_name: str, saver: LocalSaverService) -> Transcript:
    """
    Step 3: Get transcription from audio.

    Args:
        base_name: Base filename without extension
        saver: Local saver service for saving/loading transcription

    Returns:
        Transcript object
    """
    if saver.transcription_exists(base_name):
        print_progress("Transcription already exists, loading from file")
        return saver.load_transcription(base_name)

    print_progress(f"Transcribing audio: {base_name}")
    audio_path = get_audio_path(base_name)

    stt_service = ElevenLabsSTTService()
    transcript = stt_service.transcribe(audio_path)

    saver.save_transcription(base_name, transcript)
    print_progress("Transcription saved")

    return transcript


def prompt_llm_for_editing(base_name: str, saver: LocalSaverService) -> None:
    """
    Step 4: Prompt LLM for editing decisions and create editable result.

    Args:
        base_name: Base filename without extension
        saver: Local saver service
    """
    print_progress("Loading transcript")
    transcript = saver.load_transcription(base_name)

    print_progress("Sending to LLM for editing analysis")
    llm = OpenRouterLLMService()
    decision = llm.get_edits(transcript)

    print_progress("Saving editing decision (LLM response)")
    decision_path = saver.save_editing_decision(base_name, decision)

    print_progress("Converting to editable format")
    editing_result = convert_editing_decision_to_result(decision, transcript)
    result_path = saver.save_editing_result(base_name, editing_result)

    print(f"\nThoughts: {decision.thoughts}")
    print(f"Sentences to remove: {decision.sentences_to_remove}")
    print(f"\nLLM response saved to: {decision_path.name}")
    print(f"Editable result saved to: {result_path.name}")


def generate_adjusted_sentences(base_name: str, saver: LocalSaverService) -> None:
    """
    Step 5: Generate adjusted sentences with silence removal.

    Args:
        base_name: Base filename without extension
        saver: Local saver service
    """
    if saver.adjusted_sentences_exist(base_name):
        print_progress("Adjusted sentences already exist, skipping")
        print_progress(
            "If you want to regenerate, delete the adjusted sentences file and run again"
        )
        return

    print_progress("Loading transcript and editing result")
    transcript = saver.load_transcription(base_name)
    editing_result = saver.load_editing_result(base_name)

    print_progress("Generating adjusted sentences with silence removal")
    video_service = VideoService(ASSETS_DIR)
    adjusted_sentences = video_service.generate_adjusted_sentences(
        base_name=base_name,
        transcript=transcript,
        editing_result=editing_result,
        use_downsampled=True,
    )

    adjusted_path = saver.save_adjusted_sentences(base_name, adjusted_sentences)
    print_progress(f"Adjusted sentences saved to: {adjusted_path.name}")


def create_edited_video(base_name: str, saver: LocalSaverService) -> None:
    """
    Step 6: Create edited video using adjusted sentences.

    Args:
        base_name: Base filename without extension
        saver: Local saver service
    """
    print_progress("Loading adjusted sentences")
    adjusted_sentences = saver.load_adjusted_sentences(base_name)

    print_progress("Creating edited video (downsampled)")
    video_service = VideoService(ASSETS_DIR)
    edited_video_path = video_service.create_edited_video(
        base_name=base_name,
        adjusted_sentences=adjusted_sentences,
        use_downsampled=True,
        force=False,
    )

    print_progress(f"Edited video created: {edited_video_path.name}")


def create_final_cut(base_name: str, saver: LocalSaverService) -> None:
    """
    Step 7a: Create final cut from original high-res video using adjusted sentences.
    Uses MLT framework for efficient video processing.

    Args:
        base_name: Base filename without extension
        saver: Local saver service
    """
    final_cut_path = get_final_cut_path(base_name)
    if final_cut_path.exists():
        print_progress("Final cut already exists, skipping")
        return

    print_progress("Loading adjusted sentences")
    adjusted_sentences = saver.load_adjusted_sentences(base_name)

    print_progress("Creating final cut from original high-res video using MLT")
    mlt_service = MLTVideoService()
    final_cut_path = mlt_service.create_final_cut_with_mlt(
        base_name=base_name,
        adjusted_sentences=adjusted_sentences,
        force=False,
    )

    print_progress(f"Final cut created: {final_cut_path.name}")


def extract_final_cut_audio(base_name: str, saver: LocalSaverService) -> None:
    """
    Step 7b: Extract audio from final cut video.

    Args:
        base_name: Base filename without extension
        saver: Local saver service
    """
    if saver.final_cut_audio_exists(base_name):
        print_progress("Final cut audio already exists, skipping")
        return

    print_progress(f"Extracting audio from final cut: {base_name}")

    video_service = VideoService(ASSETS_DIR)
    video_service.extract_final_cut_audio(base_name, force=False)

    print_progress("Final cut audio extracted")


def downsample_final_cut(base_name: str, saver: LocalSaverService) -> None:
    """
    Step 7c: Downsample the final cut video.

    Args:
        base_name: Base filename without extension
        saver: Local saver service
    """
    downsampled_path = get_final_cut_downsampled_path(base_name)
    if downsampled_path.exists():
        print_progress("Downsampled final cut already exists, skipping")
        return

    print_progress("Downsampling final cut")
    video_service = VideoService(ASSETS_DIR)
    downsampled_path = video_service.downsample_final_cut(
        base_name=base_name,
        force=False,
    )

    print_progress(f"Downsampled final cut created: {downsampled_path.name}")


def transcribe_final_cut(base_name: str, saver: LocalSaverService) -> None:
    """
    Step 7d: Transcribe the final cut video.

    Args:
        base_name: Base filename without extension
        saver: Local saver service
    """
    if saver.final_cut_transcription_exists(base_name):
        print_progress("Final cut transcription already exists, skipping")
        return

    print_progress("Loading final cut audio")
    audio_path = get_final_cut_audio_path(base_name)

    print_progress(f"Transcribing final cut audio: {base_name}")
    stt_service = ElevenLabsSTTService()
    transcript = stt_service.transcribe(audio_path)

    # Save transcription with stage 7 naming
    transcription_path = get_final_cut_transcription_path(base_name)
    transcription_path.write_text(transcript.model_dump_json(indent=2))
    print_progress(f"Final cut transcription saved: {transcription_path.name}")
