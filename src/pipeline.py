"""
Pipeline orchestration functions.
Each function represents a step in the video editing pipeline.
"""

import subprocess
from pathlib import Path

from src.services.video import VideoService, MLTVideoService
from src.services.stt.elevenlabs import ElevenLabsSTTService
from src.services.llm.openrouter import OpenRouterLLMService
from src.services.local_saver import LocalSaverService
from src.services.agents import (
    SentenceSelectionAgent,
    TimestampAdjustmentAgent,
    GoogleDocImagePlacer,
)
from src.services.html_parser import GoogleDocHTMLParser
from src.models import (
    Transcript,
    GoogleDocScript,
    GoogleDocImagePlacements,
)
from src.constants import ASSETS_DIR
from src.util import (
    get_input_video_path,
    get_audio_path,
    print_progress,
    convert_editing_decision_to_result,
    get_edited_video_path,
    get_stage_11_with_google_doc_images_path,
    get_google_doc_html_path,
    get_google_doc_images_folder,
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
        force=True,  # Force regeneration for feedback loop
    )

    print_progress(f"Edited video created: {edited_video_path.name}")


def feedback_loop_for_cut(base_name: str, saver: LocalSaverService) -> None:
    """
    Step 7: Two-stage interactive feedback loop for refining the cut.

    Stage 1: Sentence Selection - User reviews which sentences to keep/remove (s4)
    Stage 2: Timestamp Adjustment - User reviews and adjusts timestamps (s5)

    Args:
        base_name: Base filename without extension
        saver: Local saver service
    """
    print("\n" + "=" * 60)
    print("FEEDBACK LOOP - Two-Stage Cut Review")
    print("=" * 60)

    # Get the path to the downsampled edited video
    edited_video_path = get_edited_video_path(base_name, use_downsampled=True)

    # Load necessary data
    transcript = saver.load_transcription(base_name)
    from src.util import get_editing_result_path

    editing_result_path = get_editing_result_path(base_name)

    # =====================================================================
    # STAGE 1: SENTENCE SELECTION (s4 - editing_result.json)
    # =====================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: SENTENCE SELECTION")
    print("Which sentences should be kept or removed?")
    print("=" * 60)

    sentence_agent = SentenceSelectionAgent()
    stage1_iteration = 1
    max_iterations = 10  # Safety limit

    while stage1_iteration <= max_iterations:
        print(f"\n--- Stage 1 - Iteration {stage1_iteration} ---")

        # Reload editing result on each iteration
        editing_result = saver.load_editing_result(base_name)

        # Show current sentence status
        print("\n📋 Current sentence selection:")
        kept_count = sum(
            1 for sr in editing_result.sentence_results.values() if sr.keep
        )
        removed_count = sum(
            1 for sr in editing_result.sentence_results.values() if not sr.keep
        )
        print(f"   ✓ Kept: {kept_count} sentences")
        print(f"   ✗ Removed: {removed_count} sentences")

        # Regenerate adjusted sentences and video from current editing result
        print("\n🎬 Generating video with current sentence selection...")
        video_service = VideoService(ASSETS_DIR)
        adjusted_sentences = video_service.generate_adjusted_sentences(
            base_name=base_name,
            transcript=transcript,
            editing_result=editing_result,
            use_downsampled=True,
        )
        saver.save_adjusted_sentences(base_name, adjusted_sentences)

        edited_video_path = video_service.create_edited_video(
            base_name=base_name,
            adjusted_sentences=adjusted_sentences,
            use_downsampled=True,
            force=True,
        )

        print(f"\n📹 Video location: {edited_video_path}")
        print("Please review the video to see which sentences are included.")

        # Get user feedback
        print("\n💬 Is the sentence selection good?")
        print("   (Type 'looks good', 'approve', or 'perfect' if satisfied)")
        print("   (Or provide feedback like 'remove sentence 5' or 'keep sentence 3')")
        user_feedback = input("\nYour feedback: ").strip()

        if not user_feedback:
            print("⚠ No feedback provided. Please try again.")
            continue

        # Process feedback with sentence selection agent
        try:
            print("\n🤖 Processing feedback with Sentence Selection Agent...")
            updated_editing_result, is_approved = sentence_agent.process_feedback(
                editing_result=editing_result,
                user_feedback=user_feedback,
            )

            if is_approved:
                print(
                    "\n✅ Sentence selection approved! Moving to timestamp adjustment stage."
                )
                # Save final editing result from stage 1
                saver.save_editing_result(base_name, updated_editing_result)
                break

            # Save updated editing result
            print("\n💾 Saving updated sentence selection...")
            saver.save_editing_result(base_name, updated_editing_result)

            stage1_iteration += 1

        except Exception as e:
            print(f"\n❌ Error processing feedback: {str(e)}")
            print("Please try again with different feedback.")
            continue

    if stage1_iteration > max_iterations:
        print(
            f"\n⚠ Warning: Reached maximum iterations ({max_iterations}) for sentence selection"
        )
        print("Proceeding with current state.")

    # =====================================================================
    # STAGE 2: TIMESTAMP ADJUSTMENT (s5 - adjusted_sentences.json)
    # =====================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: TIMESTAMP ADJUSTMENT")
    print("Fine-tune the timestamps of selected sentences")
    print("=" * 60)

    # Regenerate adjusted sentences from approved editing result
    print("\n🔄 Regenerating adjusted sentences from approved sentence selection...")
    editing_result = saver.load_editing_result(base_name)
    video_service = VideoService(ASSETS_DIR)
    adjusted_sentences = video_service.generate_adjusted_sentences(
        base_name=base_name,
        transcript=transcript,
        editing_result=editing_result,
        use_downsampled=True,
    )
    saver.save_adjusted_sentences(base_name, adjusted_sentences)

    # Regenerate video
    edited_video_path = video_service.create_edited_video(
        base_name=base_name,
        adjusted_sentences=adjusted_sentences,
        use_downsampled=True,
        force=True,
    )

    print(f"\n📹 Video location: {edited_video_path}")
    print("Please review the video to check timestamps and pacing.")

    timestamp_agent = TimestampAdjustmentAgent()
    stage2_iteration = 1

    while stage2_iteration <= max_iterations:
        print(f"\n--- Stage 2 - Iteration {stage2_iteration} ---")

        # Get user feedback
        print("\n💬 How do the timestamps look?")
        print("   (Type 'looks good', 'approve', or 'perfect' if satisfied)")
        print(
            "   (Or provide feedback like 'cut 2 seconds from the beginning' or 'reduce pause between sentence 3 and 4')"
        )
        user_feedback = input("\nYour feedback: ").strip()

        if not user_feedback:
            print("⚠ No feedback provided. Please try again.")
            continue

        # Reload adjusted sentences on each iteration
        adjusted_sentences = saver.load_adjusted_sentences(base_name)

        # Process feedback with timestamp adjustment agent
        try:
            print("\n🤖 Processing feedback with Timestamp Adjustment Agent...")
            updated_sentences, is_approved = timestamp_agent.process_feedback(
                adjusted_sentences=adjusted_sentences,
                user_feedback=user_feedback,
            )

            if is_approved:
                print("\n✅ Timestamps approved! Cut refinement complete.")
                # Save final adjusted sentences
                saver.save_adjusted_sentences(base_name, updated_sentences)
                break

            # Save updated sentences
            print("\n💾 Saving updated timestamps...")
            saver.save_adjusted_sentences(base_name, updated_sentences)

            # Regenerate the video with updated sentences
            print("\n🎬 Regenerating video with timestamp adjustments...")
            video_service = VideoService(ASSETS_DIR)
            edited_video_path = video_service.create_edited_video(
                base_name=base_name,
                adjusted_sentences=updated_sentences,
                use_downsampled=True,
                force=True,
            )

            print(f"\n✓ Updated video created: {edited_video_path.name}")
            print("Please review the updated video.")

            stage2_iteration += 1

        except Exception as e:
            print(f"\n❌ Error processing feedback: {str(e)}")
            print("Please try again with different feedback.")
            continue

    if stage2_iteration > max_iterations:
        print(
            f"\n⚠ Warning: Reached maximum iterations ({max_iterations}) for timestamp adjustment"
        )
        print("Proceeding with current state.")

    print("\n" + "=" * 60)
    print("Two-stage feedback loop complete!")
    print("=" * 60)


def parse_google_doc_script(
    base_name: str, saver: LocalSaverService
) -> GoogleDocScript:
    """
    Step 8: Parse Google Doc HTML to extract text lines and associated images.
    Saves the parsed script to s8_google_doc_script.json.

    Args:
        base_name: Base filename without extension
        saver: Local saver service

    Returns:
        GoogleDocScript with parsed lines and image associations

    Raises:
        FileNotFoundError: If Google Doc HTML doesn't exist
    """
    # Check if script already exists
    if saver.google_doc_script_exists(base_name):
        print_progress("Google Doc script already exists, loading from file")
        script = saver.load_google_doc_script(base_name)

        # Print summary
        lines_with_images = sum(1 for line in script.lines if line.image_filename)
        print_progress(f"Loaded {len(script.lines)} lines from saved script")
        print_progress(f"  - Lines with text: {len(script.lines)}")
        print_progress(f"  - Lines with images: {lines_with_images}")

        return script

    print_progress(f"Parsing Google Doc HTML for: {base_name}")

    # Check if HTML exists
    if not saver.google_doc_html_exists(base_name):
        html_path = get_google_doc_html_path(base_name)
        raise FileNotFoundError(
            f"Google Doc HTML not found: {html_path}\n"
            f"Expected location: assets/{base_name}/google_doc/{base_name}.html"
        )

    # Load HTML content
    html_content = saver.load_google_doc_html(base_name)

    # Parse HTML
    parser = GoogleDocHTMLParser()
    script = parser.parse_html(html_content)

    print_progress(f"Parsed {len(script.lines)} lines from Google Doc")

    # Save the parsed script
    script_path = saver.save_google_doc_script(base_name, script)
    print_progress(f"Saved script to: {script_path.name}")

    # Print summary
    lines_with_images = sum(1 for line in script.lines if line.image_filename)
    print_progress(f"  - Lines with text: {len(script.lines)}")
    print_progress(f"  - Lines with images: {lines_with_images}")

    # Print sample of parsed content
    print("\nSample of parsed content:")
    for i, line in enumerate(script.lines[:5], 1):
        image_info = f" [image: {line.image_filename}]" if line.image_filename else ""
        print(f"  {i}. {line.text[:60]}...{image_info}")

    if len(script.lines) > 5:
        print(f"  ... and {len(script.lines) - 5} more lines")

    return script


def render_shotcut_mlt(force: bool = False) -> Path:
    """
    Render video from Shotcut MLT file (for testing).
    This function is kept for future use but not included in the main pipeline.

    Args:
        force: If True, regenerate even if file exists

    Returns:
        Path to rendered video file
    """
    print("\n" + "=" * 60)
    print("Render Shotcut MLT (Testing)")
    print("=" * 60)

    # Hard-coded paths for testing
    mlt_path = Path(
        "/Users/deanoglellc/Desktop/Brogle/Shorts/workspace/video_editing/shotcut/shotcut_xml.mlt"
    )
    output_path = Path(
        "/Users/deanoglellc/Desktop/Brogle/Shorts/workspace/video_editing/shotcut/shotcut_output.mp4"
    )

    if output_path.exists() and not force:
        print_progress(f"Output already exists: {output_path}")
        return output_path

    if not mlt_path.exists():
        raise FileNotFoundError(f"MLT file not found: {mlt_path}")

    print_progress(f"Rendering video from: {mlt_path}")

    cmd = [
        "melt",
        str(mlt_path),
        "-consumer",
        f"avformat:{output_path}",
        "vcodec=libx264",
        "acodec=aac",
        "crf=23",
        "preset=fast",
        "movflags=+faststart",
        "real_time=-1",
        "rescale=bilinear",
        "deinterlace_method=yadif",
        "top_field_first=2",
    ]
    print_progress("Running melt command...")
    print_progress(f"Command: {' '.join(cmd)}")

    subprocess.run(cmd, capture_output=True, text=True, check=True)

    print("\n" + "=" * 60)
    print("Render Complete!")
    print("=" * 60)
    print(f"\nRendered video: {output_path}")

    return output_path


def place_google_doc_images(
    base_name: str, saver: LocalSaverService
) -> GoogleDocImagePlacements:
    """
    Step 9: Place images from Google Doc script onto video timeline.

    Args:
        base_name: Base filename without extension
        saver: Local saver service

    Returns:
        GoogleDocImagePlacements with image timing information

    Raises:
        FileNotFoundError: If required files don't exist
    """
    # Check if placements already exist
    if saver.google_doc_image_placements_exist(base_name):
        print_progress("Google Doc image placements already exist, loading from file")
        placements = saver.load_google_doc_image_placements(base_name)
        print_progress(f"Loaded {len(placements.placements)} image placements")
        for i, placement in enumerate(placements.placements, 1):
            sentence_range = (
                f"{placement.sentence_indexes[0]}-{placement.sentence_indexes[-1]}"
                if len(placement.sentence_indexes) > 1
                else placement.sentence_indexes[0]
            )
            print_progress(
                f"  {i}. {Path(placement.filepath).name}: sentences {sentence_range}"
            )
        return placements

    print_progress(f"Placing Google Doc images for: {base_name}")

    # Load required data
    print_progress("Loading Google Doc script and adjusted sentences")
    google_doc_script = saver.load_google_doc_script(base_name)
    adjusted_sentences = saver.load_adjusted_sentences(base_name)

    # Get Google Doc images folder
    google_doc_images_folder = get_google_doc_images_folder(base_name)
    if not google_doc_images_folder.exists():
        raise FileNotFoundError(
            f"Google Doc images folder not found: {google_doc_images_folder}"
        )

    # Create agent and place images
    print_progress("Using LLM to match script images to video timeline")
    agent = GoogleDocImagePlacer()
    placements = agent.place_images(
        google_doc_script=google_doc_script,
        adjusted_sentences=adjusted_sentences,
        google_doc_images_folder=google_doc_images_folder,
    )

    # Save placements
    placements_path = saver.save_google_doc_image_placements(base_name, placements)
    print_progress(f"Saved image placements to: {placements_path.name}")

    # Print summary
    print_progress(f"Successfully placed {len(placements.placements)} images")
    for i, placement in enumerate(placements.placements, 1):
        sentence_range = (
            f"{placement.sentence_indexes[0]}-{placement.sentence_indexes[-1]}"
            if len(placement.sentence_indexes) > 1
            else placement.sentence_indexes[0]
        )
        num_sentences = len(placement.sentence_indexes)
        print_progress(
            f"  {i}. {Path(placement.filepath).name}: sentences {sentence_range} ({num_sentences} sentence{'s' if num_sentences > 1 else ''})"
        )

    return placements


def create_video_with_google_doc_images(
    base_name: str, saver: LocalSaverService, force: bool = False
) -> Path:
    """
    Step 10: Create video with Google Doc image overlays.

    Args:
        base_name: Base filename without extension
        saver: Local saver service
        force: If True, regenerate even if file exists

    Returns:
        Path to video with Google Doc images

    Raises:
        FileNotFoundError: If required files don't exist
    """
    output_path = get_stage_11_with_google_doc_images_path(base_name)
    if output_path.exists() and not force:
        print_progress("Video with Google Doc images already exists, skipping")
        return output_path

    print_progress("Loading adjusted sentences and Google Doc image placements")
    adjusted_sentences = saver.load_adjusted_sentences(base_name)
    image_placements = saver.load_google_doc_image_placements(base_name)

    print_progress("Creating video with Google Doc image overlays using MLT")
    mlt_service = MLTVideoService()
    video_path = mlt_service.create_video_with_google_doc_images(
        base_name=base_name,
        adjusted_sentences=adjusted_sentences,
        image_placements=image_placements,
        force=force,
    )

    print_progress(f"Video with Google Doc images created: {video_path.name}")
    return video_path


def create_full_res_cut_video(
    base_name: str, saver: LocalSaverService, force: bool = False
) -> Path:
    """
    Step 11: Cut full resolution video using MLT based on adjusted sentences.

    Args:
        base_name: Base filename without extension
        saver: Local saver service
        force: If True, regenerate even if file exists

    Returns:
        Path to full resolution cut video

    Raises:
        FileNotFoundError: If required files don't exist
    """
    print_progress("Loading adjusted sentences")
    adjusted_sentences = saver.load_adjusted_sentences(base_name)

    print_progress("Creating full resolution cut video using MLT")
    mlt_service = MLTVideoService()
    video_path = mlt_service.create_full_res_cut_video(
        base_name=base_name,
        adjusted_sentences=adjusted_sentences,
        force=force,
    )

    print_progress(f"Full resolution cut video created: {video_path.name}")
    return video_path


def create_full_res_video_with_images(
    base_name: str, saver: LocalSaverService, force: bool = False
) -> Path:
    """
    Step 12: Create full resolution video with Google Doc image overlays using MLT.

    Args:
        base_name: Base filename without extension
        saver: Local saver service
        force: If True, regenerate even if file exists

    Returns:
        Path to full resolution video with Google Doc images

    Raises:
        FileNotFoundError: If required files don't exist
    """
    print_progress("Loading adjusted sentences and Google Doc image placements")
    adjusted_sentences = saver.load_adjusted_sentences(base_name)
    image_placements = saver.load_google_doc_image_placements(base_name)

    print_progress(
        "Creating full resolution video with Google Doc image overlays using MLT"
    )
    mlt_service = MLTVideoService()
    video_path = mlt_service.create_full_res_video_with_images(
        base_name=base_name,
        adjusted_sentences=adjusted_sentences,
        image_placements=image_placements,
        force=force,
    )

    print_progress(
        f"Full resolution video with Google Doc images created: {video_path.name}"
    )
    return video_path


def create_full_res_video_single_pass(
    base_name: str, saver: LocalSaverService, force: bool = False
) -> Path:
    """
    Step 11: Create full resolution video with cuts AND images in a single MLT pass.

    This is more efficient than the two-step approach (Steps 11+12) as it does
    both cutting and image overlay in one rendering operation.

    Args:
        base_name: Base filename without extension
        saver: Local saver service
        force: If True, regenerate even if file exists

    Returns:
        Path to full resolution video with cuts and images

    Raises:
        FileNotFoundError: If required files don't exist
    """
    print_progress("Loading adjusted sentences and Google Doc image placements")
    adjusted_sentences = saver.load_adjusted_sentences(base_name)
    image_placements = saver.load_google_doc_image_placements(base_name)

    print_progress("Creating full resolution video with cuts and images (single pass)")
    mlt_service = MLTVideoService()
    video_path = mlt_service.create_full_res_video_with_images_single_pass(
        base_name=base_name,
        adjusted_sentences=adjusted_sentences,
        image_placements=image_placements,
        force=force,
    )

    print_progress(f"Full resolution video created: {video_path.name}")
    return video_path
