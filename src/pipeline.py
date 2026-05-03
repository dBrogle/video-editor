import shutil
import subprocess
import zipfile
from pathlib import Path

from src.services.video import VideoService, MLTVideoService
from src.services.stt.deepgram import DeepgramSTTService
from src.services.llm.openrouter import OpenRouterLLMService
from src.services.local_saver import LocalSaverService
from src.services.agents import (
    SentenceSelectionAgent,
    TimestampAdjustmentAgent,
    GoogleDocImagePlacer,
)
from src.services.html_parser import GoogleDocHTMLParser
from src.models import Transcript, GoogleDocScript, GoogleDocImagePlacements
from src.constants import ASSETS_DIR, HD_1080P_HEIGHT
from src.util import (
    get_input_video_path,
    get_audio_path,
    get_base_folder,
    get_google_doc_folder,
    get_google_doc_html_path,
    get_google_doc_images_folder,
    print_progress,
    convert_editing_decision_to_result,
    get_stage_11_with_google_doc_images_path,
    get_1080p_downsample_video_path,
    get_1080p_with_images_video_path,
    get_sentence_selection_video_path,
    get_adjusted_sentences_video_path,
)


def _prescale_google_doc_images(images_folder: Path, target_width: int = 1080, target_height: int = 1920) -> None:
    """Scale up any Google Doc images that are smaller than the video safe zone, in-place."""
    from PIL import Image
    from src.constants import (
        IMAGE_SAFE_ZONE_TOP_PERCENT,
        IMAGE_SAFE_ZONE_BOTTOM_PERCENT,
        IMAGE_SAFE_ZONE_LEFT_PERCENT,
        IMAGE_SAFE_ZONE_RIGHT_PERCENT,
    )

    sz_width = int(target_width * (IMAGE_SAFE_ZONE_RIGHT_PERCENT - IMAGE_SAFE_ZONE_LEFT_PERCENT))
    sz_height = int(target_height * (IMAGE_SAFE_ZONE_BOTTOM_PERCENT - IMAGE_SAFE_ZONE_TOP_PERCENT))

    for img_path in sorted(images_folder.iterdir()):
        if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
            continue
        with Image.open(img_path) as img:
            w, h = img.size
            if w >= sz_width and h >= sz_height:
                continue
            width_scale = sz_width / w
            height_scale = sz_height / h
            scale = max(width_scale, height_scale)
            new_w, new_h = int(w * scale), int(h * scale)
            print_progress(f"    Prescaling {img_path.name}: {w}x{h} -> {new_w}x{new_h}")
            scaled = img.resize((new_w, new_h), Image.LANCZOS)
            scaled.save(img_path)
            scaled.close()


def _preprocess_google_doc_zip(base_name: str, force: bool = False) -> bool:
    base_folder = get_base_folder(base_name)
    google_doc_folder = get_google_doc_folder(base_name)
    target_html_path = get_google_doc_html_path(base_name)
    target_images_folder = get_google_doc_images_folder(base_name)

    if target_html_path.exists() and target_images_folder.exists() and not force:
        print_progress("  - Google Doc already extracted, skipping")
        if target_images_folder.exists():
            _prescale_google_doc_images(target_images_folder)
        return True

    zip_files = list(base_folder.glob("*.zip"))
    if not zip_files:
        print_progress("  - No Google Doc zip found, skipping")
        return False

    if len(zip_files) > 1:
        raise ValueError(f"Multiple zip files found in {base_folder}: {[z.name for z in zip_files]}")

    zip_path = zip_files[0]
    print_progress(f"  - Extracting Google Doc zip: {zip_path.name}")

    temp_extract_dir = base_folder / "_temp_zip_extract"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir()

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_extract_dir)

        html_files = list(temp_extract_dir.rglob("*.html"))
        if not html_files:
            raise ValueError(f"No .html file found in zip: {zip_path.name}")
        if len(html_files) > 1:
            raise ValueError(f"Multiple .html files found in zip: {[h.name for h in html_files]}")

        images_folders = [d for d in temp_extract_dir.rglob("*") if d.is_dir() and d.name == "images"]
        if not images_folders:
            raise ValueError(f"No 'images' folder found in zip: {zip_path.name}")
        if len(images_folders) > 1:
            raise ValueError(f"Multiple 'images' folders found in zip: {[str(i) for i in images_folders]}")

        html_file = html_files[0]
        images_folder = images_folders[0]

        google_doc_folder.mkdir(parents=True, exist_ok=True)

        if target_html_path.exists():
            target_html_path.unlink()
        shutil.move(str(html_file), str(target_html_path))

        if target_images_folder.exists():
            shutil.rmtree(target_images_folder)
        shutil.move(str(images_folder), str(target_images_folder))

        print_progress(f"    HTML: {target_html_path.name}")
        print_progress(f"    Images: {target_images_folder.name}/")
        _prescale_google_doc_images(target_images_folder)
        return True

    finally:
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)


def stage_1_preprocess_video_and_files(
    base_name: str, saver: LocalSaverService, force: bool = False
) -> None:
    print_progress(f"Preprocessing video and files: {base_name}")

    print_progress("  - Checking video rotation...")
    mlt_service = MLTVideoService()
    output_path = mlt_service.rotate_video_if_needed(base_name, force=force)
    print_progress(f"    Video ready: {output_path.name}")

    if saver.downsampled_video_exists(base_name) and not force:
        print_progress("  - Downsampled video already exists, skipping")
    else:
        print_progress("  - Downsampling video...")
        input_path = get_input_video_path(base_name)
        video_service = VideoService(ASSETS_DIR)
        video_service.generate_proxy_video(input_path, force=force)
        print_progress("    Downsampled video created")

    if saver.audio_exists(base_name) and not force:
        print_progress("  - Audio file already exists, skipping")
    else:
        print_progress("  - Extracting audio...")
        input_path = get_input_video_path(base_name)
        video_service = VideoService(ASSETS_DIR)
        video_service.extract_audio(input_path, force=force)
        print_progress("    Audio extracted")

    _preprocess_google_doc_zip(base_name, force=force)

    print_progress("Preprocessing complete")


def stage_2_get_transcription(base_name: str, saver: LocalSaverService) -> Transcript:
    if saver.transcription_exists(base_name):
        print_progress("Transcription already exists, loading from file")
        return saver.load_transcription(base_name)

    print_progress(f"Transcribing audio: {base_name}")
    audio_path = get_audio_path(base_name)

    stt_service = DeepgramSTTService()
    transcript = stt_service.transcribe(audio_path)

    saver.save_transcription(base_name, transcript)
    print_progress("Transcription saved")

    return transcript


def stage_3_initial_edit_with_llm(base_name: str, saver: LocalSaverService, force: bool = False) -> None:
    if saver.editing_decision_exists(base_name) and not force:
        print_progress("Editing decision already exists, skipping")
        print_progress("If you want to regenerate, delete s3_editing_decision.json and s3_editing_result.json and run again")
        return

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


def stage_5_generate_adjusted_sentences(
    base_name: str, saver: LocalSaverService, skip_silence_removal: bool = False
) -> None:
    if saver.adjusted_sentences_exist(base_name):
        print_progress("Adjusted sentences already exist, skipping")
        print_progress(
            "If you want to regenerate, delete s5_adjusted_sentences.json and run again"
        )
        return

    print_progress("Loading transcript and editing result")
    transcript = saver.load_transcription(base_name)

    editing_result = saver.load_final_editing_result(base_name)
    print_progress("Using final sentence selection from step 4")

    if skip_silence_removal:
        print_progress("Generating adjusted sentences (skipping silence removal)")
    else:
        print_progress("Generating adjusted sentences with silence removal")

    video_service = VideoService(ASSETS_DIR)
    adjusted_sentences = video_service.generate_adjusted_sentences(
        base_name=base_name,
        transcript=transcript,
        editing_result=editing_result,
        use_downsampled=True,
        skip_silence_removal=skip_silence_removal,
    )

    adjusted_path = saver.save_adjusted_sentences(base_name, adjusted_sentences)
    print_progress(f"Adjusted sentences saved to: {adjusted_path.name}")


def stage_6_iterate_adjusted_sentences(
    base_name: str, saver: LocalSaverService, skip_silence_removal: bool = False
) -> None:
    if saver.final_adjusted_sentences_exist(base_name):
        print_progress("Final adjusted sentences already exist, skipping iteration")
        print_progress(
            "If you want to re-iterate, delete s6_final_adjusted_sentences.json and run again"
        )
        return

    print("\n" + "=" * 60)
    print("TIMESTAMP ADJUSTMENT ITERATION")
    print("Fine-tune the timestamps of selected sentences")
    print("=" * 60)

    # Load necessary data
    transcript = saver.load_transcription(base_name)

    # Try to load final editing result first, fall back to working copy
    if saver.final_editing_result_exists(base_name):
        editing_result = saver.load_final_editing_result(base_name)
        print_progress("Using final sentence selection from step 4")
    else:
        editing_result = saver.load_editing_result(base_name)
        print_progress("Using working sentence selection from step 3")

    # Regenerate adjusted sentences from editing result
    print("\n🔄 Generating adjusted sentences from sentence selection...")
    video_service = VideoService(ASSETS_DIR)
    adjusted_sentences = video_service.generate_adjusted_sentences(
        base_name=base_name,
        transcript=transcript,
        editing_result=editing_result,
        use_downsampled=True,
        skip_silence_removal=skip_silence_removal,
    )
    saver.save_adjusted_sentences(base_name, adjusted_sentences)

    # Generate initial video to s6_adjusted_sentences_video.mp4
    adjusted_sentences_video_path = get_adjusted_sentences_video_path(base_name)
    video_service.create_edited_video(
        base_name=base_name,
        adjusted_sentences=adjusted_sentences,
        use_downsampled=True,
        force=True,
        output_path=adjusted_sentences_video_path,
    )

    print(f"\n📹 Video location: {adjusted_sentences_video_path}")
    print("Please review the video to check timestamps and pacing.")

    timestamp_agent = TimestampAdjustmentAgent()
    iteration = 1
    max_iterations = 10  # Safety limit

    while iteration <= max_iterations:
        print(f"\n--- Iteration {iteration} ---")

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

        # Reload adjusted sentences on each iteration (from s5_adjusted_sentences.json)
        adjusted_sentences = saver.load_adjusted_sentences(base_name)

        # Process feedback with timestamp adjustment agent
        try:
            print("\n🤖 Processing feedback with Timestamp Adjustment Agent...")
            updated_sentences, is_approved = timestamp_agent.process_feedback(
                adjusted_sentences=adjusted_sentences,
                user_feedback=user_feedback,
            )

            if is_approved:
                print("\n✅ Timestamps approved!")
                # Save to s5_adjusted_sentences.json (update working copy)
                saver.save_adjusted_sentences(base_name, updated_sentences)
                # Save to s6_final_adjusted_sentences.json (final approved copy)
                final_path = saver.save_final_adjusted_sentences(
                    base_name, updated_sentences
                )
                print(f"✓ Final timestamps saved to: {final_path.name}")
                break

            # Save updated sentences back to s5_adjusted_sentences.json
            print("\n💾 Saving updated timestamps...")
            saver.save_adjusted_sentences(base_name, updated_sentences)

            # Regenerate the video with updated sentences
            print("\n🎬 Regenerating video with timestamp adjustments...")
            video_service = VideoService(ASSETS_DIR)
            video_service.create_edited_video(
                base_name=base_name,
                adjusted_sentences=updated_sentences,
                use_downsampled=True,
                force=True,
                output_path=adjusted_sentences_video_path,
            )

            print(f"\n✓ Updated video created: {adjusted_sentences_video_path.name}")
            print("Please review the updated video.")

            iteration += 1

        except Exception as e:
            print(f"\n❌ Error processing feedback: {str(e)}")
            print("Please try again with different feedback.")
            continue

    if iteration > max_iterations:
        print(
            f"\n⚠ Warning: Reached maximum iterations ({max_iterations}) for timestamp adjustment"
        )
        print("Proceeding with current state.")

    print("\n" + "=" * 60)
    print("Timestamp adjustment iteration complete!")
    print("=" * 60)


def stage_4_iterate_sentence_selection(
    base_name: str, saver: LocalSaverService, skip_silence_removal: bool = False
) -> None:
    if saver.final_editing_result_exists(base_name):
        print_progress("Final sentence selection already exists, skipping iteration")
        print_progress(
            "If you want to re-iterate, delete s4_final_editing_result.json and run again"
        )
        return

    print("\n" + "=" * 60)
    print("SENTENCE SELECTION ITERATION")
    print("Which sentences should be kept or removed?")
    print("=" * 60)

    # Load necessary data
    transcript = saver.load_transcription(base_name)

    sentence_agent = SentenceSelectionAgent()
    iteration = 1
    max_iterations = 10  # Safety limit

    while iteration <= max_iterations:
        print(f"\n--- Iteration {iteration} ---")

        # Reload editing result on each iteration (from s3_editing_result.json)
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
            skip_silence_removal=skip_silence_removal,
        )

        # Generate video to s4_sentence_selection_video.mp4
        sentence_selection_video_path = get_sentence_selection_video_path(base_name)
        video_service.create_edited_video(
            base_name=base_name,
            adjusted_sentences=adjusted_sentences,
            use_downsampled=True,
            force=True,
            output_path=sentence_selection_video_path,
        )

        print(f"\n📹 Video location: {sentence_selection_video_path}")
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
                print("\n✅ Sentence selection approved!")
                # Save to s3_editing_result.json (update working copy)
                saver.save_editing_result(base_name, updated_editing_result)
                # Save to s4_final_editing_result.json (final approved copy)
                final_path = saver.save_final_editing_result(
                    base_name, updated_editing_result
                )
                print(f"✓ Final selection saved to: {final_path.name}")
                break

            # Save updated editing result back to s3_editing_result.json
            print("\n💾 Saving updated sentence selection...")
            saver.save_editing_result(base_name, updated_editing_result)

            iteration += 1

        except Exception as e:
            print(f"\n❌ Error processing feedback: {str(e)}")
            print("Please try again with different feedback.")
            continue

    if iteration > max_iterations:
        print(
            f"\n⚠ Warning: Reached maximum iterations ({max_iterations}) for sentence selection"
        )
        print("Proceeding with current state.")

    print("\n" + "=" * 60)
    print("Sentence selection iteration complete!")
    print("=" * 60)


def stage_7_parse_google_doc_script(
    base_name: str, saver: LocalSaverService
) -> GoogleDocScript:
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


def stage_8_place_google_doc_images(
    base_name: str, saver: LocalSaverService
) -> GoogleDocImagePlacements:
    if saver.google_doc_image_placements_exist(base_name):
        print_progress("Google Doc image placements already exist, loading from file")
        placements = saver.load_google_doc_image_placements(base_name)
        print_progress(f"Loaded {len(placements.placements)} image placements")
        for i, placement in enumerate(placements.placements, 1):
            print_progress(
                f"  {i}. {Path(placement.filepath).name}: sentence {placement.sentence_index} [{placement.start_fraction:.1f}-{placement.end_fraction:.1f}]"
            )
        return placements

    print_progress(f"Placing Google Doc images for: {base_name}")

    # Load required data
    print_progress("Loading Google Doc script and adjusted sentences")
    google_doc_script = saver.load_google_doc_script(base_name)
    adjusted_sentences = saver.load_best_adjusted_sentences(base_name)

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
        print_progress(
            f"  {i}. {Path(placement.filepath).name}: sentence {placement.sentence_index} [{placement.start_fraction:.1f}-{placement.end_fraction:.1f}]"
        )

    return placements


def stage_9_create_video_with_google_doc_images(
    base_name: str, saver: LocalSaverService, force: bool = False
) -> Path:
    output_path = get_stage_11_with_google_doc_images_path(base_name)
    if output_path.exists() and not force:
        print_progress("Video with Google Doc images already exists, skipping")
        return output_path

    print_progress("Loading adjusted sentences and Google Doc image placements")
    adjusted_sentences = saver.load_best_adjusted_sentences(base_name)
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


def stage_10_downsample_to_1080p(base_name: str, force: bool = False) -> Path:
    output_path = get_1080p_downsample_video_path(base_name)

    if output_path.exists() and not force:
        print_progress(f"1080p downsampled video already exists: {output_path}")
        return output_path

    print_progress("Downsampling full resolution video to 1080x1920...")
    input_path = get_input_video_path(base_name)

    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-vf",
        f"scale=1080:{HD_1080P_HEIGHT}",  # 1080 width, 1920 height for vertical video
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",  # Good quality
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-y",  # Overwrite output file
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print_progress(f"1080p downsampled video created: {output_path.name}")
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg 1080p downsampling failed:\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {e.returncode}\n"
            f"stderr: {e.stderr}"
        ) from e


def stage_11_create_1080p_video_with_images(
    base_name: str, saver: LocalSaverService, force: bool = False
) -> Path:
    output_path = get_1080p_with_images_video_path(base_name)

    if output_path.exists() and not force:
        print_progress(f"1080p video with images already exists: {output_path}")
        return output_path

    print_progress("Loading adjusted sentences and Google Doc image placements")
    adjusted_sentences = saver.load_best_adjusted_sentences(base_name)
    image_placements = saver.load_google_doc_image_placements(base_name)

    print_progress("Creating 1080p video with cuts and images using MLT")
    mlt_service = MLTVideoService()
    video_path = mlt_service.create_1080p_video_with_images(
        base_name=base_name,
        adjusted_sentences=adjusted_sentences,
        image_placements=image_placements,
        force=force,
    )

    print_progress(f"1080p video with images created: {video_path.name}")
    return video_path


def stage_12_create_full_res_video_single_pass(
    base_name: str, saver: LocalSaverService, force: bool = False
) -> Path:
    print_progress("Loading adjusted sentences and Google Doc image placements")
    adjusted_sentences = saver.load_best_adjusted_sentences(base_name)
    image_placements = saver.load_google_doc_image_placements(base_name)

    print_progress(
        "Creating full resolution video with cuts and images (single pass from 1080p)"
    )
    mlt_service = MLTVideoService()
    video_path = mlt_service.create_full_res_video_with_images_single_pass(
        base_name=base_name,
        adjusted_sentences=adjusted_sentences,
        image_placements=image_placements,
        force=force,
    )

    print_progress(f"Full resolution video created: {video_path.name}")
    return video_path
