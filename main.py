"""
Main entry point for the video editing pipeline.
Provides an interactive menu for running pipeline steps.
"""

from src.services.local_saver import LocalSaverService
from src.util import extract_filename_without_extension, get_input_video_path
from src.pipeline import (
    rotate_video,
    downsample_video,
    extract_audio,
    get_transcription,
    prompt_llm_for_editing,
    generate_adjusted_sentences,
    create_edited_video,
    feedback_loop_for_cut,
    parse_google_doc_script,
    place_google_doc_images,
    create_video_with_google_doc_images,
    create_full_res_cut_video,
    create_full_res_video_with_images,
    create_full_res_video_single_pass,
)


def display_menu() -> list[int]:
    """
    Display menu and get user's step selections.

    Returns:
        List of selected step numbers
    """
    print("\n" + "=" * 50)
    print("VIDEO EDITING PIPELINE")
    print("=" * 50)
    print("\nAvailable steps:")
    print("  0. Rotate video if needed (check rotation metadata)")
    print("  1. Downsample video")
    print("  2. Extract audio")
    print("  3. Get transcription")
    print("  4. Prompt LLM for editing")
    print("  5. Generate adjusted sentences (silence removal)")
    print("  6. Create edited video (downsampled)")
    print(
        "  7. Two-stage feedback loop - Review sentence selection & adjust timestamps"
    )
    print("  8. Parse Google Doc script (extract text & images)")
    print("  9. Place Google Doc images (LLM-based placement)")
    print(" 10. Create video with Google Doc images (downsampled)")
    print(" 11. Cut full resolution video + add images (single pass)")
    print("\n Advanced (two-step approach):")
    print(" 12. Cut full resolution video only (no images)")
    print(" 13. Add images to full resolution video (requires step 12)")
    print("\n 99. Run all steps (0-11, using single-pass approach)")
    print("\nEnter step numbers separated by commas (e.g., 0,1,2,3)")
    print("or enter 99 to run all steps.")

    while True:
        choice = input("\nYour selection: ").strip()

        if choice == "99":
            return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        try:
            steps = [int(s.strip()) for s in choice.split(",")]
            valid_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            if all(step in valid_steps for step in steps):
                return sorted(set(steps))
            else:
                print("Error: Please enter valid step numbers (0-13)")
        except ValueError:
            print("Error: Invalid input. Please enter numbers separated by commas")


def get_input_filename() -> str:
    """
    Get and validate input filename from user.

    Returns:
        Base filename without extension
    """
    print("\n" + "-" * 50)

    saver = LocalSaverService()
    last_filename = saver.get_last_filename()

    while True:
        if last_filename:
            prompt = f"Enter input video filename (default: {last_filename}): "
        else:
            prompt = "Enter input video filename (e.g., IMG_0901.MOV): "

        filename = input(prompt).strip()

        if not filename:
            if last_filename:
                print(f"Using last filename: {last_filename}")
                return last_filename
            else:
                print("Error: Filename cannot be empty")
                continue

        base_name = extract_filename_without_extension(filename)
        input_path = get_input_video_path(base_name)

        if input_path.exists():
            print(f"✓ Found: {input_path.name}")
            saver.save_last_filename(base_name)
            return base_name
        else:
            print("Error: Video file not found in assets/")
            print(f"Expected path: {input_path}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != "y":
                raise FileNotFoundError(f"Video file not found: {filename}")


def run_pipeline(base_name: str, steps: list[int]) -> None:
    """
    Run selected pipeline steps.

    Args:
        base_name: Base filename without extension
        steps: List of step numbers to run
    """
    print("\n" + "=" * 50)
    print(f"RUNNING PIPELINE: {base_name}")
    print("=" * 50)

    saver = LocalSaverService()

    step_functions = {
        0: ("Rotate video if needed", lambda: rotate_video(base_name, saver)),
        1: ("Downsample video", lambda: downsample_video(base_name, saver)),
        2: ("Extract audio", lambda: extract_audio(base_name, saver)),
        3: ("Get transcription", lambda: get_transcription(base_name, saver)),
        4: ("Prompt LLM", lambda: prompt_llm_for_editing(base_name, saver)),
        5: (
            "Generate adjusted sentences",
            lambda: generate_adjusted_sentences(base_name, saver),
        ),
        6: ("Create edited video", lambda: create_edited_video(base_name, saver)),
        7: (
            "Two-stage feedback loop - Sentence selection & timestamp adjustment",
            lambda: feedback_loop_for_cut(base_name, saver),
        ),
        8: (
            "Parse Google Doc script (extract text & images)",
            lambda: parse_google_doc_script(base_name, saver),
        ),
        9: (
            "Place Google Doc images (LLM-based placement)",
            lambda: place_google_doc_images(base_name, saver),
        ),
        10: (
            "Create video with Google Doc images (downsampled)",
            lambda: create_video_with_google_doc_images(base_name, saver, force=False),
        ),
        11: (
            "Cut full resolution video + add images (single pass)",
            lambda: create_full_res_video_single_pass(base_name, saver, force=False),
        ),
        12: (
            "Cut full resolution video only (no images)",
            lambda: create_full_res_cut_video(base_name, saver, force=False),
        ),
        13: (
            "Add images to full resolution video (requires step 12)",
            lambda: create_full_res_video_with_images(base_name, saver, force=False),
        ),
    }

    for step_num in steps:
        step_name, step_func = step_functions[step_num]
        print(f"\n--- Step {step_num}: {step_name} ---")

        try:
            step_func()
        except Exception as e:
            print(f"\n✗ Error in step {step_num}: {str(e)}")
            print("Pipeline stopped due to error.")
            raise

    print("\n" + "=" * 50)
    print("✓ PIPELINE COMPLETE")
    print("=" * 50)


def main() -> None:
    """Main entry point."""
    try:
        steps = display_menu()
        base_name = get_input_filename()
        run_pipeline(base_name, steps)
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user.")
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
