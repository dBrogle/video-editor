from src.services.local_saver import LocalSaverService
from src.util import extract_filename_without_extension, get_input_video_path, reset_pipeline
from src.pipeline import (
    stage_1_preprocess_video_and_files,
    stage_2_get_transcription,
    stage_3_initial_edit_with_llm,
    stage_4_iterate_sentence_selection,
    stage_5_generate_adjusted_sentences,
    stage_6_iterate_adjusted_sentences,
    stage_7_parse_google_doc_script,
    stage_8_place_google_doc_images,
    stage_9_create_video_with_google_doc_images,
    stage_10_downsample_to_1080p,
    stage_11_create_1080p_video_with_images,
    stage_12_create_full_res_video_single_pass,
)


def display_menu() -> list[int] | str:
    print("\n" + "=" * 50)
    print("VIDEO EDITING PIPELINE")
    print("=" * 50)
    print("\nAvailable steps:")
    print("  0. Run all steps (1-11, using 1080p approach)")
    print("  1. Preprocess video and files (rotate, downsample, audio, unzip Google Doc)")
    print("  2. Get transcription")
    print("  3. Initial edit with LLM (sentence selection)")
    print("  4. Iterate with LLM on sentence selection")
    print("  5. Generate adjusted sentences (silence removal)")
    print("  6. Iterate with LLM on adjusted sentences (timestamp adjustments)")
    print("  7. Parse Google Doc script (extract text & images)")
    print("  8. Place Google Doc images (LLM-based placement)")
    print("  9. Create downsampled video with Google Doc images")
    print(" 10. Downsample full res video to 1080p (1080x1920)")
    print(" 11. Create 1080p video with images (single pass)")
    print("\nAdvanced:")
    print(" 12. Create full resolution video with images (from 1080p, single pass)")
    print("\nUtilities:")
    print("  r. Reset pipeline (delete all generated files)")
    print("\nEnter step numbers separated by commas (e.g., 1,2,3)")
    print("or enter 0 to run all steps, or 'r' to reset.")

    while True:
        choice = input("\nYour selection: ").strip().lower()

        if choice == "r":
            return "r"

        if choice == "0":
            return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        try:
            steps = [int(s.strip()) for s in choice.split(",")]
            valid_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            if all(step in valid_steps for step in steps):
                return sorted(set(steps))
            else:
                print("Error: Please enter valid step numbers (0-12) or 'r'")
        except ValueError:
            print(
                "Error: Invalid input. Please enter numbers separated by commas or 'r'"
            )


def get_input_filename() -> str:
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


def run_pipeline(
    base_name: str, steps: list[int], skip_silence_removal: bool = False
) -> None:
    print("\n" + "=" * 50)
    print(f"RUNNING PIPELINE: {base_name}")
    print("=" * 50)

    saver = LocalSaverService()

    step_functions = {
        1: (
            "Preprocess video and files",
            lambda: stage_1_preprocess_video_and_files(base_name, saver),
        ),
        2: ("Get transcription", lambda: stage_2_get_transcription(base_name, saver)),
        3: (
            "Initial edit with LLM (sentence selection)",
            lambda: stage_3_initial_edit_with_llm(base_name, saver),
        ),
        4: (
            "Iterate with LLM on sentence selection",
            lambda: stage_4_iterate_sentence_selection(base_name, saver, skip_silence_removal),
        ),
        5: (
            "Generate adjusted sentences (silence removal)",
            lambda: stage_5_generate_adjusted_sentences(base_name, saver, skip_silence_removal),
        ),
        6: (
            "Iterate with LLM on adjusted sentences (timestamp adjustments)",
            lambda: stage_6_iterate_adjusted_sentences(base_name, saver, skip_silence_removal),
        ),
        7: (
            "Parse Google Doc script (extract text & images)",
            lambda: stage_7_parse_google_doc_script(base_name, saver),
        ),
        8: (
            "Place Google Doc images (LLM-based placement)",
            lambda: stage_8_place_google_doc_images(base_name, saver),
        ),
        9: (
            "Create downsampled video with Google Doc images",
            lambda: stage_9_create_video_with_google_doc_images(base_name, saver, force=False),
        ),
        10: (
            "Downsample full res video to 1080p (1080x1920)",
            lambda: stage_10_downsample_to_1080p(base_name, force=False),
        ),
        11: (
            "Create 1080p video with images (single pass)",
            lambda: stage_11_create_1080p_video_with_images(base_name, saver, force=False),
        ),
        12: (
            "Create full resolution video with images (from 1080p, single pass)",
            lambda: stage_12_create_full_res_video_single_pass(base_name, saver, force=False),
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
    try:
        menu_choice = display_menu()
        base_name = get_input_filename()

        if menu_choice == "r":
            print("\n" + "=" * 50)
            print("RESET PIPELINE")
            print("=" * 50)
            print(f"\nThis will delete all generated files for: {base_name}")
            print("The original video file will be preserved.")
            confirm = input("\nAre you sure? (y/N): ").strip().lower()

            if confirm == "y" or confirm == "yes":
                reset_pipeline(base_name)
                print("\n✓ Pipeline reset complete!")
            else:
                print("\nReset cancelled.")
            return

        run_pipeline(base_name, menu_choice, skip_silence_removal=False)
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user.")
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
