"""
Main entry point for the video editing pipeline.
Provides an interactive menu for running pipeline steps.
"""

from src.services.local_saver import LocalSaverService
from src.util import extract_filename_without_extension, get_input_video_path
from src.pipeline import (
    downsample_video,
    extract_audio,
    get_transcription,
    prompt_llm_for_editing,
    generate_adjusted_sentences,
    create_edited_video,
    create_final_cut,
    extract_final_cut_audio,
    downsample_final_cut,
    transcribe_final_cut,
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
    print("  1. Downsample video")
    print("  2. Extract audio")
    print("  3. Get transcription")
    print("  4. Prompt LLM for editing")
    print("  5. Generate adjusted sentences (silence removal)")
    print("  6. Create edited video (downsampled)")
    print("  7. Create final cut, extract audio, downsample & transcribe")
    print("  0. Run all steps")
    print("\nEnter step numbers separated by commas (e.g., 1,2,3)")
    print("or enter 0 to run all steps.")

    while True:
        choice = input("\nYour selection: ").strip()

        if choice == "0":
            return [1, 2, 3, 4, 5, 6, 7]

        try:
            steps = [int(s.strip()) for s in choice.split(",")]
            if all(1 <= step <= 7 for step in steps):
                return sorted(set(steps))
            else:
                print("Error: Please enter numbers between 1 and 7")
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
            "Create final cut, extract audio, downsample & transcribe",
            lambda: (
                create_final_cut(base_name, saver),
                extract_final_cut_audio(base_name, saver),
                downsample_final_cut(base_name, saver),
                transcribe_final_cut(base_name, saver),
            ),
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
