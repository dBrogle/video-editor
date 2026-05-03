import argparse
import sys

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
from src.stream_pipeline import (
    StreamSaver,
    stream_stage_1_preprocess,
    stream_stage_2_transcribe,
    stream_stage_3_llm_edit,
    stream_stage_4_iterate_selection,
    stream_stage_5_adjust_timestamps,
    stream_stage_6_iterate_timestamps,
    stream_stage_7_final_render,
)


def select_pipeline() -> str:
    print("\n" + "=" * 50)
    print("VIDEO EDITING PIPELINE")
    print("=" * 50)
    print("\nSelect pipeline:")
    print("  1. Shorts (scripted video with Google Doc images)")
    print("  2. Streams (long-form livestream cutting)")

    while True:
        choice = input("\nYour selection (1 or 2): ").strip()
        if choice in ("1", "2"):
            return "shorts" if choice == "1" else "streams"
        print("Error: Please enter 1 or 2")


def display_shorts_menu() -> list[int] | str:
    print("\n" + "=" * 50)
    print("SHORTS PIPELINE")
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


def display_streams_menu() -> list[int] | str:
    print("\n" + "=" * 50)
    print("STREAMS PIPELINE")
    print("=" * 50)
    print("\nAvailable steps:")
    print("  0. Run all steps (1-7)")
    print("  1. Preprocess stream (downsample proxy, extract audio)")
    print("  2. Transcribe stream audio")
    print("  3. Chunked LLM edit (identify sections to remove)")
    print("  4. Iterate on sentence selection")
    print("  5. Generate adjusted sentences (timestamp trimming)")
    print("  6. Iterate on adjusted sentences (timestamp fine-tuning)")
    print("  7. Final render (cuts only, native resolution)")
    print("\nUtilities:")
    print("  r. Reset pipeline (delete all generated files)")
    print("\nEnter step numbers separated by commas (e.g., 1,2,3)")
    print("or enter 0 to run all steps, or 'r' to reset.")

    while True:
        choice = input("\nYour selection: ").strip().lower()

        if choice == "r":
            return "r"

        if choice == "0":
            return [1, 2, 3, 4, 5, 6, 7]

        try:
            steps = [int(s.strip()) for s in choice.split(",")]
            valid_steps = [0, 1, 2, 3, 4, 5, 6, 7]
            if all(step in valid_steps for step in steps):
                return sorted(set(steps))
            else:
                print("Error: Please enter valid step numbers (0-7) or 'r'")
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
            print(f"Found: {input_path.name}")
            saver.save_last_filename(base_name)
            return base_name
        else:
            print("Error: Video file not found in assets/")
            print(f"Expected path: {input_path}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != "y":
                raise FileNotFoundError(f"Video file not found: {filename}")


def run_shorts_pipeline(
    base_name: str, steps: list[int], skip_silence_removal: bool = False
) -> None:
    print("\n" + "=" * 50)
    print(f"RUNNING SHORTS PIPELINE: {base_name}")
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
            print(f"\nError in step {step_num}: {str(e)}")
            print("Pipeline stopped due to error.")
            raise

    print("\n" + "=" * 50)
    print("SHORTS PIPELINE COMPLETE")
    print("=" * 50)


def run_streams_pipeline(base_name: str, steps: list[int]) -> None:
    print("\n" + "=" * 50)
    print(f"RUNNING STREAMS PIPELINE: {base_name}")
    print("=" * 50)

    saver = StreamSaver()

    step_functions = {
        1: (
            "Preprocess stream",
            lambda: stream_stage_1_preprocess(base_name),
        ),
        2: (
            "Transcribe stream audio",
            lambda: stream_stage_2_transcribe(base_name, saver),
        ),
        3: (
            "Chunked LLM edit",
            lambda: stream_stage_3_llm_edit(base_name, saver),
        ),
        4: (
            "Iterate on sentence selection",
            lambda: stream_stage_4_iterate_selection(base_name, saver),
        ),
        5: (
            "Generate adjusted sentences",
            lambda: stream_stage_5_adjust_timestamps(base_name, saver),
        ),
        6: (
            "Iterate on adjusted sentences",
            lambda: stream_stage_6_iterate_timestamps(base_name, saver),
        ),
        7: (
            "Final render",
            lambda: stream_stage_7_final_render(base_name, saver),
        ),
    }

    for step_num in steps:
        step_name, step_func = step_functions[step_num]
        print(f"\n--- Step {step_num}: {step_name} ---")

        try:
            step_func()
        except Exception as e:
            print(f"\nError in step {step_num}: {str(e)}")
            print("Pipeline stopped due to error.")
            raise

    print("\n" + "=" * 50)
    print("STREAMS PIPELINE COMPLETE")
    print("=" * 50)


def main() -> None:
    try:
        pipeline = select_pipeline()
        base_name = get_input_filename()

        if pipeline == "shorts":
            menu_choice = display_shorts_menu()

            if menu_choice == "r":
                print("\n" + "=" * 50)
                print("RESET PIPELINE")
                print("=" * 50)
                print(f"\nThis will delete all generated files for: {base_name}")
                print("The original video file will be preserved.")
                confirm = input("\nAre you sure? (y/N): ").strip().lower()

                if confirm == "y" or confirm == "yes":
                    reset_pipeline(base_name)
                    print("\nPipeline reset complete!")
                else:
                    print("\nReset cancelled.")
                return

            run_shorts_pipeline(base_name, menu_choice, skip_silence_removal=False)

        else:  # streams
            menu_choice = display_streams_menu()

            if menu_choice == "r":
                print("\n" + "=" * 50)
                print("RESET PIPELINE")
                print("=" * 50)
                print(f"\nThis will delete all stream_ files for: {base_name}")
                print("The original video file will be preserved.")
                confirm = input("\nAre you sure? (y/N): ").strip().lower()

                if confirm == "y" or confirm == "yes":
                    reset_pipeline(base_name)
                    print("\nPipeline reset complete!")
                else:
                    print("\nReset cancelled.")
                return

            run_streams_pipeline(base_name, menu_choice)

    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user.")
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        raise


def cli() -> None:
    """CLI entry point for running individual pipeline steps or tools."""
    parser = argparse.ArgumentParser(description="Video editing pipeline CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Run pipeline step(s)
    run_parser = subparsers.add_parser("run", help="Run pipeline step(s)")
    run_parser.add_argument("video", help="Video base name (e.g., d216_18)")
    run_parser.add_argument("steps", help="Comma-separated step numbers (e.g., 5,6)")
    run_parser.add_argument("--pipeline", default="shorts", choices=["shorts", "streams"])
    run_parser.add_argument("--skip-silence-removal", action="store_true")

    # Audio levels tool
    audio_parser = subparsers.add_parser("audio-levels", help="Get audio RMS levels for a time range")
    audio_parser.add_argument("video", help="Video base name")
    audio_parser.add_argument("start", type=float, help="Start time in seconds")
    audio_parser.add_argument("end", type=float, help="End time in seconds")
    audio_parser.add_argument("--resolution", type=float, default=0.01, help="Time resolution in seconds (default: 0.01)")

    # Word timestamps tool
    words_parser = subparsers.add_parser("word-timestamps", help="Get word timestamps for a sentence")
    words_parser.add_argument("video", help="Video base name")
    words_parser.add_argument("sentence", type=int, help="Sentence index (from transcription, 1-based)")

    # Split sentence tool
    split_parser = subparsers.add_parser("split-sentence", help="Split a sentence in adjusted_sentences at a timestamp")
    split_parser.add_argument("video", help="Video base name")
    split_parser.add_argument("sentence_index", help="Sentence index in adjusted sentences (e.g., '10')")
    split_parser.add_argument("split_time", type=float, help="Timestamp to split at (seconds)")
    split_parser.add_argument("--target", default="s5", choices=["s5", "s6"], help="Which adjusted sentences file to modify")

    # Preview video tool
    preview_parser = subparsers.add_parser("preview", help="Generate preview video from adjusted sentences")
    preview_parser.add_argument("video", help="Video base name")
    preview_parser.add_argument("--source", default="s5", choices=["s5", "s6"], help="Which adjusted sentences to use")

    args = parser.parse_args()

    if args.command is None:
        # No CLI args, run interactive menu
        main()
        return

    from src.cli_tools import run_tool
    run_tool(args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli()
    else:
        main()
