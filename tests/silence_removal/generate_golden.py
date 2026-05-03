"""
Interactive tool for generating/updating golden answers for silence removal tests.

Usage:
  python tests/silence_removal/generate_golden.py <video_name>
  python tests/silence_removal/generate_golden.py d216_18
  python tests/silence_removal/generate_golden.py --all

Shows the current detection result vs golden answer for each case,
and lets you accept the detection result as the new golden answer.
"""

import json
import sys
from pathlib import Path

from src.models import LLMTranscriptSentence
from src.services.video.video_service import VideoService

TEST_DATA_DIR = Path(__file__).parent / "data"


def run_detection(svc: VideoService, audio_path: Path, case: dict) -> dict:
    sentence = LLMTranscriptSentence(
        sentence=case["sentence_text"],
        start=case["original_start"],
        end=case["original_end"],
    )
    result = svc._get_adjusted_sentence(
        audio_path=audio_path,
        sentence=sentence,
        sentence_index=0,
    )
    return {
        "adjusted_start": round(result.adjusted_start, 3),
        "adjusted_end": round(result.adjusted_end, 3),
    }


def process_video(video_name: str, interactive: bool = True) -> None:
    data_dir = TEST_DATA_DIR / video_name
    cases_file = data_dir / "cases.json"
    audio_file = data_dir / "audio.mp3"

    if not cases_file.exists():
        print(f"No cases.json found in {data_dir}")
        return
    if not audio_file.exists():
        print(f"No audio.mp3 found in {data_dir}")
        return

    data = json.loads(cases_file.read_text())
    svc = VideoService()

    print(f"\n{'=' * 70}")
    print(f"  {video_name} - {len(data['cases'])} test cases")
    print(f"{'=' * 70}\n")

    modified = False

    for case in data["cases"]:
        detected = run_detection(svc, audio_file, case)

        start_diff = abs(detected["adjusted_start"] - case["expected_start"])
        end_diff = abs(detected["adjusted_end"] - case["expected_end"])
        tolerance = case.get("tolerance", 0.05)

        start_ok = start_diff <= tolerance
        end_ok = end_diff <= tolerance
        status = "PASS" if (start_ok and end_ok) else "FAIL"

        print(f"  [{status}] {case['name']}: \"{case['sentence_text'][:60]}...\"")
        print(f"         Original:  {case['original_start']:.3f} - {case['original_end']:.3f}")
        print(f"         Golden:    {case['expected_start']:.3f} - {case['expected_end']:.3f}")
        print(f"         Detected:  {detected['adjusted_start']:.3f} - {detected['adjusted_end']:.3f}")
        print(f"         Diff:      start={start_diff:.3f}s {'OK' if start_ok else 'FAIL'}  end={end_diff:.3f}s {'OK' if end_ok else 'FAIL'}")

        if interactive and not (start_ok and end_ok):
            choice = input(f"         Update golden to detected values? [y/N/q]: ").strip().lower()
            if choice == "q":
                break
            if choice == "y":
                case["expected_start"] = detected["adjusted_start"]
                case["expected_end"] = detected["adjusted_end"]
                modified = True
                print(f"         -> Updated!")

        print()

    if modified:
        cases_file.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Saved updated cases to {cases_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_golden.py <video_name|--all> [--non-interactive]")
        print(f"\nAvailable videos: {[d.name for d in TEST_DATA_DIR.iterdir() if d.is_dir()]}")
        sys.exit(1)

    interactive = "--non-interactive" not in sys.argv
    target = sys.argv[1]

    if target == "--all":
        for data_dir in sorted(TEST_DATA_DIR.iterdir()):
            if data_dir.is_dir():
                process_video(data_dir.name, interactive=interactive)
    else:
        process_video(target, interactive=interactive)


if __name__ == "__main__":
    main()
