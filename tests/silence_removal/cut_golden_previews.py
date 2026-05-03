"""Cut 'correct answer' clips from source videos for every silence_removal test case.

Each clip spans [expected_start, expected_end] from the matching source video,
so you can visually/audibly verify whether the golden answer is what you want.
"""

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "golden_previews"

SOURCES = {
    "b216_2": REPO / "cc_wsp/videos/b216_2/downsampled.mp4",
    "d216_18": REPO / "assets/d216_18/s1_downsampled.mp4",
    "d265_week_ai": REPO / "assets/d265_week_ai/s1_downsampled.mp4",
    "d267_minimax": REPO / "cc_wsp/videos/b267_minimax/downsampled.mp4",
}


def cut(src: Path, start: float, end: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-i", str(src),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(out),
        ],
        check=True,
    )


def main() -> None:
    for video_name, src in SOURCES.items():
        cases_file = DATA_DIR / video_name / "cases.json"
        if not cases_file.exists():
            print(f"! missing cases.json for {video_name}")
            continue
        if not src.exists():
            print(f"! missing source video {src}")
            continue
        cases = json.loads(cases_file.read_text())["cases"]
        for c in cases:
            out = OUT_DIR / video_name / f"{c['name']}.mp4"
            print(f"  {video_name}/{c['name']}: {c['expected_start']:.3f}-{c['expected_end']:.3f}")
            cut(src, c["expected_start"], c["expected_end"], out)
    print(f"\nDone. Clips in {OUT_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
