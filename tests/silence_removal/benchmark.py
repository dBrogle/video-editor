"""
Benchmark different silence detection algorithms and parameter combinations
against golden test cases.

Usage:
  python tests/silence_removal/benchmark.py
  python tests/silence_removal/benchmark.py --top 10
  python tests/silence_removal/benchmark.py --algorithm rms_highfreq
"""

import json
import sys
import itertools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import librosa

from tests.silence_removal.algorithms import (
    rms_basic,
    rms_highfreq,
    rms_spectral_flux,
    rms_envelope,
    rms_adaptive_padding,
    rms_hybrid,
    multiband_zcr,
)

TEST_DATA_DIR = Path(__file__).parent / "data"
FPS = 30
TOLERANCE = 2 / FPS  # 2 frames at 30fps = 0.067s


@dataclass
class CaseResult:
    name: str
    start_diff: float
    end_diff: float
    passed: bool


@dataclass
class AlgoResult:
    algorithm: str
    params: dict
    cases: list
    pass_count: int
    total: int
    avg_end_diff: float
    max_end_diff: float
    avg_start_diff: float
    score: float  # lower is better


def load_all_cases():
    cases = []
    for data_dir in sorted(TEST_DATA_DIR.iterdir()):
        if not data_dir.is_dir():
            continue
        cases_file = data_dir / "cases.json"
        audio_file = data_dir / "audio.mp3"
        if not cases_file.exists() or not audio_file.exists():
            continue
        data = json.loads(cases_file.read_text())
        for case in data["cases"]:
            case["_audio_path"] = str(audio_file)
            case["_video_name"] = data_dir.name
        cases.extend(data["cases"])
    return cases


def get_video_threshold(audio_path: str) -> float:
    audio_array, sr = librosa.load(audio_path, sr=22050, mono=True)
    rms = librosa.feature.rms(y=audio_array, frame_length=512, hop_length=256)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    speech_level = np.percentile(rms_db, 85)
    return speech_level - 15


def run_algorithm(algo_fn, cases, video_thresholds, params):
    results = []
    for case in cases:
        audio_path = case["_audio_path"]
        start = case["original_start"]
        end = case["original_end"]

        audio_array, sr = librosa.load(
            audio_path, sr=22050, mono=True, offset=start, duration=end - start
        )

        vt = video_thresholds[audio_path]
        start_offset, end_offset = algo_fn(audio_array, sr, vt, **params)

        adjusted_start = start + start_offset
        adjusted_end = start + end_offset

        start_diff = abs(adjusted_start - case["expected_start"])
        end_diff = abs(adjusted_end - case["expected_end"])

        passed = start_diff <= TOLERANCE and end_diff <= TOLERANCE

        results.append(CaseResult(
            name=f"{case['_video_name']}/{case['name']}",
            start_diff=start_diff,
            end_diff=end_diff,
            passed=passed,
        ))

    pass_count = sum(1 for r in results if r.passed)
    end_diffs = [r.end_diff for r in results]
    start_diffs = [r.start_diff for r in results]

    # Score: prioritize pass count, then average error
    # Lower is better. Penalize failures heavily.
    avg_diff = np.mean(end_diffs + start_diffs)
    score = (len(results) - pass_count) * 10 + avg_diff

    return AlgoResult(
        algorithm=algo_fn.__name__,
        params=params,
        cases=results,
        pass_count=pass_count,
        total=len(results),
        avg_end_diff=np.mean(end_diffs),
        max_end_diff=np.max(end_diffs),
        avg_start_diff=np.mean(start_diffs),
        score=score,
    )


def param_grid(base_params, sweeps):
    """Generate all combinations from a base + sweep dict."""
    keys = list(sweeps.keys())
    values = list(sweeps.values())
    for combo in itertools.product(*values):
        p = dict(base_params)
        for k, v in zip(keys, combo):
            p[k] = v
        yield p


def main():
    top_n = 15
    filter_algo = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--top" and i + 2 < len(sys.argv):
            top_n = int(sys.argv[i + 2])
        if arg == "--algorithm" and i + 2 < len(sys.argv):
            filter_algo = sys.argv[i + 2]

    print("Loading test cases...")
    cases = load_all_cases()
    print(f"Loaded {len(cases)} test cases")
    print(f"Tolerance: {TOLERANCE:.3f}s ({TOLERANCE * FPS:.0f} frames at {FPS}fps)\n")

    # Cache video thresholds
    print("Computing video-level thresholds...")
    audio_paths = set(c["_audio_path"] for c in cases)
    video_thresholds = {p: get_video_threshold(p) for p in audio_paths}

    # Define algorithm + parameter sweep combinations
    algo_configs = {
        "rms_basic": {
            "fn": rms_basic,
            "sweeps": {
                "percentile": [80, 85],
                "threshold_offset_db": [15, 18, 20, 22],
                "padding": [0.02, 0.04, 0.06, 0.08],
            },
        },
        "rms_highfreq": {
            "fn": rms_highfreq,
            "sweeps": {
                "threshold_offset_db": [15, 18, 20],
                "padding": [0.02, 0.04, 0.06],
                "highfreq_cutoff": [2000, 3000, 4000],
                "highfreq_threshold_offset_db": [18, 22, 25],
            },
        },
        "rms_spectral_flux": {
            "fn": rms_spectral_flux,
            "sweeps": {
                "threshold_offset_db": [15, 18, 20],
                "padding": [0.02, 0.04, 0.06],
                "flux_percentile": [20, 30, 40, 50],
            },
        },
        "rms_envelope": {
            "fn": rms_envelope,
            "sweeps": {
                "threshold_offset_db": [15, 18, 20, 22],
                "padding": [0.02, 0.04, 0.06, 0.08],
                "smooth_frames": [3, 5, 7, 9],
            },
        },
        "rms_adaptive_padding": {
            "fn": rms_adaptive_padding,
            "sweeps": {
                "threshold_offset_db": [15, 18, 20, 22],
                "start_padding": [0.02, 0.03],
                "min_end_padding": [0.02, 0.04],
                "max_end_padding": [0.08, 0.10, 0.12, 0.15],
                "tail_window": [8, 12, 16],
                "highfreq_cutoff": [3000, 4000],
            },
        },
        "rms_hybrid": {
            "fn": rms_hybrid,
            "sweeps": {
                "threshold_offset_db": [18, 20, 22],
                "start_padding": [0.02],
                "min_end_padding": [0.02, 0.04],
                "max_end_padding": [0.08, 0.10, 0.12],
                "smooth_frames": [3, 5, 7],
                "highfreq_cutoff": [3000, 4000],
                "hf_threshold_offset": [18, 22],
                "tail_window": [8, 12, 16],
            },
        },
        "multiband_zcr": {
            "fn": multiband_zcr,
            "sweeps": {
                "rms_offset_db": [18, 20],
                "noise_floor_margin_db": [4, 6, 8],
                "hf_offset_db": [20, 22],
                "zcr_centroid_min": [0.08],
                "centroid_min": [1800],
                "end_hold_frames": [10, 14, 18],
                "end_padding": [0.01, 0.02],
                "hf_decline_limit": [10, 14, 18],
            },
        },
    }

    if filter_algo:
        algo_configs = {k: v for k, v in algo_configs.items() if k == filter_algo}

    all_results = []

    for algo_name, config in algo_configs.items():
        fn = config["fn"]
        sweeps = config["sweeps"]
        combos = list(param_grid({}, sweeps))
        print(f"Testing {algo_name}: {len(combos)} parameter combinations...")

        for params in combos:
            result = run_algorithm(fn, cases, video_thresholds, params)
            all_results.append(result)

    # Sort by score (lower is better)
    all_results.sort(key=lambda r: r.score)

    # Print top results
    print(f"\n{'=' * 90}")
    print(f"  TOP {top_n} RESULTS (out of {len(all_results)} combinations)")
    print(f"  Tolerance: {TOLERANCE:.3f}s ({TOLERANCE * FPS:.0f} frames)")
    print(f"{'=' * 90}\n")

    for i, r in enumerate(all_results[:top_n]):
        param_str = ", ".join(f"{k}={v}" for k, v in sorted(r.params.items()))
        print(f"  #{i + 1}  {r.algorithm}  [{r.pass_count}/{r.total} passed]  "
              f"score={r.score:.3f}  avg_end={r.avg_end_diff:.3f}  max_end={r.max_end_diff:.3f}")
        print(f"       {param_str}")

        # Show per-case details for top 3
        if i < 3:
            for c in r.cases:
                status = "OK" if c.passed else "FAIL"
                print(f"         {status:4s}  {c.name:<40s}  start={c.start_diff:.3f}  end={c.end_diff:.3f}")
        print()

    # Print the winner
    best = all_results[0]
    print(f"{'=' * 90}")
    print(f"  BEST: {best.algorithm}")
    print(f"  Params: {best.params}")
    print(f"  {best.pass_count}/{best.total} passed, avg_end_diff={best.avg_end_diff:.3f}s")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
