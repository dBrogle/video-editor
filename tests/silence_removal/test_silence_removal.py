"""
Tests for silence removal logic.

Compares the multiband_zcr algorithm against manually verified golden answers.
Run with: python -m pytest tests/silence_removal/test_silence_removal.py -v
"""

import json
from pathlib import Path

import numpy as np
import librosa
import pytest

from tests.silence_removal.algorithms import multiband_zcr

TEST_DATA_DIR = Path(__file__).parent / "data"
FPS = 30
TOLERANCE = 3 / FPS  # 3 frames at 30fps = 0.1s

# Pre/post-roll: load extra audio around sentence boundaries so the algorithm
# can find onsets that Deepgram timestamps miss (often 100-200ms late)
PRE_ROLL = 0.3   # seconds before original_start
POST_ROLL = 0.0  # seconds after original_end (extension loop already searches forward)

# Best params from benchmark
BEST_PARAMS = {
    "rms_offset_db": 18,
    "noise_floor_margin_db": 8,
    "hf_offset_db": 20,
    "zcr_centroid_min": 0.08,
    "centroid_min": 1800,
    "end_hold_frames": 14,
    "start_padding": 0.02,
    "end_padding": 0.01,
    "hf_decline_limit": 14,
    "flux_threshold": 0.2,
    "decline_rate_threshold": 2.5,
}


def load_test_cases():
    """Load all test cases from all data directories."""
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
            cases.append(case)
    return cases


ALL_CASES = load_test_cases()


@pytest.fixture(scope="module")
def video_thresholds():
    """Cache video-level thresholds per audio file."""
    thresholds = {}
    for case in ALL_CASES:
        path = case["_audio_path"]
        if path not in thresholds:
            audio, sr = librosa.load(path, sr=22050, mono=True)
            rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            thresholds[path] = np.percentile(rms_db, 85) - 15
    return thresholds


def case_id(case):
    return f"{case['_video_name']}/{case['name']}"


@pytest.mark.parametrize("case", ALL_CASES, ids=[case_id(c) for c in ALL_CASES])
def test_silence_removal(case, video_thresholds):
    audio_path = case["_audio_path"]
    start = case["original_start"]
    end = case["original_end"]

    # Load with pre/post-roll so the algorithm can search beyond Deepgram timestamps
    load_start = max(0, start - PRE_ROLL)
    actual_pre_roll = start - load_start
    load_duration = (end - start) + actual_pre_roll + POST_ROLL

    audio_array, sr = librosa.load(
        audio_path, sr=22050, mono=True, offset=load_start, duration=load_duration
    )

    vt = video_thresholds[audio_path]
    start_offset, end_offset = multiband_zcr(
        audio_array, sr, vt,
        pre_roll=actual_pre_roll,
        post_roll=POST_ROLL,
        **BEST_PARAMS,
    )

    # Convert offsets (relative to loaded audio start) to absolute timestamps
    adjusted_start = load_start + start_offset
    adjusted_end = load_start + end_offset

    start_diff = abs(adjusted_start - case["expected_start"])
    end_diff = abs(adjusted_end - case["expected_end"])

    tolerance = case.get("tolerance", TOLERANCE)

    assert start_diff <= tolerance, (
        f"[{case['name']}] Start: got {adjusted_start:.3f}, "
        f"expected {case['expected_start']:.3f} (diff {start_diff:.3f}s)"
    )
    assert end_diff <= tolerance, (
        f"[{case['name']}] End: got {adjusted_end:.3f}, "
        f"expected {case['expected_end']:.3f} (diff {end_diff:.3f}s)"
    )
