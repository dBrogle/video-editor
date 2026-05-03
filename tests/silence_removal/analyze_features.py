"""
Analyze audio features at golden cut points to understand what distinguishes
speech from silence at the boundaries. This informs algorithm design.

Usage: python -m tests.silence_removal.analyze_features
"""

import json
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

TEST_DATA_DIR = Path(__file__).parent / "data"

# Analysis window: how many seconds before/after the golden cut point to show
WINDOW = 0.15
RESOLUTION = 0.01  # seconds per analysis step


def compute_features(audio_array, sr):
    """Compute multiple audio features at high resolution."""
    frame_length = 512
    hop_length = 128  # Higher resolution than production (256)
    time_per_frame = hop_length / sr

    # 1. RMS energy (broadband)
    rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # 2. High-frequency energy (>3kHz) — sibilants, plosives
    nyquist = sr / 2
    sos_high = butter(4, min(3000 / nyquist, 0.99), btype='high', output='sos')
    audio_hf = sosfilt(sos_high, audio_array)
    rms_hf = librosa.feature.rms(y=audio_hf, frame_length=frame_length, hop_length=hop_length)[0]
    rms_hf_db = librosa.amplitude_to_db(rms_hf, ref=np.max)

    # 3. Very high frequency energy (>6kHz) — fricatives, breath
    sos_vhf = butter(4, min(6000 / nyquist, 0.99), btype='high', output='sos')
    audio_vhf = sosfilt(sos_vhf, audio_array)
    rms_vhf = librosa.feature.rms(y=audio_vhf, frame_length=frame_length, hop_length=hop_length)[0]
    rms_vhf_db = librosa.amplitude_to_db(rms_vhf, ref=np.max)

    # 4. Spectral centroid — high = consonants/noise, low = vowels
    spec_centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sr,
                                                       n_fft=frame_length, hop_length=hop_length)[0]

    # 5. Spectral flux — frame-to-frame change (transients)
    S = np.abs(librosa.stft(audio_array, n_fft=frame_length, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    flux = np.pad(flux, (1, 0))
    if len(flux) > len(rms):
        flux = flux[:len(rms)]
    elif len(flux) < len(rms):
        flux = np.pad(flux, (0, len(rms) - len(flux)))

    # 6. Zero crossing rate — high for unvoiced consonants
    zcr = librosa.feature.zero_crossing_rate(audio_array, frame_length=frame_length,
                                               hop_length=hop_length)[0]

    # 7. Spectral rolloff — frequency below which 85% of energy lies
    rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sr, n_fft=frame_length,
                                                 hop_length=hop_length, roll_percent=0.85)[0]

    return {
        "time_per_frame": time_per_frame,
        "rms_db": rms_db,
        "hf_db": rms_hf_db,
        "vhf_db": rms_vhf_db,
        "centroid": spec_centroid,
        "flux": flux,
        "zcr": zcr,
        "rolloff": rolloff,
    }


def analyze_boundary(features, clip_start, golden_time, side, clip_duration):
    """Analyze features around a golden cut point."""
    tpf = features["time_per_frame"]
    # Golden time as offset from clip start
    golden_offset = golden_time - clip_start

    # Window around golden point
    window_start = max(0, golden_offset - WINDOW)
    window_end = min(clip_duration, golden_offset + WINDOW)

    frame_start = int(window_start / tpf)
    frame_end = int(window_end / tpf)
    golden_frame = int(golden_offset / tpf)

    n_frames = len(features["rms_db"])
    frame_start = max(0, min(frame_start, n_frames - 1))
    frame_end = max(0, min(frame_end, n_frames))
    golden_frame = max(0, min(golden_frame, n_frames - 1))

    print(f"\n  {'START' if side == 'start' else 'END'} boundary analysis (golden={golden_time:.3f}s):")
    print(f"  {'Time':>8s}  {'RMS':>6s}  {'HF':>6s}  {'VHF':>6s}  {'Cent':>6s}  {'Flux':>6s}  {'ZCR':>6s}  {'Roll':>6s}")
    print(f"  {'-' * 64}")

    for i in range(frame_start, frame_end):
        t = clip_start + i * tpf
        marker = " <<" if i == golden_frame else ""
        print(f"  {t:8.3f}  {features['rms_db'][i]:6.1f}  {features['hf_db'][i]:6.1f}  "
              f"{features['vhf_db'][i]:6.1f}  {features['centroid'][i]:6.0f}  "
              f"{features['flux'][i]:6.1f}  {features['zcr'][i]:6.3f}  "
              f"{features['rolloff'][i]:6.0f}{marker}")

    # Stats: compare "inside speech" vs "outside speech" around golden point
    if side == "end":
        # Frames before golden = speech, frames after = silence
        speech_frames = slice(max(0, golden_frame - 15), golden_frame)
        silence_frames = slice(golden_frame, min(n_frames, golden_frame + 15))
    else:
        # Frames after golden = speech, frames before = silence
        silence_frames = slice(max(0, golden_frame - 15), golden_frame)
        speech_frames = slice(golden_frame, min(n_frames, golden_frame + 15))

    print(f"\n  Stats around golden {side}:")
    for feat_name in ["rms_db", "hf_db", "vhf_db", "centroid", "flux", "zcr"]:
        vals = features[feat_name]
        sp = vals[speech_frames]
        sl = vals[silence_frames]
        if len(sp) > 0 and len(sl) > 0:
            print(f"    {feat_name:>8s}:  speech_mean={np.mean(sp):7.2f}  silence_mean={np.mean(sl):7.2f}  "
                  f"diff={np.mean(sp) - np.mean(sl):7.2f}  speech_min={np.min(sp):7.2f}")


def main():
    for data_dir in sorted(TEST_DATA_DIR.iterdir()):
        if not data_dir.is_dir():
            continue
        cases_file = data_dir / "cases.json"
        audio_file = data_dir / "audio.mp3"
        if not cases_file.exists() or not audio_file.exists():
            continue

        data = json.loads(cases_file.read_text())
        print(f"\n{'=' * 80}")
        print(f"  {data_dir.name}")
        print(f"{'=' * 80}")

        # Video-level stats
        full_audio, sr = librosa.load(str(audio_file), sr=22050, mono=True)
        full_rms = librosa.feature.rms(y=full_audio, frame_length=512, hop_length=256)[0]
        full_rms_db = librosa.amplitude_to_db(full_rms, ref=np.max)
        video_speech_level = np.percentile(full_rms_db, 85)
        print(f"  Video speech level (85th pct): {video_speech_level:.1f} dB")
        print(f"  Video noise floor (5th pct): {np.percentile(full_rms_db, 5):.1f} dB")

        for case in data["cases"]:
            start = case["original_start"]
            end = case["original_end"]
            duration = end - start

            print(f"\n{'-' * 80}")
            print(f"  CASE: {case['name']}")
            print(f"  \"{case['sentence_text'][:70]}\"")
            print(f"  Original: {start:.3f} - {end:.3f} ({duration:.3f}s)")
            print(f"  Golden:   {case['expected_start']:.3f} - {case['expected_end']:.3f}")

            audio_array, sr = librosa.load(
                str(audio_file), sr=22050, mono=True, offset=start, duration=duration
            )

            features = compute_features(audio_array, sr)

            # Analyze start boundary
            analyze_boundary(features, start, case["expected_start"], "start", duration)

            # Analyze end boundary
            analyze_boundary(features, start, case["expected_end"], "end", duration)


if __name__ == "__main__":
    main()
