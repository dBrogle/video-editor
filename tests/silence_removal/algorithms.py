"""
Silence detection algorithm variants for testing.

Each algorithm takes (audio_array, sr, video_threshold) and returns (start_offset, end_offset)
representing the detected speech boundaries as offsets from the beginning of the audio clip.
"""

import numpy as np
import librosa


def rms_basic(audio_array, sr, video_threshold, **params):
    """Current algorithm: RMS energy with adaptive thresholding."""
    percentile = params.get("percentile", 85)
    threshold_offset_db = params.get("threshold_offset_db", 15)
    padding = params.get("padding", 0.02)
    frame_length = params.get("frame_length", 512)
    hop_length = params.get("hop_length", 256)
    clip_db_diff = params.get("clip_db_diff", 5)

    rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    clip_speech_level = np.percentile(rms_db, percentile)
    clip_threshold = clip_speech_level - threshold_offset_db

    if video_threshold - clip_threshold > clip_db_diff:
        silence_threshold = video_threshold
    else:
        silence_threshold = clip_threshold

    speech_frames = np.where(rms_db > silence_threshold)[0]
    if len(speech_frames) == 0:
        return 0.0, len(audio_array) / sr

    start_offset = (speech_frames[0] * hop_length) / sr
    end_offset = ((speech_frames[-1] + 1) * hop_length) / sr

    start_offset = max(0, start_offset - padding)
    end_offset = min(len(audio_array) / sr, end_offset + padding)

    return start_offset, end_offset


def rms_highfreq(audio_array, sr, video_threshold, **params):
    """
    RMS + high-frequency energy detection.
    Sibilants (s, z), plosives (k, t, p) and fricatives have energy
    concentrated above ~3kHz that standard RMS misses when overall amplitude is low.
    Uses a combined signal: regular RMS OR high-freq energy above threshold.
    """
    percentile = params.get("percentile", 85)
    threshold_offset_db = params.get("threshold_offset_db", 15)
    padding = params.get("padding", 0.02)
    frame_length = params.get("frame_length", 512)
    hop_length = params.get("hop_length", 256)
    clip_db_diff = params.get("clip_db_diff", 5)
    highfreq_cutoff = params.get("highfreq_cutoff", 3000)
    highfreq_threshold_offset_db = params.get("highfreq_threshold_offset_db", 20)

    # Standard RMS
    rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # High-frequency RMS: bandpass above cutoff
    from scipy.signal import butter, sosfilt
    nyquist = sr / 2
    high_b = min(highfreq_cutoff / nyquist, 0.99)
    sos = butter(4, high_b, btype='high', output='sos')
    audio_highfreq = sosfilt(sos, audio_array)

    rms_hf = librosa.feature.rms(y=audio_highfreq, frame_length=frame_length, hop_length=hop_length)[0]
    rms_hf_db = librosa.amplitude_to_db(rms_hf, ref=np.max)

    # Thresholds
    clip_speech_level = np.percentile(rms_db, percentile)
    clip_threshold = clip_speech_level - threshold_offset_db
    if video_threshold - clip_threshold > clip_db_diff:
        silence_threshold = video_threshold
    else:
        silence_threshold = clip_threshold

    hf_speech_level = np.percentile(rms_hf_db, percentile)
    hf_threshold = hf_speech_level - highfreq_threshold_offset_db

    # Speech = above RMS threshold OR above high-freq threshold
    speech_mask = (rms_db > silence_threshold) | (rms_hf_db > hf_threshold)
    speech_frames = np.where(speech_mask)[0]

    if len(speech_frames) == 0:
        return 0.0, len(audio_array) / sr

    start_offset = (speech_frames[0] * hop_length) / sr
    end_offset = ((speech_frames[-1] + 1) * hop_length) / sr

    start_offset = max(0, start_offset - padding)
    end_offset = min(len(audio_array) / sr, end_offset + padding)

    return start_offset, end_offset


def rms_spectral_flux(audio_array, sr, video_threshold, **params):
    """
    RMS + spectral flux for boundary detection.
    Spectral flux measures frame-to-frame change in the spectrum, which is
    high at speech onsets/offsets — good for catching transient consonants
    that have low amplitude but distinct spectral change.
    """
    percentile = params.get("percentile", 85)
    threshold_offset_db = params.get("threshold_offset_db", 15)
    padding = params.get("padding", 0.02)
    frame_length = params.get("frame_length", 512)
    hop_length = params.get("hop_length", 256)
    clip_db_diff = params.get("clip_db_diff", 5)
    flux_percentile = params.get("flux_percentile", 30)

    # Standard RMS
    rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Spectral flux: magnitude of frame-to-frame spectral difference
    S = np.abs(librosa.stft(audio_array, n_fft=frame_length, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    # Pad to match rms length
    flux = np.pad(flux, (1, 0))
    if len(flux) > len(rms):
        flux = flux[:len(rms)]
    elif len(flux) < len(rms):
        flux = np.pad(flux, (0, len(rms) - len(flux)))

    flux_threshold = np.percentile(flux, flux_percentile)

    # RMS threshold
    clip_speech_level = np.percentile(rms_db, percentile)
    clip_threshold = clip_speech_level - threshold_offset_db
    if video_threshold - clip_threshold > clip_db_diff:
        silence_threshold = video_threshold
    else:
        silence_threshold = clip_threshold

    # Speech = above RMS threshold OR has significant spectral activity
    speech_mask = (rms_db > silence_threshold) | (flux > flux_threshold)
    speech_frames = np.where(speech_mask)[0]

    if len(speech_frames) == 0:
        return 0.0, len(audio_array) / sr

    start_offset = (speech_frames[0] * hop_length) / sr
    end_offset = ((speech_frames[-1] + 1) * hop_length) / sr

    start_offset = max(0, start_offset - padding)
    end_offset = min(len(audio_array) / sr, end_offset + padding)

    return start_offset, end_offset


def rms_adaptive_padding(audio_array, sr, video_threshold, **params):
    """
    RMS with adaptive end-padding based on where speech drops off.
    Instead of fixed padding, look at how sharply the signal drops.
    If it drops gradually (consonant trailing off), extend more.
    If it drops sharply (clean end), extend less.
    Also uses high-freq detection for sibilants.
    """
    percentile = params.get("percentile", 85)
    threshold_offset_db = params.get("threshold_offset_db", 18)
    start_padding = params.get("start_padding", 0.02)
    min_end_padding = params.get("min_end_padding", 0.02)
    max_end_padding = params.get("max_end_padding", 0.12)
    frame_length = params.get("frame_length", 512)
    hop_length = params.get("hop_length", 256)
    clip_db_diff = params.get("clip_db_diff", 5)
    tail_window = params.get("tail_window", 10)  # frames to analyze after last speech frame
    highfreq_cutoff = params.get("highfreq_cutoff", 3000)

    # Standard RMS
    rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # High-freq RMS for sibilant detection
    from scipy.signal import butter, sosfilt
    nyquist = sr / 2
    sos = butter(4, min(highfreq_cutoff / nyquist, 0.99), btype='high', output='sos')
    audio_hf = sosfilt(sos, audio_array)
    rms_hf = librosa.feature.rms(y=audio_hf, frame_length=frame_length, hop_length=hop_length)[0]
    rms_hf_db = librosa.amplitude_to_db(rms_hf, ref=np.max)

    # Threshold
    clip_speech_level = np.percentile(rms_db, percentile)
    clip_threshold = clip_speech_level - threshold_offset_db
    if video_threshold - clip_threshold > clip_db_diff:
        silence_threshold = video_threshold
    else:
        silence_threshold = clip_threshold

    # HF threshold (more lenient)
    hf_speech_level = np.percentile(rms_hf_db, percentile)
    hf_threshold = hf_speech_level - 22

    speech_mask = (rms_db > silence_threshold) | (rms_hf_db > hf_threshold)
    speech_frames = np.where(speech_mask)[0]

    if len(speech_frames) == 0:
        return 0.0, len(audio_array) / sr

    first_frame = speech_frames[0]
    last_frame = speech_frames[-1]

    # Adaptive end padding: look at the tail after last speech frame
    # If there's still some energy (gradual dropoff = consonant), extend more
    tail_start = last_frame + 1
    tail_end = min(tail_start + tail_window, len(rms_db))
    if tail_end > tail_start:
        tail_energy = rms_db[tail_start:tail_end]
        # How much energy is in the tail relative to the silence threshold?
        # More energy = more gradual dropoff = extend more
        tail_above = np.mean(tail_energy > (silence_threshold - 5))
        end_padding = min_end_padding + (max_end_padding - min_end_padding) * tail_above
    else:
        end_padding = min_end_padding

    start_offset = (first_frame * hop_length) / sr
    end_offset = ((last_frame + 1) * hop_length) / sr

    start_offset = max(0, start_offset - start_padding)
    end_offset = min(len(audio_array) / sr, end_offset + end_padding)

    return start_offset, end_offset


def rms_hybrid(audio_array, sr, video_threshold, **params):
    """
    Hybrid approach combining:
    1. RMS + high-freq energy for detecting sibilants/plosives
    2. Max-pool smoothing to bridge brief dips in consonants
    3. Adaptive end padding based on trailing energy gradient
    """
    percentile = params.get("percentile", 85)
    threshold_offset_db = params.get("threshold_offset_db", 20)
    start_padding = params.get("start_padding", 0.02)
    min_end_padding = params.get("min_end_padding", 0.02)
    max_end_padding = params.get("max_end_padding", 0.10)
    frame_length = params.get("frame_length", 512)
    hop_length = params.get("hop_length", 256)
    clip_db_diff = params.get("clip_db_diff", 5)
    smooth_frames = params.get("smooth_frames", 5)
    highfreq_cutoff = params.get("highfreq_cutoff", 3000)
    hf_threshold_offset = params.get("hf_threshold_offset", 20)
    tail_window = params.get("tail_window", 12)

    # Standard RMS with smoothing
    rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Max-pool smoothing to bridge brief dips (plosive releases, etc.)
    from scipy.ndimage import maximum_filter1d
    rms_db_smooth = maximum_filter1d(rms_db, size=smooth_frames)

    # High-freq RMS for sibilant detection
    from scipy.signal import butter, sosfilt
    nyquist = sr / 2
    sos = butter(4, min(highfreq_cutoff / nyquist, 0.99), btype='high', output='sos')
    audio_hf = sosfilt(sos, audio_array)
    rms_hf = librosa.feature.rms(y=audio_hf, frame_length=frame_length, hop_length=hop_length)[0]
    rms_hf_db = librosa.amplitude_to_db(rms_hf, ref=np.max)

    # Thresholds
    clip_speech_level = np.percentile(rms_db, percentile)
    clip_threshold = clip_speech_level - threshold_offset_db
    if video_threshold - clip_threshold > clip_db_diff:
        silence_threshold = video_threshold
    else:
        silence_threshold = clip_threshold

    hf_speech_level = np.percentile(rms_hf_db, percentile)
    hf_threshold = hf_speech_level - hf_threshold_offset

    # Speech = smoothed RMS above threshold OR high-freq energy above HF threshold
    speech_mask = (rms_db_smooth > silence_threshold) | (rms_hf_db > hf_threshold)
    speech_frames = np.where(speech_mask)[0]

    if len(speech_frames) == 0:
        return 0.0, len(audio_array) / sr

    first_frame = speech_frames[0]
    last_frame = speech_frames[-1]

    # Adaptive end padding: check energy gradient in the tail
    tail_start = last_frame + 1
    tail_end = min(tail_start + tail_window, len(rms_db))
    if tail_end > tail_start:
        tail_energy = rms_db[tail_start:tail_end]
        # Fraction of tail frames near the threshold (gradual dropoff = consonant)
        near_threshold = np.mean(tail_energy > (silence_threshold - 5))
        end_padding = min_end_padding + (max_end_padding - min_end_padding) * near_threshold
    else:
        end_padding = min_end_padding

    start_offset = (first_frame * hop_length) / sr
    end_offset = ((last_frame + 1) * hop_length) / sr

    start_offset = max(0, start_offset - start_padding)
    end_offset = min(len(audio_array) / sr, end_offset + end_padding)

    return start_offset, end_offset


def multiband_zcr(audio_array, sr, video_threshold, **params):
    """
    Multiband speech boundary detection with spectral flux onset detection
    and energy derivative end detection.

    Key improvements over basic approaches:
    - Spectral flux catches transient onsets (plosives, fricatives) that RMS misses
    - Energy derivative detects sustained signal decline for smarter end detection
    - Pre/post-roll aware: searches backward/forward from original boundaries
      when speech is detected at the edge, without false-triggering on
      previous/next sentence tails

    Strategy:
    - START: Find first speech in original window, then refine backward through
      contiguous speech into pre-roll (stops at silence gap)
    - END: Extend past last RMS frame through trailing consonants, using HF tracking
      and energy derivative to stop when signal is genuinely fading
    """
    # Parameters
    frame_length = params.get("frame_length", 512)
    hop_length = params.get("hop_length", 128)  # Higher resolution
    start_padding = params.get("start_padding", 0.02)
    end_padding = params.get("end_padding", 0.01)
    pre_roll = params.get("pre_roll", 0.0)
    post_roll = params.get("post_roll", 0.0)

    # Thresholds
    rms_percentile = params.get("rms_percentile", 85)
    rms_offset_db = params.get("rms_offset_db", 18)
    noise_floor_percentile = params.get("noise_floor_percentile", 10)
    noise_floor_margin_db = params.get("noise_floor_margin_db", 8)
    hf_offset_db = params.get("hf_offset_db", 20)
    zcr_centroid_min = params.get("zcr_centroid_min", 0.08)
    centroid_min = params.get("centroid_min", 1800)
    end_hold_frames = params.get("end_hold_frames", 14)
    clip_db_diff = params.get("clip_db_diff", 5)

    # New params
    flux_threshold = params.get("flux_threshold", 0.15)
    decline_rate_threshold = params.get("decline_rate_threshold", 1.5)

    time_per_frame = hop_length / sr
    pre_roll_frames = int(pre_roll / time_per_frame)

    # === Compute features ===
    rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # High-frequency RMS (>3kHz) — catches sibilants and plosives
    from scipy.signal import butter, sosfilt
    nyquist = sr / 2
    sos_hf = butter(4, min(3000 / nyquist, 0.99), btype='high', output='sos')
    audio_hf = sosfilt(sos_hf, audio_array)
    rms_hf = librosa.feature.rms(y=audio_hf, frame_length=frame_length, hop_length=hop_length)[0]
    rms_hf_db = librosa.amplitude_to_db(rms_hf, ref=np.max)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_array, frame_length=frame_length,
                                               hop_length=hop_length)[0]

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sr,
                                                  n_fft=frame_length, hop_length=hop_length)[0]

    # Spectral flux (positive only — detects energy appearing = speech onsets)
    S = np.abs(librosa.stft(audio_array, n_fft=frame_length, hop_length=hop_length))
    flux_raw = np.sum(np.maximum(np.diff(S, axis=1), 0), axis=0)
    flux_raw = np.pad(flux_raw, (1, 0))  # align with frame 0

    n_frames = len(rms_db)
    if len(flux_raw) > n_frames:
        flux_raw = flux_raw[:n_frames]
    elif len(flux_raw) < n_frames:
        flux_raw = np.pad(flux_raw, (0, n_frames - len(flux_raw)))

    # Normalize flux relative to original window only — pre/post-roll content
    # shouldn't influence what counts as a significant onset
    flux_orig = flux_raw[pre_roll_frames:max(pre_roll_frames + 1, n_frames - int(post_roll / time_per_frame))]
    flux_max = np.max(flux_orig) if len(flux_orig) > 0 else np.max(flux_raw)
    flux_norm = flux_raw / flux_max if flux_max > 0 else flux_raw
    mask_flux = flux_norm > flux_threshold

    # Energy derivative (smoothed) — for detecting sustained decline at ends
    from scipy.ndimage import uniform_filter1d
    rms_smooth = uniform_filter1d(rms_db, size=5)
    rms_deriv = np.diff(rms_smooth, prepend=rms_smooth[0])

    # === Compute thresholds (from original window only, not pre/post-roll) ===
    # Pre/post-roll audio is only used for boundary refinement, not statistics
    post_roll_frames = int(post_roll / time_per_frame)
    orig_end_frame = max(pre_roll_frames + 1, n_frames - post_roll_frames)
    rms_db_orig = rms_db[pre_roll_frames:orig_end_frame]
    rms_hf_db_orig = rms_hf_db[pre_roll_frames:orig_end_frame]

    clip_speech_level = np.percentile(rms_db_orig, rms_percentile)
    clip_rms_threshold = clip_speech_level - rms_offset_db
    if video_threshold - clip_rms_threshold > clip_db_diff:
        rms_threshold = video_threshold
    else:
        rms_threshold = clip_rms_threshold

    noise_floor = np.percentile(rms_db_orig, noise_floor_percentile)
    noise_threshold = noise_floor + noise_floor_margin_db

    hf_speech_level = np.percentile(rms_hf_db_orig, rms_percentile)
    hf_threshold = hf_speech_level - hf_offset_db

    # === Build speech masks ===
    mask_rms = rms_db > rms_threshold
    mask_hf = rms_hf_db > hf_threshold
    mask_consonant = (zcr > zcr_centroid_min) & (centroid > centroid_min) & (rms_db > noise_threshold)

    # For start detection: require HF and flux detections to also be above noise
    # floor. This prevents residual HF energy from previous sentences or room
    # noise from triggering false starts.
    mask_hf_start = mask_hf & (rms_db > noise_threshold)
    mask_flux_clean = mask_flux.copy()
    if pre_roll_frames > 0:
        mask_flux_clean[:pre_roll_frames + 5] = False
    start_mask = mask_rms | mask_hf_start | mask_consonant | mask_flux_clean

    # End mask excludes flux (noisy for offsets)
    end_mask = mask_rms | mask_hf | mask_consonant

    # === Start detection: find in original window, then refine backward ===
    # First find speech in the original window (at/after pre_roll_frames)
    start_region = start_mask[pre_roll_frames:]
    start_frames_in_region = np.where(start_region)[0]

    if len(start_frames_in_region) == 0:
        # No speech found in original window — try full range
        all_start_frames = np.where(start_mask)[0]
        if len(all_start_frames) == 0:
            return 0.0, len(audio_array) / sr
        first_frame = all_start_frames[0]
    else:
        first_frame_in_region = start_frames_in_region[0] + pre_roll_frames

        # Refine backward into pre-roll ONLY if speech starts right at the
        # original window boundary (within 3 frames), suggesting the real
        # onset is slightly earlier than Deepgram reported
        first_frame = first_frame_in_region
        if first_frame_in_region - pre_roll_frames < 3 and pre_roll_frames > 0:
            # Verify this is real sustained speech at the boundary, not a
            # single noisy frame. Require 3+ consecutive speech frames.
            consecutive = 0
            for f in range(first_frame_in_region, min(first_frame_in_region + 5, n_frames)):
                if mask_rms[f]:
                    consecutive += 1
                else:
                    break
            if consecutive >= 3:
                # Use RMS-only mask for backward search (more conservative)
                # Limit to ~50ms to avoid reaching the previous sentence
                max_backward_frames = int(0.05 / time_per_frame)
                backward_limit = max(0, first_frame_in_region - max_backward_frames)
                gap = 0
                for f in range(first_frame_in_region - 1, backward_limit - 1, -1):
                    if mask_rms[f]:
                        first_frame = f
                        gap = 0
                    else:
                        gap += 1
                        if gap > 1:
                            break

    # === End detection ===
    end_frames = np.where(end_mask)[0]
    if len(end_frames) == 0:
        return 0.0, len(audio_array) / sr

    # Find last RMS frame within original window only (not post-roll)
    # The extension loop below handles searching forward into post-roll
    rms_frames = np.where(mask_rms[:orig_end_frame])[0]
    if len(rms_frames) == 0:
        last_frame = end_frames[-1]
    else:
        last_rms_frame = rms_frames[-1]
        last_frame = last_rms_frame
        search_end = min(n_frames, last_rms_frame + int(0.5 / time_per_frame))
        gap_count = 0
        total_gap_frames = 0
        peak_hf_in_extension = rms_hf_db[last_rms_frame] if last_rms_frame < len(rms_hf_db) else -80
        hf_decline_limit = params.get("hf_decline_limit", 14)
        post_gap_rms_limit = params.get("post_gap_rms_limit", 8)

        hf_has_declined = False
        max_extension_frames = int(params.get("max_extension_sec", 0.25) / time_per_frame)

        for f in range(last_rms_frame + 1, search_end):
            # Energy derivative: if energy is declining steeply for several
            # consecutive frames, speech is ending — stop early
            if f > last_rms_frame + 3:
                recent_deriv = np.mean(rms_deriv[f-3:f+1])
                if recent_deriv < -decline_rate_threshold and rms_db[f] < rms_threshold:
                    break

            if end_mask[f]:
                if not hf_has_declined and rms_hf_db[f] > peak_hf_in_extension:
                    peak_hf_in_extension = rms_hf_db[f]

                hf_decline = peak_hf_in_extension - rms_hf_db[f]

                if hf_decline > hf_decline_limit:
                    hf_has_declined = True

                if hf_decline > hf_decline_limit * 1.2 and rms_db[f] < rms_threshold:
                    break

                if hf_has_declined and hf_decline < hf_decline_limit * 0.3 and rms_db[f] < rms_threshold:
                    break

                if f - last_rms_frame > max_extension_frames:
                    break

                if total_gap_frames > end_hold_frames * 2:
                    rms_speech = np.percentile(rms_db, rms_percentile)
                    if rms_db[f] < rms_speech - post_gap_rms_limit * 2:
                        break

                last_frame = f
                gap_count = 0
            else:
                gap_count += 1
                total_gap_frames += 1
                if gap_count > end_hold_frames:
                    break

    start_offset = (first_frame * hop_length) / sr
    end_offset = ((last_frame + 1) * hop_length) / sr

    start_offset = max(0, start_offset - start_padding)
    end_offset = min(len(audio_array) / sr, end_offset + end_padding)

    return start_offset, end_offset


def rms_envelope(audio_array, sr, video_threshold, **params):
    """
    RMS with smoothed envelope and lower threshold.
    Instead of using raw RMS frames, smooth the envelope to avoid
    missing brief consonants that dip below threshold for 1-2 frames.
    """
    percentile = params.get("percentile", 85)
    threshold_offset_db = params.get("threshold_offset_db", 18)
    padding = params.get("padding", 0.04)
    frame_length = params.get("frame_length", 512)
    hop_length = params.get("hop_length", 256)
    clip_db_diff = params.get("clip_db_diff", 5)
    smooth_frames = params.get("smooth_frames", 5)

    rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Smooth with max-pool to bridge short dips (e.g., a plosive release)
    from scipy.ndimage import maximum_filter1d
    rms_db_smooth = maximum_filter1d(rms_db, size=smooth_frames)

    clip_speech_level = np.percentile(rms_db, percentile)
    clip_threshold = clip_speech_level - threshold_offset_db
    if video_threshold - clip_threshold > clip_db_diff:
        silence_threshold = video_threshold
    else:
        silence_threshold = clip_threshold

    speech_frames = np.where(rms_db_smooth > silence_threshold)[0]
    if len(speech_frames) == 0:
        return 0.0, len(audio_array) / sr

    start_offset = (speech_frames[0] * hop_length) / sr
    end_offset = ((speech_frames[-1] + 1) * hop_length) / sr

    start_offset = max(0, start_offset - padding)
    end_offset = min(len(audio_array) / sr, end_offset + padding)

    return start_offset, end_offset
