"""
Video processing service for generating proxy assets.
Handles low-res video generation and audio extraction using ffmpeg.
"""

import subprocess
import os
from pathlib import Path

import numpy as np
import librosa
import time
from scipy.signal import butter, sosfilt
from scipy.ndimage import maximum_filter1d
from moviepy import VideoFileClip, concatenate_videoclips

# Set MoviePy config to use absolute paths
os.environ["MOVIEPY_AUDIO_BUFFERSIZE"] = "200000"

from cc_wsp.src.constants import (
    VIDEOS_DIR,
    LOW_RES_HEIGHT,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_BITRATE,
    VIDEO_CODEC,
    VIDEO_PRESET,
    AUDIO_CODEC,
)
from cc_wsp.src.util import (
    extract_filename_without_extension,
    get_downsampled_video_path,
    validate_file_exists,
    print_progress,
    ensure_directory_exists,
    get_audio_path,
    get_edited_video_path,
    prepare_transcript_for_prompt,
    get_input_video_path,
)
from cc_wsp.src.models import (
    Transcript,
    EditingResult,
    AdjustedSentence,
    AdjustedSentences,
    LLMTranscriptSentence,
)

# Silence detection constants (multiband_zcr algorithm)
SPEECH_LEVEL_PERCENTILE = 85
RMS_OFFSET_DB = 18
NOISE_FLOOR_MARGIN_DB = 8
HF_OFFSET_DB = 20
ZCR_CENTROID_MIN = 0.08
CENTROID_MIN = 1800
END_HOLD_FRAMES = 14
START_PADDING = 0.04
END_PADDING = 0.01
HF_DECLINE_LIMIT = 14
CLIP_DB_DIFFERENCE_THRESHOLD = 5


class VideoService:
    """
    Service for video processing operations.
    Generates proxy assets (low-res video, audio) using ffmpeg.
    """

    def __init__(self, assets_dir: Path | str | None = None):
        """
        Initialize video service.

        Args:
            assets_dir: Directory for storing assets. Defaults to constant.
        """
        self.assets_dir = Path(assets_dir) if assets_dir else VIDEOS_DIR
        ensure_directory_exists(self.assets_dir)
        self._video_level_threshold_cache = {}  # Cache for video-level thresholds

    def generate_proxy_video(
        self, input_path: str | Path, height: int = LOW_RES_HEIGHT, force: bool = False
    ) -> Path:
        """
        Generate a low-resolution proxy video.

        Args:
            input_path: Path to input video file
            height: Target height in pixels (width auto-calculated)
            force: If True, regenerate even if file exists

        Returns:
            Path to generated proxy video

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If ffmpeg fails
        """
        input_path = Path(input_path)
        validate_file_exists(input_path)

        # Build output path
        base_filename = extract_filename_without_extension(input_path)
        output_path = get_downsampled_video_path(base_filename)

        # Skip if exists and not forcing
        if output_path.exists() and not force:
            print_progress(f"Proxy video already exists: {output_path}")
            return output_path

        print_progress(f"Generating {height}p proxy video...")

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vf",
            f"scale=-2:{height}",  # -2 ensures width is divisible by 2
            "-c:v",
            VIDEO_CODEC,
            "-preset",
            VIDEO_PRESET,
            "-crf",
            "28",  # Higher CRF = lower quality/size
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-y",  # Overwrite output file
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print_progress(f"Proxy video created: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg proxy generation failed:\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {e.returncode}\n"
                f"stderr: {e.stderr}"
            ) from e

    def extract_audio(
        self,
        input_path: str | Path,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        force: bool = False,
    ) -> Path:
        """
        Extract audio from video as mono MP3 file.

        Args:
            input_path: Path to input video file
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels (1 = mono)
            force: If True, regenerate even if file exists

        Returns:
            Path to extracted audio file

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If ffmpeg fails
        """
        input_path = Path(input_path)
        validate_file_exists(input_path)

        # Build output path
        base_filename = extract_filename_without_extension(input_path)
        output_path = get_audio_path(base_filename)

        # Skip if exists and not forcing
        if output_path.exists() and not force:
            print_progress(f"Audio file already exists: {output_path}")
            return output_path

        print_progress(f"Extracting audio ({sample_rate}Hz, {channels}ch, MP3)...")

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vn",  # No video
            "-acodec",
            AUDIO_CODEC,
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-b:a",
            AUDIO_BITRATE,
            "-y",  # Overwrite output file
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print_progress(f"Audio extracted: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg audio extraction failed:\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {e.returncode}\n"
                f"stderr: {e.stderr}"
            ) from e

    def process_video(
        self, input_path: str | Path, force: bool = False
    ) -> tuple[Path, Path]:
        """
        Process video: generate both proxy video and extract audio.

        Args:
            input_path: Path to input video file
            force: If True, regenerate even if files exist

        Returns:
            Tuple of (proxy_video_path, audio_path)

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If processing fails
        """
        input_path = Path(input_path)
        validate_file_exists(input_path)

        print_progress(f"Processing video: {input_path.name}")

        # Generate proxy video
        proxy_video = self.generate_proxy_video(input_path, force=force)

        # Extract audio
        audio = self.extract_audio(input_path, force=force)

        print_progress("Video processing complete")

        return proxy_video, audio

    def get_video_info(self, video_path: str | Path) -> dict:
        """
        Get video metadata using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If ffprobe fails
        """
        video_path = Path(video_path)
        validate_file_exists(video_path)

        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            import json

            return json.loads(result.stdout)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffprobe failed:\nExit code: {e.returncode}\nstderr: {e.stderr}"
            ) from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {str(e)}") from e

    def _get_video_level_speech_threshold(self, audio_path: Path) -> float:
        """
        Calculate the video-level speech threshold by analyzing the entire audio file.
        This is cached per audio file to avoid repeated calculations.

        Args:
            audio_path: Path to the extracted audio file

        Returns:
            The silence threshold in dB for the entire video
        """
        # Check cache first
        cache_key = str(audio_path)
        if cache_key in self._video_level_threshold_cache:
            return self._video_level_threshold_cache[cache_key]

        print_progress("Calculating video-level speech threshold...")

        # Load entire audio file
        audio_array, sr = librosa.load(str(audio_path), sr=22050, mono=True)

        # Calculate RMS energy for entire file
        frame_length = 512
        hop_length = 256

        rms = librosa.feature.rms(
            y=audio_array, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Calculate video-level speech threshold
        speech_level_db = np.percentile(rms_db, SPEECH_LEVEL_PERCENTILE)
        video_threshold = speech_level_db - RMS_OFFSET_DB

        # Cache the result
        self._video_level_threshold_cache[cache_key] = video_threshold

        print_progress(
            f"Video-level threshold: {video_threshold:.2f} dB (speech level: {speech_level_db:.2f} dB)"
        )

        return video_threshold

    def _get_adjusted_sentence(
        self,
        audio_path: Path,
        sentence: LLMTranscriptSentence,
        sentence_index: int,
    ) -> AdjustedSentence:
        """
        Detect speech boundaries using multiband analysis.

        Uses different strategies for starts vs ends:
        - Starts: RMS-only (clean onsets have sharp energy rise)
        - Ends: Multi-signal — extends through trailing consonants using:
          - High-frequency energy (>3kHz) for sibilants (s, z)
          - ZCR + spectral centroid for plosives (k, t, p)
          - Noise floor check for trailing quiet voiced sounds
          - HF decline detection to stop when consonant fades into room noise
        """
        start = sentence.start
        end = sentence.end
        frame_length = 512
        hop_length = 128  # Higher resolution for boundary detection

        audio_array, sr = librosa.load(
            str(audio_path), sr=22050, mono=True, offset=start, duration=end - start,
        )

        time_per_frame = hop_length / sr

        # === Compute features ===
        rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # High-frequency RMS (>3kHz) — catches sibilants and plosives
        nyquist = sr / 2
        sos_hf = butter(4, min(3000 / nyquist, 0.99), btype='high', output='sos')
        audio_hf = sosfilt(sos_hf, audio_array)
        rms_hf = librosa.feature.rms(y=audio_hf, frame_length=frame_length, hop_length=hop_length)[0]
        rms_hf_db = librosa.amplitude_to_db(rms_hf, ref=np.max)

        # Zero crossing rate and spectral centroid — high for unvoiced consonants
        zcr = librosa.feature.zero_crossing_rate(audio_array, frame_length=frame_length, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

        n_frames = len(rms_db)

        # === Compute thresholds ===
        video_threshold = self._get_video_level_speech_threshold(audio_path)

        clip_speech_level = np.percentile(rms_db, SPEECH_LEVEL_PERCENTILE)
        clip_rms_threshold = clip_speech_level - RMS_OFFSET_DB
        if video_threshold - clip_rms_threshold > CLIP_DB_DIFFERENCE_THRESHOLD:
            rms_threshold = video_threshold
            threshold_source = "video-level"
        else:
            rms_threshold = clip_rms_threshold
            threshold_source = "clip-level"

        noise_floor = np.percentile(rms_db, 10)
        noise_threshold = noise_floor + NOISE_FLOOR_MARGIN_DB

        hf_speech_level = np.percentile(rms_hf_db, SPEECH_LEVEL_PERCENTILE)
        hf_threshold = hf_speech_level - HF_OFFSET_DB

        # === Build speech masks ===
        mask_rms = rms_db > rms_threshold
        mask_hf = rms_hf_db > hf_threshold
        mask_consonant = (zcr > ZCR_CENTROID_MIN) & (centroid > CENTROID_MIN) & (rms_db > noise_threshold)

        # Start detection: multi-signal (catches sibilant/plosive onsets)
        start_mask = mask_rms | mask_hf | mask_consonant
        start_frames = np.where(start_mask)[0]

        # End detection: combined mask
        end_mask = mask_rms | mask_hf | mask_consonant
        end_frames = np.where(end_mask)[0]

        if len(start_frames) == 0 or len(end_frames) == 0:
            if sentence_index:
                print(f"  Sentence {sentence_index}: No speech detected, keeping original")
            return AdjustedSentence(
                original_start=sentence.start, original_end=sentence.end,
                adjusted_start=sentence.start, adjusted_end=sentence.end,
                text=sentence.sentence, index=str(sentence_index),
                threshold_source=threshold_source,
            )

        first_frame = start_frames[0]

        # End: extend past last RMS frame through trailing consonants
        rms_frames = np.where(mask_rms)[0]
        last_rms_frame = rms_frames[-1] if len(rms_frames) > 0 else end_frames[-1]
        last_frame = last_rms_frame

        search_end = min(n_frames, last_rms_frame + int(0.5 / time_per_frame))
        gap_count = 0
        peak_hf = rms_hf_db[last_rms_frame] if last_rms_frame < len(rms_hf_db) else -80

        for f in range(last_rms_frame + 1, search_end):
            if end_mask[f]:
                if rms_hf_db[f] > peak_hf:
                    peak_hf = rms_hf_db[f]
                # Stop if HF has declined significantly (consonant fading into noise)
                hf_decline = peak_hf - rms_hf_db[f]
                if hf_decline > HF_DECLINE_LIMIT and rms_db[f] < rms_threshold:
                    break
                last_frame = f
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > END_HOLD_FRAMES:
                    break

        # Convert to timestamps
        start_offset = (first_frame * hop_length) / sr
        end_offset = ((last_frame + 1) * hop_length) / sr

        if first_frame == 0:
            # Speech already at frame 0 — transcription boundary may be mid-word
            # Apply padding as pre-roll before original start
            adjusted_start = max(0, start - START_PADDING)
        else:
            adjusted_start = max(start, start + start_offset - START_PADDING)
        adjusted_end = min(end, start + end_offset + END_PADDING)

        if sentence_index:
            print(
                f"  Sentence {sentence_index}: Trimmed {adjusted_start - start:.3f}s from start, "
                f"{end - adjusted_end:.3f}s from end"
            )
            print(f"    {start:.2f}s -> {adjusted_start:.2f}s to {end:.2f}s -> {adjusted_end:.2f}s")

        return AdjustedSentence(
            original_start=sentence.start, original_end=sentence.end,
            adjusted_start=adjusted_start, adjusted_end=adjusted_end,
            text=sentence.sentence, index=str(sentence_index),
            threshold_source=threshold_source,
        )

    def generate_adjusted_sentences(
        self,
        base_name: str,
        transcript: Transcript,
        editing_result: EditingResult,
        use_downsampled: bool = True,
        skip_silence_removal: bool = False,
    ) -> AdjustedSentences:
        """
        Generate adjusted sentences with optional silence-trimmed timestamps.

        This method processes each kept sentence from the editing result,
        and optionally analyzes the audio to detect speech boundaries and generates
        adjusted timestamps with silence removed from start and end.

        Args:
            base_name: Base filename without extension
            transcript: Transcript object with word-level timestamps
            editing_result: EditingResult with sentence keep/remove decisions
            use_downsampled: If True, use the downsampled video (default)
            skip_silence_removal: If True, skip silence removal and use original timestamps (default: False)

        Returns:
            AdjustedSentences object with trimmed timestamps (or original if skipped)

        Raises:
            FileNotFoundError: If input video doesn't exist
            RuntimeError: If audio analysis fails
        """
        # Get input video path
        if use_downsampled:
            input_path = get_downsampled_video_path(base_name)
        else:
            input_path = get_input_video_path(base_name)

        validate_file_exists(input_path)

        # Get sentences from transcript
        sentences = prepare_transcript_for_prompt(transcript)

        # Filter sentences based on editing result (1-indexed)
        kept_sentences = [
            sentence
            for i, sentence in enumerate(sentences, 1)
            if editing_result.sentence_results[str(i)].keep
        ]

        if not kept_sentences:
            raise ValueError("No sentences left after filtering")

        print_progress(f"Processing {len(kept_sentences)} kept sentences...")

        # If skipping silence removal, just use original timestamps
        if skip_silence_removal:
            print_progress("Skipping silence removal - using original timestamps")
            adjusted_sentence_list = []
            for idx, sentence in enumerate(kept_sentences, 1):
                adjusted_sentence_list.append(
                    AdjustedSentence(
                        original_start=sentence.start,
                        original_end=sentence.end,
                        adjusted_start=sentence.start,
                        adjusted_end=sentence.end,
                        text=sentence.sentence,
                        index=str(idx),
                        threshold_source="skipped",
                        words=sentence.words,
                    )
                )
            print_progress(
                f"Generated timestamps for {len(adjusted_sentence_list)} sentences (no silence removal)"
            )
            return AdjustedSentences(sentences=adjusted_sentence_list)

        # Get audio file path for silence removal
        audio_path = get_audio_path(base_name)
        validate_file_exists(audio_path)

        print_progress(
            f"Generating adjusted sentences with silence removal from {audio_path.name}..."
        )

        try:
            # Load video once (still needed for creating the edited video later)
            video = VideoFileClip(str(input_path))

            # Process each kept sentence
            adjusted_sentence_list = []

            for idx, sentence in enumerate(kept_sentences, 1):
                # Detect speech boundaries in this sentence's time range
                adjusted_sentence = self._get_adjusted_sentence(
                    audio_path, sentence, sentence_index=idx
                )

                # Create adjusted sentence record
                adjusted_sentence_list.append(adjusted_sentence)

            # Clean up video
            video.close()

            print_progress(
                f"Generated adjusted timestamps for {len(adjusted_sentence_list)} sentences"
            )

            return AdjustedSentences(sentences=adjusted_sentence_list)

        except Exception as e:
            raise RuntimeError(
                f"Failed to generate adjusted sentences: {str(e)}"
            ) from e

    def create_edited_video(
        self,
        base_name: str,
        adjusted_sentences: AdjustedSentences,
        use_downsampled: bool = True,
        force: bool = False,
        output_path: Path | None = None,
    ) -> Path:
        """
        Create an edited video using pre-computed adjusted sentences.

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences with silence-trimmed timestamps
            use_downsampled: If True, edit the downsampled video (default)
            force: If True, regenerate even if file exists
            output_path: Optional custom output path (if None, uses default path)

        Returns:
            Path to edited video file

        Raises:
            FileNotFoundError: If input video doesn't exist
            RuntimeError: If video editing fails
        """
        # Get input and output paths
        if use_downsampled:
            input_path = get_downsampled_video_path(base_name)
        else:
            input_path = get_input_video_path(base_name)

        if output_path is None:
            output_path = get_edited_video_path(base_name, use_downsampled)

        # Skip if exists and not forcing
        if output_path.exists() and not force:
            print_progress(f"Edited video already exists: {output_path}")
            return output_path

        validate_file_exists(input_path)

        print_progress(f"Creating edited video from {input_path.name}...")

        if not adjusted_sentences.sentences:
            raise ValueError("No sentences provided - cannot create video")

        print_progress(f"Using {len(adjusted_sentences.sentences)} adjusted sentences")

        try:
            # Load video
            video = VideoFileClip(str(input_path))

            # Create clips for each adjusted sentence
            clips = []

            for adj_sentence in adjusted_sentences.sentences:
                # Extract clip using adjusted timestamps
                clip = video.subclipped(
                    adj_sentence.adjusted_start, adj_sentence.adjusted_end
                )
                clips.append(clip)

            # Concatenate all clips
            print_progress("Concatenating video clips...")
            final_video = concatenate_videoclips(clips)

            print_progress(f"Writing edited video to {output_path.name}...")
            final_video.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,
                ffmpeg_params=[
                    "-x264-params",
                    "colorprim=bt709:transfer=bt709:colormatrix=bt709",
                    "-color_primaries", "bt709",
                    "-color_trc", "bt709",
                    "-colorspace", "bt709",
                    "-color_range", "tv",
                ],
            )

            # Clean up
            video.close()
            final_video.close()
            for clip in clips:
                clip.close()

            print_progress(f"Edited video created: {output_path}")
            return output_path

        except Exception as e:
            raise RuntimeError(f"Video editing failed: {str(e)}") from e
