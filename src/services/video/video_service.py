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
from moviepy import VideoFileClip, concatenate_videoclips

# Set MoviePy config to use absolute paths
os.environ["MOVIEPY_AUDIO_BUFFERSIZE"] = "200000"

from src.constants import (
    ASSETS_DIR,
    LOW_RES_HEIGHT,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    VIDEO_CODEC,
    VIDEO_PRESET,
    AUDIO_CODEC,
)
from src.util import (
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
from src.models import (
    Transcript,
    EditingResult,
    AdjustedSentence,
    AdjustedSentences,
    LLMTranscriptSentence,
)

# Silence detection constants
SPEECH_LEVEL_PERCENTILE = 85  # Percentile of RMS dB to use as speech level reference
SILENCE_THRESHOLD_OFFSET_DB = (
    15  # dB below speech level to consider as silence (15-25 typical range)
)
SILENCE_PADDING = 0.02  # Seconds to pad before/after detected speech
CLIP_DB_DIFFERENCE_THRESHOLD = 5  # If clip's speech level differs from video by more than this (dB), use video-level threshold


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
        self.assets_dir = Path(assets_dir) if assets_dir else ASSETS_DIR
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
        Extract audio from video as mono WAV file.

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

        print_progress(f"Extracting audio ({sample_rate}Hz, {channels}ch)...")

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
        video_threshold = speech_level_db - SILENCE_THRESHOLD_OFFSET_DB

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
        Detect speech boundaries by analyzing audio amplitude using librosa.
        Returns adjusted (start, end) timestamps with silence trimmed.

        Args:
            audio_path: Path to the extracted audio file
            start: Start time of segment in original video (seconds)
            end: End time of segment in original video (seconds)
            sentence_index: Optional sentence index for debugging (1-based)

        Returns:
            Tuple of (adjusted_start, adjusted_end) in seconds relative to original video
        """
        start = sentence.start
        end = sentence.end
        # Load only the specific time segment directly with librosa from the audio file
        audio_array, sr = librosa.load(
            str(audio_path),
            sr=22050,
            mono=True,
            offset=start,
            duration=end - start,
        )

        # Calculate RMS (Root Mean Square) energy per frame
        frame_length = 512  # ~23ms at 22050 Hz
        hop_length = 256  # 50% overlap

        rms = librosa.feature.rms(
            y=audio_array, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # ADAPTIVE THRESHOLDING
        # Find the speech level (use configurable percentile - typically 75th-85th)
        # This captures the typical speech energy level
        clip_speech_level_db = np.percentile(rms_db, SPEECH_LEVEL_PERCENTILE)
        clip_threshold = clip_speech_level_db - SILENCE_THRESHOLD_OFFSET_DB

        # Get video-level threshold for comparison
        video_threshold = self._get_video_level_speech_threshold(audio_path)

        # Check if clip's speech level deviates significantly from video level
        # If the clip is mostly silence (e.g., >80% silence), its speech level will be much lower
        # In that case, use the video-level threshold instead
        speech_level_lower = video_threshold - clip_threshold

        threshold_source = ""
        if speech_level_lower > CLIP_DB_DIFFERENCE_THRESHOLD:
            # Clip is mostly quiet - use video-level threshold
            print("Using video level threshold")
            silence_threshold = video_threshold
            threshold_source = "video-level"
        else:
            # Clip is representative - use clip-level threshold
            silence_threshold = clip_threshold
            threshold_source = "clip-level"

        # DEBUG: Output for sentences 3, 4, 5
        if sentence_index in [3, 4, 5]:
            # Calculate sampling interval to show ~100th of second (0.01s)
            # Each RMS frame represents hop_length/sr seconds
            time_per_frame = hop_length / sr
            samples_per_0_01s = max(1, int(0.01 / time_per_frame))

            print(f"\n{'=' * 60}")
            print(f"DEBUG: Sentence {sentence_index}")
            print(
                f"Time range: {start:.3f}s - {end:.3f}s (duration: {end - start:.3f}s)"
            )
            print(f"Audio array length: {len(audio_array)} samples at {sr} Hz")
            print(f"RMS frames: {len(rms)} frames")
            print(f"{'=' * 60}")

            # Show percentile analysis
            print(f"\nAdaptive Threshold Analysis:")
            print(f"  RMS dB percentiles:")
            for p in [50, 60, 70, 75, 80, 85, 90, 95]:
                print(f"    {p}th: {np.percentile(rms_db, p):.2f} dB")
            print(f"  Using {threshold_source} threshold")
            print(
                f"  Speech level ({SPEECH_LEVEL_PERCENTILE}th percentile): {silence_threshold:.2f} dB"
            )
            print(
                f"  Silence threshold (speech - {SILENCE_THRESHOLD_OFFSET_DB}dB): {silence_threshold:.2f} dB"
            )
            print(
                f"  Frames above threshold: {np.sum(rms_db > silence_threshold)}/{len(rms_db)}"
            )

            print(f"\nRaw Statistics:")
            print(f"  RMS dB Min: {rms_db.min():.2f}")
            print(f"  RMS dB Max: {rms_db.max():.2f}")
            print(f"  RMS dB Mean: {rms_db.mean():.2f}")
            print(f"  RMS dB Median: {np.median(rms_db):.2f}")
            print(f"{'=' * 60}\n")

        # Find first and last frames above threshold
        speech_frames = np.where(rms_db > silence_threshold)[0]

        if len(speech_frames) == 0:
            # No speech detected, return original times
            if sentence_index:
                print(
                    f"  Sentence {sentence_index}: No speech detected, keeping original {start:.2f}s-{end:.2f}s"
                )
            return AdjustedSentence(
                original_start=sentence.start,
                original_end=sentence.end,
                adjusted_start=sentence.start,
                adjusted_end=sentence.end,
                text=sentence.sentence,
                index=str(sentence_index),
                threshold_source=threshold_source,
            )

        first_speech_frame = speech_frames[0]
        last_speech_frame = speech_frames[-1]

        # Convert frames back to time offsets
        start_offset = (first_speech_frame * hop_length) / sr
        end_offset = ((last_speech_frame + 1) * hop_length) / sr

        # Apply to original timestamps
        adjusted_start = start + start_offset
        adjusted_end = start + end_offset

        # Apply padding (but keep within original segment bounds)
        adjusted_start = max(start, adjusted_start - SILENCE_PADDING)
        adjusted_end = min(end, adjusted_end + SILENCE_PADDING)

        if sentence_index:
            trimmed_start = adjusted_start - start
            trimmed_end = end - adjusted_end
            print(
                f"  Sentence {sentence_index}: Trimmed {trimmed_start:.3f}s from start, {trimmed_end:.3f}s from end"
            )
            print(
                f"    {start:.2f}s -> {adjusted_start:.2f}s to {end:.2f}s -> {adjusted_end:.2f}s"
            )
        return AdjustedSentence(
            original_start=sentence.start,
            original_end=sentence.end,
            adjusted_start=adjusted_start,
            adjusted_end=adjusted_end,
            text=sentence.sentence,
            index=str(sentence_index),
            threshold_source=threshold_source,
        )

    def generate_adjusted_sentences(
        self,
        base_name: str,
        transcript: Transcript,
        editing_result: EditingResult,
        use_downsampled: bool = True,
    ) -> AdjustedSentences:
        """
        Generate adjusted sentences with silence-trimmed timestamps.

        This method processes each kept sentence from the editing result,
        analyzes the audio to detect speech boundaries, and generates
        adjusted timestamps with silence removed from start and end.

        Args:
            base_name: Base filename without extension
            transcript: Transcript object with word-level timestamps
            editing_result: EditingResult with sentence keep/remove decisions
            use_downsampled: If True, use the downsampled video (default)

        Returns:
            AdjustedSentences object with trimmed timestamps

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

        # Get audio file path
        audio_path = get_audio_path(base_name)
        validate_file_exists(audio_path)

        print_progress(
            f"Generating adjusted sentences with silence removal from {audio_path.name}..."
        )

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
    ) -> Path:
        """
        Create an edited video using pre-computed adjusted sentences.

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences with silence-trimmed timestamps
            use_downsampled: If True, edit the downsampled video (default)
            force: If True, regenerate even if file exists

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
            print_progress("Concatenating video segments...")
            final_video = concatenate_videoclips(clips)

            # Write output
            print_progress(f"Writing edited video to {output_path.name}...")
            final_video.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,
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
