"""
Whisper.cpp Speech-to-Text service implementation.
Runs the local whisper.cpp binary for transcription, ideal for long-form audio
like streams where cloud APIs have duration limits.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from src.services.stt.base import SpeechToTextService
from src.models import Transcript, WordTimestamp, LLMTranscriptSentence
from src.util import validate_file_exists, print_progress


# Default path to the whisper.cpp binary and models
WHISPER_CPP_DIR = Path("/Users/brogle/workspace/brogle/claude_video_editing/whisper.cpp")
WHISPER_BINARY = WHISPER_CPP_DIR / "main"
WHISPER_MODELS_DIR = WHISPER_CPP_DIR / "models"


class WhisperSTTService(SpeechToTextService):
    """
    Whisper.cpp implementation of Speech-to-Text service.
    Runs locally, no API key needed, handles long-form audio.
    """

    def __init__(
        self,
        model_name: str = "base.en",
        whisper_binary: Path | str | None = None,
        models_dir: Path | str | None = None,
        threads: int = 4,
    ):
        self.model_name = model_name
        self.whisper_binary = Path(whisper_binary) if whisper_binary else WHISPER_BINARY
        self.models_dir = Path(models_dir) if models_dir else WHISPER_MODELS_DIR
        self.threads = threads

        self.model_path = self.models_dir / f"ggml-{model_name}.bin"

        if not self.whisper_binary.exists():
            raise FileNotFoundError(
                f"whisper.cpp binary not found: {self.whisper_binary}\n"
                f"Build it with: cd {WHISPER_CPP_DIR} && make"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Whisper model not found: {self.model_path}\n"
                f"Download it with: cd {WHISPER_CPP_DIR} && bash models/download-ggml-model.sh {model_name}"
            )

    def transcribe(self, audio_path: str | Path) -> Transcript:
        audio_path = Path(audio_path)
        validate_file_exists(audio_path)

        # whisper.cpp requires 16kHz mono WAV input
        wav_path = self._convert_to_wav(audio_path)

        try:
            # Run whisper.cpp with full JSON output
            json_data = self._run_whisper(wav_path)

            # Convert to internal Transcript model
            transcript = self._parse_whisper_json(json_data)

            # Split sentences on large word gaps (same as Deepgram service)
            transcript = self._split_sentences_by_word_gaps(transcript)

            return transcript
        finally:
            # Clean up temp WAV if we created one
            if wav_path != audio_path and wav_path.exists():
                wav_path.unlink()

    def _convert_to_wav(self, audio_path: Path) -> Path:
        """Convert audio to 16kHz mono WAV for whisper.cpp."""
        if audio_path.suffix.lower() == ".wav":
            return audio_path

        print_progress("Converting audio to 16kHz WAV for whisper.cpp...")
        wav_path = audio_path.with_suffix(".wav")

        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-y",
            str(wav_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return wav_path

    def _run_whisper(self, wav_path: Path) -> dict:
        """Run whisper.cpp and return parsed JSON output."""
        # Use a temp file for JSON output
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_base = tmp.name.replace(".json", "")

        cmd = [
            str(self.whisper_binary),
            "-m", str(self.model_path),
            "-f", str(wav_path),
            "-t", str(self.threads),
            "-l", "en",
            "-ojf",  # Full JSON output with token-level timestamps
            "-of", output_base,  # Output file path (without extension)
            "-pp",  # Print progress
        ]

        print_progress(f"Running whisper.cpp ({self.model_name} model)...")
        print_progress(f"Input: {wav_path.name}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        # whisper.cpp prints progress to stderr
        if result.stderr:
            # Print last few lines of stderr for progress visibility
            lines = result.stderr.strip().split("\n")
            for line in lines[-3:]:
                if line.strip():
                    print_progress(f"  whisper: {line.strip()}")

        if result.returncode != 0:
            raise RuntimeError(
                f"whisper.cpp failed (exit code {result.returncode}):\n{result.stderr}"
            )

        # Read the JSON output
        json_path = Path(f"{output_base}.json")
        if not json_path.exists():
            raise RuntimeError(
                f"whisper.cpp did not produce JSON output at: {json_path}"
            )

        try:
            json_data = json.loads(json_path.read_text())
            return json_data
        finally:
            json_path.unlink(missing_ok=True)

    def _parse_whisper_json(self, data: dict) -> Transcript:
        """
        Parse whisper.cpp full JSON output into internal Transcript model.

        The JSON structure has:
        - transcription: list of segments, each with:
          - timestamps: {from, to}
          - offsets: {from, to} in milliseconds
          - text: segment text
          - tokens: list of token objects (with -ojf flag), each with:
            - text: token text
            - offsets: {from, to} in milliseconds
            - p: probability
        """
        transcription = data.get("transcription", [])
        language = data.get("result", {}).get("language", "en")

        sentences: list[LLMTranscriptSentence] = []

        for segment in transcription:
            seg_start_ms = segment.get("offsets", {}).get("from", 0)
            seg_end_ms = segment.get("offsets", {}).get("to", 0)
            seg_text = segment.get("text", "").strip()

            if not seg_text:
                continue

            # Extract word-level timestamps from tokens
            words: list[WordTimestamp] = []
            tokens = segment.get("tokens", [])

            for token in tokens:
                token_text = token.get("text", "").strip()
                if not token_text:
                    continue
                # Skip special tokens (start with [)
                if token_text.startswith("["):
                    continue

                token_offsets = token.get("offsets", {})
                t_start_ms = token_offsets.get("from", seg_start_ms)
                t_end_ms = token_offsets.get("to", seg_end_ms)

                words.append(WordTimestamp(
                    word=token_text,
                    start=t_start_ms / 1000.0,
                    end=t_end_ms / 1000.0,
                ))

            # If no token-level data, create a single word entry for the segment
            if not words:
                words = [WordTimestamp(
                    word=seg_text,
                    start=seg_start_ms / 1000.0,
                    end=seg_end_ms / 1000.0,
                )]

            sentences.append(LLMTranscriptSentence(
                sentence=seg_text,
                start=seg_start_ms / 1000.0,
                end=seg_end_ms / 1000.0,
                words=words,
            ))

        # Calculate total duration
        duration = None
        if sentences:
            duration = sentences[-1].end

        return Transcript(sentences=sentences, language=language, duration=duration)
