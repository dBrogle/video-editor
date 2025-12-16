"""
Abstract base class for Speech-to-Text services.
No provider-specific types should be exposed.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from src.models import Transcript


class SpeechToTextService(ABC):
    """
    Abstract base class for speech-to-text transcription services.
    All implementations must return the internal Transcript model.
    """

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> Transcript:
        """
        Transcribe an audio file to text with word-level timestamps.

        Args:
            audio_path: Path to the audio file to transcribe

        Returns:
            Transcript object with segments and word-level timestamps

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If transcription fails
        """
        raise NotImplementedError
