"""
ElevenLabs Speech-to-Text service implementation.
Converts ElevenLabs API responses to internal models using the official SDK.
"""

import os
from pathlib import Path
from typing import Any, Dict

from elevenlabs.client import ElevenLabs  # type: ignore

from src.services.stt.base import SpeechToTextService
from src.models import Transcript, TranscriptSegment, WordTimestamp
from src.constants import ENV_ELEVENLABS_API_KEY
from src.util import validate_file_exists, prepare_transcript_for_prompt


class ElevenLabsSTTService(SpeechToTextService):
    """
    ElevenLabs implementation of Speech-to-Text service using the official SDK.
    Requests word-level timestamps and normalizes to internal format.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize ElevenLabs STT service.

        Args:
            api_key: ElevenLabs API key. If None, reads from environment.

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv(ENV_ELEVENLABS_API_KEY)
        if not self.api_key:
            raise ValueError(
                f"ElevenLabs API key not found. "
                f"Provide via constructor or {ENV_ELEVENLABS_API_KEY} env var."
            )

        # Initialize ElevenLabs client
        self.client = ElevenLabs(api_key=self.api_key)

    def transcribe(self, audio_path: str | Path) -> Transcript:
        """
        Transcribe audio using ElevenLabs STT API via official SDK.

        Args:
            audio_path: Path to audio file

        Returns:
            Internal Transcript model

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If API call fails
        """
        audio_path = Path(audio_path)
        validate_file_exists(audio_path)

        try:
            # Open audio file and call ElevenLabs API using SDK
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.speech_to_text.convert(
                    file=audio_file,
                    model_id="scribe_v1",  # Using scribe_v1 as per example
                    tag_audio_events=True,  # Tag audio events like laughter, applause
                    language_code="eng",  # Can be made configurable
                    diarize=True,  # Annotate who is speaking
                )

            # Convert SDK response to internal model
            transcript = self._convert_response(transcription)

            # Generate sentences from segments
            sentences = prepare_transcript_for_prompt(transcript)
            transcript.sentences = sentences

            return transcript

        except Exception as e:
            raise RuntimeError(f"ElevenLabs transcription failed: {str(e)}") from e

    def _convert_response(self, response: Any) -> Transcript:
        """
        Convert ElevenLabs SDK response to internal Transcript model.

        The SDK returns either:
        - SpeechToTextChunkResponseModel (single channel)
        - MultichannelSpeechToTextResponseModel (multiple channels)

        Args:
            response: ElevenLabs SDK response object

        Returns:
            Internal Transcript model
        """
        segments = []

        # Convert response to dict if it's a Pydantic model
        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        elif hasattr(response, "dict"):
            response_dict = response.dict()
        else:
            response_dict = response

        # Handle multichannel response
        if "transcripts" in response_dict:
            # MultichannelSpeechToTextResponseModel
            transcripts = response_dict.get("transcripts", [])
            # For now, just use the first channel
            if transcripts:
                response_dict = transcripts[0]
            else:
                # No transcripts available
                return Transcript(segments=[], language=None, duration=None)

        # Handle single channel response (SpeechToTextChunkResponseModel)
        full_text = response_dict.get("text", "")
        language_code = response_dict.get("language_code")
        words_data = response_dict.get("words", [])

        # Convert words to internal format
        words = self._extract_words_from_api(words_data)

        # Create a single segment with all words
        if words:
            segment = TranscriptSegment(
                text=full_text, start=words[0].start, end=words[-1].end, words=words
            )
            segments.append(segment)
        elif full_text:
            # Fallback: create segment without word timestamps
            segment = TranscriptSegment(text=full_text, start=0.0, end=0.0, words=[])
            segments.append(segment)

        # Calculate duration
        duration = None
        if segments and segments[-1].words:
            duration = segments[-1].words[-1].end

        return Transcript(segments=segments, language=language_code, duration=duration)

    def _extract_words_from_api(
        self, words_data: list[Dict[str, Any]]
    ) -> list[WordTimestamp]:
        """
        Extract word-level timestamps from API words array.

        According to API spec, each word object has:
        - text: string
        - start: number | null (seconds)
        - end: number | null (seconds)
        - type: "word" | "spacing" | "audio_event"
        - speaker_id: string | null
        - logprob: number (log probability)
        - characters: array | null (character-level details)

        Args:
            words_data: List of word objects from API

        Returns:
            List of WordTimestamp objects
        """
        words: list[WordTimestamp] = []

        for word_obj in words_data:
            # Convert to dict if it's a Pydantic model
            if hasattr(word_obj, "model_dump"):
                word_obj = word_obj.model_dump()
            elif hasattr(word_obj, "dict"):
                word_obj = word_obj.dict()

            word_type = word_obj.get("type", "word")

            # Only process actual words, skip spacing and audio events
            if word_type != "word":
                continue

            text = word_obj.get("text", "")
            start = word_obj.get("start")
            end = word_obj.get("end")

            # Skip if no timing information
            if start is None or end is None:
                continue

            words.append(WordTimestamp(word=text, start=float(start), end=float(end)))

        # Clean up timestamps for words with durations over 1 second
        words = self._clean_caption_timestamps(words)

        return words

    def _clean_caption_timestamps(
        self, words: list[WordTimestamp]
    ) -> list[WordTimestamp]:
        """
        Clean up caption timestamps for words with durations over 1 second.

        For words with duration > 1 second:
        - If word ends a sentence (has sentence-ending punctuation), reduce end time
          to 1 second after start
        - If word starts a sentence (previous word ends with punctuation), bring start
          time to 1 second before end

        Args:
            words: List of WordTimestamp objects

        Returns:
            List of WordTimestamp objects with cleaned timestamps
        """
        if not words:
            return words

        # Sentence-ending punctuation characters
        sentence_endings = {".", "!", "?"}

        cleaned_words = []

        for i, word in enumerate(words):
            duration = word.end - word.start

            # If duration is 1 second or less, keep as is
            if duration <= 1.0:
                cleaned_words.append(word)
                continue

            # Check if this word ends a sentence
            ends_sentence = any(
                word.word.rstrip().endswith(punct) for punct in sentence_endings
            )

            # Check if this word starts a sentence (previous word ended with punctuation)
            starts_sentence = False
            if i > 0:
                prev_word = words[i - 1]
                starts_sentence = any(
                    prev_word.word.rstrip().endswith(punct)
                    for punct in sentence_endings
                )

            # Adjust timestamps based on position in sentence
            if ends_sentence:
                # Word ends a sentence: reduce end time to 1 second after start
                cleaned_word = WordTimestamp(
                    word=word.word, start=word.start, end=word.start + 1.0
                )
                cleaned_words.append(cleaned_word)
            elif starts_sentence:
                # Word starts a sentence: bring start to 1 second before end
                cleaned_word = WordTimestamp(
                    word=word.word, start=word.end - 1.0, end=word.end
                )
                cleaned_words.append(cleaned_word)
            else:
                # Middle of sentence: keep original timestamps
                print(
                    "Warning, long word neither starts nor ends a sentence: ",
                    word.word,
                )
                cleaned_words.append(word)

        return cleaned_words
