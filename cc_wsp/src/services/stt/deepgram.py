"""
Deepgram Speech-to-Text service implementation.
Converts Deepgram API responses to internal models using the official SDK.
"""

import os
from pathlib import Path
from typing import Any, Dict

from deepgram import DeepgramClient  # type: ignore

from cc_wsp.src.services.stt.base import SpeechToTextService
from cc_wsp.src.models import (
    Transcript,
    WordTimestamp,
    LLMTranscriptSentence,
)
from cc_wsp.src.constants import ENV_DEEPGRAM_API_KEY
from cc_wsp.src.util import validate_file_exists


class DeepgramSTTService(SpeechToTextService):
    """
    Deepgram implementation of Speech-to-Text service using the official SDK.
    Requests word-level timestamps and normalizes to internal format.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize Deepgram STT service.

        Args:
            api_key: Deepgram API key. If None, reads from environment.

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv(ENV_DEEPGRAM_API_KEY)
        if not self.api_key:
            raise ValueError(
                f"Deepgram API key not found. "
                f"Provide via constructor or {ENV_DEEPGRAM_API_KEY} env var."
            )

        # Initialize Deepgram client
        self.client = DeepgramClient(api_key=self.api_key)

    def transcribe(self, audio_path: str | Path) -> Transcript:
        """
        Transcribe audio using Deepgram STT API via official SDK.

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
            # Open audio file and call Deepgram API using SDK
            with open(audio_path, "rb") as audio_file:
                # Read the file content
                audio_data = audio_file.read()

                # Call Deepgram API with transcription options.
                # Long recordings (e.g. a 75-min screen capture ~17MB) can
                # exceed the SDK's default write timeout during upload, so
                # give the request a generous timeout.
                response = self.client.listen.v1.media.transcribe_file(
                    request=audio_data,
                    model="nova-3",
                    language="en",
                    smart_format=True,
                    punctuate=True,
                    request_options={"timeout_in_seconds": 1200},
                )

            # Convert SDK response to internal model with sentences
            transcript = self._convert_response(response)

            # Split sentences based on word gaps before returning
            transcript = self._split_sentences_by_word_gaps(transcript)

            return transcript

        except Exception as e:
            raise RuntimeError(f"Deepgram transcription failed: {str(e)}") from e

    def _convert_response(self, response: Any) -> Transcript:
        """
        Convert Deepgram SDK response to internal Transcript model.

        The Deepgram response structure:
        {
            "metadata": {...},
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "...",
                                "confidence": 0.99,
                                "words": [
                                    {
                                        "word": "why",
                                        "start": 10.719999,
                                        "end": 10.96,
                                        "confidence": 0.99,
                                        "punctuated_word": "Why"
                                    },
                                    ...
                                ]
                            }
                        ]
                    }
                ]
            }
        }

        Args:
            response: Deepgram SDK response object

        Returns:
            Internal Transcript model
        """
        # Convert response to dict if it's a Pydantic model or has to_dict method
        if hasattr(response, "to_dict"):
            response_dict = response.to_dict()
        elif hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        elif hasattr(response, "dict"):
            response_dict = response.dict()
        else:
            response_dict = response

        # Navigate to the results
        results = response_dict.get("results", {})
        channels = results.get("channels", [])

        if not channels:
            return Transcript(sentences=[], language=None, duration=None)

        # Get the first channel (usually there's only one)
        channel = channels[0]
        alternatives = channel.get("alternatives", [])

        if not alternatives:
            return Transcript(sentences=[], language=None, duration=None)

        # Get the best alternative (first one, highest confidence)
        alternative = alternatives[0]
        words_data = alternative.get("words", [])

        # Convert words to internal format and generate sentences
        words = self._extract_words_from_api(words_data)
        sentences = self._create_sentences_from_words(words)

        # Calculate duration
        duration = None
        if sentences and sentences[-1].words:
            duration = sentences[-1].words[-1].end

        # Extract language from metadata if available
        metadata = response_dict.get("metadata", {})
        language = metadata.get("language", "en")

        return Transcript(sentences=sentences, language=language, duration=duration)

    def _extract_words_from_api(
        self, words_data: list[Dict[str, Any]]
    ) -> list[WordTimestamp]:
        """
        Extract word-level timestamps from Deepgram API words array.

        According to Deepgram API spec, each word object has:
        - word: string (original word)
        - punctuated_word: string (word with punctuation)
        - start: float (seconds)
        - end: float (seconds)
        - confidence: float (0-1)

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

            # Use punctuated_word if available, otherwise fall back to word
            text = word_obj.get("punctuated_word") or word_obj.get("word", "")
            start = word_obj.get("start")
            end = word_obj.get("end")

            # Skip if no timing information
            if start is None or end is None:
                continue

            words.append(WordTimestamp(word=text, start=float(start), end=float(end)))

        return words

    def _create_sentences_from_words(
        self, words: list[WordTimestamp]
    ) -> list[LLMTranscriptSentence]:
        """
        Create sentences from words by detecting sentence-ending punctuation.

        A sentence ends when a word's last character is sentence-ending punctuation
        (., ?, or !). This uses Deepgram's punctuated_word field which includes
        proper punctuation.

        Args:
            words: List of WordTimestamp objects with punctuated words

        Returns:
            List of LLMTranscriptSentence objects
        """
        if not words:
            return []

        # Sentence-ending punctuation characters
        sentence_endings = {".", "!", "?"}

        sentences: list[LLMTranscriptSentence] = []
        current_words: list[WordTimestamp] = []
        current_start: float | None = None

        for word in words:
            # Set start time if this is the first word in the sentence
            if current_start is None:
                current_start = word.start

            # Add the word to current sentence
            current_words.append(word)

            # Check if the word ends with sentence-ending punctuation
            word_text = word.word.rstrip()  # Remove trailing whitespace
            if word_text and word_text[-1] in sentence_endings:
                # Complete the current sentence
                if current_words and current_start is not None:
                    sentence_text = " ".join(w.word for w in current_words)
                    sentences.append(
                        LLMTranscriptSentence(
                            sentence=sentence_text,
                            start=current_start,
                            end=word.end,
                            words=current_words.copy(),
                        )
                    )

                # Reset for next sentence
                current_words = []
                current_start = None

        # Handle any remaining words that didn't end with punctuation
        if current_words and current_start is not None:
            sentence_text = " ".join(w.word for w in current_words)
            sentences.append(
                LLMTranscriptSentence(
                    sentence=sentence_text,
                    start=current_start,
                    end=current_words[-1].end,
                    words=current_words,
                )
            )

        return sentences
