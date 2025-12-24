"""
Abstract base class for Speech-to-Text services.
No provider-specific types should be exposed.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from src.models import Transcript, LLMTranscriptSentence, WordTimestamp
from src.constants import TIME_BETWEEN_WORDS_THRESHOLD

logger = logging.getLogger(__name__)


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
            Transcript object with sentences and word-level timestamps

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If transcription fails
        """
        raise NotImplementedError

    def _split_sentences_by_word_gaps(self, transcript: Transcript) -> Transcript:
        """
        Split sentences where consecutive words have large time gaps between them.

        This method examines each sentence and splits it into multiple sentences
        if there are gaps between consecutive words that exceed TIME_BETWEEN_WORDS_THRESHOLD.
        This helps handle cases where the STT provider incorrectly groups separate
        phrases into a single sentence due to missing punctuation.

        Args:
            transcript: The transcript to process

        Returns:
            A new Transcript with sentences split based on word gaps
        """
        if not transcript.sentences:
            return transcript

        new_sentences: list[LLMTranscriptSentence] = []

        for sentence in transcript.sentences:
            # If sentence has no words or only one word, keep as is
            if len(sentence.words) <= 1:
                new_sentences.append(sentence)
                continue

            # Find split points where gap exceeds threshold
            split_indices: list[int] = []
            for i in range(len(sentence.words) - 1):
                current_word = sentence.words[i]
                next_word = sentence.words[i + 1]
                gap = next_word.start - current_word.end

                if gap > TIME_BETWEEN_WORDS_THRESHOLD:
                    # Split after current word (i+1 is the start of next segment)
                    split_indices.append(i + 1)
                    logger.info(
                        f"Splitting sentence: gap of {gap:.3f}s between "
                        f"'{current_word.word}' (ends at {current_word.end:.3f}s) and "
                        f"'{next_word.word}' (starts at {next_word.start:.3f}s)"
                    )

            # If no splits needed, keep sentence as is
            if not split_indices:
                new_sentences.append(sentence)
                continue

            # Split the sentence into multiple sentences
            start_idx = 0
            for split_idx in split_indices:
                # Create a new sentence from start_idx to split_idx
                segment_words = sentence.words[start_idx:split_idx]
                if segment_words:
                    segment_text = " ".join(w.word for w in segment_words)
                    new_sentences.append(
                        LLMTranscriptSentence(
                            sentence=segment_text,
                            start=segment_words[0].start,
                            end=segment_words[-1].end,
                            words=segment_words,
                        )
                    )
                start_idx = split_idx

            # Add the final segment
            final_words = sentence.words[start_idx:]
            if final_words:
                final_text = " ".join(w.word for w in final_words)
                new_sentences.append(
                    LLMTranscriptSentence(
                        sentence=final_text,
                        start=final_words[0].start,
                        end=final_words[-1].end,
                        words=final_words,
                    )
                )

        # Return new transcript with split sentences
        return Transcript(
            sentences=new_sentences,
            language=transcript.language,
            duration=transcript.duration,
        )
