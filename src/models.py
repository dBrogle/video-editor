"""
Internal, service-agnostic Pydantic models.
These are the source of truth for all data representations in the system.
"""

from typing import Optional
from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    """
    Represents a single word with timing information.
    """

    word: str = Field(..., description="The word text")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class TranscriptSegment(BaseModel):
    """
    Represents a segment of transcribed text with word-level timestamps.
    """

    text: str = Field(..., description="Full text of the segment")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    words: list[WordTimestamp] = Field(
        default_factory=list, description="Word-level timestamps"
    )


class Transcript(BaseModel):
    """
    Complete transcript of an audio file.
    This is the canonical representation used throughout the system.
    """

    segments: list[TranscriptSegment] = Field(
        default_factory=list, description="List of transcript segments"
    )
    sentences: list["LLMTranscriptSentence"] = Field(
        default_factory=list,
        description="List of sentences extracted from segments for editing",
    )
    language: Optional[str] = Field(None, description="Detected language code")
    duration: Optional[float] = Field(
        None, description="Total audio duration in seconds"
    )

    @property
    def full_text(self) -> str:
        """Get the complete transcript text."""
        return " ".join(segment.text for segment in self.segments)

    @property
    def word_count(self) -> int:
        """Get total word count across all segments."""
        return sum(len(segment.words) for segment in self.segments)


class LLMTranscriptSentence(BaseModel):
    """
    Represents a sentence extracted from a transcript for LLM prompts.
    Sentences are formed by grouping words until punctuation is encountered.
    """

    sentence: str = Field(..., description="The complete sentence text")
    start: float = Field(..., description="Start time of the sentence in seconds")
    end: float = Field(..., description="End time of the sentence in seconds")

    def __str__(self) -> str:
        """Format sentence as [{start}-{end}]-{sentence}"""
        return f"[{self.start}-{self.end}]-{self.sentence}"


class EditingDecision(BaseModel):
    """
    LLM response for video editing decisions.
    """

    thoughts: str = Field(..., description="LLM reasoning about editing choices")
    sentences_to_remove: list[int] = Field(
        ..., description="List of sentence indices to remove (1-indexed)"
    )


class SentenceResult(BaseModel):
    """
    Result for a single sentence in the editing process.
    """

    text: str = Field(..., description="The sentence text")
    keep: bool = Field(..., description="Whether to keep this sentence")


class EditingResult(BaseModel):
    """
    Human-editable format for editing decisions.
    Maps sentence numbers to their text and keep/remove status.
    """

    sentence_results: dict[str, SentenceResult] = Field(
        ..., description="Dictionary mapping sentence number (1-indexed) to result"
    )


class AdjustedSentence(BaseModel):
    """
    Represents a sentence with adjusted timestamps after silence removal.
    """

    original_start: float = Field(
        ..., description="Original start time in the video (seconds)"
    )
    original_end: float = Field(
        ..., description="Original end time in the video (seconds)"
    )
    adjusted_start: float = Field(
        ..., description="Adjusted start time after silence removal (seconds)"
    )
    adjusted_end: float = Field(
        ..., description="Adjusted end time after silence removal (seconds)"
    )
    text: str = Field(..., description="The sentence text")
    index: str = Field(..., description="The index of the sentence")
    threshold_source: str = Field(..., description="Source of audio thresholding")


class AdjustedSentences(BaseModel):
    """
    Collection of sentences with adjusted timestamps after silence trimming.
    """

    sentences: list[AdjustedSentence] = Field(
        ..., description="List of sentences with adjusted timestamps"
    )
