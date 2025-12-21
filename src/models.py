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
    words: list[WordTimestamp] = Field(
        default_factory=list, description="Word-level timestamps for the sentence"
    )

    def __str__(self) -> str:
        """Format sentence as [{start}-{end}]-{sentence}"""
        return f"[{self.start}-{self.end}]-{self.sentence}"

    def to_dict_for_prompt(self, include_words: bool = False) -> dict:
        """
        Convert to dictionary for LLM prompts.

        Args:
            include_words: Whether to include word-level timestamps

        Returns:
            Dictionary representation
        """
        result = {
            "sentence": self.sentence,
            "start": self.start,
            "end": self.end,
        }
        if include_words and self.words:
            result["words"] = [
                {"word": w.word, "start": w.start, "end": w.end} for w in self.words
            ]
        return result


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
    words: list[WordTimestamp] = Field(
        default_factory=list, description="Word-level timestamps for the sentence"
    )


class AdjustedSentences(BaseModel):
    """
    Collection of sentences with adjusted timestamps after silence trimming.
    """

    sentences: list[AdjustedSentence] = Field(
        ..., description="List of sentences with adjusted timestamps"
    )


class ImageDescription(BaseModel):
    """
    Description of an image to be generated, with sentence associations.
    """

    description: str = Field(..., description="Human-readable description of the image")
    detailed_prompt: str = Field(
        ..., description="Detailed prompt optimized for image generation"
    )
    sentence_ids: list[str] = Field(
        ..., description="List of sentence IDs where this image should be shown"
    )


class ImageMetadata(BaseModel):
    """
    Metadata for a generated image.
    """

    filename: str = Field(..., description="Image filename (e.g., 'image_001.png')")
    prompt: str = Field(..., description="The prompt used to generate this image")
    sentence_ids: list[str] = Field(
        ..., description="List of sentence IDs where this image is shown"
    )
    generated_at: str = Field(..., description="ISO timestamp of generation")
    generator_service: str = Field(
        ..., description="Service used to generate (e.g., 'dalle', 'stability')"
    )


class ImagesMetadataFile(BaseModel):
    """
    Container for all image metadata in a project.
    """

    images: list[ImageMetadata] = Field(
        default_factory=list, description="List of all image metadata"
    )


class GoogleDocLine(BaseModel):
    """
    Represents a line of text from a Google Doc with optional associated image.
    """

    text: str = Field(..., description="The line text content")
    image_filename: Optional[str] = Field(
        None, description="Associated image filename if present (e.g., 'image1.png')"
    )


class GoogleDocScript(BaseModel):
    """
    Container for all lines parsed from a Google Doc HTML.
    """

    lines: list[GoogleDocLine] = Field(
        default_factory=list, description="List of script lines with images"
    )


class GoogleDocImagePlacement(BaseModel):
    """
    Represents a placed image with sentence associations from Google Doc script.
    """

    filepath: str = Field(..., description="Path to the image file")
    sentence_indexes: list[str] = Field(
        ..., description="List of sentence indexes where this image should be shown"
    )


class GoogleDocImagePlacements(BaseModel):
    """
    Container for all image placements from Google Doc script.
    """

    placements: list[GoogleDocImagePlacement] = Field(
        default_factory=list, description="List of image placements with timing"
    )
