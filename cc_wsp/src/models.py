from typing import Optional
from pydantic import BaseModel, Field, model_validator


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class Transcript(BaseModel):
    sentences: list["LLMTranscriptSentence"] = Field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None

    @property
    def full_text(self) -> str:
        return " ".join(sentence.sentence for sentence in self.sentences)

    @property
    def word_count(self) -> int:
        return sum(len(sentence.words) for sentence in self.sentences)


class LLMTranscriptSentence(BaseModel):
    sentence: str
    start: float
    end: float
    words: list[WordTimestamp] = Field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.start}-{self.end}]-{self.sentence}"

    def to_dict_for_prompt(self, include_words: bool = False) -> dict:
        result = {"sentence": self.sentence, "start": self.start, "end": self.end}
        if include_words and self.words:
            result["words"] = [
                {"word": w.word, "start": w.start, "end": w.end} for w in self.words
            ]
        return result


class EditingDecision(BaseModel):
    thoughts: str
    sentences_to_remove: list[int]


class SentenceResult(BaseModel):
    text: str
    keep: bool


class EditingResult(BaseModel):
    sentence_results: dict[str, SentenceResult]


class AdjustedSentence(BaseModel):
    original_start: float
    original_end: float
    adjusted_start: float
    adjusted_end: float
    text: str
    index: str
    threshold_source: str
    words: list[WordTimestamp] = Field(default_factory=list)


class AdjustedSentences(BaseModel):
    sentences: list[AdjustedSentence]


class ImageDescription(BaseModel):
    description: str
    detailed_prompt: str
    sentence_ids: list[str]


class ImageMetadata(BaseModel):
    filename: str
    prompt: str
    sentence_ids: list[str]
    generated_at: str
    generator_service: str


class ImagesMetadataFile(BaseModel):
    images: list[ImageMetadata] = Field(default_factory=list)


class GoogleDocLine(BaseModel):
    text: str
    image_filenames: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_single_image(cls, data):
        """Backward compat: convert old image_filename (singular) to image_filenames (list)."""
        if isinstance(data, dict) and "image_filename" in data:
            old = data.pop("image_filename")
            if "image_filenames" not in data:
                data["image_filenames"] = [old] if old else []
        return data


class GoogleDocScript(BaseModel):
    lines: list[GoogleDocLine] = Field(default_factory=list)


class GoogleDocImagePlacement(BaseModel):
    filepath: str
    sentence_index: str
    start_fraction: float = 0.0
    end_fraction: float = 1.0
    # Images sharing the same (sentence_index, start_fraction, end_fraction) form a
    # concurrent group shown at the same time. `layout` controls how that group is
    # composited into the safe zone:
    #   "tile"       -> auto grid (rows/cols chosen to match the safe-zone aspect)
    #   "horizontal" -> single row, images side by side
    #   "vertical"   -> single column, images stacked top-to-bottom
    # Ignored for solo placements. The whole group renders with the first member's layout.
    layout: str = "tile"


class GoogleDocImagePlacements(BaseModel):
    placements: list[GoogleDocImagePlacement] = Field(default_factory=list)


class ZoomFilter(BaseModel):
    sentence_index: str
    zoom_factor: float = 1.3  # 1.0 = no zoom, 1.5 = 50% zoom in
    x_offset: float = 0.0  # horizontal pan: -1 (left) to 1 (right), 0 = centered
    y_offset: float = 0.0  # vertical pan: -1 (up) to 1 (down), 0 = centered


class ZoomFilters(BaseModel):
    filters: list[ZoomFilter] = Field(default_factory=list)


class FaceData(BaseModel):
    top_frac: float
    bottom_frac: float
    left_frac: float
    right_frac: float
    samples_with_face: int
    total_samples: int

    @property
    def center_y(self) -> float:
        return (self.top_frac + self.bottom_frac) / 2
