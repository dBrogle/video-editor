from typing import Optional
from pydantic import BaseModel, Field


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
    image_filename: Optional[str] = None


class GoogleDocScript(BaseModel):
    lines: list[GoogleDocLine] = Field(default_factory=list)


class GoogleDocImagePlacement(BaseModel):
    filepath: str
    sentence_index: str
    start_fraction: float = 0.0
    end_fraction: float = 1.0


class GoogleDocImagePlacements(BaseModel):
    placements: list[GoogleDocImagePlacement] = Field(default_factory=list)
