"""
Local file saver service for storing and loading transcriptions.
"""

import json
from pathlib import Path

from src.models import (
    Transcript,
    EditingDecision,
    EditingResult,
    AdjustedSentences,
    GoogleDocScript,
    GoogleDocImagePlacements,
)
from src.util import (
    get_transcription_path,
    get_downsampled_video_path,
    get_audio_path,
    get_editing_decision_path,
    get_editing_result_path,
    get_adjusted_sentences_path,
    get_google_doc_html_path,
    get_google_doc_folder,
    get_google_doc_images_folder,
    get_google_doc_script_path,
    get_google_doc_image_placements_path,
)
from src.constants import ASSETS_DIR


class LocalSaverService:
    """
    Service for saving and loading transcriptions locally.
    Also checks for existence of other pipeline assets.
    """

    LAST_FILENAME_PATH = ASSETS_DIR / ".last_filename"

    def save_transcription(self, base_name: str, transcript: Transcript) -> Path:
        """
        Save transcription to JSON file.

        Args:
            base_name: Base filename without extension
            transcript: Transcript object to save

        Returns:
            Path to saved transcription file
        """
        path = get_transcription_path(base_name)
        path.write_text(transcript.model_dump_json(indent=2))
        return path

    def load_transcription(self, base_name: str) -> Transcript:
        """
        Load transcription from JSON file.

        Args:
            base_name: Base filename without extension

        Returns:
            Transcript object

        Raises:
            FileNotFoundError: If transcription file doesn't exist
        """
        path = get_transcription_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Transcription not found: {path}")

        data = json.loads(path.read_text())
        return Transcript(**data)

    def transcription_exists(self, base_name: str) -> bool:
        """
        Check if transcription file exists.

        Args:
            base_name: Base filename without extension

        Returns:
            True if transcription exists, False otherwise
        """
        return get_transcription_path(base_name).exists()

    def downsampled_video_exists(self, base_name: str) -> bool:
        """
        Check if downsampled video exists.

        Args:
            base_name: Base filename without extension

        Returns:
            True if downsampled video exists, False otherwise
        """
        return get_downsampled_video_path(base_name).exists()

    def audio_exists(self, base_name: str) -> bool:
        """
        Check if audio file exists.

        Args:
            base_name: Base filename without extension

        Returns:
            True if audio exists, False otherwise
        """
        return get_audio_path(base_name).exists()

    def save_last_filename(self, base_name: str) -> None:
        """Save the last used filename."""
        self.LAST_FILENAME_PATH.write_text(base_name)

    def get_last_filename(self) -> str | None:
        """Get the last used filename, or None if none exists."""
        if self.LAST_FILENAME_PATH.exists():
            return self.LAST_FILENAME_PATH.read_text().strip()
        return None

    def save_editing_decision(self, base_name: str, decision: EditingDecision) -> Path:
        """
        Save editing decision to JSON file.

        Args:
            base_name: Base filename without extension
            decision: EditingDecision object to save

        Returns:
            Path to saved editing decision file
        """
        path = get_editing_decision_path(base_name)
        path.write_text(decision.model_dump_json(indent=2))
        return path

    def load_editing_decision(self, base_name: str) -> EditingDecision:
        """
        Load editing decision from JSON file.

        Args:
            base_name: Base filename without extension

        Returns:
            EditingDecision object

        Raises:
            FileNotFoundError: If editing decision file doesn't exist
        """
        path = get_editing_decision_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Editing decision not found: {path}")

        data = json.loads(path.read_text())
        return EditingDecision(**data)

    def editing_decision_exists(self, base_name: str) -> bool:
        """
        Check if editing decision file exists.

        Args:
            base_name: Base filename without extension

        Returns:
            True if editing decision exists, False otherwise
        """
        return get_editing_decision_path(base_name).exists()

    def save_editing_result(self, base_name: str, result: EditingResult) -> Path:
        """
        Save editing result to JSON file.

        Args:
            base_name: Base filename without extension
            result: EditingResult object to save

        Returns:
            Path to saved editing result file
        """
        path = get_editing_result_path(base_name)
        path.write_text(result.model_dump_json(indent=2))
        return path

    def load_editing_result(self, base_name: str) -> EditingResult:
        """
        Load editing result from JSON file.

        Args:
            base_name: Base filename without extension

        Returns:
            EditingResult object

        Raises:
            FileNotFoundError: If editing result file doesn't exist
        """
        path = get_editing_result_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Editing result not found: {path}")

        data = json.loads(path.read_text())
        return EditingResult(**data)

    def editing_result_exists(self, base_name: str) -> bool:
        """
        Check if editing result file exists.

        Args:
            base_name: Base filename without extension

        Returns:
            True if editing result exists, False otherwise
        """
        return get_editing_result_path(base_name).exists()

    def save_adjusted_sentences(
        self, base_name: str, adjusted_sentences: AdjustedSentences
    ) -> Path:
        """
        Save adjusted sentences to JSON file.

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences object to save

        Returns:
            Path to saved adjusted sentences file
        """
        path = get_adjusted_sentences_path(base_name)
        path.write_text(adjusted_sentences.model_dump_json(indent=2))
        return path

    def load_adjusted_sentences(self, base_name: str) -> AdjustedSentences:
        """
        Load adjusted sentences from JSON file.

        Args:
            base_name: Base filename without extension

        Returns:
            AdjustedSentences object

        Raises:
            FileNotFoundError: If adjusted sentences file doesn't exist
        """
        path = get_adjusted_sentences_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Adjusted sentences not found: {path}")

        data = json.loads(path.read_text())
        return AdjustedSentences(**data)

    def adjusted_sentences_exist(self, base_name: str) -> bool:
        """
        Check if adjusted sentences file exists.

        Args:
            base_name: Base filename without extension

        Returns:
            True if adjusted sentences exists, False otherwise
        """
        return get_adjusted_sentences_path(base_name).exists()

    def load_google_doc_html(self, base_name: str) -> str:
        """
        Load Google Doc HTML file content.

        Args:
            base_name: Base filename without extension

        Returns:
            HTML content as string

        Raises:
            FileNotFoundError: If HTML file doesn't exist
        """
        path = get_google_doc_html_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Google Doc HTML not found: {path}")

        return path.read_text(encoding="utf-8")

    def google_doc_html_exists(self, base_name: str) -> bool:
        """
        Check if Google Doc HTML file exists.

        Args:
            base_name: Base filename without extension

        Returns:
            True if HTML file exists, False otherwise
        """
        return get_google_doc_html_path(base_name).exists()

    def get_google_doc_images_path(self, base_name: str) -> Path:
        """
        Get path to Google Doc images folder.

        Args:
            base_name: Base filename without extension

        Returns:
            Path to images folder
        """
        return get_google_doc_images_folder(base_name)

    def save_google_doc_script(self, base_name: str, script: GoogleDocScript) -> Path:
        """
        Save Google Doc script to JSON file.

        Args:
            base_name: Base filename without extension
            script: GoogleDocScript object to save

        Returns:
            Path to saved script file
        """
        path = get_google_doc_script_path(base_name)
        path.write_text(script.model_dump_json(indent=2))
        return path

    def load_google_doc_script(self, base_name: str) -> GoogleDocScript:
        """
        Load Google Doc script from JSON file.

        Args:
            base_name: Base filename without extension

        Returns:
            GoogleDocScript object

        Raises:
            FileNotFoundError: If script file doesn't exist
        """
        path = get_google_doc_script_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Google Doc script not found: {path}")

        data = json.loads(path.read_text())
        return GoogleDocScript(**data)

    def google_doc_script_exists(self, base_name: str) -> bool:
        """
        Check if Google Doc script file exists.

        Args:
            base_name: Base filename without extension

        Returns:
            True if script file exists, False otherwise
        """
        return get_google_doc_script_path(base_name).exists()

    def save_google_doc_image_placements(
        self, base_name: str, placements: GoogleDocImagePlacements
    ) -> Path:
        """
        Save Google Doc image placements to JSON file.

        Args:
            base_name: Base filename without extension
            placements: GoogleDocImagePlacements object to save

        Returns:
            Path to saved placements file
        """
        path = get_google_doc_image_placements_path(base_name)
        path.write_text(placements.model_dump_json(indent=2))
        return path

    def load_google_doc_image_placements(
        self, base_name: str
    ) -> GoogleDocImagePlacements:
        """
        Load Google Doc image placements from JSON file.

        Args:
            base_name: Base filename without extension

        Returns:
            GoogleDocImagePlacements object

        Raises:
            FileNotFoundError: If placements file doesn't exist
        """
        path = get_google_doc_image_placements_path(base_name)
        if not path.exists():
            raise FileNotFoundError(f"Image placements not found: {path}")

        data = json.loads(path.read_text())
        return GoogleDocImagePlacements(**data)

    def google_doc_image_placements_exist(self, base_name: str) -> bool:
        """
        Check if Google Doc image placements file exists.

        Args:
            base_name: Base filename without extension

        Returns:
            True if placements file exists, False otherwise
        """
        return get_google_doc_image_placements_path(base_name).exists()
