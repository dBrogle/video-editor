"""
Google Doc Image Placer Agent - Places images from Google Doc script into video timeline.
"""

import json
from pathlib import Path
from typing import Optional

from src.models import (
    AdjustedSentences,
    GoogleDocScript,
    GoogleDocImagePlacement,
    GoogleDocImagePlacements,
)
from src.services.llm.base import LLMService
from src.services.llm.openrouter import OpenRouterLLMService


GOOGLE_DOC_IMAGE_PLACER_PROMPT = """You are a video editing assistant helping to place images from a script onto a video timeline.

You have two pieces of information:
1. The Google Doc script with text lines and associated images
2. The actual video sentences with their indexes

Your task is to match the script's image placements to the actual video sentences and determine which sentence indexes each image should appear over.

Google Doc Script (what was planned):
{google_doc_script_json}

Actual Video Sentences (indexed):
{adjusted_sentences_json}

Instructions:
- Match each image from the Google Doc script to the corresponding sentences in the actual video
- The speaker may deviate slightly from the script, so use semantic matching (not exact text matching)
- Determine which sentence indexes each image should be shown over based on when that topic is discussed
- Images typically span 1-3 sentences, but can span more if the topic warrants it
- Consider the flow and avoid too many rapid image changes
- Images should never overlap (each sentence should have at most one image)
- If an image in the script doesn't match any video content, skip it

Respond with a JSON object in this exact format:
{{
    "thoughts": "Your analysis of how the script maps to the video and your placement decisions",
    "placements": [
        {{
            "filepath": "image1.png",
            "sentence_indexes": ["1", "2"]
        }},
        {{
            "filepath": "image2.png",
            "sentence_indexes": ["5", "6", "7"]
        }}
    ]
}}

Important:
- sentence_indexes should be strings (e.g., "1", "2", not 1, 2)
- Only include images that have a clear match in the video
- Images should never overlap - each sentence should appear in at most one placement
- Placements should be in chronological order (earlier sentence indexes first)
- The filepath should match the image filename from the Google Doc script (e.g., "image1.png", "image2.png")
"""


class GoogleDocImagePlacer:
    """
    Agent for placing images from Google Doc script onto video timeline.
    Uses an LLM to intelligently match script content to actual video sentences.
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize the Google Doc Image Placer.

        Args:
            llm_service: LLM service to use. If None, creates default OpenRouterLLMService.
        """
        self.llm_service = llm_service or OpenRouterLLMService(
            temperature=0.3,  # Lower temperature for more consistent placement decisions
            max_tokens=4000,
        )

    def place_images(
        self,
        google_doc_script: GoogleDocScript,
        adjusted_sentences: AdjustedSentences,
        google_doc_images_folder: Path,
    ) -> GoogleDocImagePlacements:
        """
        Place images from Google Doc script onto video timeline by matching to sentence indexes.

        Args:
            google_doc_script: Parsed Google Doc script with text and image associations
            adjusted_sentences: Video sentences with indexes
            google_doc_images_folder: Path to folder containing Google Doc images

        Returns:
            GoogleDocImagePlacements with image paths and sentence indexes

        Raises:
            RuntimeError: If LLM fails to generate valid response
        """
        # Convert inputs to JSON for the prompt
        script_json = self._script_to_json(google_doc_script)
        sentences_json = self._sentences_to_json(adjusted_sentences)

        # Build prompt
        prompt = GOOGLE_DOC_IMAGE_PLACER_PROMPT.format(
            google_doc_script_json=script_json,
            adjusted_sentences_json=sentences_json,
        )

        # Get LLM response
        try:
            response_text = self.llm_service.complete(prompt=prompt)

            # Parse JSON response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            response_data = json.loads(response_text)

            # Validate response structure
            if "placements" not in response_data:
                raise ValueError("Response missing 'placements' field")

            thoughts = response_data.get("thoughts", "")
            print(f"\n🤖 Agent thoughts: {thoughts}\n")

            # Convert to GoogleDocImagePlacement objects with full paths
            placements = []
            for placement_data in response_data["placements"]:
                # Get the image filename from the response
                image_filename = placement_data["filepath"]

                # Build full path to the image
                full_image_path = google_doc_images_folder / image_filename

                # Verify image exists
                if not full_image_path.exists():
                    print(f"   ⚠ Warning: Image not found: {full_image_path}")
                    continue

                # Get sentence indexes (ensure they are strings)
                sentence_indexes = [
                    str(idx) for idx in placement_data["sentence_indexes"]
                ]

                placement = GoogleDocImagePlacement(
                    filepath=str(full_image_path),
                    sentence_indexes=sentence_indexes,
                )
                placements.append(placement)

                sentence_range = (
                    f"{sentence_indexes[0]}-{sentence_indexes[-1]}"
                    if len(sentence_indexes) > 1
                    else sentence_indexes[0]
                )
                print(f"   ✓ Placed {image_filename}: sentences {sentence_range}")

            return GoogleDocImagePlacements(placements=placements)

        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text}"
            ) from e
        except KeyError as e:
            raise RuntimeError(
                f"LLM response missing required field: {e}\nResponse: {response_text}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to place images: {e}") from e

    def _script_to_json(self, google_doc_script: GoogleDocScript) -> str:
        """
        Convert Google Doc script to JSON format for the prompt.

        Args:
            google_doc_script: GoogleDocScript object

        Returns:
            JSON string representation
        """
        script_lines = []
        for line in google_doc_script.lines:
            line_dict = {
                "text": line.text,
                "image_filename": line.image_filename,
            }
            script_lines.append(line_dict)
        return json.dumps(script_lines, indent=2)

    def _sentences_to_json(self, adjusted_sentences: AdjustedSentences) -> str:
        """
        Convert adjusted sentences to JSON format for the prompt (index and text only).

        Args:
            adjusted_sentences: AdjustedSentences object

        Returns:
            JSON string representation
        """
        sentences_dict = {}
        for sentence in adjusted_sentences.sentences:
            sentences_dict[sentence.index] = sentence.text
        return json.dumps(sentences_dict, indent=2)
