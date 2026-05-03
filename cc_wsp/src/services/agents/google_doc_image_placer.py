"""
Google Doc Image Placer Agent - Places images from Google Doc script into video timeline.
"""

import json
from pathlib import Path
from typing import Optional

from cc_wsp.src.models import (
    AdjustedSentences,
    GoogleDocScript,
    GoogleDocImagePlacement,
    GoogleDocImagePlacements,
)
from cc_wsp.src.services.llm.base import LLMService
from cc_wsp.src.services.llm.openrouter import OpenRouterLLMService


GOOGLE_DOC_IMAGE_PLACER_PROMPT = """You are a video editing assistant helping to place images from a script onto a video timeline.

You have two pieces of information:
1. The Google Doc script with text lines and associated images (each line can have multiple images via image_filenames list)
2. The actual video sentences with their indexes

Your task is to match the script's image placements to the actual video sentences and determine which images should appear and when.

Google Doc Script (what was planned):
{google_doc_script_json}

Actual Video Sentences (indexed):
{adjusted_sentences_json}

Instructions:
- Match each image from the Google Doc script to the corresponding sentences in the actual video
- The speaker may deviate slightly from the script, so use semantic matching (not exact text matching)
- A script line may have MULTIPLE images (image_filenames list). Place each image appropriately.
- You can place MULTIPLE images within a single sentence using fractional timing
- start_fraction and end_fraction are values from 0.0 to 1.0, representing the portion of the sentence duration
  - For example: start_fraction=0.0, end_fraction=0.5 means the image shows for the first half of the sentence
  - start_fraction=0.5, end_fraction=1.0 means the image shows for the second half
  - start_fraction=0.0, end_fraction=1.0 means the image shows for the entire sentence
- Images CAN overlap in time (e.g., two images showing simultaneously on the same sentence with overlapping fractions).
  The rendering system supports multiple overlay tracks, so overlapping images will both be visible.
- Consider the flow and pacing - avoid too many rapid image changes
- If an image in the script doesn't match any video content, don't include it

Respond with a JSON object in this exact format:
{{
    "thoughts": "Your analysis of how the script maps to the video and your placement decisions",
    "placements": [
        {{
            "sentence_index": "1",
            "filepath": "image1.png",
            "start_fraction": 0.0,
            "end_fraction": 1.0
        }},
        {{
            "sentence_index": "2",
            "filepath": "image1.png",
            "start_fraction": 0.0,
            "end_fraction": 0.5
        }},
        {{
            "sentence_index": "2",
            "filepath": "image2.png",
            "start_fraction": 0.5,
            "end_fraction": 1.0
        }},
        {{
            "sentence_index": "4",
            "filepath": "image2.png",
            "start_fraction": 0.0,
            "end_fraction": 1.0
        }}
    ]
}}

Important:
- placements is a LIST of placement objects (not a dictionary)
- Each placement has: sentence_index (string), filepath (string), start_fraction (float), end_fraction (float)
- Multiple placements can target the same sentence_index with different or overlapping fractional ranges
- Sentences without images simply have no placement entries for them
- start_fraction must be less than end_fraction
- Both fractions must be between 0.0 and 1.0
- The filepath should match the image filename from the Google Doc script (e.g., "image1.png", "image2.png")
- Script lines may have "instructions" (text in [square brackets] from the script). These describe assets like GIFs, B-roll, or title cards.
  Match instruction descriptions to available files in the images folder (e.g., "gif of the robot dog" -> "robot_dog.gif").
- Available files in the images folder: {{available_files}}

Remember, your output MUST be valid json, so your output should start with ```json and end with a closing bracket.
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

        # List all available image/gif files in the folder
        available_files = sorted(
            f.name for f in google_doc_images_folder.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
        )
        available_files_str = ", ".join(available_files)

        # Build prompt
        prompt = GOOGLE_DOC_IMAGE_PLACER_PROMPT.format(
            google_doc_script_json=script_json,
            adjusted_sentences_json=sentences_json,
            available_files=available_files_str,
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

            placements_list = response_data["placements"]
            placements = []

            for placement_data in placements_list:
                image_filename = placement_data.get("filepath")
                sentence_idx = placement_data.get("sentence_index")
                start_fraction = placement_data.get("start_fraction", 0.0)
                end_fraction = placement_data.get("end_fraction", 1.0)

                if image_filename is None or sentence_idx is None:
                    continue

                full_image_path = google_doc_images_folder / image_filename
                if full_image_path.exists():
                    placement = GoogleDocImagePlacement(
                        filepath=str(full_image_path),
                        sentence_index=str(sentence_idx),
                        start_fraction=start_fraction,
                        end_fraction=end_fraction,
                    )
                    placements.append(placement)
                    print(
                        f"   ✓ Placed {image_filename}: sentence {sentence_idx} [{start_fraction:.1f}-{end_fraction:.1f}]"
                    )
                else:
                    print(f"   ⚠ Warning: Image not found: {full_image_path}")

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
                "image_filenames": line.image_filenames,
            }
            if line.instructions:
                line_dict["instructions"] = line.instructions
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
