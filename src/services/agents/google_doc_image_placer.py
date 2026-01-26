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

Your task is to match the script's image placements to the actual video sentences and determine which image (if any) should appear for each sentence.

Google Doc Script (what was planned):
{google_doc_script_json}

Actual Video Sentences (indexed):
{adjusted_sentences_json}

Instructions:
- Match each image from the Google Doc script to the corresponding sentences in the actual video
- The speaker may deviate slightly from the script, so use semantic matching (not exact text matching)
- For each sentence index, determine which image (if any) should be shown
- Each sentence can have at most ONE image
- If multiple images could fit a sentence, choose the most relevant one
- If no image fits a sentence, set filepath to null
- Consider the flow and avoid too many rapid image changes
- If an image in the script doesn't match any video content, don't place it anywhere

Respond with a JSON object in this exact format:
{{
    "thoughts": "Your analysis of how the script maps to the video and your placement decisions",
    "placements": {{
        "1": {{
            "filepath": "image1.png"
        }},
        "2": {{
            "filepath": "image1.png"
        }},
        "3": {{
            "filepath": null
        }},
        "4": {{
            "filepath": "image2.png"
        }}
    }}
}}

Important:
- The placements object MUST be keyed by sentence index (as strings: "1", "2", "3", etc.)
- Each value should be an object with a "filepath" field
- The filepath should be either a string (image filename like "image1.png") or null (no image)
- You MUST include an entry for EVERY sentence index that exists in the video
- The filepath should match the image filename from the Google Doc script (e.g., "image1.png", "image2.png")
- Each sentence can only have ONE image. If multiple images could fit, choose the most relevant one.
- The same image can appear on multiple consecutive sentences if appropriate

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

            # Parse the new sentence-index-keyed format
            # placements is now a dict like: {"1": {"filepath": "image1.png"}, "2": {"filepath": null}, ...}
            placements_dict = response_data["placements"]

            # Group consecutive sentences with the same image into single placements
            # This converts the sentence-keyed format back to the list-based format expected by the pipeline
            placements = []
            current_image = None
            current_sentence_indexes = []

            # Get all sentence indexes in order
            all_sentence_indexes = sorted(placements_dict.keys(), key=int)

            for sentence_idx in all_sentence_indexes:
                placement_info = placements_dict[sentence_idx]
                image_filename = placement_info.get("filepath")

                # If this sentence has an image
                if image_filename is not None:
                    # If it's a different image than we're currently tracking
                    if image_filename != current_image:
                        # Save the previous placement if it exists
                        if current_image is not None and current_sentence_indexes:
                            full_image_path = google_doc_images_folder / current_image
                            if full_image_path.exists():
                                placement = GoogleDocImagePlacement(
                                    filepath=str(full_image_path),
                                    sentence_indexes=current_sentence_indexes,
                                )
                                placements.append(placement)

                                sentence_range = (
                                    f"{current_sentence_indexes[0]}-{current_sentence_indexes[-1]}"
                                    if len(current_sentence_indexes) > 1
                                    else current_sentence_indexes[0]
                                )
                                print(
                                    f"   ✓ Placed {current_image}: sentences {sentence_range}"
                                )
                            else:
                                print(
                                    f"   ⚠ Warning: Image not found: {full_image_path}"
                                )

                        # Start tracking the new image
                        current_image = image_filename
                        current_sentence_indexes = [sentence_idx]
                    else:
                        # Same image, add to current group
                        current_sentence_indexes.append(sentence_idx)
                else:
                    # No image for this sentence, save any current placement
                    if current_image is not None and current_sentence_indexes:
                        full_image_path = google_doc_images_folder / current_image
                        if full_image_path.exists():
                            placement = GoogleDocImagePlacement(
                                filepath=str(full_image_path),
                                sentence_indexes=current_sentence_indexes,
                            )
                            placements.append(placement)

                            sentence_range = (
                                f"{current_sentence_indexes[0]}-{current_sentence_indexes[-1]}"
                                if len(current_sentence_indexes) > 1
                                else current_sentence_indexes[0]
                            )
                            print(
                                f"   ✓ Placed {current_image}: sentences {sentence_range}"
                            )
                        else:
                            print(f"   ⚠ Warning: Image not found: {full_image_path}")

                        current_image = None
                        current_sentence_indexes = []

            # Don't forget the last placement if we were tracking one
            if current_image is not None and current_sentence_indexes:
                full_image_path = google_doc_images_folder / current_image
                if full_image_path.exists():
                    placement = GoogleDocImagePlacement(
                        filepath=str(full_image_path),
                        sentence_indexes=current_sentence_indexes,
                    )
                    placements.append(placement)

                    sentence_range = (
                        f"{current_sentence_indexes[0]}-{current_sentence_indexes[-1]}"
                        if len(current_sentence_indexes) > 1
                        else current_sentence_indexes[0]
                    )
                    print(f"   ✓ Placed {current_image}: sentences {sentence_range}")
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
