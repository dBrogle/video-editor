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


GOOGLE_DOC_IMAGE_PLACER_PROMPT = """You are a video editing assistant placing images from a script onto a video timeline.

You have two pieces of information:
1. The Google Doc script: ordered text lines, each with its associated images (image_filenames) and any [bracket] instructions. The ordering of lines and images is the creator's intended visual structure — an image listed on a line is meant to illustrate THAT line's words.
2. The actual video sentences. Each sentence has an index, its transcribed text, its duration in seconds, and a word-level breakdown. For every word you get `s` and `e` — the word's start and end as a FRACTION of the sentence (0.0 = sentence start, 1.0 = sentence end). Use these to place each image precisely on the words it illustrates.

Google Doc Script (what was planned):
{google_doc_script_json}

Actual Video Sentences (indexed, with per-word fractions):
{adjusted_sentences_json}

The speaker improvises and reorders, and the cut merges/splits sentences, so ONE script line's content may land inside one sentence, or span two sentences, or share a sentence with neighboring lines. Match by meaning, then use the word fractions to set timing.

## How to set timing (this is the most important part)
- start_fraction / end_fraction are 0.0–1.0 positions WITHIN the matched sentence.
- Find the actual words in the sentence that correspond to the script line's content, then set start_fraction to roughly that first word's `s` and end_fraction to the last word's `e`. The image should be on screen DURING the words it depicts.
- If a sentence contains several script lines back-to-back, give each line's image(s) the fraction window of its own words — do not just split the sentence into equal thirds. Equal splits are wrong when the words aren't evenly spaced.
- A single image illustrating the whole sentence gets 0.0–1.0.
- Keep images on screen long enough to register (aim for at least ~0.8s of the sentence's duration); widen a too-narrow window rather than letting it flicker.

## Showing MULTIPLE images AT THE SAME TIME (concurrent groups)
Sometimes a beat calls for several images on screen together — e.g. a line lists multiple images that are one visual set (examples side by side, a caption + a reaction meme, a before/after), or a [bracket] instruction says to put one image "over"/"with"/"alongside" another.
- To show images together SIDE BY SIDE / STACKED (not overlapping), give them the EXACT SAME sentence_index, start_fraction, AND end_fraction. Placements that share all three are composited into one frame automatically.
- Set "layout" on those placements to control the arrangement:
    "tile"       = AUTO — the renderer picks vertical vs horizontal vs grid based on the
                   images' aspect ratios so they take up as much space as possible. Use this by default.
    "horizontal" = force a single row, side by side
    "vertical"   = force a single column, stacked top-to-bottom
  All members of a group should use the same layout. Prefer "tile" unless you specifically want to force one.
- To make one image POP UP "OVER" another for a moment (a backdrop image is up, then a reaction meme / second image appears over it briefly, then the backdrop returns), SPLIT the backdrop into two placements around the pop-up, so the pop-up is the only image during its window: backdrop A [0.30-0.46], pop-up B [0.46-0.60], backdrop A again [0.60-0.74]. The pop-up fully covers the backdrop for its window, then the backdrop comes back — which reads as the second image appearing over the first. (Do not rely on overlapping time windows for stacking; use this split-and-return pattern.)
- Prefer a concurrent group when the images form one combined beat. Prefer sequential windows when the images illustrate DIFFERENT successive words/phrases. Use the overlay technique when one image is explicitly meant to appear "over"/"on top of" another.
- Don't overcrowd: more than 4 images at once gets unreadable.

## Other rules
- A script line may have MULTIPLE images — place each, concurrently or sequentially per the guidance above.
- The same image filename may legitimately appear on more than one line; place it each place it appears.
- If a script image doesn't match any spoken content, omit it.
- [bracket] instructions describe assets (GIFs, B-roll, title cards) or placement notes. Match described assets to available files (e.g. "gif of the robot dog" -> "robot_dog.gif") and honor placement notes ("over the other one", "stack", "side by side").
- Available files in the images folder: {{available_files}}

Respond with ONLY valid JSON, starting with ```json:
{{
    "thoughts": "Brief: which script line maps to which sentence, which words each image lands on, and any concurrent groups.",
    "placements": [
        {{"sentence_index": "1", "filepath": "image1.png", "start_fraction": 0.0, "end_fraction": 0.45, "layout": "tile"}},
        {{"sentence_index": "1", "filepath": "image2.png", "start_fraction": 0.0, "end_fraction": 0.45, "layout": "tile"}},
        {{"sentence_index": "1", "filepath": "image3.png", "start_fraction": 0.55, "end_fraction": 1.0, "layout": "tile"}},
        {{"sentence_index": "2", "filepath": "image4.png", "start_fraction": 0.0, "end_fraction": 1.0, "layout": "tile"}}
    ]
}}
(In that example image1 and image2 show together for the first ~45% of sentence 1 — same fractions, so they're one concurrent group — then image3 shows for the rest.)

Important:
- placements is a LIST (not a dict). Each item: sentence_index (string), filepath (string), start_fraction (float), end_fraction (float), layout (string, default "tile").
- start_fraction < end_fraction, both in 0.0–1.0.
- filepath matches the image filename from the script (e.g. "image1.png").
- Sentences without images simply have no entries.

Your output MUST be valid json: start with ```json and end with a closing fence.
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
                layout = placement_data.get("layout", "tile") or "tile"
                if layout not in ("tile", "horizontal", "vertical"):
                    layout = "tile"

                if image_filename is None or sentence_idx is None:
                    continue

                full_image_path = google_doc_images_folder / image_filename
                if full_image_path.exists():
                    placement = GoogleDocImagePlacement(
                        filepath=str(full_image_path),
                        sentence_index=str(sentence_idx),
                        start_fraction=start_fraction,
                        end_fraction=end_fraction,
                        layout=layout,
                    )
                    placements.append(placement)
                    grp = f" +{layout}" if start_fraction != end_fraction else ""
                    print(
                        f"   ✓ Placed {image_filename}: sentence {sentence_idx} [{start_fraction:.2f}-{end_fraction:.2f}]{grp}"
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
        Convert adjusted sentences to JSON for the prompt: index, text, duration,
        and per-word timing expressed as fractions of the sentence (0.0-1.0).

        The word fractions let the LLM align an image's start_fraction/end_fraction
        to the specific words it illustrates, instead of guessing even splits.
        """
        sentences = []
        for sentence in adjusted_sentences.sentences:
            dur = sentence.adjusted_end - sentence.adjusted_start
            entry: dict = {
                "index": sentence.index,
                "text": sentence.text,
                "duration_s": round(dur, 2),
            }
            if dur > 0 and getattr(sentence, "words", None):
                words = []
                for w in sentence.words:
                    s = (w.start - sentence.adjusted_start) / dur
                    e = (w.end - sentence.adjusted_start) / dur
                    words.append({
                        "w": w.word,
                        "s": round(max(0.0, min(1.0, s)), 2),
                        "e": round(max(0.0, min(1.0, e)), 2),
                    })
                entry["words"] = words
            sentences.append(entry)
        return json.dumps(sentences, indent=2)
