"""
Abstract base class for Large Language Model services.
For future LLM-driven edit decisions and prompt-based features.
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from src.util import prepare_transcript_for_prompt


EDITING_PROMPT_TEMPLATE = """You are a video editor analyzing a transcript to identify sentences that should be removed for a cleaner, more engaging final cut.

===========================================
CRITICAL RULE - RETAKES (READ THIS FIRST!)
===========================================
When the speaker says similar content multiple times (retakes), you MUST:
- ALWAYS KEEP THE LAST/LATEST VERSION (the one with the later timestamp)
- ALWAYS REMOVE THE EARLIER VERSIONS (the ones with earlier timestamps)

Why? Because if the speaker repeats themselves, they keep going until they get it right. The last version is ALWAYS the one they were satisfied with. NEVER remove the final take and keep an earlier attempt.

EXAMPLE OF RETAKES:
Sentence 5: [41.38-46.14] "But if it's learning, but if it's learning patterns from data, then that's ML."
Sentence 12: [72.66-78.26] "Like, literally but if it's learning patterns from data, then that's machine learning."
→ These are retakes! Sentence 12 is LATER and cleaner. KEEP sentence 12, REMOVE sentence 5.

Another example:
Sentence 7: [47.06-52.40] "And this can be really simple, just like predicting housing prices on some Y equals MX plus B type energy."
Sentence 13: [79.32-83.88] "And it can be really simple, just like predicting housing prices on some Y equals MX plus B type."
→ Sentence 13 is the LATER, cleaner version. KEEP sentence 13, REMOVE sentence 7.

===========================================
TWO-STEP ANALYSIS PROCESS
===========================================

STEP 1: IDENTIFY RETAKES FIRST
Look through the transcript and find groups of sentences that say similar things. Compare their timestamps. For each retake group:
- Identify which sentence comes LAST (highest timestamp)
- Mark ALL EARLIER versions for removal
- Mark the LAST version to KEEP

STEP 2: IDENTIFY OTHER CONTENT TO REMOVE
After handling retakes, identify other sentences to remove:
- Filler content or false starts (that aren't retakes)
- Off-topic tangents or off-hand comments (e.g., "Don't worry, I'll edit you guys out")
- Mistakes or unclear sections
- Incomplete thoughts
- Anything that doesn't contribute to the core message
- Keep outros, the subject almost always ends the videos with "cheers"

===========================================
TRANSCRIPT SENTENCES
===========================================
Format: "sentence_number": "[start_timestamp-end_timestamp]-sentence_text"

{sentences_json}

===========================================
REQUIRED JSON RESPONSE FORMAT
===========================================
{{
    "thoughts": "Your step-by-step reasoning. First, identify any retake groups you found and explain which is the LAST/LATEST version to keep. Then explain other content you're removing and why. BE EXPLICIT about timestamps when discussing retakes.",
    "sentences_to_remove": [list of sentence numbers to remove (1-indexed)]
}}

REMEMBER: 
- This MUST be valid JSON or the system will break
- When in doubt about retakes: KEEP THE LATER TIMESTAMP, REMOVE THE EARLIER TIMESTAMP
- NEVER remove the final/latest take and keep an earlier attempt
"""


class LLMService(ABC):
    """
    Abstract base class for LLM interaction.
    Implementations handle provider-specific API details.
    """

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: Input prompt text
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response

        Raises:
            RuntimeError: If API call fails
        """
        raise NotImplementedError

    def transcript_to_sentences_json(self, transcript: "Transcript") -> str:
        """
        Convert transcript to JSON format for LLM prompts.

        Args:
            transcript: Transcript object

        Returns:
            JSON string with format {"1": "[start-end]-sentence", "2": ...}
        """
        sentences = prepare_transcript_for_prompt(transcript)
        sentences_dict = {
            str(i): str(sentence) for i, sentence in enumerate(sentences, 1)
        }
        return json.dumps(sentences_dict, indent=2)

    def _save_debug_log(self, prompt: str, output: str) -> None:
        """
        Save prompt and output to a debug file for inspection.

        Args:
            prompt: The input prompt sent to the LLM
            output: The output response from the LLM
        """
        # Create debug directory if it doesn't exist
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)

        # Generate timestamp string for filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"prompt_{timestamp_str}.txt"
        filepath = debug_dir / filename

        # Write prompt and output to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("PROMPT\n")
            f.write("=" * 80 + "\n\n")
            f.write(prompt)
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("OUTPUT\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n")
