"""
Abstract base class for Large Language Model services.
For future LLM-driven edit decisions and prompt-based features.
"""

import json
from abc import ABC, abstractmethod
from typing import Any

from src.util import prepare_transcript_for_prompt


EDITING_PROMPT_TEMPLATE = """You are a video editor analyzing a transcript to identify sentences that should be removed for a cleaner, more engaging final cut.

Review the following transcript sentences and identify which ones should be removed. Consider removing:
- Filler content, repetitions, or false starts
- Off-topic tangents
- Mistakes or unclear sections
- Anything that doesn't contribute to the core message
- If there's one or multiple repetitions, the last one should be kept. Since if the subject in the video repeats until the sentence is good.

Transcript sentences:
{sentences_json}

Respond with your analysis in JSON format:
{{
    "thoughts": "Your reasoning about what to remove and why. Make note of any repetitions, keeping the last one (since that will always be the take good enough to continue), and also make note of any sentences that seem like off-hand comments.",
    "sentences_to_remove": [list of sentence numbers to remove (1-indexed)]
}}"""


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
