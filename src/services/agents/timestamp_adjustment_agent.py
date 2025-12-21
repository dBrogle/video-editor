"""
Timestamp Adjustment Agent - Interactive agent for adjusting sentence timestamps based on user feedback.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.models import AdjustedSentences
from src.services.llm.base import LLMService
from src.services.llm.openrouter import OpenRouterLLMService


TIMESTAMP_ADJUSTMENT_AGENT_PROMPT = """You are a video editing assistant helping to adjust sentence timestamps based on user feedback.

Current state of the video (adjusted sentences with timestamps):
{adjusted_sentences_json}

Note: Each sentence includes word-level timestamps in the "words" array. You can use these precise timestamps 
to make fine-grained cuts at the word level when adjusting sentence boundaries.

User feedback:
{user_feedback}

You have the following tools available to adjust timestamps:

1. adjust_timestamp - Adjust the start or end timestamp of a sentence
   Parameters:
   - sentence_index: The index of the sentence (as a string, e.g., "1", "2", etc.)
   - field: Either "adjusted_start" or "adjusted_end"
   - new_value: The new timestamp value in seconds (float)
   
   Tip: Use the word-level timestamps to make precise cuts. For example, if you want to cut the first 
   two words from a sentence, look at the "words" array and set the adjusted_start to the start time 
   of the third word.

2. approve - Approve the current timestamps and finish editing
   No parameters needed.

Based on the user's feedback, respond with a JSON object containing:
{{
    "thoughts": "Your analysis of the user's feedback and what actions to take",
    "actions": [
        {{
            "tool": "tool_name",
            "parameters": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}
    ]
}}

If the user approves the video (e.g., "looks good", "perfect", "approve"), use the "approve" tool.
If the user wants adjustments, analyze their feedback and use the appropriate tools.

Examples:
- "Cut 2 seconds from the beginning" -> adjust_timestamp on sentence 1's adjusted_start
- "The pause between sentence 3 and 4 is too long" -> adjust timestamps to reduce gap
- "Start sentence 5 at word 3" -> adjust_timestamp using word-level timing
- "Looks good" -> approve

Note: Sentence selection (keep/remove) is handled in a separate stage. This agent only adjusts timestamps.
"""


class TimestampAdjustmentAgent:
    """
    Agent for adjusting sentence timestamps based on user feedback.
    Uses an LLM to interpret feedback and execute timestamp adjustment actions.
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize the Timestamp Adjustment Agent.

        Args:
            llm_service: LLM service to use. If None, creates default OpenRouterLLMService.
        """
        self.llm_service = llm_service or OpenRouterLLMService(
            temperature=0.3,  # Lower temperature for more consistent editing decisions
            max_tokens=2000,
        )

    def process_feedback(
        self,
        adjusted_sentences: AdjustedSentences,
        user_feedback: str,
    ) -> tuple[AdjustedSentences, bool]:
        """
        Process user feedback and adjust timestamps in adjusted sentences.

        Args:
            adjusted_sentences: Current state of adjusted sentences
            user_feedback: User's feedback text

        Returns:
            Tuple of (updated_adjusted_sentences, is_approved)
            - updated_adjusted_sentences: Modified sentences based on feedback
            - is_approved: True if user approved, False if more iterations needed

        Raises:
            RuntimeError: If LLM response cannot be parsed or actions fail
        """
        # Convert adjusted sentences to JSON for the prompt (include words for timestamp adjustment)
        sentences_json = self._sentences_to_json(adjusted_sentences, include_words=True)

        # Build prompt
        prompt = TIMESTAMP_ADJUSTMENT_AGENT_PROMPT.format(
            adjusted_sentences_json=sentences_json,
            user_feedback=user_feedback,
        )

        # Get LLM response
        response_text = self.llm_service.complete(prompt)

        # Parse response
        try:
            response_data = self._parse_response(response_text)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse agent response: {str(e)}\nResponse: {response_text}"
            ) from e

        # Execute actions
        thoughts = response_data.get("thoughts", "")
        actions = response_data.get("actions", [])

        print(f"\n🤖 Agent thoughts: {thoughts}\n")

        is_approved = False
        updated_sentences = adjusted_sentences

        for action in actions:
            tool = action.get("tool")
            params = action.get("parameters", {})

            print(f"   Executing: {tool} with {params}")

            if tool == "approve":
                is_approved = True
                print("   ✓ Timestamps approved!")
            elif tool == "adjust_timestamp":
                updated_sentences = self._adjust_timestamp(updated_sentences, params)
            else:
                print(f"   ⚠ Warning: Unknown tool '{tool}'")

        return updated_sentences, is_approved

    def _sentences_to_json(
        self, adjusted_sentences: AdjustedSentences, include_words: bool = True
    ) -> str:
        """
        Convert adjusted sentences to a compact, human-readable format for the prompt.

        Args:
            adjusted_sentences: AdjustedSentences object
            include_words: Whether to include word-level timestamps

        Returns:
            Formatted string representation
        """
        lines = []
        for sentence in adjusted_sentences.sentences:
            # Sentence header with index, text, and adjusted timestamps
            lines.append(
                f"Sentence {sentence.index}:\n"
                f'"{sentence.text}" [{sentence.adjusted_start:.2f} - {sentence.adjusted_end:.2f}]'
            )

            # Include word-level timestamps if available and requested
            if include_words and hasattr(sentence, "words") and sentence.words:
                for word in sentence.words:
                    lines.append(f"[{word.start:.2f} - {word.end:.2f}] {word.word}")

            # Add blank line between sentences for readability
            lines.append("")

        return "\n".join(lines)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response JSON.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If response cannot be parsed
        """
        # Extract JSON from response (handle markdown code blocks)
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response")

        json_text = response_text[start : end + 1]
        return json.loads(json_text)

    def _adjust_timestamp(
        self,
        adjusted_sentences: AdjustedSentences,
        params: Dict[str, Any],
    ) -> AdjustedSentences:
        """
        Adjust a timestamp for a sentence.

        Args:
            adjusted_sentences: Current sentences
            params: Tool parameters (sentence_index, field, new_value)

        Returns:
            Updated AdjustedSentences

        Raises:
            ValueError: If parameters are invalid
        """
        sentence_index = params.get("sentence_index")
        field = params.get("field")
        new_value = params.get("new_value")

        if not all([sentence_index, field, new_value is not None]):
            raise ValueError(
                f"Missing required parameters for adjust_timestamp: {params}"
            )

        # Find the sentence
        for sentence in adjusted_sentences.sentences:
            if sentence.index == sentence_index:
                # Update the field
                if field not in [
                    "original_start",
                    "original_end",
                    "adjusted_start",
                    "adjusted_end",
                ]:
                    raise ValueError(f"Invalid field: {field}")

                # Validate new_value is a number
                if new_value is None:
                    raise ValueError("new_value cannot be None")
                try:
                    new_value_float = float(new_value)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid timestamp value: {new_value}") from e

                setattr(sentence, field, new_value_float)
                print(
                    f"   ✓ Adjusted sentence {sentence_index} {field} to {new_value_float}s"
                )
                return adjusted_sentences

        raise ValueError(f"Sentence with index {sentence_index} not found")
