"""
Sentence Selection Agent - Interactive agent for selecting which sentences to keep/remove.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.models import EditingResult, Transcript
from src.services.llm.base import LLMService
from src.services.llm.openrouter import OpenRouterLLMService


SENTENCE_SELECTION_AGENT_PROMPT = """You are a video editing assistant helping to select which sentences to keep or remove from a video based on user feedback.

Current editing state (which sentences are kept or removed):
{editing_result_json}

User feedback:
{user_feedback}

You have the following tools available:

1. keep_sentence - Mark a sentence to be kept in the final video
   Parameters:
   - sentence_index: The index of the sentence to keep (as a string, e.g., "1", "2", etc.)

2. remove_sentence - Mark a sentence to be removed from the final video
   Parameters:
   - sentence_index: The index of the sentence to remove (as a string)

3. approve - Approve the current sentence selection and move to timestamp adjustments
   No parameters needed.

Based on the user's feedback, respond with a JSON object containing:
{{
    "thoughts": "Your analysis of the user's feedback and what actions to take",
    "actions": [
        {{
            "tool": "tool_name",
            "parameters": {{
                "param1": "value1"
            }}
        }}
    ]
}}

If the user approves the sentence selection (e.g., "looks good", "perfect", "approve"), use the "approve" tool.
If the user wants adjustments, analyze their feedback and use the appropriate tools.

Examples:
- "Remove sentence 5" -> remove_sentence with sentence_index "5"
- "Keep sentence 3" -> keep_sentence with sentence_index "3"
- "Remove sentences 10 through 15" -> multiple remove_sentence actions
- "The middle section is too long" -> analyze and suggest sentence removals
- "Looks good" -> approve
"""


class SentenceSelectionAgent:
    """
    Agent for selecting which sentences to keep or remove based on user feedback.
    Uses an LLM to interpret feedback and execute sentence selection actions.
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize the Sentence Selection Agent.

        Args:
            llm_service: LLM service to use. If None, creates default OpenRouterLLMService.
        """
        self.llm_service = llm_service or OpenRouterLLMService(
            temperature=0.3,  # Lower temperature for more consistent editing decisions
            max_tokens=2000,
        )

    def process_feedback(
        self,
        editing_result: EditingResult,
        user_feedback: str,
    ) -> tuple[EditingResult, bool]:
        """
        Process user feedback and update sentence selection (keep/remove).

        Args:
            editing_result: Current editing result with sentence keep/remove decisions
            user_feedback: User's feedback text

        Returns:
            Tuple of (updated_editing_result, is_approved)
            - updated_editing_result: Modified editing result based on feedback
            - is_approved: True if user approved, False if more iterations needed

        Raises:
            RuntimeError: If LLM response cannot be parsed or actions fail
        """
        # Convert editing result to JSON for the prompt
        editing_result_json = self._editing_result_to_json(editing_result)

        # Build prompt
        prompt = SENTENCE_SELECTION_AGENT_PROMPT.format(
            editing_result_json=editing_result_json,
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
        updated_result = editing_result

        for action in actions:
            tool = action.get("tool")
            params = action.get("parameters", {})

            print(f"   Executing: {tool} with {params}")

            if tool == "approve":
                is_approved = True
                print("   ✓ Sentence selection approved!")
            elif tool == "keep_sentence":
                updated_result = self._keep_sentence(updated_result, params)
            elif tool == "remove_sentence":
                updated_result = self._remove_sentence(updated_result, params)
            else:
                print(f"   ⚠ Warning: Unknown tool '{tool}'")

        return updated_result, is_approved

    def _editing_result_to_json(self, editing_result: EditingResult) -> str:
        """
        Convert editing result to JSON format for the prompt.

        Args:
            editing_result: EditingResult object

        Returns:
            JSON string representation showing sentences and their keep/remove status
        """
        sentences_list = []
        for index, sentence_result in editing_result.sentence_results.items():
            sentences_list.append(
                {
                    "index": index,
                    "text": sentence_result.text,
                    "keep": sentence_result.keep,
                }
            )
        return json.dumps(sentences_list, indent=2)

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

    def _keep_sentence(
        self,
        editing_result: EditingResult,
        params: Dict[str, Any],
    ) -> EditingResult:
        """
        Mark a sentence to be kept in the final video.

        Args:
            editing_result: Current editing result
            params: Tool parameters (sentence_index)

        Returns:
            Updated EditingResult

        Raises:
            ValueError: If parameters are invalid
        """
        sentence_index = params.get("sentence_index")
        if not sentence_index:
            raise ValueError(f"Missing required parameter sentence_index: {params}")

        # Check if sentence exists
        if sentence_index not in editing_result.sentence_results:
            raise ValueError(
                f"Sentence with index {sentence_index} not found in editing result"
            )

        # Set keep to True
        editing_result.sentence_results[sentence_index].keep = True
        print(f"   ✓ Marked sentence {sentence_index} to be KEPT")

        return editing_result

    def _remove_sentence(
        self,
        editing_result: EditingResult,
        params: Dict[str, Any],
    ) -> EditingResult:
        """
        Mark a sentence to be removed from the final video.

        Args:
            editing_result: Current editing result
            params: Tool parameters (sentence_index)

        Returns:
            Updated EditingResult

        Raises:
            ValueError: If parameters are invalid
        """
        sentence_index = params.get("sentence_index")
        if not sentence_index:
            raise ValueError(f"Missing required parameter sentence_index: {params}")

        # Check if sentence exists
        if sentence_index not in editing_result.sentence_results:
            raise ValueError(
                f"Sentence with index {sentence_index} not found in editing result"
            )

        # Set keep to False
        editing_result.sentence_results[sentence_index].keep = False
        print(f"   ✓ Marked sentence {sentence_index} to be REMOVED")

        return editing_result
