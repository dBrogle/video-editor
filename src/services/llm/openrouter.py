"""
OpenRouter LLM service implementation.
Provides access to multiple LLM providers through OpenRouter API.
"""

import os
import json
import requests  # type: ignore
from typing import Any, Dict, Optional

from src.services.llm.base import LLMService, EDITING_PROMPT_TEMPLATE
from src.models import Transcript, EditingDecision
from src.constants import (
    ENV_OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    OpenRouterModel,
)


class OpenRouterLLMService(LLMService):
    """
    OpenRouter implementation of LLM service.
    Supports various models through a unified interface.
    """

    def __init__(
        self,
        model: OpenRouterModel | str = OpenRouterModel.GEMINI_25_FLASH,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """
        Initialize OpenRouter LLM service.

        Args:
            model: Model to use (OpenRouterModel enum or string identifier)
            api_key: OpenRouter API key. If None, reads from environment.
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens in response

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv(ENV_OPENROUTER_API_KEY)
        if not self.api_key:
            raise ValueError(
                f"OpenRouter API key not found. "
                f"Provide via constructor or {ENV_OPENROUTER_API_KEY} env var."
            )

        # Convert enum to string value if needed
        self.model = model.value if isinstance(model, OpenRouterModel) else model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = OPENROUTER_API_URL

    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a completion using OpenRouter API.

        Args:
            prompt: Input prompt text
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional OpenRouter parameters

        Returns:
            Generated text response

        Raises:
            RuntimeError: If API call fails
        """
        try:
            response = self._call_api(
                prompt=prompt,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs,
            )

            # Extract text from response
            output = self._extract_text(response)

            # Save debug log
            self._save_debug_log(prompt, output)

            return output

        except requests.RequestException as e:
            raise RuntimeError(f"OpenRouter API request failed: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"OpenRouter completion failed: {str(e)}") from e

    def get_edits(self, transcript: Transcript) -> EditingDecision:
        """
        Get editing decisions for a transcript.

        Args:
            transcript: Transcript object

        Returns:
            EditingDecision object with thoughts and sentences to remove

        Raises:
            RuntimeError: If API call or parsing fails
        """
        # Convert transcript to JSON format
        sentences_json = self.transcript_to_sentences_json(transcript)

        # Build prompt
        prompt = EDITING_PROMPT_TEMPLATE.format(sentences_json=sentences_json)

        # Call LLM
        response_text = self.complete(prompt)

        # Parse response
        try:
            # Get the first "{" and the last "}"
            start = response_text.find("{")
            end = response_text.rfind("}")
            response_text = response_text[start : end + 1]
            response_json = json.loads(response_text)
            return EditingDecision(**response_json)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse LLM response: {str(e)} on text: {response_text}"
            ) from e

    def _call_api(
        self, prompt: str, temperature: float, max_tokens: int, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Call OpenRouter API with the given parameters.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            API response as dictionary

        Raises:
            requests.RequestException: If API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        response = requests.post(
            self.api_url, headers=headers, json=payload, timeout=120
        )

        response.raise_for_status()
        return response.json()

    def _extract_text(self, response: Dict[str, Any]) -> str:
        """
        Extract text from OpenRouter API response.

        Args:
            response: API response dictionary

        Returns:
            Generated text

        Raises:
            ValueError: If response format is unexpected
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")

            message = choices[0].get("message", {})
            content = message.get("content", "")

            return content

        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected OpenRouter response format: {response}"
            ) from e
