"""
OpenRouter LLM service implementation.
Provides access to multiple LLM providers through OpenRouter API.
"""

import os
import json
import requests  # type: ignore
from typing import Any, Dict, Optional

from cc_wsp.src.services.llm.base import LLMService, EDITING_PROMPT_TEMPLATE, STREAM_EDITING_PROMPT_TEMPLATE, STREAM_HIGHLIGHTS_PROMPT_TEMPLATE
from cc_wsp.src.models import Transcript, EditingDecision, EditingResult, SentenceResult
from cc_wsp.src.constants import (
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
        model: OpenRouterModel | str = OpenRouterModel.CLAUDE_SONNET_45,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 5000,
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

    def get_edits(self, transcript: Transcript, script_text: str | None = None) -> EditingDecision:
        """
        Get editing decisions for a transcript.

        Args:
            transcript: Transcript object
            script_text: Optional script outline from the Google Doc

        Returns:
            EditingDecision object with thoughts and sentences to remove

        Raises:
            RuntimeError: If API call or parsing fails
        """
        # Convert transcript to JSON format
        sentences_json = self.transcript_to_sentences_json(transcript)

        # Build prompt
        prompt = EDITING_PROMPT_TEMPLATE.format(sentences_json=sentences_json)

        if script_text:
            prompt += f"""

===========================================
SCRIPT OUTLINE (from the creator's notes)
===========================================
Below is the rough script/outline the speaker was working from. They may improvise, go off-script, or rephrase things, but this is the general intended flow. Use this to understand the intended structure and identify which takes best match each script point.

{script_text}
"""

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

    def get_stream_edits(
        self,
        transcript: Transcript,
        chunk_size: int = 100,
        overlap: int = 5,
    ) -> EditingResult:
        """
        Get editing decisions for a long stream transcript by processing in chunks.

        Splits the transcript into overlapping chunks, sends each to the LLM,
        and merges the removal decisions.

        Args:
            transcript: Full transcript object
            chunk_size: Number of sentences per chunk
            overlap: Number of overlapping sentences between chunks

        Returns:
            EditingResult with keep/remove decisions for all sentences
        """
        sentences = transcript.sentences
        total = len(sentences)
        all_sentences_to_remove: set[int] = set()

        print(f"Processing {total} sentences in chunks of {chunk_size}...")

        chunk_start = 0
        chunk_num = 0

        while chunk_start < total:
            chunk_num += 1
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_sentences = sentences[chunk_start:chunk_end]

            # Build the sentences JSON with original 1-based indices
            sentences_dict = {}
            for i, sentence in enumerate(chunk_sentences):
                original_index = chunk_start + i + 1  # 1-based
                sentences_dict[str(original_index)] = str(sentence)

            sentences_json = json.dumps(sentences_dict, indent=2)

            # Build context about where this chunk sits
            start_time = chunk_sentences[0].start
            end_time = chunk_sentences[-1].end
            chunk_context = (
                f"This is chunk {chunk_num} of the stream "
                f"(sentences {chunk_start + 1}-{chunk_end}, "
                f"timestamps {start_time:.1f}s - {end_time:.1f}s)."
            )

            prompt = STREAM_EDITING_PROMPT_TEMPLATE.format(
                chunk_context=chunk_context,
                sentences_json=sentences_json,
            )

            print(f"  Chunk {chunk_num}: sentences {chunk_start + 1}-{chunk_end} ({start_time:.0f}s - {end_time:.0f}s)")

            response_text = self.complete(prompt, max_tokens=8000)

            try:
                start = response_text.find("{")
                end = response_text.rfind("}")
                response_json = json.loads(response_text[start : end + 1])
                chunk_removals = response_json.get("sentences_to_remove", [])
                thoughts = response_json.get("thoughts", "")
                print(f"    LLM thoughts: {thoughts[:120]}...")
                print(f"    Removing {len(chunk_removals)} sentences from this chunk")
                all_sentences_to_remove.update(chunk_removals)
            except Exception as e:
                print(f"    Warning: Failed to parse chunk {chunk_num} response: {e}")
                print(f"    Skipping this chunk (keeping all sentences)")

            # Advance by chunk_size minus overlap
            chunk_start += chunk_size - overlap

        # Build EditingResult from merged removals
        sentence_results = {}
        for i, sentence in enumerate(sentences, 1):
            sentence_results[str(i)] = SentenceResult(
                text=sentence.sentence,
                keep=(i not in all_sentences_to_remove),
            )

        kept = sum(1 for sr in sentence_results.values() if sr.keep)
        removed = len(all_sentences_to_remove)
        print(f"\nStream edit complete: {kept} kept, {removed} removed out of {total} total")

        return EditingResult(sentence_results=sentence_results)

    def get_stream_highlights(
        self,
        transcript: Transcript,
        kept_indices: set[int] | None = None,
        chunk_size: int = 100,
        overlap: int = 5,
        target_minutes: float = 25,
        current_minutes: float | None = None,
    ) -> EditingResult:
        """
        Get highlight editing decisions for a stream transcript by processing in chunks.
        Can be called iteratively — pass kept_indices to only process remaining sentences.

        Returns:
            EditingResult with keep/remove decisions for all sentences
        """
        sentences = transcript.sentences
        total = len(sentences)

        # Build list of (original_1based_index, sentence) for sentences still in play
        if kept_indices is None:
            active = [(i + 1, s) for i, s in enumerate(sentences)]
        else:
            active = [(i + 1, s) for i, s in enumerate(sentences) if (i + 1) in kept_indices]

        total_active = len(active)
        total_duration = sum(s.end - s.start for _, s in active)
        target_seconds = target_minutes * 60

        if current_minutes is not None:
            cut_percent = max(50, int((1 - target_seconds / (current_minutes * 60)) * 100))
        else:
            cut_percent = max(50, int((1 - target_seconds / total_duration) * 100))

        target_context = (
            f"Current duration: {total_duration / 60:.1f} minutes. "
            f"Target: under {target_minutes:.0f} minutes. "
            f"You need to cut roughly {cut_percent}% of this content."
        )

        all_sentences_to_remove: set[int] = set()

        print(f"Processing {total_active} sentences in chunks of {chunk_size} (target: {target_minutes:.0f} min)...")

        chunk_start = 0
        chunk_num = 0

        while chunk_start < total_active:
            chunk_num += 1
            chunk_end = min(chunk_start + chunk_size, total_active)
            chunk_items = active[chunk_start:chunk_end]

            # Build sentences JSON with original indices
            sentences_dict = {}
            for orig_idx, sentence in chunk_items:
                sentences_dict[str(orig_idx)] = str(sentence)

            sentences_json = json.dumps(sentences_dict, indent=2)

            start_time = chunk_items[0][1].start
            end_time = chunk_items[-1][1].end
            chunk_context = (
                f"This is chunk {chunk_num} of the stream highlights edit "
                f"(sentences from {start_time:.1f}s - {end_time:.1f}s, "
                f"covering {(end_time - start_time) / 60:.1f} minutes)."
            )

            prompt = STREAM_HIGHLIGHTS_PROMPT_TEMPLATE.format(
                chunk_context=chunk_context,
                target_context=target_context,
                cut_percent=cut_percent,
                sentences_json=sentences_json,
            )

            print(f"  Chunk {chunk_num}: {start_time:.0f}s - {end_time:.0f}s ({len(chunk_items)} sentences)")

            # Retry up to 3 times on timeout
            response_text = None
            for attempt in range(3):
                try:
                    response_text = self.complete(prompt, max_tokens=8000)
                    break
                except RuntimeError as e:
                    if "timed out" in str(e).lower() and attempt < 2:
                        print(f"    Timeout, retrying ({attempt + 2}/3)...")
                        continue
                    raise

            if response_text is None:
                print(f"    Warning: All retries failed for chunk {chunk_num}")
                chunk_start += chunk_size - overlap
                continue

            try:
                start = response_text.find("{")
                end = response_text.rfind("}")
                response_json = json.loads(response_text[start : end + 1])
                chunk_removals = response_json.get("sentences_to_remove", [])
                thoughts = response_json.get("thoughts", "")
                print(f"    Removing {len(chunk_removals)} / {len(chunk_items)} sentences")
                print(f"    LLM: {thoughts[:150]}...")
                all_sentences_to_remove.update(chunk_removals)
            except Exception as e:
                print(f"    Warning: Failed to parse chunk {chunk_num}: {e}")

            chunk_start += chunk_size - overlap

        # Build EditingResult — merge with existing kept_indices
        sentence_results = {}
        for i, sentence in enumerate(sentences, 1):
            if kept_indices is not None and i not in kept_indices:
                # Already removed in a previous pass
                keep = False
            elif i in all_sentences_to_remove:
                keep = False
            else:
                keep = True
            sentence_results[str(i)] = SentenceResult(
                text=sentence.sentence,
                keep=keep,
            )

        kept = sum(1 for sr in sentence_results.values() if sr.keep)
        removed = total - kept
        kept_duration = sum(
            sentences[i].end - sentences[i].start
            for i in range(total)
            if sentence_results[str(i + 1)].keep
        )
        print(f"\nPass complete: {kept} kept, {removed} removed. Estimated duration: {kept_duration / 60:.1f} min")

        return EditingResult(sentence_results=sentence_results)

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
            self.api_url, headers=headers, json=payload, timeout=300
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
