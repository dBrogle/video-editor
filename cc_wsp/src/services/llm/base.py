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

from cc_wsp.src.util import prepare_transcript_for_prompt


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


STREAM_EDITING_PROMPT_TEMPLATE = """You are a video editor analyzing a transcript chunk from a long livestream (~2 hours). Your job is to identify sentences that should be REMOVED to create a tighter, more engaging cut.

===========================================
CONTEXT
===========================================
This is a LIVESTREAM, not a scripted video. The speaker is streaming live, so expect:
- Natural conversation flow with tangents
- Interactions with chat/audience
- Dead air, awkward pauses, or "be right back" moments
- Off-topic rambling that doesn't contribute to the main content
- Technical difficulties, bathroom breaks, drink breaks
- Repetitive explanations or going in circles

===========================================
WHAT TO REMOVE
===========================================
Mark sentences for removal if they are:
1. **Dead air / filler**: Long pauses, "um", "uh", silence, or "let me think" with no payoff
2. **Off-topic tangents**: Unrelated stories or rambling that doesn't serve the stream's purpose
3. **Chat interaction fluff**: Reading chat messages that add nothing ("oh someone said hi", "thanks for the sub")
4. **Technical issues**: "Hold on my mic is broken", "let me restart OBS", etc.
5. **Breaks**: "I'll be right back", bathroom/drink breaks, AFK moments
6. **Redundant repetition**: Saying the same thing multiple times without adding value
7. **False starts and stumbles**: Incomplete thoughts that get restarted

===========================================
WHAT TO KEEP
===========================================
Keep sentences that are:
1. **Core content**: The main topic/activity being streamed
2. **Good stories or anecdotes**: Entertaining tangents that viewers would enjoy
3. **Meaningful chat interaction**: Answering interesting questions, funny moments
4. **Key transitions**: "Alright let's move on to..." or natural topic changes
5. **Highlights**: Funny moments, reactions, exciting gameplay, good insights

===========================================
IMPORTANT NOTES
===========================================
- When in doubt, KEEP the sentence. It's better to keep slightly too much than to cut important content.
- This is one chunk of a longer stream. Don't worry about overall narrative arc.
- Preserve the natural flow — don't create jarring jumps by removing single sentences from the middle of a thought.
- If a group of sentences forms a coherent thought, either keep ALL of them or remove ALL of them.

{chunk_context}

===========================================
TRANSCRIPT SENTENCES
===========================================
Format: "sentence_number": "[start_timestamp-end_timestamp]-sentence_text"

{sentences_json}

===========================================
REQUIRED JSON RESPONSE FORMAT
===========================================
{{
    "thoughts": "Brief reasoning about what you're removing and why. Group your reasoning by type (dead air, tangents, etc.)",
    "sentences_to_remove": [list of sentence numbers to remove (using the numbers shown above)]
}}

REMEMBER:
- This MUST be valid JSON or the system will break
- Use the sentence numbers exactly as shown in the transcript above
- When in doubt, KEEP the content
"""


STREAM_HIGHLIGHTS_PROMPT_TEMPLATE = """You are a video editor creating a highlights reel from a long coding/tech livestream. Your job is to identify sentences that should be REMOVED. The goal is to cut the stream down to a tight, engaging highlights video.

===========================================
CONTEXT
===========================================
{chunk_context}

This is a LIVESTREAM recording. The speaker is coding/working live. We want to extract the BEST moments — primarily technical highlights, plus 1-2 funny or authentic moments per chunk.

===========================================
TARGET
===========================================
{target_context}

===========================================
WHAT TO REMOVE (be aggressive!)
===========================================
1. **Dead air / silence / thinking**: "um", "uh", "let me think", long pauses, typing in silence
2. **Repetitive debugging**: Going back and forth on the same error without progress
3. **Reading docs/code silently**: Just reading without explaining
4. **Mundane setup**: Installing packages, configuring environment, waiting for builds
5. **Chat fluff**: "Thanks for the sub", "welcome", reading irrelevant chat messages
6. **Breaks / AFK**: "Be right back", bathroom breaks, getting water
7. **Technical difficulties**: "My mic is broken", "OBS crashed", etc.
8. **Redundant explanations**: Saying the same thing multiple times
9. **Low-energy filler**: Rambling without substance, "so yeah", "anyway"
10. **Routine typing narration**: "Let me just type this out", "okay so I'll add this here" (unless explaining something interesting)

===========================================
WHAT TO KEEP
===========================================
1. **Technical insights**: Explaining concepts, architecture decisions, "aha" moments
2. **Problem-solving breakthroughs**: When something clicks or gets fixed
3. **Interesting code explanations**: Walking through how something works
4. **Funny/authentic moments**: Genuine reactions, good jokes, entertaining failures (keep 1-2 per chunk)
5. **Key results**: Demos, things working, showing off results
6. **Good transitions**: "Alright, now let's tackle..." (brief ones only)

===========================================
IMPORTANT RULES
===========================================
- Be AGGRESSIVE with cuts. This stream needs to lose {cut_percent}%+ of its content.
- When in doubt, REMOVE. We'd rather have a tight 20-minute video than a loose 35-minute one.
- Keep coherent groups: if removing a sentence creates a jarring jump, remove the whole group.
- Preserve enough context that a viewer can follow the technical narrative.

===========================================
TRANSCRIPT SENTENCES
===========================================
Format: "sentence_number": "[start_timestamp-end_timestamp]-sentence_text"

{sentences_json}

===========================================
REQUIRED JSON RESPONSE FORMAT
===========================================
{{
    "thoughts": "Brief reasoning: what technical highlights are you keeping? What are you cutting and why?",
    "sentences_to_remove": [list of sentence numbers to remove (using the numbers shown above)]
}}

REMEMBER:
- This MUST be valid JSON
- Use the sentence numbers exactly as shown above
- Be aggressive — cut everything that isn't a highlight
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
