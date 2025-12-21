"""
Image Planning Agent - Generates image descriptions and prompts based on video transcript.
"""

import json
from typing import Optional

from src.models import ImageDescription, AdjustedSentences
from src.services.llm.base import LLMService
from src.services.llm.openrouter import OpenRouterLLMService


FIRST_PASS_PROMPT = """You are a creative visual assistant helping to enhance a video with AI-generated images.

General instruction from the user:
{general_instruction}

Video transcript (sentence ID -> text):
{sentences_json}

Your task is to analyze the transcript and suggest images that would enhance the video. For each image:
1. Identify which sentences it should appear over (by sentence ID)
2. Provide a brief description of what the image should show
3. Create a detailed, optimized prompt for an AI image generator

Guidelines:
- Images should be relevant to what's being said in those sentences
- Images should enhance understanding or add visual interest
- Don't suggest too many images - quality over quantity
- Each image should typically span 1-3 sentences
- Avoid overlapping images (each sentence should have at most one image)

Respond with a JSON object in this exact format:
{{
    "thoughts": "Your analysis of the transcript and image strategy",
    "images": [
        {{
            "description": "Brief human-readable description",
            "sentence_ids": ["1", "2"],
            "detailed_prompt": "Detailed prompt for image generator, including style, composition, lighting, mood, etc."
        }}
    ]
}}

Example response:
{{
    "thoughts": "The video discusses dogs and their behavior. I'll create 2 images: one for the introduction about Corgis, and one for the section about training.",
    "images": [
        {{
            "description": "A happy Corgi dog",
            "sentence_ids": ["1", "2"],
            "detailed_prompt": "A professional photograph of a happy Corgi dog with orange and white fur, sitting on green grass, bright natural lighting, shallow depth of field, high quality, photorealistic"
        }},
        {{
            "description": "Dog training session",
            "sentence_ids": ["5", "6"],
            "detailed_prompt": "A warm scene of a person training a dog with treats, indoor setting with soft lighting, focus on the connection between human and dog, professional photography, photorealistic"
        }}
    ]
}}

Now analyze the transcript and suggest images:"""


class ImagePlanningAgent:
    """
    Agent for planning and generating image descriptions based on video transcript.
    Uses an LLM to analyze content and create optimized image prompts.
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize the Image Planning Agent.

        Args:
            llm_service: LLM service to use. If None, creates default OpenRouterLLMService.
        """
        self.llm_service = llm_service or OpenRouterLLMService(
            temperature=0.7,  # Higher temperature for more creative image ideas
            max_tokens=3000,
        )

    def plan_images_first_pass(
        self,
        adjusted_sentences: AdjustedSentences,
        general_instruction: str = "Create engaging images that enhance the video content",
    ) -> list[ImageDescription]:
        """
        First pass: Generate image descriptions and prompts based on transcript.

        Args:
            adjusted_sentences: Sentences with timestamps from the video
            general_instruction: General guidance for image generation

        Returns:
            List of ImageDescription objects with prompts and sentence associations

        Raises:
            RuntimeError: If LLM fails to generate valid response
        """
        # Create sentences dict for the prompt (sentence_id -> text)
        sentences_dict = {
            sentence.index: sentence.text for sentence in adjusted_sentences.sentences
        }
        sentences_json = json.dumps(sentences_dict, indent=2)

        # Format the prompt
        prompt = FIRST_PASS_PROMPT.format(
            general_instruction=general_instruction,
            sentences_json=sentences_json,
        )

        # Get LLM response
        try:
            response_text = self.llm_service.complete(prompt=prompt)

            # Parse JSON response
            # Remove markdown code blocks if present
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
            if "images" not in response_data:
                raise ValueError("Response missing 'images' field")

            # Convert to ImageDescription objects
            image_descriptions = []
            for img_data in response_data["images"]:
                image_desc = ImageDescription(
                    description=img_data["description"],
                    detailed_prompt=img_data["detailed_prompt"],
                    sentence_ids=img_data["sentence_ids"],
                )
                image_descriptions.append(image_desc)

            return image_descriptions

        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text}"
            )
        except KeyError as e:
            raise RuntimeError(
                f"LLM response missing required field: {e}\nResponse: {response_text}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate image descriptions: {e}")
