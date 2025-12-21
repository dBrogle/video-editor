"""
OpenRouter image generation service.
"""

import asyncio
import base64
import os
from pathlib import Path
import httpx
from src.services.image_generation.base import ImageGeneratorService
from src.util import print_progress
from src.constants import (
    ENV_OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    OpenRouterImageModel,
)


class OpenRouterImageGenerator(ImageGeneratorService):
    """
    Image generation service using OpenRouter's image generation models.
    Supports Gemini and Flux models.
    """

    def __init__(
        self,
        model: OpenRouterImageModel | str = OpenRouterImageModel.GEMINI_25_FLASH_IMAGE,
        api_key: str | None = None,
    ):
        """
        Initialize OpenRouter image generation service.

        Args:
            model: Image model to use (OpenRouterImageModel enum or string)
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv(ENV_OPENROUTER_API_KEY)
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Convert enum to string value if needed
        self.model = model.value if isinstance(model, OpenRouterImageModel) else model
        self.api_url = OPENROUTER_API_URL

    async def generate_image(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
    ) -> Path:
        """
        Generate an image using OpenRouter.

        Args:
            prompt: Text description of the image to generate
            output_path: Where to save the generated image
            width: Image width in pixels (not used for OpenRouter, kept for interface compatibility)
            height: Image height in pixels (not used for OpenRouter, kept for interface compatibility)

        Returns:
            Path to the saved image file

        Raises:
            RuntimeError: If image generation fails
        """
        print_progress(f"Generating image with {self.model}: {prompt[:50]}...")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Request image generation
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        "modalities": ["image", "text"],
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Extract image from response
                if not data.get("choices"):
                    raise RuntimeError("No choices in OpenRouter response")

                message = data["choices"][0]["message"]
                if not message.get("images"):
                    raise RuntimeError("No images in OpenRouter response")

                # Get the first image (base64 data URL)
                image_data_url = message["images"][0]["image_url"]["url"]

                # Parse base64 data URL (format: data:image/png;base64,...)
                if not image_data_url.startswith("data:"):
                    raise RuntimeError(
                        f"Unexpected image URL format: {image_data_url[:50]}"
                    )

                # Extract base64 data
                header, base64_data = image_data_url.split(",", 1)
                image_bytes = base64.b64decode(base64_data)

                # Save to file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(image_bytes)

                print_progress(f"Image saved to: {output_path}")
                return output_path

        except httpx.HTTPStatusError as e:
            error_msg = (
                f"OpenRouter API error: {e.response.status_code} - {e.response.text}"
            )
            print_progress(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to generate image: {str(e)}"
            print_progress(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e

    async def generate_images_batch(
        self,
        prompts: list[tuple[str, Path]],
        width: int = 1024,
        height: int = 1024,
        max_concurrent: int = 3,
    ) -> list[Path]:
        """
        Generate multiple images in parallel with concurrency control.

        Args:
            prompts: List of (prompt, output_path) tuples
            width: Image width in pixels (not used for OpenRouter)
            height: Image height in pixels (not used for OpenRouter)
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of paths to successfully generated images

        Raises:
            RuntimeError: If all image generations fail
        """
        print_progress(
            f"Generating {len(prompts)} images with max {max_concurrent} concurrent requests..."
        )

        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def generate_with_semaphore(prompt: str, path: Path) -> Path | None:
            async with semaphore:
                try:
                    return await self.generate_image(prompt, path, width, height)
                except Exception as e:
                    print_progress(
                        f"Failed to generate image for prompt '{prompt[:30]}...': {e}"
                    )
                    return None

        # Generate all images concurrently
        tasks = [generate_with_semaphore(prompt, path) for prompt, path in prompts]
        completed = await asyncio.gather(*tasks, return_exceptions=False)

        # Filter out failed generations
        results = [path for path in completed if path is not None]

        if not results:
            raise RuntimeError("All image generations failed")

        print_progress(f"Successfully generated {len(results)}/{len(prompts)} images")
        return results
