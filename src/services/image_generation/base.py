"""
Abstract base class for image generation services.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class ImageGeneratorService(ABC):
    """
    Abstract base class for AI image generation services.

    Implementations should handle:
    - API authentication
    - Image generation from text prompts
    - Downloading and saving generated images
    - Error handling and retries
    """

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
    ) -> Path:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            output_path: Where to save the generated image
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Path to the saved image file

        Raises:
            RuntimeError: If image generation fails
        """
        pass

    @abstractmethod
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
            width: Image width in pixels
            height: Image height in pixels
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of paths to successfully generated images

        Raises:
            RuntimeError: If all image generations fail
        """
        pass
