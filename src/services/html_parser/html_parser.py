"""
Google Doc HTML Parser Service.

Extracts text lines and associated images from Google Doc HTML exports.
"""

from pathlib import Path
from bs4 import BeautifulSoup, Tag
from src.models import GoogleDocScript, GoogleDocLine


class GoogleDocHTMLParser:
    """
    Service for parsing Google Doc HTML exports.

    Extracts text content and image associations from HTML structure.
    Images are assigned to the most recent non-empty text line.
    """

    def parse_html(self, html_content: str) -> GoogleDocScript:
        """
        Parse Google Doc HTML and extract text lines with associated images.

        Args:
            html_content: Raw HTML content from Google Doc export

        Returns:
            GoogleDocScript with lines and image associations
        """
        soup = BeautifulSoup(html_content, "html.parser")

        lines = []
        current_text = None

        # Find all body elements (p, h1, h2, h3, h4, h5, h6)
        body = soup.find("body")
        if not body:
            return GoogleDocScript(lines=[])

        # Iterate through all elements in the body
        for element in body.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
            # Extract text content
            text = self._extract_text(element)

            # Check if element contains an image
            img_tag = element.find("img")

            if text:
                # This element has text content
                if img_tag:
                    # Element has both text and image - create line with image
                    image_filename = self._extract_image_filename(img_tag)
                    lines.append(
                        GoogleDocLine(text=text, image_filename=image_filename)
                    )
                    current_text = None
                else:
                    # Element has only text - save as current text
                    current_text = text
                    lines.append(GoogleDocLine(text=text, image_filename=None))
            elif img_tag:
                # Element has only image, no text
                # Assign image to the most recent text line
                if current_text and lines:
                    # Find the last line and update it with this image
                    image_filename = self._extract_image_filename(img_tag)
                    lines[-1].image_filename = image_filename
                    current_text = None

        return GoogleDocScript(lines=lines)

    def parse_html_file(self, html_path: Path) -> GoogleDocScript:
        """
        Parse Google Doc HTML file and extract text lines with associated images.

        Args:
            html_path: Path to HTML file

        Returns:
            GoogleDocScript with lines and image associations

        Raises:
            FileNotFoundError: If HTML file doesn't exist
        """
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_path}")

        html_content = html_path.read_text(encoding="utf-8")
        return self.parse_html(html_content)

    def _extract_text(self, element: Tag) -> str:
        """
        Extract clean text from an HTML element.

        Args:
            element: BeautifulSoup Tag element

        Returns:
            Cleaned text content (stripped, with normalized whitespace)
        """
        # Get all text, excluding text within img tags
        text = element.get_text(separator=" ", strip=True)

        # Normalize whitespace
        text = " ".join(text.split())

        # Decode HTML entities (like &rsquo; -> ')
        # BeautifulSoup already handles this, but let's be explicit
        return text

    def _extract_image_filename(self, img_tag: Tag) -> str | None:
        """
        Extract image filename from img tag's src attribute.

        Args:
            img_tag: BeautifulSoup img Tag

        Returns:
            Image filename (e.g., 'image1.png') or None if not found
        """
        src = img_tag.get("src")
        if not src:
            return None

        # Extract filename from path like "images/image1.png"
        filename = Path(src).name
        return filename
