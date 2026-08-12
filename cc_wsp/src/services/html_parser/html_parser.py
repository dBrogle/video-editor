"""
Google Doc HTML Parser Service.

Extracts text lines and associated images from Google Doc HTML exports.
"""

import re
from pathlib import Path
from bs4 import BeautifulSoup, Tag
from cc_wsp.src.models import GoogleDocScript, GoogleDocLine


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

            # A single paragraph may hold several images (Google Docs inlines them
            # side by side when they're pasted on one line), so take all of them.
            img_tags = element.find_all("img")
            filenames = [
                f for f in (self._extract_image_filename(t) for t in img_tags) if f
            ]

            if text:
                # Check if text is a bracketed instruction (e.g., "[gif of robot dog]")
                instruction = self._extract_instruction(text)
                if instruction:
                    # Attach instruction to the most recent text line
                    if lines:
                        lines[-1].instructions.append(instruction)
                elif filenames:
                    # Element has both text and image(s) - create line with images
                    lines.append(
                        GoogleDocLine(text=text, image_filenames=filenames)
                    )
                    current_text = None
                else:
                    # Element has only text - save as current text
                    current_text = text
                    lines.append(GoogleDocLine(text=text, image_filenames=[]))
            elif filenames:
                # Element has only image(s), no text
                # Assign them to the most recent text line
                # Don't clear current_text — multiple images may follow the same text
                if lines:
                    lines[-1].image_filenames.extend(filenames)

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

    def _extract_instruction(self, text: str) -> str | None:
        """
        Check if text is a bracketed instruction like [gif of robot dog].

        Args:
            text: Text content to check

        Returns:
            Instruction content (without brackets) if text is a bracketed instruction, else None
        """
        match = re.fullmatch(r"\[(.+)\]", text.strip())
        return match.group(1) if match else None

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
