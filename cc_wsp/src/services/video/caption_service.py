"""
Post-render caption service: burns word-level captions and title cards onto final video.
Uses Pillow for text rendering and FFmpeg for compositing.
"""

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from cc_wsp.src.constants import (
    CAPTION_FONT_PATH,
    CAPTION_FONT_INDEX,
    CAPTION_FONT_SIZE,
    CAPTION_Y_PERCENT,
    CAPTION_OUTLINE_WIDTH,
    CAPTION_MAX_WORDS_PER_CHUNK,
    CAPTION_MAX_CHARS_PER_CHUNK,
    CAPTION_COLOR,
    CAPTION_OUTLINE_COLOR,
    CAPTION_BG_COLOR,
    CAPTION_BG_OPACITY,
    CAPTION_BG_CORNER_RADIUS,
    CAPTION_BG_PADDING_H,
    CAPTION_BG_PADDING_V,
    TITLE_CARD_FONT_PATH,
    TITLE_CARD_FONT_INDEX,
    TITLE_CARD_BG_COLOR,
    TITLE_CARD_TEXT_COLOR,
    TITLE_CARD_FONT_SIZE,
    TITLE_CARD_Y_PERCENT,
    TITLE_CARD_CORNER_RADIUS,
    TITLE_CARD_PADDING_H,
    TITLE_CARD_PADDING_V,
    TITLE_CARD_MAX_WIDTH_PERCENT,
    HD_1080P_HEIGHT,
)
from cc_wsp.src.models import AdjustedSentences, Transcript, GoogleDocScript


VIDEO_WIDTH = 1080
VIDEO_HEIGHT = HD_1080P_HEIGHT  # 1920 for vertical video


@dataclass
class WordTiming:
    word: str
    start: float
    end: float


@dataclass
class CaptionChunk:
    text: str
    start: float
    end: float


@dataclass
class TitleCardConfig:
    text: str
    start: float
    end: float
    sentence_index: str | None = None


def build_word_timeline(
    adjusted: AdjustedSentences, transcription: Transcript
) -> list[WordTiming]:
    """Map word timestamps from transcription to final video time."""
    all_words = [w for s in transcription.sentences for w in s.words]

    timeline: list[WordTiming] = []
    video_cursor = 0.0

    for adj in adjusted.sentences:
        duration = adj.adjusted_end - adj.adjusted_start

        for w in all_words:
            if w.start < adj.adjusted_end and w.end > adj.adjusted_start:
                clipped_start = max(w.start, adj.adjusted_start)
                clipped_end = min(w.end, adj.adjusted_end)
                offset = clipped_start - adj.adjusted_start
                word_dur = clipped_end - clipped_start
                if word_dur > 0.01:
                    timeline.append(
                        WordTiming(
                            word=w.word,
                            start=video_cursor + offset,
                            end=video_cursor + offset + word_dur,
                        )
                    )

        video_cursor += duration

    return timeline


def group_words_into_chunks(
    timeline: list[WordTiming],
    max_words: int = CAPTION_MAX_WORDS_PER_CHUNK,
    max_chars: int = CAPTION_MAX_CHARS_PER_CHUNK,
) -> list[CaptionChunk]:
    """Group words into display chunks, splitting by character count and natural breaks."""
    if not timeline:
        return []

    chunks: list[CaptionChunk] = []
    current_words: list[WordTiming] = []

    def flush():
        if current_words:
            text = " ".join(w.word for w in current_words)
            chunks.append(
                CaptionChunk(
                    text=text,
                    start=current_words[0].start,
                    end=current_words[-1].end,
                )
            )
            current_words.clear()

    for wt in timeline:
        # Check if adding this word would exceed the character limit
        candidate = " ".join(w.word for w in current_words + [wt])
        if current_words and len(candidate) > max_chars:
            flush()

        current_words.append(wt)

        # Also break on natural punctuation
        is_natural_break = wt.word.rstrip().endswith((",", ".", "?", "!", ";", ":"))
        if is_natural_break:
            flush()

    flush()

    for i in range(len(chunks) - 1):
        if chunks[i].end > chunks[i + 1].start:
            chunks[i] = CaptionChunk(
                text=chunks[i].text,
                start=chunks[i].start,
                end=chunks[i + 1].start,
            )

    return chunks


def render_caption_png(
    text: str,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    y_percent: float = CAPTION_Y_PERCENT,
    font_path: str = CAPTION_FONT_PATH,
    font_index: int = CAPTION_FONT_INDEX,
    font_size: int = CAPTION_FONT_SIZE,
    outline_width: int = CAPTION_OUTLINE_WIDTH,
    text_color: tuple = CAPTION_COLOR,
    outline_color: tuple = CAPTION_OUTLINE_COLOR,
    bg_color: tuple = CAPTION_BG_COLOR,
    bg_opacity: int = CAPTION_BG_OPACITY,
    bg_corner_radius: int = CAPTION_BG_CORNER_RADIUS,
    bg_padding_h: int = CAPTION_BG_PADDING_H,
    bg_padding_v: int = CAPTION_BG_PADDING_V,
) -> Image.Image:
    """Render a single caption frame as a transparent PNG."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size=font_size, index=font_index)

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Center horizontally, position at y_percent vertically
    x = (width - tw) // 2
    y = int(height * y_percent) - th // 2

    # Translucent rounded pill behind the text for readability. Sized to the
    # ink bbox (bbox offsets matter for fonts with ascenders/descenders), then
    # expanded by the outline width and padding.
    if bg_opacity > 0:
        ink_l = x + bbox[0] - outline_width - bg_padding_h
        ink_t = y + bbox[1] - outline_width - bg_padding_v
        ink_r = x + bbox[2] + outline_width + bg_padding_h
        ink_b = y + bbox[3] + outline_width + bg_padding_v
        draw.rounded_rectangle(
            (ink_l, ink_t, ink_r, ink_b),
            radius=bg_corner_radius,
            fill=(*bg_color, bg_opacity),
        )

    # Draw black outline by rendering text at offsets
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text(
                (x + dx, y + dy),
                text,
                font=font,
                fill=(*outline_color, 255),
            )

    # Draw white text on top
    draw.text((x, y), text, font=font, fill=(*text_color, 255))

    return img


def render_title_card_png(
    text: str,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    y_percent: float = TITLE_CARD_Y_PERCENT,
    font_path: str = TITLE_CARD_FONT_PATH,
    font_index: int = TITLE_CARD_FONT_INDEX,
    font_size: int = TITLE_CARD_FONT_SIZE,
    bg_color: tuple = TITLE_CARD_BG_COLOR,
    text_color: tuple = TITLE_CARD_TEXT_COLOR,
    corner_radius: int = TITLE_CARD_CORNER_RADIUS,
    padding_h: int = TITLE_CARD_PADDING_H,
    padding_v: int = TITLE_CARD_PADDING_V,
    max_width_pct: float = TITLE_CARD_MAX_WIDTH_PERCENT,
) -> Image.Image:
    """Render a blue rounded-rectangle title card as a transparent PNG."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size=font_size, index=font_index)

    # Constrain text wrapping to max_width_pct of video width
    max_box_width = int(width * max_width_pct)
    max_text_width = max_box_width - 2 * padding_h
    # Preserve explicit newlines from the source text; word-wrap within each.
    lines = []
    for paragraph in text.split("\n"):
        if paragraph == "":
            lines.append("")
        else:
            lines.extend(_wrap_text(draw, paragraph, font, max_text_width))
    line_height = font_size + 6

    total_text_height = line_height * len(lines)
    text_block_width = max(draw.textlength(line, font=font) for line in lines)
    box_width = text_block_width + 2 * padding_h
    box_height = total_text_height + 2 * padding_v

    box_x = (width - box_width) // 2
    box_y = int(height * y_percent)

    # Draw rounded rectangle — fully opaque
    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_width, box_y + box_height],
        radius=corner_radius,
        fill=(*bg_color, 255),
    )

    # First line centered (header); subsequent lines left-aligned to the
    # inner padding edge so list items stack neatly.
    text_y = box_y + padding_v
    for i, line in enumerate(lines):
        if i == 0:
            lw = draw.textlength(line, font=font)
            lx = (width - lw) // 2
        else:
            lx = box_x + padding_h
        draw.text((lx, text_y), line, font=font, fill=(*text_color, 255))
        text_y += line_height

    return img


def _wrap_text(
    draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: float
) -> list[str]:
    """Word-wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test = f"{current_line} {word}".strip()
        if draw.textlength(test, font=font) <= max_width:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def detect_title_text(script: GoogleDocScript) -> str | None:
    """Auto-detect title text from the Google Doc script 'Text: ...' line."""
    for line in script.lines:
        m = re.match(r'^Text:\s*["\u201c](.+?)["\u201d]', line.text)
        if m:
            return m.group(1)
    return None


def extract_title_instructions(
    script: GoogleDocScript,
    adjusted: "AdjustedSentences",
) -> list[TitleCardConfig]:
    """
    Extract [Title text: "..."] instructions from the script and map them
    to sentence timings in the final video.
    """
    from cc_wsp.src.services.agents.google_doc_image_placer import GoogleDocImagePlacer

    # Build a timeline mapping: cumulative offset for each sentence in the cut video
    cumulative = 0.0
    sentence_timing: dict[str, tuple[float, float]] = {}
    for s in adjusted.sentences:
        duration = s.adjusted_end - s.adjusted_start
        sentence_timing[s.index] = (cumulative, cumulative + duration)
        cumulative += duration

    # Match script lines with title instructions to video sentences
    title_cards = []
    for line in script.lines:
        for instr in line.instructions:
            m = re.match(r'^Title text:\s*["\u201c](.+?)(?:["\u201d]|$)', instr)
            if not m:
                continue
            title_text = m.group(1).replace("\\n", "\n")

            # Find the best matching sentence for this script line. Filter
            # out common stop words so "Out of microsoft" matches on
            # "microsoft" rather than scoring 1 on any sentence containing
            # "of".
            stop_words = {
                "a", "an", "the", "and", "or", "but", "of", "in", "on",
                "at", "to", "for", "with", "from", "by", "as", "is", "it",
                "this", "that", "these", "those", "they", "their", "them",
                "so", "our", "be", "was", "were", "are", "has", "have", "had",
                "will", "would", "can", "could", "should",
            }

            def _content_words(text: str) -> set[str]:
                tokens = re.findall(r"[a-z0-9']+", text.lower())
                return {t for t in tokens if t not in stop_words}

            l_words = _content_words(line.text)
            best_idx = None
            best_score = 0
            for s in adjusted.sentences:
                s_words = _content_words(s.text)
                overlap = len(s_words & l_words)
                if overlap > best_score:
                    best_score = overlap
                    best_idx = s.index

            if best_idx and best_idx in sentence_timing:
                start, end = sentence_timing[best_idx]
                title_cards.append(TitleCardConfig(
                    text=title_text, start=start, end=end,
                    sentence_index=best_idx,
                ))

    return title_cards


def burn_captions(
    video_path: Path,
    output_path: Path,
    chunks: list[CaptionChunk],
    title_config: TitleCardConfig | None = None,
    title_cards: list[TitleCardConfig] | None = None,
    caption_y_percent: float = CAPTION_Y_PERCENT,
):
    """Composite caption PNGs onto the video using FFmpeg overlay filters."""
    with tempfile.TemporaryDirectory(prefix="captions_") as tmpdir:
        tmp = Path(tmpdir)
        inputs = ["-i", str(video_path)]
        overlays = []
        input_idx = 1  # 0 is the video

        # Render and save title card(s)
        all_titles = []
        if title_config:
            all_titles.append(title_config)
        if title_cards:
            all_titles.extend(title_cards)

        for ti, tc in enumerate(all_titles):
            # Scale down the title font when the card has many lines so it
            # doesn't overwhelm the frame. Base 56px, shrink by 6px per
            # extra line beyond 2, floor at 36px.
            line_count = tc.text.count("\n") + 1
            tc_font_size = max(36, TITLE_CARD_FONT_SIZE - 6 * max(0, line_count - 2))
            tc_img = render_title_card_png(tc.text, font_size=tc_font_size)
            tc_path = tmp / f"title_card_{ti}.png"
            tc_img.save(str(tc_path))
            inputs.extend(["-i", str(tc_path)])
            overlays.append(
                (input_idx, tc.start, tc.end)
            )
            input_idx += 1

        # Render and save each caption chunk
        for i, chunk in enumerate(chunks):
            cap_img = render_caption_png(chunk.text, y_percent=caption_y_percent)
            cap_path = tmp / f"cap_{i:04d}.png"
            cap_img.save(str(cap_path))
            inputs.extend(["-i", str(cap_path)])
            overlays.append((input_idx, chunk.start, chunk.end))
            input_idx += 1

        # Build filter_complex chain
        # Each overlay takes the previous result and composites the next image
        filter_parts = []
        prev = "0:v"
        for j, (idx, start, end) in enumerate(overlays):
            out = f"v{j}"
            enable = f"between(t\\,{start:.3f}\\,{end:.3f})"
            filter_parts.append(
                f"[{prev}][{idx}:v]overlay=0:0:enable='{enable}'[{out}]"
            )
            prev = out

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{prev}]",
            "-map",
            "0:a",
            "-c:v",
            "libx264",
            "-crf",
            "19",
            "-preset",
            "fast",
            "-c:a",
            "copy",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

        print(f"=> Burning {len(chunks)} captions + {'title card' if title_config else 'no title'}")
        print(f"=> Running ffmpeg ({len(overlays)} overlay filters)...")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error:\n{result.stderr[-2000:]}")
            raise RuntimeError("FFmpeg caption burn failed")

        print(f"=> Captioned video: {output_path}")
