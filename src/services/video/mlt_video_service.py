"""
MLT-based video editing service.
Uses MLT XML files and the melt command-line tool for video processing.
"""

import os
import subprocess
import tempfile
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

from src.models import AdjustedSentences
from src.util import (
    get_input_video_path,
    get_final_cut_path,
    print_progress,
)


class MLTVideoService:
    """
    Video editing service using MLT framework.
    Uses MLT XML files and melt command for efficient video processing.
    """

    def __init__(self):
        """Initialize the MLT video service."""

    def _get_video_properties(self, video_path: Path) -> dict:
        """
        Get video properties (resolution, framerate) using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with width, height, fps, frame_rate_num, frame_rate_den
        """
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        video_stream = data["streams"][0]

        width = video_stream["width"]
        height = video_stream["height"]

        # Parse frame rate (format: "num/den" or just "num")
        r_frame_rate = video_stream["r_frame_rate"]
        if "/" in r_frame_rate:
            num, den = map(int, r_frame_rate.split("/"))
            frame_rate_num = num
            frame_rate_den = den
            fps = num / den
        else:
            frame_rate_num = int(r_frame_rate)
            frame_rate_den = 1
            fps = float(r_frame_rate)

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_rate_num": frame_rate_num,
            "frame_rate_den": frame_rate_den,
        }

    def _create_mlt_xml(
        self,
        video_path: Path,
        adjusted_sentences: AdjustedSentences,
        output_mlt_path: Path,
    ) -> None:
        """
        Create MLT XML file from adjusted sentences.

        Args:
            video_path: Path to source video file
            adjusted_sentences: AdjustedSentences with timestamps
            output_mlt_path: Path where MLT XML file will be saved
        """
        # Get video properties
        print("OGDEAN 6")
        props = self._get_video_properties(video_path)
        print("OGDEAN 7")

        # Create root element
        root = ET.Element(
            "mlt",
            {
                "LC_NUMERIC": "C",
                "version": "7.0.1",
                "root": str(video_path.parent),
            },
        )

        # Add profile
        ET.SubElement(
            root,
            "profile",
            {
                "description": f"Custom {props['width']}x{props['height']} {props['fps']:.2f} fps",
                "width": str(props["width"]),
                "height": str(props["height"]),
                "progressive": "1",
                "frame_rate_num": str(props["frame_rate_num"]),
                "frame_rate_den": str(props["frame_rate_den"]),
            },
        )

        # Add producer (source video)
        producer = ET.SubElement(root, "producer", {"id": "source_video"})

        resource_prop = ET.SubElement(producer, "property", {"name": "resource"})
        resource_prop.text = str(video_path)

        service_prop = ET.SubElement(producer, "property", {"name": "mlt_service"})
        service_prop.text = "avformat"

        # Add playlist with entries
        playlist = ET.SubElement(root, "playlist", {"id": "main_playlist"})

        for sentence in adjusted_sentences.sentences:
            # Convert time to frames
            start_frame = int(sentence.adjusted_start * props["fps"])
            end_frame = int(sentence.adjusted_end * props["fps"])

            # MLT uses inclusive frames, so we subtract 1 from end
            # (frame 0 to frame 149 = 150 frames = 5 seconds at 30fps)
            ET.SubElement(
                playlist,
                "entry",
                {
                    "producer": "source_video",
                    "in": str(start_frame),
                    "out": str(end_frame - 1),  # MLT 'out' is inclusive
                },
            )

        # Add tractor (main timeline)
        tractor = ET.SubElement(root, "tractor", {"id": "main_tractor"})
        ET.SubElement(tractor, "track", {"producer": "main_playlist"})

        # Pretty print XML
        xml_string = ET.tostring(root, encoding="unicode")
        dom = minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="  ")

        # Remove extra blank lines
        lines = [line for line in pretty_xml.split("\n") if line.strip()]
        pretty_xml = "\n".join(lines)

        # Write to file
        with open(output_mlt_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        print_progress(f"Created MLT XML file: {output_mlt_path}")

    def create_final_cut_with_mlt(
        self,
        base_name: str,
        adjusted_sentences: AdjustedSentences,
        force: bool = False,
    ) -> Path:
        """
        Create final cut video using MLT XML and melt command.

        This method creates a temporary MLT XML file defining the clips to include,
        then uses the melt command to render the final video.

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences with silence-trimmed timestamps
            force: If True, regenerate even if file exists

        Returns:
            Path to final cut video file

        Raises:
            FileNotFoundError: If input video doesn't exist
            RuntimeError: If melt command fails
        """
        input_path = get_input_video_path(base_name)
        output_path = get_final_cut_path(base_name)

        if output_path.exists() and not force:
            print_progress(f"Final cut already exists: {output_path}")
            return output_path

        print_progress(f"Creating final cut with MLT from: {input_path.name}...")

        print_progress(
            f"Processing {len(adjusted_sentences.sentences)} adjusted sentences with MLT"
        )

        temp_mlt_fd, temp_mlt_path_str = tempfile.mkstemp(suffix=".mlt", text=True)
        temp_mlt_path = Path(temp_mlt_path_str)

        try:
            os.close(temp_mlt_fd)
            self._create_mlt_xml(input_path, adjusted_sentences, temp_mlt_path)

            cmd = [
                "melt",
                str(temp_mlt_path),
                "-consumer",
                f"avformat:{output_path}",
                "vcodec=libx264",
                "acodec=aac",
                "crf=18",
                "preset=medium",
            ]

            print_progress("Running melt command...")
            print_progress(f"Command: {' '.join(cmd)}")

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            print_progress(f"Final cut created with MLT: {output_path}")
            return output_path

        finally:
            if temp_mlt_path.exists():
                temp_mlt_path.unlink()
                print_progress("Cleaned up temporary MLT file")
