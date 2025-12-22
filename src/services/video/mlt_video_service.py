"""
MLT-based video editing service.
Uses MLT XML files and the melt command-line tool for video processing.
"""

import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from src.models import AdjustedSentences, ImagesMetadataFile, GoogleDocImagePlacements
from src.util import (
    get_stage_7_with_images_path,
    get_stage_7_mlt_xml_path,
    get_stage_11_with_google_doc_images_path,
    get_stage_11_mlt_xml_path,
    get_images_folder,
    get_edited_video_path,
    get_full_res_cut_video_path,
    get_full_res_cut_mlt_path,
    get_full_res_with_images_video_path,
    get_full_res_with_images_mlt_path,
    get_input_video_path,
    print_progress,
)
from src.constants import (
    IMAGE_SAFE_ZONE_TOP_PERCENT,
    IMAGE_SAFE_ZONE_BOTTOM_PERCENT,
    IMAGE_SAFE_ZONE_LEFT_PERCENT,
    IMAGE_SAFE_ZONE_RIGHT_PERCENT,
)
from src.services.video.mlt_util import (
    frames_to_timecode,
    get_video_properties,
    calculate_safe_zone,
    create_mlt_root_and_profile,
    add_black_producer,
    add_video_chain,
    add_image_producer,
    create_base_playlists,
    create_main_tractor,
    save_pretty_xml,
    add_mix_transition,
    add_composite_transition,
    add_cairo_transition,
)


class MLTVideoService:
    """
    Video editing service using MLT framework.
    Uses MLT XML files and melt command for efficient video processing.
    """

    def __init__(self):
        """Initialize the MLT video service."""

    def rotate_video_if_needed(self, base_name: str, force: bool = False) -> Path:
        """
        Check if a video needs rotation and create a properly oriented .mp4 file.

        This handles videos that have rotation metadata (like iPhone videos shot in portrait
        but stored as landscape with rotation=-90). The method detects rotation metadata
        and creates a new .mp4 file with pixels physically rotated to the correct orientation.

        Args:
            base_name: Base filename without extension
            force: If True, regenerate even if .mp4 exists

        Returns:
            Path to the properly oriented video file (either .mp4 or original if no rotation needed)

        Raises:
            FileNotFoundError: If source video doesn't exist
            RuntimeError: If melt command fails
        """
        from src.constants import ASSETS_DIR
        from src.services.video.mlt_util import (
            get_video_rotation,
            create_rotation_mlt_xml,
        )

        # Look for source video files in order of preference
        folder = ASSETS_DIR / base_name
        source_path = None
        for ext in [".MOV", ".mov", ".mp4", ".MP4"]:
            test_path = folder / f"{base_name}{ext}"
            if test_path.exists():
                # Skip if it's already an .mp4 (unless forcing)
                if ext.lower() == ".mp4" and not force:
                    print_progress(f"Video already in .mp4 format: {test_path.name}")
                    return test_path
                source_path = test_path
                break

        if not source_path:
            raise FileNotFoundError(f"No video file found for {base_name} in {folder}")

        output_path = folder / f"{base_name}.mp4"

        # Skip if output exists and not forcing
        if output_path.exists() and not force:
            print_progress(f"Rotated video already exists: {output_path.name}")
            return output_path

        print_progress(f"Checking rotation metadata for: {source_path.name}")

        # Get video properties including rotation
        width, height, fps_num, fps_den, rotation = get_video_rotation(source_path)

        print_progress(f"Video dimensions: {width}x{height}, rotation: {rotation}°")

        # Determine if we need to rotate and what the output dimensions should be
        needs_rotation = rotation != 0

        if not needs_rotation:
            print_progress("No rotation needed, video is already properly oriented")
            # If source is not .mp4, we should still convert it
            if source_path.suffix.lower() != ".mp4":
                print_progress(f"Converting {source_path.suffix} to .mp4 format...")
                needs_rotation = True  # Set to true to trigger conversion
            else:
                return source_path

        # For -90 or 270 degree rotation, swap width and height
        if abs(rotation) == 90 or abs(rotation) == 270:
            output_width = height
            output_height = width
            print_progress(f"Will rotate video to: {output_width}x{output_height}")
        else:
            output_width = width
            output_height = height

        # Create MLT XML for rotation
        mlt_xml_path = folder / f"{base_name}_rotate.mlt"

        print_progress(f"Creating MLT XML for rotation: {mlt_xml_path.name}")
        create_rotation_mlt_xml(
            source_path=source_path,
            output_width=output_width,
            output_height=output_height,
            fps_num=fps_num,
            fps_den=fps_den,
            mlt_xml_path=mlt_xml_path,
        )

        print_progress("Running melt command to rotate video...")

        # Run melt command
        cmd = [
            "melt",
            str(mlt_xml_path),
            "-consumer",
            f"avformat:{output_path}",
            "vcodec=libx264",
            "crf=18",
            "preset=faster",
            "acodec=aac",
            "pix_fmt=yuv420p",
        ]

        print_progress(f"Command: {' '.join(cmd)}")

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        print_progress(f"Rotated video created: {output_path.name}")
        print_progress(f"MLT XML saved for debugging: {mlt_xml_path.name}")

        # Clean up old mp4 if it exists
        old_mp4 = folder / f"{base_name}_old.mp4"
        if old_mp4.exists():
            old_mp4.unlink()
            print_progress(f"Cleaned up old file: {old_mp4.name}")

        return output_path

    def _build_sentence_timeline(self, adjusted_sentences: AdjustedSentences) -> dict:
        """
        Build a mapping of sentence IDs to their cumulative times in the cut video.

        Args:
            adjusted_sentences: Sentences with timing info

        Returns:
            Dictionary mapping sentence_id -> {"start": float, "end": float}
        """
        sentence_cumulative_times = {}
        cumulative_time = 0.0

        for sentence in adjusted_sentences.sentences:
            sentence_duration = sentence.adjusted_end - sentence.adjusted_start
            sentence_cumulative_times[sentence.index] = {
                "start": cumulative_time,
                "end": cumulative_time + sentence_duration,
            }
            cumulative_time += sentence_duration

        return sentence_cumulative_times

    def _calculate_image_timings_with_delay(
        self,
        images_metadata: ImagesMetadataFile,
        sentence_timeline: dict,
        fps: float,
    ) -> list[tuple[int, int, int] | None]:
        """
        Calculate timing for all images with first-image delay.

        Args:
            images_metadata: Metadata for images to overlay
            sentence_timeline: Mapping of sentence IDs to timing info
            fps: Frames per second

        Returns:
            List of (start_frame, end_frame, image_index) tuples or None
        """
        image_timings: list[tuple[int, int, int] | None] = []

        for i, img_meta in enumerate(images_metadata.images):
            if not img_meta.sentence_ids:
                image_timings.append(None)
                continue

            sentence_starts = []
            for sent_id in img_meta.sentence_ids:
                if sent_id in sentence_timeline:
                    times = sentence_timeline[sent_id]
                    sentence_starts.append(times["start"])

            if not sentence_starts:
                image_timings.append(None)
                continue

            img_start = min(sentence_starts)
            image_delay_frames = 67 if i == 0 else 0
            img_start_frame = int(img_start * fps) + image_delay_frames
            img_end_frame = img_start_frame + 120

            image_timings.append((img_start_frame, img_end_frame, i))

        return image_timings

    def _calculate_google_doc_image_timings(
        self,
        image_placements: GoogleDocImagePlacements,
        sentence_timeline: dict,
        fps: float,
    ) -> list[tuple[int, int, int] | None]:
        """
        Calculate timing for Google Doc images based on sentence duration.

        Args:
            image_placements: Google Doc image placements with sentence associations
            sentence_timeline: Mapping of sentence IDs to timing info
            fps: Frames per second

        Returns:
            List of (start_frame, end_frame, image_index) tuples or None
        """
        image_timings: list[tuple[int, int, int] | None] = []

        for i, placement in enumerate(image_placements.placements):
            if not placement.sentence_indexes:
                image_timings.append(None)
                continue

            sentence_starts = []
            sentence_ends = []
            for sent_id in placement.sentence_indexes:
                if sent_id in sentence_timeline:
                    times = sentence_timeline[sent_id]
                    sentence_starts.append(times["start"])
                    sentence_ends.append(times["end"])

            if not sentence_starts:
                image_timings.append(None)
                continue

            img_start = min(sentence_starts)
            img_end = max(sentence_ends)
            img_start_frame = int(img_start * fps)
            img_end_frame = int(img_end * fps)

            if img_end_frame <= img_start_frame:
                img_end_frame = img_start_frame + 1

            image_timings.append((img_start_frame, img_end_frame, i))

        return image_timings

    def _create_overlay_playlist(
        self,
        root: ET.Element,
        image_timings: list[tuple[int, int, int] | None],
        fps: float,
    ) -> ET.Element:
        """
        Create playlist for image overlays with blanks between images.

        Args:
            root: MLT XML root element
            image_timings: List of (start_frame, end_frame, image_index) tuples or None
            fps: Frames per second

        Returns:
            Playlist element
        """
        playlist = ET.SubElement(root, "playlist", {"id": "playlist1"})

        # Add Shotcut properties
        ET.SubElement(playlist, "property", {"name": "shotcut:video"}).text = "1"
        ET.SubElement(playlist, "property", {"name": "shotcut:name"}).text = "V2"

        # Build a timeline of all image events
        events: list[dict[str, int]] = []
        for timing in image_timings:
            if timing:
                start_frame, end_frame, image_index = timing
                duration = end_frame - start_frame
                events.append(
                    {
                        "frame": start_frame,
                        "image_index": image_index,
                        "duration": duration,
                    }
                )

        # Sort events by frame
        events.sort(key=lambda x: x["frame"])

        # Build playlist with blanks and entries
        current_playlist_frame = 0

        for event in events:
            event_frame = event["frame"]
            event_image_index = event["image_index"]
            image_duration = 120  # Fixed duration

            # Add blank before this image if needed
            if event_frame > current_playlist_frame:
                blank_frames = event_frame - current_playlist_frame
                blank_timecode = frames_to_timecode(blank_frames, fps)
                ET.SubElement(playlist, "blank", {"length": blank_timecode})
                current_playlist_frame += blank_frames

            # Add image entry
            entry_timecode = frames_to_timecode(image_duration, fps)
            ET.SubElement(
                playlist,
                "entry",
                {
                    "producer": f"producer_{event_image_index}",
                    "in": "00:00:00.000",
                    "out": entry_timecode,
                },
            )

            current_playlist_frame += image_duration

        return playlist

    def _create_overlay_playlist_with_dynamic_duration(
        self,
        root: ET.Element,
        image_timings: list[tuple[int, int, int] | None],
        fps: float,
    ) -> ET.Element:
        """
        Create playlist for image overlays with blanks between images.
        Images have dynamic durations based on sentence timing.

        Args:
            root: MLT XML root element
            image_timings: List of (start_frame, end_frame, image_index) tuples or None
            fps: Frames per second

        Returns:
            Playlist element
        """
        playlist = ET.SubElement(root, "playlist", {"id": "playlist1"})

        # Add Shotcut properties
        ET.SubElement(playlist, "property", {"name": "shotcut:video"}).text = "1"
        ET.SubElement(playlist, "property", {"name": "shotcut:name"}).text = "V2"

        # Build a timeline of all image events
        events: list[dict] = []
        for timing in image_timings:
            if timing:
                start_frame, end_frame, image_index = timing
                duration = end_frame - start_frame
                events.append(
                    {
                        "frame": start_frame,
                        "image_index": image_index,
                        "duration": duration,
                    }
                )

        # Sort events by frame
        events.sort(key=lambda x: x["frame"])

        # Build playlist with blanks and entries
        current_playlist_frame = 0

        for event in events:
            event_frame = event["frame"]
            event_image_index = event["image_index"]
            image_duration = event["duration"]

            # Add blank before this image if needed
            if event_frame > current_playlist_frame:
                blank_frames = event_frame - current_playlist_frame
                blank_timecode = frames_to_timecode(blank_frames, fps)
                ET.SubElement(playlist, "blank", {"length": blank_timecode})
                current_playlist_frame += blank_frames

            # Add image entry with dynamic duration
            entry_timecode = frames_to_timecode(image_duration, fps)
            ET.SubElement(
                playlist,
                "entry",
                {
                    "producer": f"producer_{event_image_index}",
                    "in": "00:00:00.000",
                    "out": entry_timecode,
                },
            )

            current_playlist_frame += image_duration

        return playlist

    def _create_mlt_xml_with_images(
        self,
        video_path: Path,
        adjusted_sentences: AdjustedSentences,
        images_metadata: ImagesMetadataFile,
        images_folder: Path,
        output_mlt_path: Path,
    ) -> None:
        """
        Create MLT XML file with image overlays on an already-cut video.
        Uses playlist-based structure with affine filters and blend transitions.

        Args:
            video_path: Path to already-edited video file (e.g., s6_downsampled_edited.mp4)
            adjusted_sentences: Sentences with timing info (to map sentence IDs to times in cut video)
            images_metadata: Metadata for images to overlay
            images_folder: Path to folder containing images
            output_mlt_path: Path where MLT XML file will be saved
        """
        # Get video properties
        props = get_video_properties(video_path)

        # Calculate safe zone for image positioning (in pixels)
        safe_zone = calculate_safe_zone(
            props,
            IMAGE_SAFE_ZONE_TOP_PERCENT,
            IMAGE_SAFE_ZONE_BOTTOM_PERCENT,
            IMAGE_SAFE_ZONE_LEFT_PERCENT,
            IMAGE_SAFE_ZONE_RIGHT_PERCENT,
        )

        # Build sentence timeline mapping
        sentence_timeline = self._build_sentence_timeline(adjusted_sentences)

        # Calculate total duration in frames
        total_duration = sum(
            s.adjusted_end - s.adjusted_start for s in adjusted_sentences.sentences
        )
        total_frames = int(total_duration * props["fps"])
        total_timecode = frames_to_timecode(total_frames, props["fps"])

        # Create root element and profile
        root = create_mlt_root_and_profile(props)

        # Add black background producer
        add_black_producer(root, total_timecode)

        # Add video chain (not simple producer)
        add_video_chain(root, video_path, total_timecode)

        # Calculate timing for all images
        image_timings = self._calculate_image_timings_with_delay(
            images_metadata, sentence_timeline, props["fps"]
        )

        # Add image producers (without filters)
        for i, img_meta in enumerate(images_metadata.images):
            img_path = images_folder / img_meta.filename
            timing = image_timings[i]
            if timing:
                add_image_producer(root, i, img_path)

        # Create base playlists (background and video)
        create_base_playlists(root, total_timecode)

        # Create overlay playlist with blanks
        self._create_overlay_playlist(root, image_timings, props["fps"])

        # Create main tractor with tracks and transitions
        create_main_tractor(root, total_timecode, safe_zone)

        # Save XML
        save_pretty_xml(root, output_mlt_path)

        print_progress(f"Created MLT XML file with images: {output_mlt_path}")

    def create_video_with_images(
        self,
        base_name: str,
        adjusted_sentences: AdjustedSentences,
        images_metadata: ImagesMetadataFile,
        force: bool = False,
    ) -> Path:
        """
        Create video with image overlays using MLT XML and melt command.
        Uses the already-cut video from Stage 6 as the base.

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences with timestamps
            images_metadata: Metadata for images to overlay
            force: If True, regenerate even if file exists

        Returns:
            Path to video file with images

        Raises:
            FileNotFoundError: If edited video or images don't exist
            RuntimeError: If melt command fails
        """
        # Use the already-cut downsampled video from Stage 6
        input_path = get_edited_video_path(base_name, use_downsampled=True)
        output_path = get_stage_7_with_images_path(base_name)
        mlt_xml_path = get_stage_7_mlt_xml_path(base_name)
        images_folder = get_images_folder(base_name)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Edited video not found: {input_path}. Please run Stage 6 first."
            )

        if output_path.exists() and not force:
            print_progress(f"Video with images already exists: {output_path}")
            return output_path

        print_progress(f"Creating video with images from: {input_path.name}...")
        print_progress(f"Adding {len(images_metadata.images)} image overlays")

        # Create MLT XML file (saved for debugging)
        self._create_mlt_xml_with_images(
            input_path,
            adjusted_sentences,
            images_metadata,
            images_folder,
            mlt_xml_path,
        )

        cmd = [
            "melt",
            str(mlt_xml_path),
            "-consumer",
            f"avformat:{output_path}",
            "vcodec=libx264",
            "acodec=aac",
            "crf=18",
            "preset=medium",
            "pix_fmt=yuv420p",
        ]

        print_progress("Running melt command...")
        print_progress(f"Command: {' '.join(cmd)}")

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        print_progress(f"Video with images created: {output_path}")
        print_progress(f"MLT XML saved for debugging: {mlt_xml_path}")
        return output_path

    def _create_mlt_xml_with_google_doc_images(
        self,
        video_path: Path,
        adjusted_sentences: AdjustedSentences,
        image_placements: GoogleDocImagePlacements,
        output_mlt_path: Path,
    ) -> None:
        """
        Create MLT XML file with Google Doc image overlays on an already-cut video.
        Uses playlist-based structure with affine filters and blend transitions.
        Images are timed based on sentence duration, not fixed duration.

        Args:
            video_path: Path to already-edited video file (e.g., s6_downsampled_edited.mp4)
            adjusted_sentences: Sentences with timing info (to map sentence IDs to times in cut video)
            image_placements: Google Doc image placements with sentence associations
            output_mlt_path: Path where MLT XML file will be saved
        """
        # Get video properties
        props = get_video_properties(video_path)

        # Calculate safe zone for image positioning (in pixels)
        safe_zone = calculate_safe_zone(
            props,
            IMAGE_SAFE_ZONE_TOP_PERCENT,
            IMAGE_SAFE_ZONE_BOTTOM_PERCENT,
            IMAGE_SAFE_ZONE_LEFT_PERCENT,
            IMAGE_SAFE_ZONE_RIGHT_PERCENT,
        )

        # Build sentence timeline mapping
        sentence_timeline = self._build_sentence_timeline(adjusted_sentences)

        # Calculate total duration in frames
        total_duration = sum(
            s.adjusted_end - s.adjusted_start for s in adjusted_sentences.sentences
        )
        total_frames = int(total_duration * props["fps"])
        total_timecode = frames_to_timecode(total_frames, props["fps"])

        # Create root element and profile
        root = create_mlt_root_and_profile(props)

        # Add black background producer
        add_black_producer(root, total_timecode)

        # Add video chain (not simple producer)
        add_video_chain(root, video_path, total_timecode)

        # Calculate timing for all images based on sentence indexes
        image_timings = self._calculate_google_doc_image_timings(
            image_placements, sentence_timeline, props["fps"]
        )

        # Add image producers (without filters)
        for i, placement in enumerate(image_placements.placements):
            img_path = Path(placement.filepath)
            timing = image_timings[i]
            if timing:
                add_image_producer(root, i, img_path)

        # Create base playlists (background and video)
        create_base_playlists(root, total_timecode)

        # Create overlay playlist with blanks and dynamic image durations
        self._create_overlay_playlist_with_dynamic_duration(
            root, image_timings, props["fps"]
        )

        # Create main tractor with tracks and transitions
        create_main_tractor(root, total_timecode, safe_zone)

        # Save XML
        save_pretty_xml(root, output_mlt_path)

        print_progress(
            f"Created MLT XML file with Google Doc images: {output_mlt_path}"
        )

    def create_video_with_google_doc_images(
        self,
        base_name: str,
        adjusted_sentences: AdjustedSentences,
        image_placements: GoogleDocImagePlacements,
        force: bool = False,
    ) -> Path:
        """
        Create video with Google Doc image overlays using MLT XML and melt command.
        Uses the already-cut video from Stage 6 as the base.
        Images are timed based on sentence duration, not fixed duration.

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences with timestamps
            image_placements: Google Doc image placements with sentence associations
            force: If True, regenerate even if file exists

        Returns:
            Path to video file with Google Doc images

        Raises:
            FileNotFoundError: If edited video or images don't exist
            RuntimeError: If melt command fails
        """
        # Use the already-cut downsampled video from Stage 6
        input_path = get_edited_video_path(base_name, use_downsampled=True)
        output_path = get_stage_11_with_google_doc_images_path(base_name)
        mlt_xml_path = get_stage_11_mlt_xml_path(base_name)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Edited video not found: {input_path}. Please run Stage 6 first."
            )

        if output_path.exists() and not force:
            print_progress(
                f"Video with Google Doc images already exists: {output_path}"
            )
            return output_path

        print_progress(
            f"Creating video with Google Doc images from: {input_path.name}..."
        )
        print_progress(f"Adding {len(image_placements.placements)} image overlays")

        # Verify all image files exist
        missing_images = []
        for placement in image_placements.placements:
            img_path = Path(placement.filepath)
            if not img_path.exists():
                missing_images.append(str(img_path))

        if missing_images:
            raise FileNotFoundError(
                "Missing image files:\n"
                + "\n".join(f"  - {img}" for img in missing_images)
            )

        # Create MLT XML file (saved for debugging)
        self._create_mlt_xml_with_google_doc_images(
            input_path,
            adjusted_sentences,
            image_placements,
            mlt_xml_path,
        )

        cmd = [
            "melt",
            str(mlt_xml_path),
            "-consumer",
            f"avformat:{output_path}",
            "vcodec=libx264",
            "acodec=aac",
            "crf=18",
            "preset=medium",
            "pix_fmt=yuv420p",
        ]

        print_progress("Running melt command...")
        print_progress(f"Command: {' '.join(cmd)}")

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        print_progress(f"Video with Google Doc images created: {output_path}")
        print_progress(f"MLT XML saved for debugging: {mlt_xml_path}")
        return output_path

    def _create_mlt_xml_for_cutting(
        self,
        video_path: Path,
        adjusted_sentences: AdjustedSentences,
        output_mlt_path: Path,
    ) -> None:
        """
        Create MLT XML file to cut video based on adjusted sentences.
        Extracts clips from the original video and concatenates them.

        Args:
            video_path: Path to original video file (full resolution)
            adjusted_sentences: Sentences with timing info (to extract clips)
            output_mlt_path: Path where MLT XML file will be saved
        """
        # Get video properties
        props = get_video_properties(video_path)

        # Create root element and profile
        root = create_mlt_root_and_profile(props)

        # Calculate total output duration
        total_duration = sum(
            s.adjusted_end - s.adjusted_start for s in adjusted_sentences.sentences
        )
        total_frames = int(total_duration * props["fps"])
        total_timecode = frames_to_timecode(total_frames, props["fps"])

        # Add black background producer
        add_black_producer(root, total_timecode)

        # Create a chain for each sentence clip
        for i, sentence in enumerate(adjusted_sentences.sentences):
            clip_duration = sentence.adjusted_end - sentence.adjusted_start
            clip_frames = int(clip_duration * props["fps"])
            clip_timecode = frames_to_timecode(clip_frames, props["fps"])

            # Convert timestamps to frames for in/out points
            in_frame = int(sentence.adjusted_start * props["fps"])
            out_frame = int(sentence.adjusted_end * props["fps"]) - 1
            in_timecode = frames_to_timecode(in_frame, props["fps"])
            out_timecode = frames_to_timecode(out_frame, props["fps"])

            # Create chain for this clip
            chain = ET.SubElement(
                root,
                "chain",
                {
                    "id": f"chain_clip_{i}",
                    "in": in_timecode,
                    "out": out_timecode,
                },
            )

            # Add properties
            ET.SubElement(chain, "property", {"name": "length"}).text = clip_timecode
            ET.SubElement(chain, "property", {"name": "eof"}).text = "pause"
            ET.SubElement(chain, "property", {"name": "resource"}).text = str(
                video_path
            )
            ET.SubElement(
                chain, "property", {"name": "mlt_service"}
            ).text = "avformat-novalidate"
            ET.SubElement(chain, "property", {"name": "seekable"}).text = "1"
            ET.SubElement(chain, "property", {"name": "audio_index"}).text = "1"
            ET.SubElement(chain, "property", {"name": "video_index"}).text = "0"
            ET.SubElement(chain, "property", {"name": "mute_on_pause"}).text = "0"

            # Add hash and creation time
            import hashlib

            file_hash = hashlib.md5(f"{video_path}_{i}".encode()).hexdigest()
            ET.SubElement(chain, "property", {"name": "shotcut:hash"}).text = file_hash

            from datetime import datetime

            creation_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            ET.SubElement(
                chain, "property", {"name": "creation_time"}
            ).text = creation_time

            ET.SubElement(chain, "property", {"name": "ignore_points"}).text = "0"
            ET.SubElement(
                chain, "property", {"name": "shotcut:caption"}
            ).text = f"Clip {i + 1}"
            ET.SubElement(chain, "property", {"name": "xml"}).text = "was here"

        # Create playlist with all clips concatenated
        playlist = ET.SubElement(root, "playlist", {"id": "playlist0"})
        ET.SubElement(playlist, "property", {"name": "shotcut:video"}).text = "1"
        ET.SubElement(playlist, "property", {"name": "shotcut:name"}).text = "V1"

        # Add all clips to the playlist
        for i, sentence in enumerate(adjusted_sentences.sentences):
            clip_duration = sentence.adjusted_end - sentence.adjusted_start
            clip_frames = int(clip_duration * props["fps"])
            clip_timecode = frames_to_timecode(clip_frames - 1, props["fps"])

            ET.SubElement(
                playlist,
                "entry",
                {
                    "producer": f"chain_clip_{i}",
                    "in": "00:00:00.000",
                    "out": clip_timecode,
                },
            )

        # Create main tractor
        tractor = ET.SubElement(
            root,
            "tractor",
            {
                "id": "tractor0",
                "title": "Shotcut version 22.12.21",
                "in": "00:00:00.000",
                "out": total_timecode,
            },
        )

        # Add Shotcut properties
        ET.SubElement(tractor, "property", {"name": "shotcut"}).text = "1"
        ET.SubElement(
            tractor, "property", {"name": "shotcut:projectAudioChannels"}
        ).text = "2"
        ET.SubElement(tractor, "property", {"name": "shotcut:projectFolder"}).text = "0"

        # Add track
        ET.SubElement(tractor, "track", {"producer": "playlist0"})

        # Save XML
        save_pretty_xml(root, output_mlt_path)

        print_progress(f"Created MLT XML file for cutting: {output_mlt_path}")

    def create_full_res_cut_video(
        self,
        base_name: str,
        adjusted_sentences: AdjustedSentences,
        force: bool = False,
    ) -> Path:
        """
        Create full resolution cut video using MLT XML and melt command.
        Cuts the original full resolution video based on adjusted sentences.

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences with timestamps
            force: If True, regenerate even if file exists

        Returns:
            Path to full resolution cut video file

        Raises:
            FileNotFoundError: If original video doesn't exist
            RuntimeError: If melt command fails
        """
        input_path = get_input_video_path(base_name)
        output_path = get_full_res_cut_video_path(base_name)
        mlt_xml_path = get_full_res_cut_mlt_path(base_name)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Original video not found: {input_path}. Cannot create full resolution cut."
            )

        if output_path.exists() and not force:
            print_progress(f"Full resolution cut video already exists: {output_path}")
            return output_path

        print_progress(f"Creating full resolution cut video from: {input_path.name}...")
        print_progress(f"Cutting {len(adjusted_sentences.sentences)} segments")

        # Create MLT XML file (saved for debugging)
        self._create_mlt_xml_for_cutting(
            input_path,
            adjusted_sentences,
            mlt_xml_path,
        )

        cmd = [
            "melt",
            str(mlt_xml_path),
            "-consumer",
            f"avformat:{output_path}",
            "vcodec=libx264",
            "acodec=aac",
            "crf=18",
            "preset=medium",
            "pix_fmt=yuv420p",
        ]

        print_progress("Running melt command...")
        print_progress(f"Command: {' '.join(cmd)}")

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        print_progress(f"Full resolution cut video created: {output_path}")
        print_progress(f"MLT XML saved for debugging: {mlt_xml_path}")
        return output_path

    def _create_mlt_xml_for_cutting_with_images(
        self,
        video_path: Path,
        adjusted_sentences: AdjustedSentences,
        image_placements: GoogleDocImagePlacements,
        output_mlt_path: Path,
    ) -> None:
        """
        Create MLT XML file to cut video AND add image overlays in a single pass.
        Combines cutting based on adjusted sentences with image overlay based on placements.

        Args:
            video_path: Path to original video file (full resolution)
            adjusted_sentences: Sentences with timing info (to extract clips)
            image_placements: Google Doc image placements with sentence associations
            output_mlt_path: Path where MLT XML file will be saved
        """
        # Get video properties
        props = get_video_properties(video_path)

        # Create root element and profile
        root = create_mlt_root_and_profile(props)

        # Calculate total output duration (sum of all sentence clips)
        total_duration = sum(
            s.adjusted_end - s.adjusted_start for s in adjusted_sentences.sentences
        )
        total_frames = int(total_duration * props["fps"])
        total_timecode = frames_to_timecode(total_frames, props["fps"])

        # Add black background producer
        add_black_producer(root, total_timecode)

        # Create a chain for each sentence clip (cutting)
        for i, sentence in enumerate(adjusted_sentences.sentences):
            clip_duration = sentence.adjusted_end - sentence.adjusted_start
            clip_frames = int(clip_duration * props["fps"])
            clip_timecode = frames_to_timecode(clip_frames, props["fps"])

            # Convert timestamps to frames for in/out points
            in_frame = int(sentence.adjusted_start * props["fps"])
            out_frame = int(sentence.adjusted_end * props["fps"]) - 1
            in_timecode = frames_to_timecode(in_frame, props["fps"])
            out_timecode = frames_to_timecode(out_frame, props["fps"])

            # Create chain for this clip
            chain = ET.SubElement(
                root,
                "chain",
                {
                    "id": f"chain_clip_{i}",
                    "in": in_timecode,
                    "out": out_timecode,
                },
            )

            # Add properties
            ET.SubElement(chain, "property", {"name": "length"}).text = clip_timecode
            ET.SubElement(chain, "property", {"name": "eof"}).text = "pause"
            ET.SubElement(chain, "property", {"name": "resource"}).text = str(
                video_path
            )
            ET.SubElement(
                chain, "property", {"name": "mlt_service"}
            ).text = "avformat-novalidate"
            ET.SubElement(chain, "property", {"name": "seekable"}).text = "1"
            ET.SubElement(chain, "property", {"name": "audio_index"}).text = "1"
            ET.SubElement(chain, "property", {"name": "video_index"}).text = "0"
            ET.SubElement(chain, "property", {"name": "mute_on_pause"}).text = "0"

            # Add hash and creation time
            import hashlib

            file_hash = hashlib.md5(f"{video_path}_{i}".encode()).hexdigest()
            ET.SubElement(chain, "property", {"name": "shotcut:hash"}).text = file_hash

            from datetime import datetime

            creation_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            ET.SubElement(
                chain, "property", {"name": "creation_time"}
            ).text = creation_time

            ET.SubElement(chain, "property", {"name": "ignore_points"}).text = "0"
            ET.SubElement(
                chain, "property", {"name": "shotcut:caption"}
            ).text = f"Clip {i + 1}"
            ET.SubElement(chain, "property", {"name": "xml"}).text = "was here"

        # Add image producers
        for i, placement in enumerate(image_placements.placements):
            img_path = Path(placement.filepath)
            add_image_producer(root, i, img_path)

        # Build sentence timeline for image timing (relative to cut video)
        sentence_timeline = self._build_sentence_timeline(adjusted_sentences)

        # Calculate image timings based on cut video timeline
        image_timings = self._calculate_google_doc_image_timings(
            image_placements, sentence_timeline, props["fps"]
        )

        # Create video playlist with all clips concatenated (V1)
        video_playlist = ET.SubElement(root, "playlist", {"id": "playlist0"})
        ET.SubElement(video_playlist, "property", {"name": "shotcut:video"}).text = "1"
        ET.SubElement(video_playlist, "property", {"name": "shotcut:name"}).text = "V1"

        # Add all clips to the video playlist
        for i, sentence in enumerate(adjusted_sentences.sentences):
            clip_duration = sentence.adjusted_end - sentence.adjusted_start
            clip_frames = int(clip_duration * props["fps"])
            clip_timecode = frames_to_timecode(clip_frames - 1, props["fps"])

            ET.SubElement(
                video_playlist,
                "entry",
                {
                    "producer": f"chain_clip_{i}",
                    "in": "00:00:00.000",
                    "out": clip_timecode,
                },
            )

        # Create image overlay playlist with blanks (V2)
        self._create_overlay_playlist_with_dynamic_duration(
            root, image_timings, props["fps"]
        )

        # Calculate safe zone for image positioning
        safe_zone = calculate_safe_zone(
            props,
            IMAGE_SAFE_ZONE_TOP_PERCENT,
            IMAGE_SAFE_ZONE_BOTTOM_PERCENT,
            IMAGE_SAFE_ZONE_LEFT_PERCENT,
            IMAGE_SAFE_ZONE_RIGHT_PERCENT,
        )

        # Create main tractor with tracks and transitions
        tractor = ET.SubElement(
            root,
            "tractor",
            {
                "id": "tractor0",
                "title": "Shotcut version 22.12.21",
                "in": "00:00:00.000",
                "out": total_timecode,
            },
        )

        # Add Shotcut properties
        ET.SubElement(tractor, "property", {"name": "shotcut"}).text = "1"
        ET.SubElement(
            tractor, "property", {"name": "shotcut:projectAudioChannels"}
        ).text = "2"
        ET.SubElement(tractor, "property", {"name": "shotcut:projectFolder"}).text = "0"

        # Add tracks - Must use 3-track structure (background + video + images)
        # This matches the working downsampled workflow structure
        ET.SubElement(tractor, "track", {"producer": "black"})  # Track 0: Background
        ET.SubElement(tractor, "track", {"producer": "playlist0"})  # Track 1: Video
        ET.SubElement(tractor, "track", {"producer": "playlist1"})  # Track 2: Images

        # Add transitions matching 3-track structure
        # Mix transition for audio (track 0 to 1)
        add_mix_transition(tractor, "transition0", 0, 1)

        # Cairo blend transition (disabled, track 0 to 1)
        add_cairo_transition(tractor, "transition1", 0, 1)

        # Mix transition for audio (track 0 to 2)
        add_mix_transition(tractor, "transition2", 0, 2)

        # Composite transition for image overlay (track 1 to 2)
        add_composite_transition(tractor, "transition3", 1, 2, safe_zone)

        # Save XML
        save_pretty_xml(root, output_mlt_path)

        print_progress(
            f"Created MLT XML file for cutting with images: {output_mlt_path}"
        )

    def create_full_res_video_with_images_single_pass(
        self,
        base_name: str,
        adjusted_sentences: AdjustedSentences,
        image_placements: GoogleDocImagePlacements,
        force: bool = False,
    ) -> Path:
        """
        Create full resolution video with cuts AND image overlays in a single MLT pass.
        Cuts the original video based on adjusted sentences and adds images in one operation.

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences with timestamps
            image_placements: Google Doc image placements with sentence associations
            force: If True, regenerate even if file exists

        Returns:
            Path to full resolution video file with cuts and images

        Raises:
            FileNotFoundError: If original video or images don't exist
            RuntimeError: If melt command fails
        """
        input_path = get_input_video_path(base_name)
        output_path = get_full_res_with_images_video_path(base_name)
        mlt_xml_path = get_full_res_with_images_mlt_path(base_name)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Original video not found: {input_path}. Cannot create full resolution video."
            )

        if output_path.exists() and not force:
            print_progress(
                f"Full resolution video with cuts and images already exists: {output_path}"
            )
            return output_path

        print_progress(f"Creating full resolution video from: {input_path.name}...")
        print_progress(f"Cutting {len(adjusted_sentences.sentences)} segments")
        print_progress(f"Adding {len(image_placements.placements)} image overlays")

        # Verify all image files exist
        missing_images = []
        for placement in image_placements.placements:
            img_path = Path(placement.filepath)
            if not img_path.exists():
                missing_images.append(str(img_path))

        if missing_images:
            raise FileNotFoundError(
                "Missing image files:\n"
                + "\n".join(f"  - {img}" for img in missing_images)
            )

        # Create MLT XML file for cutting with images in one pass
        self._create_mlt_xml_for_cutting_with_images(
            input_path,
            adjusted_sentences,
            image_placements,
            mlt_xml_path,
        )

        cmd = [
            "melt",
            str(mlt_xml_path),
            "-consumer",
            f"avformat:{output_path}",
            "vcodec=libx264",
            "acodec=aac",
            "crf=18",
            "preset=medium",
            "pix_fmt=yuv420p",
        ]

        print_progress("Running melt command (single pass - cutting + images)...")
        print_progress(f"Command: {' '.join(cmd)}")

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        print_progress(
            f"Full resolution video with cuts and images created: {output_path}"
        )
        print_progress(f"MLT XML saved for debugging: {mlt_xml_path}")
        return output_path

    def create_full_res_video_with_images(
        self,
        base_name: str,
        adjusted_sentences: AdjustedSentences,
        image_placements: GoogleDocImagePlacements,
        force: bool = False,
    ) -> Path:
        """
        Create full resolution video with Google Doc image overlays using MLT XML and melt command.
        Uses the full resolution cut video as the base.
        Images are timed based on sentence duration, not fixed duration.

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences with timestamps
            image_placements: Google Doc image placements with sentence associations
            force: If True, regenerate even if file exists

        Returns:
            Path to full resolution video file with Google Doc images

        Raises:
            FileNotFoundError: If cut video or images don't exist
            RuntimeError: If melt command fails
        """
        # Use the full resolution cut video as the base
        input_path = get_full_res_cut_video_path(base_name)
        output_path = get_full_res_with_images_video_path(base_name)
        mlt_xml_path = get_full_res_with_images_mlt_path(base_name)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Full resolution cut video not found: {input_path}. Please run Stage 11 first."
            )

        if output_path.exists() and not force:
            print_progress(
                f"Full resolution video with Google Doc images already exists: {output_path}"
            )
            return output_path

        print_progress(
            f"Creating full resolution video with Google Doc images from: {input_path.name}..."
        )
        print_progress(f"Adding {len(image_placements.placements)} image overlays")

        # Verify all image files exist
        missing_images = []
        for placement in image_placements.placements:
            img_path = Path(placement.filepath)
            if not img_path.exists():
                missing_images.append(str(img_path))

        if missing_images:
            raise FileNotFoundError(
                "Missing image files:\n"
                + "\n".join(f"  - {img}" for img in missing_images)
            )

        # Create MLT XML file (saved for debugging)
        self._create_mlt_xml_with_google_doc_images(
            input_path,
            adjusted_sentences,
            image_placements,
            mlt_xml_path,
        )

        cmd = [
            "melt",
            str(mlt_xml_path),
            "-consumer",
            f"avformat:{output_path}",
            "vcodec=libx264",
            "acodec=aac",
            "crf=18",
            "preset=medium",
            "pix_fmt=yuv420p",
        ]

        print_progress("Running melt command...")
        print_progress(f"Command: {' '.join(cmd)}")

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        print_progress(
            f"Full resolution video with Google Doc images created: {output_path}"
        )
        print_progress(f"MLT XML saved for debugging: {mlt_xml_path}")
        return output_path
