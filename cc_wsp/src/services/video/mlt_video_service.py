"""
MLT-based video editing service.
Uses MLT XML files and the melt command-line tool for video processing.
"""

import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from cc_wsp.src.models import AdjustedSentences, ImagesMetadataFile, GoogleDocImagePlacements, ZoomFilters
from cc_wsp.src.util import (
    get_stage_11_with_google_doc_images_path,
    get_stage_11_mlt_xml_path,
    get_best_edited_video_path,
    get_full_res_with_images_video_path,
    get_full_res_with_images_mlt_path,
    print_progress,
)
from cc_wsp.src.constants import (
    IMAGE_SAFE_ZONE_TOP_PERCENT,
    IMAGE_SAFE_ZONE_BOTTOM_PERCENT,
    IMAGE_SAFE_ZONE_LEFT_PERCENT,
    IMAGE_SAFE_ZONE_RIGHT_PERCENT,
)
from cc_wsp.src.services.video.mlt_util import (
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
    add_zoom_filter,
)

HIGH_RES_CRF = 19
LOW_RES_CRF = 24


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
        from cc_wsp.src.constants import VIDEOS_DIR
        from cc_wsp.src.services.video.mlt_util import (
            get_video_rotation,
            create_rotation_mlt_xml,
        )

        # Look for source video files in order of preference
        folder = VIDEOS_DIR / base_name
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
            f"crf={HIGH_RES_CRF}",
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
        Calculate timing for Google Doc images based on sentence duration and fractional placement.

        Args:
            image_placements: Google Doc image placements with sentence associations
            sentence_timeline: Mapping of sentence IDs to timing info
            fps: Frames per second

        Returns:
            List of (start_frame, end_frame, image_index) tuples or None
        """
        image_timings: list[tuple[int, int, int] | None] = []

        for i, placement in enumerate(image_placements.placements):
            sent_id = placement.sentence_index
            if sent_id not in sentence_timeline:
                image_timings.append(None)
                continue

            times = sentence_timeline[sent_id]
            sentence_start = times["start"]
            sentence_end = times["end"]
            sentence_duration = sentence_end - sentence_start

            img_start = sentence_start + (sentence_duration * placement.start_fraction)
            img_end = sentence_start + (sentence_duration * placement.end_fraction)

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

    def _assign_images_to_tracks(
        self,
        image_timings: list[tuple[int, int, int] | None],
    ) -> list[list[dict]]:
        """
        Assign images to tracks so that overlapping images go on separate tracks.
        Uses a greedy algorithm: for each image (sorted by start frame), assign it
        to the first track where it doesn't overlap with the last event.

        Args:
            image_timings: List of (start_frame, end_frame, image_index) tuples or None

        Returns:
            List of tracks, where each track is a list of event dicts sorted by frame.
        """
        events = []
        for timing in image_timings:
            if timing:
                start_frame, end_frame, image_index = timing
                events.append({
                    "frame": start_frame,
                    "end_frame": end_frame,
                    "image_index": image_index,
                    "duration": end_frame - start_frame,
                })

        events.sort(key=lambda x: x["frame"])

        tracks: list[list[dict]] = []
        # track_ends[i] = end frame of the last event on track i
        track_ends: list[int] = []

        for event in events:
            placed = False
            for track_idx, track_end in enumerate(track_ends):
                if event["frame"] >= track_end:
                    tracks[track_idx].append(event)
                    track_ends[track_idx] = event["end_frame"]
                    placed = True
                    break
            if not placed:
                tracks.append([event])
                track_ends.append(event["end_frame"])

        return tracks

    def _create_overlay_playlist_for_track(
        self,
        root: ET.Element,
        track_events: list[dict],
        track_index: int,
        fps: float,
    ) -> ET.Element:
        """
        Create a single overlay playlist for one track of non-overlapping images.

        Args:
            root: MLT XML root element
            track_events: List of event dicts (sorted by frame) for this track
            track_index: 0-based track index (used for playlist ID naming)
            fps: Frames per second

        Returns:
            Playlist element
        """
        playlist_id = f"playlist{track_index + 1}"
        track_name = f"V{track_index + 2}"

        playlist = ET.SubElement(root, "playlist", {"id": playlist_id})
        ET.SubElement(playlist, "property", {"name": "shotcut:video"}).text = "1"
        ET.SubElement(playlist, "property", {"name": "shotcut:name"}).text = track_name

        current_playlist_frame = 0

        for event in track_events:
            event_frame = event["frame"]
            image_duration = event["duration"]

            if event_frame > current_playlist_frame:
                blank_frames = event_frame - current_playlist_frame
                blank_timecode = frames_to_timecode(blank_frames, fps)
                ET.SubElement(playlist, "blank", {"length": blank_timecode})
                current_playlist_frame += blank_frames

            entry_timecode = frames_to_timecode(image_duration, fps)
            ET.SubElement(
                playlist,
                "entry",
                {
                    "producer": f"producer_{event['image_index']}",
                    "in": "00:00:00.000",
                    "out": entry_timecode,
                },
            )

            current_playlist_frame += image_duration

        return playlist

    def _create_overlay_playlists_with_dynamic_duration(
        self,
        root: ET.Element,
        image_timings: list[tuple[int, int, int] | None],
        fps: float,
    ) -> int:
        """
        Create overlay playlists for image overlays, using multiple tracks
        when images overlap in time.

        Args:
            root: MLT XML root element
            image_timings: List of (start_frame, end_frame, image_index) tuples or None
            fps: Frames per second

        Returns:
            Number of overlay tracks created
        """
        tracks = self._assign_images_to_tracks(image_timings)

        if not tracks:
            # No images — create one empty overlay playlist for the tractor structure
            playlist = ET.SubElement(root, "playlist", {"id": "playlist1"})
            ET.SubElement(playlist, "property", {"name": "shotcut:video"}).text = "1"
            ET.SubElement(playlist, "property", {"name": "shotcut:name"}).text = "V2"
            return 1

        for track_idx, track_events in enumerate(tracks):
            self._create_overlay_playlist_for_track(root, track_events, track_idx, fps)

        return len(tracks)

    def _create_mlt_xml_with_google_doc_images(
        self,
        video_path: Path,
        adjusted_sentences: AdjustedSentences,
        image_placements: GoogleDocImagePlacements,
        output_mlt_path: Path,
        safe_zone_config: dict | None = None,
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
        sz = safe_zone_config or {}
        safe_zone = calculate_safe_zone(
            props,
            sz.get("image_safe_zone_top", IMAGE_SAFE_ZONE_TOP_PERCENT),
            sz.get("image_safe_zone_bottom", IMAGE_SAFE_ZONE_BOTTOM_PERCENT),
            sz.get("image_safe_zone_left", IMAGE_SAFE_ZONE_LEFT_PERCENT),
            sz.get("image_safe_zone_right", IMAGE_SAFE_ZONE_RIGHT_PERCENT),
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

        # Add image producers with centering
        for i, placement in enumerate(image_placements.placements):
            img_path = Path(placement.filepath)
            timing = image_timings[i]
            if timing:
                add_image_producer(root, i, img_path, safe_zone, center_image=True)

        # Create base playlists (background and video)
        create_base_playlists(root, total_timecode)

        # Create overlay playlists (multiple tracks if images overlap)
        num_overlay_tracks = self._create_overlay_playlists_with_dynamic_duration(
            root, image_timings, props["fps"]
        )

        # Create main tractor with tracks and transitions
        create_main_tractor(root, total_timecode, safe_zone, num_overlay_tracks)

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
        # Use the best available edited video (prefers s6_adjusted_sentences_video.mp4)
        input_path = get_best_edited_video_path(base_name)
        output_path = get_stage_11_with_google_doc_images_path(base_name)
        mlt_xml_path = get_stage_11_mlt_xml_path(base_name)

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
            f"crf={HIGH_RES_CRF}",
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

    def _create_mlt_xml_for_cutting_with_images(
        self,
        video_path: Path,
        adjusted_sentences: AdjustedSentences,
        image_placements: GoogleDocImagePlacements,
        output_mlt_path: Path,
        zoom_filters: ZoomFilters | None = None,
        safe_zone_config: dict | None = None,
        title_card_sentence_indices: set[str] | None = None,
    ) -> None:
        """
        Create MLT XML file to cut video AND add image overlays in a single pass.
        Combines cutting based on adjusted sentences with image overlay based on placements.

        Args:
            video_path: Path to original video file (full resolution)
            adjusted_sentences: Sentences with timing info (to extract clips)
            image_placements: Google Doc image placements with sentence associations
            output_mlt_path: Path where MLT XML file will be saved
            zoom_filters: Optional zoom/pan filters to apply to specific sentence clips
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

            # Apply zoom filter if one exists for this sentence
            if zoom_filters:
                for zf in zoom_filters.filters:
                    if zf.sentence_index == sentence.index:
                        add_zoom_filter(
                            chain,
                            f"zoom_filter_{i}",
                            props["width"],
                            props["height"],
                            zf.zoom_factor,
                            zf.x_offset,
                            zf.y_offset,
                        )
                        break

        # Calculate safe zone for image positioning (needed before adding image producers)
        sz = safe_zone_config or {}
        safe_zone = calculate_safe_zone(
            props,
            sz.get("image_safe_zone_top", IMAGE_SAFE_ZONE_TOP_PERCENT),
            sz.get("image_safe_zone_bottom", IMAGE_SAFE_ZONE_BOTTOM_PERCENT),
            sz.get("image_safe_zone_left", IMAGE_SAFE_ZONE_LEFT_PERCENT),
            sz.get("image_safe_zone_right", IMAGE_SAFE_ZONE_RIGHT_PERCENT),
        )

        # Derive compressed safe zones for sentences that carry a title card.
        # Title cards sit in the top portion of the safe zone, so push their
        # images into the lower portion. For sentences with multiple images
        # (e.g. a logo banner that follows a hero image), push the 2nd-and-
        # later images into an even lower sub-band so they visually sit
        # below the first image.
        title_indices = title_card_sentence_indices or set()

        def _compress(frac_top: float) -> dict:
            shifted_top = safe_zone["top"] + int(safe_zone["height"] * frac_top)
            return {
                **safe_zone,
                "top": shifted_top,
                "height": safe_zone["bottom"] - shifted_top,
            }

        lowered_safe_zone = _compress(0.55) if title_indices else safe_zone
        extra_lowered_safe_zone = _compress(0.75) if title_indices else safe_zone

        # Count images per sentence so we can tell which is the "first"
        seen_per_sentence: dict[str, int] = {}

        # Add image producers with centering
        for i, placement in enumerate(image_placements.placements):
            img_path = Path(placement.filepath)
            sid = str(placement.sentence_index)
            if sid in title_indices:
                occurrence = seen_per_sentence.get(sid, 0)
                sz = extra_lowered_safe_zone if occurrence >= 1 else lowered_safe_zone
                seen_per_sentence[sid] = occurrence + 1
            else:
                sz = safe_zone
            add_image_producer(root, i, img_path, sz, center_image=True)

        # Build sentence timeline for image timing (relative to cut video)
        sentence_timeline = self._build_sentence_timeline(adjusted_sentences)

        # Calculate image timings based on cut video timeline
        image_timings = self._calculate_google_doc_image_timings(
            image_placements, sentence_timeline, props["fps"]
        )

        # Create background playlist wrapping the black producer
        background_playlist = ET.SubElement(root, "playlist", {"id": "background"})
        ET.SubElement(
            background_playlist,
            "entry",
            {"producer": "black", "in": "00:00:00.000", "out": total_timecode},
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

        # Create image overlay playlists (multiple tracks if images overlap)
        num_overlay_tracks = self._create_overlay_playlists_with_dynamic_duration(
            root, image_timings, props["fps"]
        )

        # Create main tractor with tracks and transitions
        create_main_tractor(root, total_timecode, safe_zone, num_overlay_tracks)

        # Save XML
        save_pretty_xml(root, output_mlt_path)

        print_progress(
            f"Created MLT XML file for cutting with images: {output_mlt_path}"
        )

    def create_stream_final_video(
        self,
        base_name: str,
        adjusted_sentences: AdjustedSentences,
        force: bool = False,
    ) -> Path:
        """
        Create the final stream video with cuts only (no image overlays).
        Operates on the original video file at native resolution (horizontal).

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences with timestamps
            force: If True, regenerate even if file exists

        Returns:
            Path to final cut video

        Raises:
            FileNotFoundError: If input video doesn't exist
            RuntimeError: If melt command fails
        """
        from cc_wsp.src.util import (
            get_input_video_path,
            get_stream_final_video_path,
            get_stream_final_mlt_path,
        )

        input_path = get_input_video_path(base_name)
        output_path = get_stream_final_video_path(base_name)
        mlt_xml_path = get_stream_final_mlt_path(base_name)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        if output_path.exists() and not force:
            print_progress(f"Stream final video already exists: {output_path}")
            return output_path

        print_progress(f"Creating stream final video from: {input_path.name}...")
        print_progress(f"Cutting {len(adjusted_sentences.sentences)} sentences")

        # Reuse the existing cuts-only MLT XML builder
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
            f"crf={HIGH_RES_CRF}",
            "preset=medium",
            "pix_fmt=yuv420p",
        ]

        print_progress("Running melt command (stream final - cuts only)...")
        print_progress(f"Command: {' '.join(cmd)}")

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        print_progress(f"Stream final video created: {output_path}")
        print_progress(f"MLT XML saved for debugging: {mlt_xml_path}")
        return output_path

    def create_1080p_video_with_images(
        self,
        base_name: str,
        adjusted_sentences: AdjustedSentences,
        image_placements: GoogleDocImagePlacements,
        force: bool = False,
        zoom_filters: ZoomFilters | None = None,
        safe_zone_config: dict | None = None,
        title_card_sentence_indices: set[str] | None = None,
    ) -> Path:
        """
        Create 1080p video with cuts AND image overlays in a single MLT pass.
        Operates on the 1080p downsampled video (s10_1080p_downsample.mp4).

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences with timestamps
            image_placements: Google Doc image placements with sentence associations
            force: If True, regenerate even if file exists
            zoom_filters: Optional zoom/pan filters to apply to specific clips

        Returns:
            Path to 1080p video file with cuts and images

        Raises:
            FileNotFoundError: If 1080p video or images don't exist
            RuntimeError: If melt command fails
        """
        from cc_wsp.src.util import (
            get_1080p_downsample_video_path,
            get_1080p_with_images_video_path,
            get_1080p_with_images_mlt_path,
        )

        input_path = get_1080p_downsample_video_path(base_name)
        output_path = get_1080p_with_images_video_path(base_name)
        mlt_xml_path = get_1080p_with_images_mlt_path(base_name)

        if not input_path.exists():
            raise FileNotFoundError(
                f"1080p downsampled video not found: {input_path}. Run step 10 first."
            )

        if output_path.exists() and not force:
            print_progress(
                f"1080p video with cuts and images already exists: {output_path}"
            )
            return output_path

        print_progress(f"Creating 1080p video from: {input_path.name}...")
        print_progress(f"Cutting {len(adjusted_sentences.sentences)} sentences")
        print_progress(f"Adding {len(image_placements.placements)} image overlays")
        if zoom_filters and zoom_filters.filters:
            print_progress(f"Applying {len(zoom_filters.filters)} zoom filters")

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
            zoom_filters=zoom_filters,
            safe_zone_config=safe_zone_config,
            title_card_sentence_indices=title_card_sentence_indices,
        )

        cmd = [
            "melt",
            str(mlt_xml_path),
            "-consumer",
            f"avformat:{output_path}",
            "vcodec=libx264",
            "acodec=aac",
            f"crf={LOW_RES_CRF}",
            "preset=fast",
            "pix_fmt=yuv420p",
        ]

        print_progress("Running melt command (1080p - cutting + images)...")
        print_progress(f"Command: {' '.join(cmd)}")

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        print_progress(f"1080p video with cuts and images created: {output_path}")
        print_progress(f"MLT XML saved for debugging: {mlt_xml_path}")
        return output_path

    def create_full_res_video_with_images_single_pass(
        self,
        base_name: str,
        adjusted_sentences: AdjustedSentences,
        image_placements: GoogleDocImagePlacements,
        force: bool = False,
        zoom_filters: ZoomFilters | None = None,
        safe_zone_config: dict | None = None,
    ) -> Path:
        """
        Create full resolution video with cuts AND image overlays in a single MLT pass.
        Uses the 1080p downsampled video (from Step 10) as input for faster processing.

        Args:
            base_name: Base filename without extension
            adjusted_sentences: AdjustedSentences with timestamps
            image_placements: Google Doc image placements with sentence associations
            force: If True, regenerate even if file exists
            zoom_filters: Optional zoom/pan filters to apply to specific clips

        Returns:
            Path to full resolution video file with cuts and images

        Raises:
            FileNotFoundError: If 1080p downsampled video or images don't exist
            RuntimeError: If melt command fails
        """
        from cc_wsp.src.util import get_1080p_downsample_video_path

        input_path = get_1080p_downsample_video_path(base_name)
        output_path = get_full_res_with_images_video_path(base_name)
        mlt_xml_path = get_full_res_with_images_mlt_path(base_name)

        if not input_path.exists():
            raise FileNotFoundError(
                f"1080p downsampled video not found: {input_path}. Run step 10 first."
            )

        if output_path.exists() and not force:
            print_progress(
                f"Full resolution video with cuts and images already exists: {output_path}"
            )
            return output_path

        print_progress(
            f"Creating full resolution video from 1080p: {input_path.name}..."
        )
        print_progress(f"Cutting {len(adjusted_sentences.sentences)} sentences")
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
            zoom_filters=zoom_filters,
            safe_zone_config=safe_zone_config,
        )

        cmd = [
            "melt",
            str(mlt_xml_path),
            "-consumer",
            f"avformat:{output_path}",
            "vcodec=libx264",
            "acodec=aac",
            f"crf={HIGH_RES_CRF}",
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
