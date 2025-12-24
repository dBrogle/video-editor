"""
Utility functions for MLT XML generation.
Independent helper functions that don't require class state.
"""

import subprocess
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom
from datetime import datetime
import hashlib
from PIL import Image


def frames_to_timecode(frames: int, fps: float) -> str:
    """
    Convert frames to MLT timecode format (HH:MM:SS.mmm).

    Args:
        frames: Number of frames
        fps: Frames per second

    Returns:
        Timecode string in format HH:MM:SS.mmm
    """
    total_seconds = frames / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def get_video_properties(video_path: Path) -> dict:
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


def calculate_safe_zone(
    props: dict,
    top_percent: float,
    bottom_percent: float,
    left_percent: float,
    right_percent: float,
) -> dict:
    """
    Calculate image safe zone dimensions and position in both pixels and percentages.

    Args:
        props: Video properties dictionary with width and height
        top_percent: Top margin as fraction (e.g., 0.1 for 10%)
        bottom_percent: Bottom edge as fraction (e.g., 0.9 for 90%)
        left_percent: Left margin as fraction (e.g., 0.1 for 10%)
        right_percent: Right edge as fraction (e.g., 0.9 for 90%)

    Returns:
        Dictionary with safe zone dimensions and position (pixels and percentages)
    """
    safe_zone_top = int(props["height"] * top_percent)
    safe_zone_bottom = int(props["height"] * bottom_percent)
    safe_zone_left = int(props["width"] * left_percent)
    safe_zone_right = int(props["width"] * right_percent)

    safe_zone_width = safe_zone_right - safe_zone_left
    safe_zone_height = safe_zone_bottom - safe_zone_top

    return {
        "top": safe_zone_top,
        "bottom": safe_zone_bottom,
        "left": safe_zone_left,
        "right": safe_zone_right,
        "width": safe_zone_width,
        "height": safe_zone_height,
        # Percentages for affine filter
        "top_percent": (safe_zone_top / props["height"]) * 100,
        "left_percent": (safe_zone_left / props["width"]) * 100,
        "width_percent": (safe_zone_width / props["width"]) * 100,
        "height_percent": (safe_zone_height / props["height"]) * 100,
    }


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """
    Get the dimensions of an image file.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


def calculate_centered_geometry(image_path: Path, safe_zone: dict) -> str:
    """
    Calculate centered geometry for an image within the safe zone.

    If the image is smaller than the safe zone in either dimension,
    it will be centered within that dimension while maintaining aspect ratio.

    Args:
        image_path: Path to the image file
        safe_zone: Safe zone dictionary with dimensions

    Returns:
        Geometry string in format "x:y:widthxheight:opacity"
    """
    # Get image dimensions
    img_width, img_height = get_image_dimensions(image_path)

    # Calculate aspect ratios
    img_aspect = img_width / img_height
    safe_aspect = safe_zone["width"] / safe_zone["height"]

    # Fit image within safe zone while maintaining aspect ratio
    if img_aspect > safe_aspect:
        # Image is wider - fit to width
        fitted_width = safe_zone["width"]
        fitted_height = int(fitted_width / img_aspect)
    else:
        # Image is taller - fit to height
        fitted_height = safe_zone["height"]
        fitted_width = int(fitted_height * img_aspect)

    # Center the fitted image within the safe zone
    x_offset = safe_zone["left"] + (safe_zone["width"] - fitted_width) // 2
    y_offset = safe_zone["top"] + (safe_zone["height"] - fitted_height) // 2

    return f"{x_offset}:{y_offset}:{fitted_width}x{fitted_height}:100"


def create_mlt_root_and_profile(props: dict) -> ET.Element:
    """
    Create MLT XML root element with profile configuration.

    Args:
        props: Video properties dictionary with width, height, frame_rate_num, frame_rate_den

    Returns:
        Root ET.Element configured with MLT profile
    """
    root = ET.Element(
        "mlt",
        {
            "LC_NUMERIC": "C",
            "version": "7.13.0",
            "title": "Shotcut version 22.12.21",
            "producer": "main_bin",
        },
    )

    ET.SubElement(
        root,
        "profile",
        {
            "description": "automatic",
            "width": str(props["width"]),
            "height": str(props["height"]),
            "progressive": "1",
            "sample_aspect_num": "1",
            "sample_aspect_den": "1",
            "display_aspect_num": "17",
            "display_aspect_den": "30",
            "frame_rate_num": str(props["frame_rate_num"]),
            "frame_rate_den": str(props["frame_rate_den"]),
            "colorspace": "601",
        },
    )
    return root


def add_black_producer(root: ET.Element, total_timecode: str) -> None:
    """
    Add black color producer to MLT XML root.

    Args:
        root: MLT XML root element
        total_timecode: Total duration timecode
    """
    producer = ET.SubElement(
        root,
        "producer",
        {"id": "black", "in": "00:00:00.000", "out": total_timecode},
    )

    length_prop = ET.SubElement(producer, "property", {"name": "length"})
    length_prop.text = total_timecode

    eof_prop = ET.SubElement(producer, "property", {"name": "eof"})
    eof_prop.text = "pause"

    resource_prop = ET.SubElement(producer, "property", {"name": "resource"})
    resource_prop.text = "0"

    aspect_prop = ET.SubElement(producer, "property", {"name": "aspect_ratio"})
    aspect_prop.text = "1"

    service_prop = ET.SubElement(producer, "property", {"name": "mlt_service"})
    service_prop.text = "color"

    image_format_prop = ET.SubElement(
        producer, "property", {"name": "mlt_image_format"}
    )
    image_format_prop.text = "rgba"

    test_audio_prop = ET.SubElement(producer, "property", {"name": "set.test_audio"})
    test_audio_prop.text = "0"


def add_video_chain(root: ET.Element, video_path: Path, total_timecode: str) -> None:
    """
    Add video chain (not simple producer) to MLT XML root.
    Shotcut uses chains to wrap media files for filters/effects.

    Args:
        root: MLT XML root element
        video_path: Path to video file
        total_timecode: Total duration timecode
    """
    chain = ET.SubElement(
        root, "chain", {"id": "chain_source_video", "out": total_timecode}
    )

    length_prop = ET.SubElement(chain, "property", {"name": "length"})
    length_prop.text = total_timecode

    eof_prop = ET.SubElement(chain, "property", {"name": "eof"})
    eof_prop.text = "pause"

    resource_prop = ET.SubElement(chain, "property", {"name": "resource"})
    resource_prop.text = str(video_path)

    service_prop = ET.SubElement(chain, "property", {"name": "mlt_service"})
    service_prop.text = "avformat-novalidate"

    seekable_prop = ET.SubElement(chain, "property", {"name": "seekable"})
    seekable_prop.text = "1"

    audio_index_prop = ET.SubElement(chain, "property", {"name": "audio_index"})
    audio_index_prop.text = "1"

    video_index_prop = ET.SubElement(chain, "property", {"name": "video_index"})
    video_index_prop.text = "0"

    mute_prop = ET.SubElement(chain, "property", {"name": "mute_on_pause"})
    mute_prop.text = "0"

    # Add creation time (current time in ISO format)
    creation_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    creation_time_prop = ET.SubElement(chain, "property", {"name": "creation_time"})
    creation_time_prop.text = creation_time

    # Add hash (simple hash based on filename)
    file_hash = hashlib.md5(str(video_path).encode()).hexdigest()
    hash_prop = ET.SubElement(chain, "property", {"name": "shotcut:hash"})
    hash_prop.text = file_hash

    ignore_points_prop = ET.SubElement(chain, "property", {"name": "ignore_points"})
    ignore_points_prop.text = "0"

    caption_prop = ET.SubElement(chain, "property", {"name": "shotcut:caption"})
    caption_prop.text = video_path.name

    xml_prop = ET.SubElement(chain, "property", {"name": "xml"})
    xml_prop.text = "was here"


def add_image_producer(
    root: ET.Element,
    image_index: int,
    image_path: Path,
    safe_zone: dict | None = None,
    center_image: bool = True,
    use_prescaled: bool = False,
) -> None:
    """
    Add image producer to MLT XML root with optional centering.

    Args:
        root: MLT XML root element
        image_index: Index of the image
        image_path: Path to image file (can be pre-scaled)
        safe_zone: Safe zone dimensions for centering (optional)
        center_image: If True and safe_zone provided, center image within safe zone
        use_prescaled: If True, assumes image is already scaled to safe zone size
    """
    # Images have a very long duration (4 hours) as they're static
    producer = ET.SubElement(
        root,
        "producer",
        {
            "id": f"producer_{image_index}",
            "in": "00:00:00.000",
            "out": "03:59:59.987",
        },
    )

    length_prop = ET.SubElement(producer, "property", {"name": "length"})
    length_prop.text = "04:00:00.000"

    eof_prop = ET.SubElement(producer, "property", {"name": "eof"})
    eof_prop.text = "pause"

    resource_prop = ET.SubElement(producer, "property", {"name": "resource"})
    resource_prop.text = str(image_path)

    ttl_prop = ET.SubElement(producer, "property", {"name": "ttl"})
    ttl_prop.text = "1"

    aspect_prop = ET.SubElement(producer, "property", {"name": "aspect_ratio"})
    aspect_prop.text = "1"

    progressive_prop = ET.SubElement(producer, "property", {"name": "progressive"})
    progressive_prop.text = "1"

    seekable_prop = ET.SubElement(producer, "property", {"name": "seekable"})
    seekable_prop.text = "1"

    format_prop = ET.SubElement(producer, "property", {"name": "format"})
    format_prop.text = "1"

    service_prop = ET.SubElement(producer, "property", {"name": "mlt_service"})
    service_prop.text = "qimage"

    # Add hash
    file_hash = hashlib.md5(str(image_path).encode()).hexdigest()
    hash_prop = ET.SubElement(producer, "property", {"name": "shotcut:hash"})
    hash_prop.text = file_hash

    # Add creation time
    creation_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    creation_time_prop = ET.SubElement(producer, "property", {"name": "creation_time"})
    creation_time_prop.text = creation_time

    # Add positioning filter if requested
    if center_image and safe_zone:
        if use_prescaled:
            # Image is already scaled to safe zone size, just position it
            x_offset = safe_zone["left"]
            y_offset = safe_zone["top"]

            # Add affine filter for positioning only (no scaling needed)
            affine_filter = ET.SubElement(producer, "filter")
            ET.SubElement(
                affine_filter, "property", {"name": "mlt_service"}
            ).text = "affine"

            # Geometry: x:y:widthxheight:opacity
            geometry = (
                f"{x_offset}:{y_offset}:{safe_zone['width']}x{safe_zone['height']}:100"
            )
            ET.SubElement(
                affine_filter, "property", {"name": "geometry"}
            ).text = geometry
            ET.SubElement(
                affine_filter, "property", {"name": "transition.fill"}
            ).text = "0"
            ET.SubElement(
                affine_filter, "property", {"name": "transition.distort"}
            ).text = "0"
        else:
            # Image needs to be scaled by MLT
            # Get image dimensions
            img_width, img_height = get_image_dimensions(image_path)

            # Scale image to FILL the entire safe zone (cover mode - may crop)
            # Scale to whichever dimension requires MORE scaling
            width_scale = safe_zone["width"] / img_width
            height_scale = safe_zone["height"] / img_height

            # Use the LARGER scale factor to ensure the image fills the entire safe zone
            # Example: 100x100 image in 200x300 safe zone -> scale by 3.0 -> 300x300 (fills height, crops width)
            # This makes images more prominent but may crop parts that extend beyond the safe zone
            scale_factor = max(width_scale, height_scale)

            # Calculate scaled dimensions
            scaled_width = int(img_width * scale_factor)
            scaled_height = int(img_height * scale_factor)

            # Calculate centering offsets within safe zone
            x_offset = safe_zone["left"] + (safe_zone["width"] - scaled_width) // 2
            y_offset = safe_zone["top"] + (safe_zone["height"] - scaled_height) // 2

            # Add affine filter for positioning and scaling
            affine_filter = ET.SubElement(producer, "filter")
            ET.SubElement(
                affine_filter, "property", {"name": "mlt_service"}
            ).text = "affine"

            # Geometry: x:y:widthxheight:opacity
            geometry = f"{x_offset}:{y_offset}:{scaled_width}x{scaled_height}:100"
            ET.SubElement(
                affine_filter, "property", {"name": "geometry"}
            ).text = geometry
            ET.SubElement(
                affine_filter, "property", {"name": "transition.fill"}
            ).text = "0"
            ET.SubElement(
                affine_filter, "property", {"name": "transition.distort"}
            ).text = "0"
            ET.SubElement(
                affine_filter, "property", {"name": "transition.halign"}
            ).text = "center"
            ET.SubElement(
                affine_filter, "property", {"name": "transition.valign"}
            ).text = "middle"

    xml_prop = ET.SubElement(producer, "property", {"name": "xml"})
    xml_prop.text = "was here"


def add_mix_transition(
    tractor: ET.Element,
    transition_id: str,
    a_track: int,
    b_track: int,
) -> None:
    """
    Add mix transition between two tracks.

    Args:
        tractor: Tractor element
        transition_id: ID for the transition
        a_track: Source track index
        b_track: Destination track index
    """
    mix_transition = ET.SubElement(tractor, "transition", {"id": transition_id})
    ET.SubElement(mix_transition, "property", {"name": "a_track"}).text = str(a_track)
    ET.SubElement(mix_transition, "property", {"name": "b_track"}).text = str(b_track)
    ET.SubElement(mix_transition, "property", {"name": "mlt_service"}).text = "mix"
    ET.SubElement(mix_transition, "property", {"name": "always_active"}).text = "1"
    ET.SubElement(mix_transition, "property", {"name": "sum"}).text = "1"


def add_cairo_transition(
    tractor: ET.Element,
    transition_id: str,
    a_track: int,
    b_track: int,
) -> None:
    """
    Add disabled Cairo blend transition.

    Args:
        tractor: Tractor element
        transition_id: ID for the transition
        a_track: Source track index
        b_track: Destination track index
    """
    cairo_transition = ET.SubElement(tractor, "transition", {"id": transition_id})
    ET.SubElement(cairo_transition, "property", {"name": "a_track"}).text = str(a_track)
    ET.SubElement(cairo_transition, "property", {"name": "b_track"}).text = str(b_track)
    ET.SubElement(cairo_transition, "property", {"name": "version"}).text = "0.1"
    ET.SubElement(
        cairo_transition, "property", {"name": "mlt_service"}
    ).text = "frei0r.cairoblend"
    ET.SubElement(cairo_transition, "property", {"name": "threads"}).text = "0"
    ET.SubElement(cairo_transition, "property", {"name": "disable"}).text = "1"


def add_composite_transition(
    tractor: ET.Element,
    transition_id: str,
    a_track: int,
    b_track: int,
    safe_zone: dict,
    center_images: bool = True,
) -> None:
    """
    Add composite transition for overlay positioning.

    Args:
        tractor: Tractor element
        transition_id: ID for the transition
        a_track: Source track index
        b_track: Destination track index
        safe_zone: Safe zone dimensions for geometry
        center_images: If True, images will be centered within safe zone (default: True)
    """
    # Use safe zone geometry as default
    # Note: Individual image centering is handled per-producer with affine filters
    geometry = (
        f"{safe_zone['left']}:{safe_zone['top']}:"
        f"{safe_zone['width']}x{safe_zone['height']}:100"
    )
    composite_transition = ET.SubElement(tractor, "transition", {"id": transition_id})
    ET.SubElement(composite_transition, "property", {"name": "a_track"}).text = str(
        a_track
    )
    ET.SubElement(composite_transition, "property", {"name": "b_track"}).text = str(
        b_track
    )
    ET.SubElement(
        composite_transition, "property", {"name": "mlt_service"}
    ).text = "composite"
    ET.SubElement(
        composite_transition, "property", {"name": "geometry"}
    ).text = geometry
    # When center_images is True, we use fill=0 to prevent stretching
    # and let affine filters handle centering per image
    fill_value = "0" if center_images else "1"
    ET.SubElement(composite_transition, "property", {"name": "fill"}).text = fill_value
    ET.SubElement(composite_transition, "property", {"name": "distort"}).text = "0"
    ET.SubElement(composite_transition, "property", {"name": "operator"}).text = "over"
    ET.SubElement(composite_transition, "property", {"name": "halign"}).text = "center"
    ET.SubElement(composite_transition, "property", {"name": "valign"}).text = "middle"


def create_base_playlists(
    root: ET.Element,
    total_timecode: str,
) -> None:
    """
    Create background and video playlists.

    Args:
        root: MLT XML root element
        total_timecode: Total duration timecode
    """
    # Background playlist
    background_playlist = ET.SubElement(root, "playlist", {"id": "background"})
    ET.SubElement(
        background_playlist,
        "entry",
        {"producer": "black", "in": "00:00:00.000", "out": total_timecode},
    )

    # Video playlist
    video_playlist = ET.SubElement(root, "playlist", {"id": "playlist0"})
    ET.SubElement(video_playlist, "property", {"name": "shotcut:video"}).text = "1"
    ET.SubElement(video_playlist, "property", {"name": "shotcut:name"}).text = "V1"
    ET.SubElement(
        video_playlist,
        "entry",
        {
            "producer": "chain_source_video",
            "in": "00:00:00.000",
            "out": total_timecode,
        },
    )


def create_main_tractor(
    root: ET.Element,
    total_timecode: str,
    safe_zone: dict,
) -> None:
    """
    Create main tractor with all tracks and transitions.

    Args:
        root: MLT XML root element
        total_timecode: Total duration timecode
        safe_zone: Safe zone dimensions for overlay positioning
    """
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

    # Add tracks
    ET.SubElement(tractor, "track", {"producer": "background"})
    ET.SubElement(tractor, "track", {"producer": "playlist0"})
    ET.SubElement(tractor, "track", {"producer": "playlist1"})

    # Transitions
    add_mix_transition(tractor, "transition0", 0, 1)
    add_cairo_transition(tractor, "transition1", 0, 1)
    add_mix_transition(tractor, "transition2", 0, 2)
    add_composite_transition(tractor, "transition3", 1, 2, safe_zone)


def save_pretty_xml(root: ET.Element, output_path: Path) -> None:
    """
    Pretty print and save XML to file.

    Args:
        root: XML root element
        output_path: Path where XML file will be saved
    """
    xml_string = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Remove extra blank lines
    lines = [line for line in pretty_xml.split("\n") if line.strip()]
    pretty_xml = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


def get_video_rotation(video_path: Path) -> tuple[int, int, int, int, int]:
    """
    Get video rotation metadata and dimensions using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height, fps_num, fps_den, rotation_degrees)
        rotation_degrees will be 0 if no rotation metadata is found
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate:stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe_data = json.loads(result.stdout)

    stream = probe_data["streams"][0]
    width = stream["width"]
    height = stream["height"]
    fps_str = stream["r_frame_rate"]
    fps_num, fps_den = map(int, fps_str.split("/"))

    # Check for rotation in tags or side_data
    rotation = 0
    if "tags" in stream and "rotate" in stream["tags"]:
        rotation = int(stream["tags"]["rotate"])
    elif "side_data_list" in stream:
        for side_data in stream["side_data_list"]:
            if "rotation" in side_data:
                rotation = int(side_data["rotation"])
                break

    return width, height, fps_num, fps_den, rotation


def create_rotation_mlt_xml(
    source_path: Path,
    output_width: int,
    output_height: int,
    fps_num: int,
    fps_den: int,
    mlt_xml_path: Path,
) -> None:
    """
    Create MLT XML file for video rotation/conversion.

    This creates a simple MLT XML that reads a source video and outputs it
    with the specified dimensions and frame rate. MLT will automatically handle
    rotation based on the profile dimensions.

    Args:
        source_path: Path to source video file
        output_width: Output video width
        output_height: Output video height
        fps_num: Frame rate numerator
        fps_den: Frame rate denominator
        mlt_xml_path: Path where MLT XML will be saved
    """
    # Build MLT XML
    root = ET.Element("mlt")
    root.set("LC_NUMERIC", "C")
    root.set("version", "7.0.0")
    root.set("producer", "main_bin")

    # Create profile with correct output dimensions
    profile = ET.SubElement(root, "profile")
    profile.set("description", "Custom")
    profile.set("width", str(output_width))
    profile.set("height", str(output_height))
    profile.set("progressive", "1")
    profile.set("sample_aspect_num", "1")
    profile.set("sample_aspect_den", "1")
    profile.set("display_aspect_num", str(output_width))
    profile.set("display_aspect_den", str(output_height))
    profile.set("frame_rate_num", str(fps_num))
    profile.set("frame_rate_den", str(fps_den))
    profile.set("colorspace", "709")

    # Create producer for source video
    producer = ET.SubElement(root, "producer")
    producer.set("id", "producer0")
    prop = ET.SubElement(producer, "property")
    prop.set("name", "resource")
    prop.text = str(source_path)

    # Create playlist
    playlist = ET.SubElement(root, "playlist")
    playlist.set("id", "main_bin")
    entry = ET.SubElement(playlist, "entry")
    entry.set("producer", "producer0")
    entry.set("in", "0")
    entry.set("out", "9999999")

    # Create tractor
    tractor = ET.SubElement(root, "tractor")
    tractor.set("id", "tractor0")
    track = ET.SubElement(tractor, "track")
    track.set("producer", "main_bin")

    # Write XML file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(mlt_xml_path), encoding="utf-8", xml_declaration=True)
