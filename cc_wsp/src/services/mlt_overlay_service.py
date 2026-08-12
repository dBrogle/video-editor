"""MLT XML for overlay-only rendering on a pre-cut crossfaded video.

The v1 path cuts AND overlays in one MLT pass; the v2 path renders the cut
and audio crossfades upstream (crossfade_service.create_crossfaded_cut), so
MLT only needs to compose images on top of the finished video. No per-clip
chains, no in/out points — just one chain, one V1 entry, and overlay tracks.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from cc_wsp.src import util, util_v2
from cc_wsp.src.constants import (
    IMAGE_SAFE_ZONE_TOP_PERCENT, IMAGE_SAFE_ZONE_BOTTOM_PERCENT,
    IMAGE_SAFE_ZONE_LEFT_PERCENT, IMAGE_SAFE_ZONE_RIGHT_PERCENT,
    HIGH_RES_CRF, LOW_RES_CRF,
)
from cc_wsp.src.models import GoogleDocImagePlacements, Transcript, ZoomFilters
from cc_wsp.src.services.video.mlt_util import (
    frames_to_timecode, get_video_properties, calculate_safe_zone,
    create_mlt_root_and_profile, add_black_producer, add_video_chain,
    add_image_producer, add_zoom_filter_ranged, create_main_tractor, save_pretty_xml,
)
from cc_wsp.src.services.video.caption_service import (
    detect_title_text, extract_title_instructions,
)


def _build_sentence_timeline(transcript: Transcript) -> dict[str, dict]:
    return {
        str(i): {"start": s.start, "end": s.end}
        for i, s in enumerate(transcript.sentences, start=1)
    }


def _calc_image_timings(
    placements: GoogleDocImagePlacements,
    timeline: dict[str, dict],
    fps: float,
) -> list[tuple[int, int, int] | None]:
    out: list[tuple[int, int, int] | None] = []
    for i, p in enumerate(placements.placements):
        sid = str(p.sentence_index)
        if sid not in timeline:
            out.append(None)
            continue
        s = timeline[sid]
        dur = s["end"] - s["start"]
        img_start = s["start"] + dur * p.start_fraction
        img_end = s["start"] + dur * p.end_fraction
        sf = int(img_start * fps)
        ef = int(img_end * fps)
        if ef <= sf:
            ef = sf + 1
        out.append((sf, ef, i))
    return out


def _group_concurrent(placements: GoogleDocImagePlacements) -> dict[int, tuple[int, int]]:
    """Map placement_index -> (group_size, position_in_group).

    Two placements are concurrent if they share (sentence_index, start_fraction,
    end_fraction). Concurrent groups are tiled in a grid; sequential placements
    (group_size == 1) get the full safe zone.
    """
    groups: dict[tuple, list[int]] = {}
    for i, p in enumerate(placements.placements):
        key = (str(p.sentence_index), round(p.start_fraction, 4), round(p.end_fraction, 4))
        groups.setdefault(key, []).append(i)
    out: dict[int, tuple[int, int]] = {}
    for indices in groups.values():
        for pos, idx in enumerate(indices):
            out[idx] = (len(indices), pos)
    return out


def _strip_boxes(
    sizes: list[tuple[int, int]], zone_w: int, zone_h: int, axis: str
) -> list[tuple[float, float, float, float]]:
    """Tight-pack images in a single strip, scaled to a common cross-dimension so
    they all share the same width (vertical) or height (horizontal), butted edge
    to edge with NO gap between them, and the whole block is as large as the safe
    zone allows and centered in it.

    Returns one (x, y, w, h) paste box per image (floats; caller rounds).
    """
    if axis == "v":
        # All share width W; image i height = W * (h_i / w_i). Stack vertically.
        inv = sum(h / w for w, h in sizes)            # == sum(1 / aspect_i)
        common_w = min(zone_w, zone_h / inv) if inv else zone_w
        disp = [(common_w, common_w * h / w) for w, h in sizes]
        total_h = sum(d[1] for d in disp)
        x = (zone_w - common_w) / 2
        y = (zone_h - total_h) / 2
        boxes = []
        for dw, dh in disp:
            boxes.append((x, y, dw, dh))
            y += dh
        return boxes
    # axis == "h": all share height H; width i = H * (w_i / h_i). Side by side.
    s = sum(w / h for w, h in sizes)                  # == sum(aspect_i)
    common_h = min(zone_h, zone_w / s) if s else zone_h
    disp = [(common_h * w / h, common_h) for w, h in sizes]
    total_w = sum(d[0] for d in disp)
    x = (zone_w - total_w) / 2
    y = (zone_h - common_h) / 2
    boxes = []
    for dw, dh in disp:
        boxes.append((x, y, dw, dh))
        x += dw
    return boxes


def _grid_boxes(
    sizes: list[tuple[int, int]], zone_w: int, zone_h: int, rows: int, cols: int
) -> list[tuple[float, float, float, float]]:
    """Equal-cell grid: each image contain-fit and centered in its cell. Used as a
    fallback candidate for 4+ images where a single strip would get too small."""
    cell_w = zone_w / cols
    cell_h = zone_h / rows
    boxes = []
    for i, (w, h) in enumerate(sizes):
        r, c = divmod(i, cols)
        scale = min(cell_w / w, cell_h / h)
        dw, dh = w * scale, h * scale
        x = c * cell_w + (cell_w - dw) / 2
        y = r * cell_h + (cell_h - dh) / 2
        boxes.append((x, y, dw, dh))
    return boxes


def _area(boxes: list[tuple[float, float, float, float]]) -> float:
    return sum(w * h for _, _, w, h in boxes)


def _layout_boxes(
    sizes: list[tuple[int, int]], zone_w: int, zone_h: int, layout: str
) -> list[tuple[float, float, float, float]]:
    """Pick paste boxes for a concurrent group.

    - "vertical"   -> tight single-column strip (equal width, no gaps).
    - "horizontal" -> tight single-row strip (equal height, no gaps).
    - "tile"/auto  -> whichever arrangement shows the images LARGEST: the vertical
                      strip, the horizontal strip, or (for 4+) a near-square grid.
                      Strips are gap-free and keep the images the same width/height,
                      so a 2-image group always butts the pair together with no gap.
    """
    n = len(sizes)
    if n == 1:
        w, h = sizes[0]
        scale = min(zone_w / w, zone_h / h)
        dw, dh = w * scale, h * scale
        return [((zone_w - dw) / 2, (zone_h - dh) / 2, dw, dh)]
    if layout == "vertical":
        return _strip_boxes(sizes, zone_w, zone_h, "v")
    if layout == "horizontal":
        return _strip_boxes(sizes, zone_w, zone_h, "h")
    # tile / auto: choose the largest-area arrangement.
    candidates = [
        _strip_boxes(sizes, zone_w, zone_h, "v"),
        _strip_boxes(sizes, zone_w, zone_h, "h"),
    ]
    if n >= 4:
        import math
        cols = math.ceil(math.sqrt(n))
        rows = (n + cols - 1) // cols
        candidates.append(_grid_boxes(sizes, zone_w, zone_h, rows, cols))
    return max(candidates, key=_area)


def composite_concurrent_images(
    image_paths: list[Path],
    target_w: int,
    target_h: int,
    output_path: Path,
    *,
    layout: str = "tile",
) -> Path:
    """Composite N images that should appear at the same time into a single
    RGBA PNG sized target_w x target_h (the safe zone).

    This is the canonical helper for "show multiple images at once". Images are
    tight-packed (no gap between them) and scaled to a shared width (vertical) or
    height (horizontal) so they read as a balanced set and fill as much of the
    safe zone as possible. `layout`: "tile" (auto-pick the largest arrangement),
    "horizontal" (one row), or "vertical" (one column / stacked). No cropping;
    aspect is preserved.
    """
    sizes: list[tuple[int, int]] = []
    for src_path in image_paths:
        with Image.open(src_path) as img:
            sizes.append((img.width, img.height))
    boxes = _layout_boxes(sizes, target_w, target_h, layout)
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    for src_path, (x, y, w, h) in zip(image_paths, boxes):
        fw, fh = max(1, int(round(w))), max(1, int(round(h)))
        with Image.open(src_path) as img:
            img_fit = img.convert("RGBA").resize((fw, fh), Image.LANCZOS)
        canvas.paste(img_fit, (int(round(x)), int(round(y))), img_fit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "PNG")
    return output_path


def _collapse_concurrent(
    placements: GoogleDocImagePlacements,
    safe_zone: dict,
    cache_dir: Path,
) -> GoogleDocImagePlacements:
    """Replace concurrent placement groups with a single placement using a
    pre-composed tiled image. Sequential placements pass through unchanged."""
    groups: dict[tuple, list[int]] = {}
    for i, p in enumerate(placements.placements):
        key = (str(p.sentence_index), round(p.start_fraction, 4), round(p.end_fraction, 4))
        groups.setdefault(key, []).append(i)

    new_list = []
    seen = set()
    for i, p in enumerate(placements.placements):
        key = (str(p.sentence_index), round(p.start_fraction, 4), round(p.end_fraction, 4))
        if len(groups[key]) == 1:
            new_list.append(p)
            continue
        if key in seen:
            continue
        seen.add(key)
        idxs = groups[key]
        image_paths = [Path(placements.placements[j].filepath) for j in idxs]
        # The group renders with the first member's layout.
        layout = getattr(placements.placements[idxs[0]], "layout", "tile") or "tile"
        h = hashlib.md5("|".join(str(ip) for ip in image_paths).encode()).hexdigest()[:12]
        composite_path = (
            cache_dir
            / f"tile_s{p.sentence_index}_{key[1]}_{key[2]}_{layout}_"
              f"{safe_zone['width']}x{safe_zone['height']}_{h}.png"
        )
        if not composite_path.exists():
            composite_concurrent_images(
                image_paths, safe_zone["width"], safe_zone["height"], composite_path,
                layout=layout,
            )
        new_p = type(p)(
            filepath=str(composite_path),
            sentence_index=p.sentence_index,
            start_fraction=p.start_fraction,
            end_fraction=p.end_fraction,
        )
        new_list.append(new_p)
    return type(placements)(placements=new_list)


def _assign_to_tracks(timings: list[tuple[int, int, int] | None]) -> list[list[dict]]:
    events = [
        {"frame": t[0], "end_frame": t[1], "image_index": t[2], "duration": t[1] - t[0]}
        for t in timings if t
    ]
    events.sort(key=lambda e: e["frame"])
    tracks: list[list[dict]] = []
    ends: list[int] = []
    for e in events:
        placed = False
        for ti, end in enumerate(ends):
            if e["frame"] >= end:
                tracks[ti].append(e)
                ends[ti] = e["end_frame"]
                placed = True
                break
        if not placed:
            tracks.append([e])
            ends.append(e["end_frame"])
    return tracks


def _add_overlay_playlist(root, events, track_index: int, fps: float):
    pl = ET.SubElement(root, "playlist", {"id": f"playlist{track_index + 1}"})
    ET.SubElement(pl, "property", {"name": "shotcut:video"}).text = "1"
    ET.SubElement(pl, "property", {"name": "shotcut:name"}).text = f"V{track_index + 2}"
    cur = 0
    for e in events:
        if e["frame"] > cur:
            ET.SubElement(pl, "blank", {"length": frames_to_timecode(e["frame"] - cur, fps)})
            cur = e["frame"]
        # MLT in/out are inclusive frame positions, so an N-frame entry ends at
        # frame N-1. Emitting `out = N` made every entry one frame too long, and
        # since consecutive images sit back-to-back on a track with no blank to
        # resync, that error accumulated: by the end of a 45-image timeline the
        # overlays lagged their intended marks by ~1.5s.
        ET.SubElement(pl, "entry", {
            "producer": f"producer_{e['image_index']}",
            "in": "00:00:00.000",
            "out": frames_to_timecode(max(e["duration"] - 1, 0), fps),
        })
        cur += e["duration"]


def build_overlay_mlt(
    cut_video_path: Path,
    transcript: Transcript,
    placements: GoogleDocImagePlacements,
    output_mlt_path: Path,
    *,
    title_card_sentence_indices: set[str] | None = None,
    safe_zone_config: dict | None = None,
    zoom_filters: ZoomFilters | None = None,
) -> None:
    props = get_video_properties(cut_video_path)
    total_dur = util_v2.probe_duration(cut_video_path)
    total_frames = int(total_dur * props["fps"])
    total_tc = frames_to_timecode(total_frames, props["fps"])
    # Make the source chain and black producer a hair longer than the timeline
    # so the very last composited frame still has a real base-video frame under
    # the overlays (otherwise MLT returns black past the chain's declared length
    # and the last frame shows the held overlay image over black).
    producer_tc = frames_to_timecode(total_frames + 30, props["fps"])

    root = create_mlt_root_and_profile(props)
    add_black_producer(root, producer_tc)
    source_chain = add_video_chain(root, cut_video_path, producer_tc)

    # Per-sentence zoom emphasis: scope each affine filter to the v2 sentence's
    # frame range on the cut. MLT honors in/out on <filter> elements.
    if zoom_filters and zoom_filters.filters:
        timeline_z = _build_sentence_timeline(transcript)
        for zi, zf in enumerate(zoom_filters.filters):
            sid = str(zf.sentence_index)
            if sid not in timeline_z:
                print(f"  zoom: sentence {sid} not in v2 transcript — skipping")
                continue
            s = timeline_z[sid]
            f_in = int(s["start"] * props["fps"])
            f_out = max(f_in, int(s["end"] * props["fps"]) - 1)
            add_zoom_filter_ranged(
                source_chain, f"zoom{zi}",
                props["width"], props["height"],
                frames_to_timecode(f_in, props["fps"]),
                frames_to_timecode(f_out, props["fps"]),
                zoom_factor=zf.zoom_factor,
                x_offset=zf.x_offset, y_offset=zf.y_offset,
            )

    sz_cfg = safe_zone_config or {}
    safe_zone = calculate_safe_zone(
        props,
        sz_cfg.get("image_safe_zone_top", IMAGE_SAFE_ZONE_TOP_PERCENT),
        sz_cfg.get("image_safe_zone_bottom", IMAGE_SAFE_ZONE_BOTTOM_PERCENT),
        sz_cfg.get("image_safe_zone_left", IMAGE_SAFE_ZONE_LEFT_PERCENT),
        sz_cfg.get("image_safe_zone_right", IMAGE_SAFE_ZONE_RIGHT_PERCENT),
    )

    placements = _collapse_concurrent(
        placements, safe_zone, output_mlt_path.parent / ".composites",
    )

    title_indices = title_card_sentence_indices or set()

    def _compress(frac_top: float) -> dict:
        shifted = safe_zone["top"] + int(safe_zone["height"] * frac_top)
        return {**safe_zone, "top": shifted, "height": safe_zone["bottom"] - shifted}

    lowered = _compress(0.55) if title_indices else safe_zone
    extra_lowered = _compress(0.75) if title_indices else safe_zone

    seen: dict[str, int] = {}
    for i, p in enumerate(placements.placements):
        sid = str(p.sentence_index)
        if sid in title_indices:
            occ = seen.get(sid, 0)
            sz = extra_lowered if occ >= 1 else lowered
            seen[sid] = occ + 1
        else:
            sz = safe_zone
        add_image_producer(root, i, Path(p.filepath), sz, center_image=True)

    bg = ET.SubElement(root, "playlist", {"id": "background"})
    ET.SubElement(bg, "entry", {"producer": "black", "in": "00:00:00.000", "out": total_tc})

    v1 = ET.SubElement(root, "playlist", {"id": "playlist0"})
    ET.SubElement(v1, "property", {"name": "shotcut:video"}).text = "1"
    ET.SubElement(v1, "property", {"name": "shotcut:name"}).text = "V1"
    ET.SubElement(v1, "entry", {
        "producer": "chain_source_video",
        "in": "00:00:00.000",
        "out": total_tc,
    })

    timeline = _build_sentence_timeline(transcript)
    timings = _calc_image_timings(placements, timeline, props["fps"])
    tracks = _assign_to_tracks(timings)

    if not tracks:
        empty = ET.SubElement(root, "playlist", {"id": "playlist1"})
        ET.SubElement(empty, "property", {"name": "shotcut:video"}).text = "1"
        ET.SubElement(empty, "property", {"name": "shotcut:name"}).text = "V2"
        num_tracks = 1
    else:
        for ti, evs in enumerate(tracks):
            _add_overlay_playlist(root, evs, ti, props["fps"])
        num_tracks = len(tracks)

    create_main_tractor(root, total_tc, safe_zone, num_tracks)

    # The <mlt producer="main_bin"> attribute names what melt should render.
    # Without an actual main_bin element melt falls back to a behaviour that
    # over-renders ~0.5s of held-frame garbage past the timeline. Point it at
    # the timeline tractor clipped to the real length.
    main_bin = ET.SubElement(root, "playlist", {"id": "main_bin"})
    ET.SubElement(main_bin, "property", {"name": "xml_retain"}).text = "1"
    ET.SubElement(main_bin, "entry", {
        "producer": "tractor0", "in": "00:00:00.000", "out": total_tc,
    })

    save_pretty_xml(root, output_mlt_path)


def _detect_title_indices_v2(transcript: Transcript, name: str) -> set[str]:
    """Re-derive title-card sentence indices against the v2 transcription.

    extract_title_instructions matches script lines against AdjustedSentences
    by content-word overlap, so we wrap the Transcript as a pseudo
    AdjustedSentences to reuse that matcher unchanged.
    """
    try:
        script = util.load_google_doc_script(name)
    except Exception:
        return set()

    from cc_wsp.src.services.place_images_v2_service import _transcript_as_adjusted
    pseudo = _transcript_as_adjusted(transcript)
    instructions = extract_title_instructions(script, pseudo)
    return {ins.sentence_index for ins in instructions if ins.sentence_index}


def render_overlay(
    name: str,
    *,
    use_downsampled: bool,
    output_path: Path,
    crf: int,
    preset: str,
    force: bool,
) -> Path:
    if output_path.exists() and not force:
        print(f"{output_path.name} exists (use --force to regenerate)")
        return output_path

    cut_path = util_v2.cut_downsampled_path(name) if use_downsampled else util_v2.cut_path(name)
    if not cut_path.exists():
        raise FileNotFoundError(f"cut video missing: {cut_path}")

    transcript = Transcript(**json.loads(util_v2.transcription_v2_path(name).read_text()))
    placements_data = json.loads(util_v2.images_v2_path(name).read_text())
    placements = GoogleDocImagePlacements(**placements_data)

    title_indices = _detect_title_indices_v2(transcript, name)
    if title_indices:
        print(f"Title-card sentences: {sorted(title_indices)}")

    zoom_filters = util.load_zooms(name) if util.zooms_path(name).exists() else None
    if zoom_filters and zoom_filters.filters:
        print(f"Zoom emphasis on v2 sentences: "
              f"{[(zf.sentence_index, zf.zoom_factor) for zf in zoom_filters.filters]}")

    mlt_path = output_path.with_suffix(".mlt")
    build_overlay_mlt(
        cut_path, transcript, placements, mlt_path,
        title_card_sentence_indices=title_indices,
        zoom_filters=zoom_filters,
    )

    raw_out = output_path.with_name(output_path.stem + "_raw.mp4")
    cmd = [
        "melt", str(mlt_path),
        "-consumer", f"avformat:{raw_out}",
        "vcodec=libx264", "acodec=aac",
        f"crf={crf}", f"preset={preset}",
        "pix_fmt=yuv420p",
    ]
    print(f"Running melt: {output_path.name}")
    subprocess.run(cmd, check=True, capture_output=True)

    # MLT writes color_space=gbr; remux with -c copy to set BT.709 stream tags.
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw_out),
        "-c", "copy",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace", "bt709", "-color_range", "tv",
        str(output_path),
    ], check=True, capture_output=True)
    raw_out.unlink()

    print(f"Done: {output_path}")
    return output_path


def preview_v2(name: str, *, force: bool = False) -> Path:
    return render_overlay(
        name,
        use_downsampled=True,
        output_path=util_v2.preview_v2_path(name),
        crf=LOW_RES_CRF,
        preset="fast",
        force=force,
    )


def render_v2(name: str, *, force: bool = False) -> Path:
    return render_overlay(
        name,
        use_downsampled=False,
        output_path=util_v2.final_v2_path(name),
        crf=HIGH_RES_CRF,
        preset="medium",
        force=force,
    )
