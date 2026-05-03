#!/usr/bin/env python3
"""
cc_wsp video editing tools for Claude Code.

Usage: python cc_wsp/tool.py <command> <video> [args...]

Commands (shorts):
  preprocess <video>                    Downsample, extract audio, unzip google doc, prescale images
  transcribe <video>                    Run Deepgram STT → transcription.json
  suggest-edits <video>                 LLM suggests sentence removals → sentences.json
  silence-detect <video>                Run silence detection on kept sentences → adjusted.json
  audio-levels <video> <start> <end>    Show audio RMS levels for a time range
  word-timestamps <video> <sentence>    Show word timestamps for a transcription sentence (1-based)
  split-sentence <video> <idx> <time>   Split a sentence in adjusted.json at a timestamp
  place-images <video>                  LLM places google doc images → images.json
  parse-doc <video>                     Parse google doc HTML → google_doc_script.json
  preview <video>                       Generate preview video from adjusted.json
  render <video>                        Final 1080p render with images
  show <video> <file>                   Print contents of a step file
  status <video>                        Show which step files exist

Commands (streams):
  stream-preprocess <video>             Downsample to 360p, extract audio
  stream-transcribe <video>             Run Deepgram STT → stream_transcription.json
  stream-edit <video>                   Iterative LLM editing to hit target duration
  stream-silence-detect <video>         Silence detection on kept sentences → stream_adjusted.json
  stream-preview <video>                Preview video from stream_adjusted.json
  stream-render <video>                 Final render (cuts only, native resolution)
  stream-show <video> <file>            Print stream step file contents
  stream-status <video>                 Show stream step file status
"""

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Add parent directory to path so we can import cc_wsp.src
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import librosa

from cc_wsp.src.constants import (
    VIDEOS_DIR, LOW_RES_HEIGHT, HD_1080P_HEIGHT, AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS, AUDIO_BITRATE, AUDIO_CODEC, VIDEO_CODEC, VIDEO_PRESET,
    LOW_RES_CRF, HIGH_RES_CRF,
)
from cc_wsp.src import util
from cc_wsp.src.models import (
    Transcript, AdjustedSentence, AdjustedSentences, LLMTranscriptSentence, WordTimestamp,
)


def _ffmpeg_has_filter(binary: str, name: str) -> bool:
    out = subprocess.run(
        [binary, "-hide_banner", "-filters"],
        capture_output=True, text=True, check=False,
    ).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def _get_tonemap_ffmpeg() -> str:
    """Return an ffmpeg binary path that has the zscale filter (for HDR tone-mapping).

    Prefers system ffmpeg if it has zscale; otherwise falls back to the
    imageio-ffmpeg bundled binary, which ships with libzimg on macOS.
    """
    if _ffmpeg_has_filter("ffmpeg", "zscale"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if _ffmpeg_has_filter(candidate, "zscale"):
            return candidate
    except Exception:
        pass
    return "ffmpeg"


def _build_scale_vf(input_path: Path, scale_expr: str) -> tuple[str, str]:
    """Build a -vf expression for scaling; tone-map HDR inputs to SDR BT.709.

    Returns (ffmpeg_binary, vf_expression). The binary may be swapped for
    imageio-ffmpeg's bundled build when tone-mapping is required.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=color_transfer,color_primaries",
         "-of", "default=nw=1:nk=1", str(input_path)],
        capture_output=True, text=True, check=True,
    )
    tokens = {t.strip().lower() for t in probe.stdout.split()}
    hdr_transfers = {"smpte2084", "arib-std-b67"}
    is_hdr = bool(tokens & hdr_transfers) or "bt2020" in tokens
    if not is_hdr:
        return "ffmpeg", scale_expr
    binary = _get_tonemap_ffmpeg()
    vf = (
        "zscale=t=linear:npl=100,format=gbrpf32le,"
        "zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
        "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
        + scale_expr
    )
    return binary, vf


def cmd_preprocess(args):
    name = args.video
    d = util.get_video_dir(name)
    input_path = util.get_input_video_path(name)

    # Downsample
    ds_path = util.downsampled_path(name)
    if ds_path.exists() and not args.force:
        print(f"Downsampled video exists: {ds_path.name}")
    else:
        print("Downsampling video...")
        binary, vf = _build_scale_vf(input_path, f"scale=-2:{LOW_RES_HEIGHT}")
        subprocess.run([
            binary, "-i", str(input_path), "-vf", vf,
            "-c:v", VIDEO_CODEC, "-preset", VIDEO_PRESET, "-crf", "28",
            "-c:a", "aac", "-b:a", "128k", "-y", str(ds_path),
        ], capture_output=True, check=True)
        print(f"Created: {ds_path.name}")

    # Extract audio
    a_path = util.audio_path(name)
    if a_path.exists() and not args.force:
        print(f"Audio exists: {a_path.name}")
    else:
        print("Extracting audio...")
        subprocess.run([
            "ffmpeg", "-i", str(input_path), "-vn", "-acodec", AUDIO_CODEC,
            "-ar", str(AUDIO_SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS),
            "-b:a", AUDIO_BITRATE, "-y", str(a_path),
        ], capture_output=True, check=True)
        print(f"Created: {a_path.name}")

    # Unzip google doc
    doc_dir = util.google_doc_dir(name)
    images_dir = util.google_doc_images_dir(name)
    zip_files = list(d.glob("*.zip"))
    if zip_files and (not doc_dir.exists() or args.force):
        print(f"Extracting google doc: {zip_files[0].name}")
        temp = d / "_temp_zip"
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        try:
            with zipfile.ZipFile(zip_files[0], 'r') as zf:
                zf.extractall(temp)
            html_files = list(temp.rglob("*.html"))
            img_folders = [x for x in temp.rglob("*") if x.is_dir() and x.name == "images"]
            if html_files and img_folders:
                doc_dir.mkdir(parents=True, exist_ok=True)
                html_dest = util.google_doc_html_path(name)
                if html_dest.exists():
                    html_dest.unlink()
                shutil.move(str(html_files[0]), str(html_dest))
                if images_dir.exists():
                    shutil.rmtree(images_dir)
                shutil.move(str(img_folders[0]), str(images_dir))
                print(f"Extracted: {html_dest.name} + images/")
        finally:
            if temp.exists():
                shutil.rmtree(temp)

    # Detect face position (used for caption/image placement)
    if not util.face_data_path(name).exists() or args.force:
        _run_face_detection(name)

    # Prescale images (uses face-derived safe zone if available)
    config = util.load_config(name)
    if images_dir.exists():
        _prescale_images(images_dir, config)

    print("Preprocessing complete.")


def _run_face_detection(name: str):
    from cc_wsp.src.services.video.face_detection import detect_face_position
    ds_path = util.downsampled_path(name)
    if not ds_path.exists():
        print("Skipping face detection: downsampled video not found.")
        return
    print("Detecting face position...")
    face = detect_face_position(ds_path, num_samples=20)
    if face is None:
        print("  No faces detected — falling back to default safe zone.")
        return
    util.save_face_data(name, face)
    print(
        f"  Face bbox (fracs): y=[{face.top_frac:.3f},{face.bottom_frac:.3f}] "
        f"x=[{face.left_frac:.3f},{face.right_frac:.3f}] "
        f"({face.samples_with_face}/{face.total_samples} samples)"
    )


def cmd_detect_face(args):
    _run_face_detection(args.video)


def _prescale_images(images_dir: Path, config: dict):
    from PIL import Image
    sz_w = int(1080 * (config["image_safe_zone_right"] - config["image_safe_zone_left"]))
    sz_h = int(HD_1080P_HEIGHT * (config["image_safe_zone_bottom"] - config["image_safe_zone_top"]))
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
            continue
        with Image.open(p) as img:
            w, h = img.size
            if w >= sz_w and h >= sz_h:
                continue
            scale = max(sz_w / w, sz_h / h)
            nw, nh = int(w * scale), int(h * scale)
            print(f"  Prescaling {p.name}: {w}x{h} -> {nw}x{nh}")
            scaled = img.resize((nw, nh), Image.LANCZOS)
            scaled.save(p)
            scaled.close()


def cmd_transcribe(args):
    name = args.video
    t_path = util.transcription_path(name)
    if t_path.exists() and not args.force:
        print(f"Transcription exists: {t_path.name}")
        return

    from cc_wsp.src.services.stt.deepgram import DeepgramSTTService
    a_path = util.audio_path(name)
    print(f"Transcribing {a_path.name}...")
    stt = DeepgramSTTService()
    transcript = stt.transcribe(a_path)
    util.save_transcription(name, transcript)
    print(f"Saved: {t_path.name} ({len(transcript.sentences)} sentences)")


def cmd_suggest_edits(args):
    name = args.video
    s_path = util.sentences_path(name)
    if s_path.exists() and not args.force:
        print(f"Sentences file exists: {s_path.name}")
        return

    transcript = util.load_transcription(name)

    # Load script outline if available
    script_text = None
    try:
        script = util.load_google_doc_script(name)
        script_text = "\n".join(line.text for line in script.lines if line.text)
    except FileNotFoundError:
        pass

    from cc_wsp.src.services.llm.openrouter import OpenRouterLLMService
    model_name = "Claude Sonnet 4.5" if not script_text else "Claude Sonnet 4.5 (with script)"
    print(f"Sending to LLM ({model_name}) for editing suggestions...")
    llm = OpenRouterLLMService()
    decision = llm.get_edits(transcript, script_text=script_text)

    result = util.convert_decision_to_result(decision, transcript)
    util.save_sentences(name, result)

    # Also save LLM reasoning
    reasoning_path = util.get_video_dir(name) / "edit_reasoning.json"
    util.save_json(reasoning_path, decision)

    kept = sum(1 for s in result.sentence_results.values() if s.keep)
    removed = sum(1 for s in result.sentence_results.values() if not s.keep)
    print(f"Saved: {s_path.name} ({kept} kept, {removed} removed)")
    print(f"LLM reasoning: {reasoning_path.name}")
    print(f"\nThoughts: {decision.thoughts[:200]}...")


def cmd_silence_detect(args):
    name = args.video
    adj_path = util.adjusted_path(name)
    if adj_path.exists() and not args.force:
        print(f"Adjusted sentences exist: {adj_path.name}")
        return

    transcript = util.load_transcription(name)
    sentences_result = util.load_sentences(name)
    a_path = util.audio_path(name)

    # Import the video service for silence detection
    # Temporarily patch constants to use cc_wsp paths
    from cc_wsp.src.services.video.video_service import VideoService
    svc = VideoService(VIDEOS_DIR)

    kept = [
        (i, s) for i, s in enumerate(transcript.sentences, 1)
        if sentences_result.sentence_results[str(i)].keep
    ]

    print(f"Running silence detection on {len(kept)} sentences...")
    adjusted_list = []
    for idx, sentence in kept:
        result = svc._get_adjusted_sentence(a_path, sentence, idx)
        adjusted_list.append(result)

    adj = AdjustedSentences(sentences=adjusted_list)
    util.save_adjusted(name, adj)
    print(f"Saved: {adj_path.name}")


def cmd_audio_levels(args):
    name, start, end = args.video, args.start, args.end
    resolution = args.resolution
    a_path = util.audio_path(name)

    sr = 22050
    audio, sr = librosa.load(str(a_path), sr=sr, mono=True, offset=start, duration=end - start)
    frame_length, hop_length = 512, 256
    tpf = hop_length / sr

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    speech_level = np.percentile(rms_db, 85)
    threshold = speech_level - 18

    print(f"Audio: {name} [{start:.3f} - {end:.3f}]")
    print(f"Speech level: {speech_level:.1f} dB, Threshold: {threshold:.1f} dB")
    print()

    frames_per_step = max(1, int(resolution / tpf))
    print(f"{'Time':>8s}  {'dB':>6s}  Bar")
    print("-" * 50)
    for i in range(0, len(rms_db), frames_per_step):
        chunk = rms_db[i:i + frames_per_step]
        db = float(np.mean(chunk))
        t = start + i * tpf
        bar_len = max(0, int((db + 80) / 2))
        marker = "#" if db > threshold else "."
        print(f"{t:8.3f}  {db:6.1f}  {marker * bar_len}")


def cmd_word_timestamps(args):
    name, sent_idx = args.video, args.sentence
    transcript = util.load_transcription(name)

    if sent_idx < 1 or sent_idx > len(transcript.sentences):
        print(f"Error: sentence {sent_idx} out of range (1-{len(transcript.sentences)})")
        return

    s = transcript.sentences[sent_idx - 1]
    print(f"Sentence {sent_idx}: \"{s.sentence}\"")
    print(f"Time: {s.start:.3f} - {s.end:.3f}")
    if not s.words:
        print("No word timestamps available.")
        return

    print(f"\n{'Word':<20s}  {'Start':>8s}  {'End':>8s}  {'Dur':>6s}  {'Gap':>9s}")
    print("-" * 60)
    for i, w in enumerate(s.words):
        gap = ""
        if i < len(s.words) - 1:
            g = s.words[i + 1].start - w.end
            gap = f"{g:.3f}s"
            if g > 0.3:
                gap += " ***"
        print(f"{w.word:<20s}  {w.start:8.3f}  {w.end:8.3f}  {w.end - w.start:6.3f}  {gap:>9s}")


def cmd_split_sentence(args):
    name, idx, split_time = args.video, args.index, args.time
    adj = util.load_adjusted(name)
    transcript = util.load_transcription(name)

    target_i = None
    for i, s in enumerate(adj.sentences):
        if s.index == idx:
            target_i = i
            break

    if target_i is None:
        print(f"Error: sentence '{idx}' not found. Available: {[s.index for s in adj.sentences]}")
        return

    s = adj.sentences[target_i]
    if split_time <= s.adjusted_start or split_time >= s.adjusted_end:
        print(f"Error: split_time {split_time} outside [{s.adjusted_start:.3f}, {s.adjusted_end:.3f}]")
        return

    # Find words from transcript by timestamp overlap
    words = []
    for ts in transcript.sentences:
        overlap = min(ts.end, s.original_end) - max(ts.start, s.original_start)
        if overlap > (s.original_end - s.original_start) * 0.5:
            words = ts.words
            break

    words_a = [w for w in words if (w.start + w.end) / 2 < split_time]
    words_b = [w for w in words if (w.start + w.end) / 2 >= split_time]
    text_a = " ".join(w.word for w in words_a) if words_a else s.text + " [part 1]"
    text_b = " ".join(w.word for w in words_b) if words_b else s.text + " [part 2]"

    s1 = AdjustedSentence(
        original_start=s.original_start, original_end=s.original_end,
        adjusted_start=s.adjusted_start, adjusted_end=split_time,
        text=text_a, index=s.index, threshold_source=s.threshold_source,
        words=[WordTimestamp(word=w.word, start=w.start, end=w.end) for w in words_a],
    )
    s2 = AdjustedSentence(
        original_start=s.original_start, original_end=s.original_end,
        adjusted_start=split_time, adjusted_end=s.adjusted_end,
        text=text_b, index=f"{s.index}b", threshold_source=s.threshold_source,
        words=[WordTimestamp(word=w.word, start=w.start, end=w.end) for w in words_b],
    )

    adj.sentences[target_i:target_i + 1] = [s1, s2]
    util.save_adjusted(name, adj)
    print(f"Split sentence {idx} at {split_time:.3f}s")
    print(f"  {s1.index}: {s1.adjusted_start:.3f}-{s1.adjusted_end:.3f}  \"{s1.text[:50]}\"")
    print(f"  {s2.index}: {s2.adjusted_start:.3f}-{s2.adjusted_end:.3f}  \"{s2.text[:50]}\"")


def cmd_parse_doc(args):
    name = args.video
    html_path = util.google_doc_html_path(name)
    if not html_path.exists():
        print(f"No google doc HTML found at {html_path}")
        return

    from cc_wsp.src.services.html_parser import GoogleDocHTMLParser
    parser = GoogleDocHTMLParser()
    html = html_path.read_text(encoding="utf-8")
    script = parser.parse_html(html)
    util.save_google_doc_script(name, script)

    lines_with_images = sum(1 for l in script.lines if l.image_filenames)
    total_images = sum(len(l.image_filenames) for l in script.lines)
    print(f"Parsed: {len(script.lines)} lines, {lines_with_images} with images ({total_images} total images)")


def cmd_place_images(args):
    name = args.video
    img_path = util.images_path(name)
    if img_path.exists() and not args.force:
        print(f"Image placements exist: {img_path.name}")
        return

    script = util.load_google_doc_script(name)
    adjusted = util.load_adjusted(name)
    img_dir = util.google_doc_images_dir(name)

    from cc_wsp.src.services.agents.google_doc_image_placer import GoogleDocImagePlacer
    agent = GoogleDocImagePlacer()
    placements = agent.place_images(
        google_doc_script=script,
        adjusted_sentences=adjusted,
        google_doc_images_folder=img_dir,
    )
    util.save_images(name, placements)
    print(f"Placed {len(placements.placements)} images")
    for i, p in enumerate(placements.placements, 1):
        print(f"  {i}. {Path(p.filepath).name}: sentence {p.sentence_index} [{p.start_fraction:.1f}-{p.end_fraction:.1f}]")


def cmd_preview(args):
    name = args.video
    adjusted = util.load_adjusted(name)
    ds_path = util.downsampled_path(name)
    out_path = util.preview_path(name)

    from cc_wsp.src.services.video.video_service import VideoService
    svc = VideoService(VIDEOS_DIR)
    svc.create_edited_video(
        base_name=name, adjusted_sentences=adjusted,
        use_downsampled=True, force=True, output_path=out_path,
    )
    print(f"Preview: {out_path}")


def cmd_render(args):
    name = args.video

    # Step 1: downsample to 1080p if needed
    ds_1080_path = util.downsampled_1080p_path(name)
    if not ds_1080_path.exists():
        input_path = util.get_input_video_path(name)
        print("Downsampling to 1080p...")
        binary, vf = _build_scale_vf(input_path, f"scale=1080:{HD_1080P_HEIGHT}")
        subprocess.run([
            binary, "-i", str(input_path), "-vf", vf,
            "-c:v", VIDEO_CODEC, "-preset", VIDEO_PRESET, "-crf", "23",
            "-c:a", "aac", "-b:a", "192k", "-y", str(ds_1080_path),
        ], capture_output=True, check=True)

    # Step 2: render with MLT
    adjusted = util.load_adjusted(name)
    img_placements = util.load_images(name) if util.images_path(name).exists() else None

    from cc_wsp.src.services.video.mlt_video_service import MLTVideoService
    from cc_wsp.src.services.video.mlt_util import (
        get_video_properties, create_mlt_root_and_profile, frames_to_timecode,
        add_black_producer, add_image_producer, calculate_safe_zone,
        create_main_tractor, save_pretty_xml, add_mix_transition,
        add_composite_transition,
    )
    from xml.etree import ElementTree as ET
    import hashlib
    from datetime import datetime

    zoom_filters = util.load_zooms(name) if util.zooms_path(name).exists() else None
    config = util.load_config(name)
    if util.face_data_path(name).exists():
        print(
            f"Safe zone from face: top={config['image_safe_zone_top']:.3f}, "
            f"bottom={config['image_safe_zone_bottom']:.3f}"
        )

    # Detect sentences that will carry a title card so the render can push
    # their images below the card area.
    title_card_sentence_indices: set[str] = set()
    try:
        from cc_wsp.src.services.video.caption_service import extract_title_instructions
        script = util.load_google_doc_script(name)
        for tc in extract_title_instructions(script, adjusted):
            # extract_title_instructions returns start/end times; we need the
            # sentence index. Re-derive from the sentence timeline.
            cumulative = 0.0
            for s in adjusted.sentences:
                dur = s.adjusted_end - s.adjusted_start
                if abs(cumulative - tc.start) < 0.05:
                    title_card_sentence_indices.add(str(s.index))
                    break
                cumulative += dur
    except FileNotFoundError:
        pass
    if title_card_sentence_indices:
        print(f"Lowering images on title-card sentences: {sorted(title_card_sentence_indices)}")

    mlt_svc = MLTVideoService()
    if img_placements:
        video_path = mlt_svc.create_1080p_video_with_images(
            base_name=name, adjusted_sentences=adjusted,
            image_placements=img_placements, force=True,
            zoom_filters=zoom_filters,
            safe_zone_config=config,
            title_card_sentence_indices=title_card_sentence_indices,
        )
    else:
        # Render cuts only
        from cc_wsp.src.services.video.video_service import VideoService
        svc = VideoService(VIDEOS_DIR)
        video_path = svc.create_edited_video(
            base_name=name, adjusted_sentences=adjusted,
            use_downsampled=False, force=True, output_path=util.final_path(name),
        )

    print(f"Final video: {video_path}")


def cmd_captions(args):
    name = args.video

    from cc_wsp.src.services.video.caption_service import (
        build_word_timeline,
        group_words_into_chunks,
        detect_title_text,
        extract_title_instructions,
        burn_captions,
        TitleCardConfig,
    )

    # Load data
    adjusted = util.load_adjusted(name)
    transcription = util.load_transcription(name)
    final = util.final_path(name)
    output = util.captioned_path(name)

    if not final.exists():
        print(f"Error: {final} not found. Run 'render' first.")
        return

    # Build word timeline and group into chunks
    print("=> Building word timeline...")
    timeline = build_word_timeline(adjusted, transcription)
    print(f"   {len(timeline)} words mapped to final video time")

    caption_y = args.caption_y if args.caption_y is not None else None
    from cc_wsp.src.constants import CAPTION_Y_PERCENT, CAPTION_MAX_WORDS_PER_CHUNK
    if caption_y is not None:
        y_pct = caption_y
    elif util.face_data_path(name).exists():
        face = util.load_face_data(name)
        y_pct = util.compute_caption_y_from_face(face)
        print(f"   Caption Y from face: {y_pct:.3f} (face bottom {face.bottom_frac:.3f})")
    else:
        y_pct = CAPTION_Y_PERCENT

    chunks = group_words_into_chunks(timeline, max_words=CAPTION_MAX_WORDS_PER_CHUNK)
    print(f"   {len(chunks)} caption chunks")

    # Title card (hook text at start)
    title_config = None
    if not args.no_title:
        title_text = args.title
        if not title_text:
            # Auto-detect from script
            try:
                script = util.load_google_doc_script(name)
                title_text = detect_title_text(script)
            except FileNotFoundError:
                pass

        if title_text:
            from cc_wsp.src.constants import TITLE_CARD_DURATION
            title_config = TitleCardConfig(
                text=title_text, start=0.0, end=TITLE_CARD_DURATION
            )
            print(f"   Title card: \"{title_text}\" (0.0-{TITLE_CARD_DURATION:.1f}s)")

    # Extract title text from [Title text: "..."] bracket instructions
    extra_title_cards = []
    try:
        script = util.load_google_doc_script(name)
        extra_title_cards = extract_title_instructions(script, adjusted)
        for tc in extra_title_cards:
            print(f"   Title instruction: \"{tc.text}\" ({tc.start:.1f}-{tc.end:.1f}s)")
    except FileNotFoundError:
        pass

    burn_captions(
        video_path=final,
        output_path=output,
        chunks=chunks,
        title_config=title_config,
        title_cards=extra_title_cards,
        caption_y_percent=y_pct,
    )
    print(f"Captioned video: {output}")


def cmd_set_zooms(args):
    """Set zoom filters for specific sentences. Saves to zooms.json."""
    name = args.video
    from cc_wsp.src.models import ZoomFilter, ZoomFilters

    # Load existing zooms or start fresh
    if util.zooms_path(name).exists() and not args.clear:
        zooms = util.load_zooms(name)
    else:
        zooms = ZoomFilters()

    if args.clear:
        util.save_zooms(name, zooms)
        print("Cleared all zoom filters")
        return

    if args.sentence:
        zf = ZoomFilter(
            sentence_index=args.sentence,
            zoom_factor=args.zoom,
            x_offset=args.x_offset,
            y_offset=args.y_offset,
        )
        # Replace existing filter for this sentence if any
        zooms.filters = [f for f in zooms.filters if f.sentence_index != args.sentence]
        zooms.filters.append(zf)
        util.save_zooms(name, zooms)
        print(f"Set zoom on sentence {args.sentence}: {args.zoom}x (offset: {args.x_offset}, {args.y_offset})")

    # Show current zooms
    if zooms.filters:
        print(f"\nCurrent zooms for {name}:")
        adjusted = util.load_adjusted(name)
        sent_map = {s.index: s.text for s in adjusted.sentences}
        for zf in sorted(zooms.filters, key=lambda z: z.sentence_index):
            text = sent_map.get(zf.sentence_index, "???")
            print(f"  [{zf.sentence_index}] {zf.zoom_factor}x (x={zf.x_offset}, y={zf.y_offset}) — {text[:60]}")
    else:
        print(f"No zoom filters for {name}")


def cmd_show(args):
    name, file = args.video, args.file
    path_map = {
        "transcription": util.transcription_path,
        "sentences": util.sentences_path,
        "adjusted": util.adjusted_path,
        "images": util.images_path,
    }

    if file not in path_map:
        print(f"Unknown file: {file}. Options: {list(path_map.keys())}")
        return

    p = path_map[file](name)
    if not p.exists():
        print(f"File not found: {p}")
        return

    if file == "transcription":
        t = util.load_transcription(name)
        for i, s in enumerate(t.sentences, 1):
            print(f"  {i:2d}: [{s.start:7.2f}-{s.end:7.2f}] \"{s.sentence}\"")
    elif file == "sentences":
        r = util.load_sentences(name)
        for idx, sr in sorted(r.sentence_results.items(), key=lambda x: int(x[0])):
            status = "KEEP" if sr.keep else "REMOVE"
            print(f"  {idx:>3s}: [{status:6s}] \"{sr.text[:70]}\"")
    elif file == "adjusted":
        a = util.load_adjusted(name)
        for s in a.sentences:
            dur = s.adjusted_end - s.adjusted_start
            print(f"  {s.index:>4s}: [{s.adjusted_start:7.2f}-{s.adjusted_end:7.2f}] ({dur:.2f}s) \"{s.text[:60]}\"")
    elif file == "images":
        im = util.load_images(name)
        for i, p in enumerate(im.placements, 1):
            print(f"  {i:2d}: {Path(p.filepath).name} on sentence {p.sentence_index} [{p.start_fraction:.1f}-{p.end_fraction:.1f}]")


def cmd_status(args):
    name = args.video
    d = util.get_video_dir(name)
    print(f"Video: {name} ({d})")
    print()

    checks = [
        ("Input video", util.get_input_video_path(name)),
        ("Audio", util.audio_path(name)),
        ("Downsampled", util.downsampled_path(name)),
        ("Google doc", util.google_doc_html_path(name)),
        ("Transcription", util.transcription_path(name)),
        ("Sentences", util.sentences_path(name)),
        ("Adjusted", util.adjusted_path(name)),
        ("Images", util.images_path(name)),
        ("Preview", util.preview_path(name)),
        ("1080p source", util.downsampled_1080p_path(name)),
        ("Final", util.final_path(name)),
    ]

    for label, path in checks:
        exists = "OK" if path.exists() else "--"
        print(f"  [{exists:>2s}] {label:<16s}  {path.name}")


# ============================================================
# Stream commands
# ============================================================

STREAM_LOW_RES_HEIGHT = 360
STREAM_CHUNK_SIZE = 100
STREAM_CHUNK_OVERLAP = 5


def cmd_stream_preprocess(args):
    name = args.video
    input_path = util.get_input_video_path(name)

    # Downsample to 360p proxy
    ds_path = util.stream_downsampled_path(name)
    if ds_path.exists() and not args.force:
        print(f"Stream proxy exists: {ds_path.name}")
    else:
        print(f"Downsampling to {STREAM_LOW_RES_HEIGHT}p proxy...")
        subprocess.run([
            "ffmpeg", "-i", str(input_path), "-vf", f"scale=-2:{STREAM_LOW_RES_HEIGHT}",
            "-c:v", VIDEO_CODEC, "-preset", VIDEO_PRESET, "-crf", "28",
            "-c:a", "aac", "-b:a", "128k", "-y", str(ds_path),
        ], check=True)
        print(f"Created: {ds_path.name}")

    # Extract audio
    a_path = util.stream_audio_path(name)
    if a_path.exists() and not args.force:
        print(f"Stream audio exists: {a_path.name}")
    else:
        print("Extracting audio...")
        subprocess.run([
            "ffmpeg", "-i", str(input_path), "-vn", "-acodec", AUDIO_CODEC,
            "-ar", str(AUDIO_SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS),
            "-b:a", AUDIO_BITRATE, "-y", str(a_path),
        ], check=True)
        print(f"Created: {a_path.name}")

    print("Stream preprocessing complete.")


def cmd_stream_transcribe(args):
    name = args.video
    t_path = util.stream_transcription_path(name)
    if t_path.exists() and not args.force:
        print(f"Stream transcription exists: {t_path.name}")
        return

    a_path = util.stream_audio_path(name)

    # For long streams, split audio into chunks to avoid Deepgram upload timeout
    chunk_minutes = 20
    total_duration = _get_duration(a_path)
    total_minutes = total_duration / 60

    from cc_wsp.src.services.stt.deepgram import DeepgramSTTService
    from cc_wsp.src.models import Transcript, LLMTranscriptSentence

    stt = DeepgramSTTService()

    if total_minutes <= chunk_minutes + 2:
        # Short enough to transcribe in one go
        print(f"Transcribing {a_path.name}...")
        transcript = stt.transcribe(a_path)
    else:
        # Split into chunks and transcribe each
        num_chunks = int(total_minutes / chunk_minutes) + 1
        print(f"Stream is {total_minutes:.0f} min — splitting into {num_chunks} chunks of ~{chunk_minutes} min")

        all_sentences: list[LLMTranscriptSentence] = []
        chunk_dir = util.get_video_dir(name)

        for i in range(num_chunks):
            start_sec = i * chunk_minutes * 60
            if start_sec >= total_duration:
                break

            chunk_path = chunk_dir / f"_audio_chunk_{i}.mp3"
            duration_sec = min(chunk_minutes * 60, total_duration - start_sec)

            print(f"  Chunk {i + 1}/{num_chunks}: {start_sec / 60:.0f}-{(start_sec + duration_sec) / 60:.0f} min")

            # Extract chunk
            subprocess.run([
                "ffmpeg", "-i", str(a_path), "-ss", str(start_sec),
                "-t", str(duration_sec), "-acodec", "copy", "-y", str(chunk_path),
            ], capture_output=True, check=True)

            # Transcribe chunk
            chunk_transcript = stt.transcribe(chunk_path)

            # Offset timestamps by chunk start
            for sentence in chunk_transcript.sentences:
                sentence.start += start_sec
                sentence.end += start_sec
                for w in sentence.words:
                    w.start += start_sec
                    w.end += start_sec
                all_sentences.append(sentence)

            # Cleanup chunk
            chunk_path.unlink()

        transcript = Transcript(
            sentences=all_sentences,
            language="en",
            duration=total_duration,
        )

    util.save_stream_transcription(name, transcript)
    duration = sum(s.end - s.start for s in transcript.sentences)
    print(f"Saved: {t_path.name} ({len(transcript.sentences)} sentences, {duration / 60:.1f} min of speech)")


def _get_duration(path: Path) -> float:
    """Get audio/video duration in seconds."""
    import json as _json
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(_json.loads(result.stdout)["format"]["duration"])


def cmd_stream_edit(args):
    """Iteratively run LLM editing passes until the stream is under the target duration."""
    name = args.video
    target_minutes = args.target_minutes
    max_passes = args.max_passes

    transcript = util.load_stream_transcription(name)
    total_sentences = len(transcript.sentences)
    total_duration = sum(s.end - s.start for s in transcript.sentences)

    print(f"Stream: {total_sentences} sentences, {total_duration / 60:.1f} min total speech")
    print(f"Target: {target_minutes} min, max {max_passes} LLM passes")

    from cc_wsp.src.services.llm.openrouter import OpenRouterLLMService
    llm = OpenRouterLLMService()

    # Load existing sentences if available (to resume)
    s_path = util.stream_sentences_path(name)
    if s_path.exists() and not args.force:
        result = util.load_stream_sentences(name)
        kept_indices = {
            int(idx) for idx, sr in result.sentence_results.items() if sr.keep
        }
        kept_duration = sum(
            transcript.sentences[i - 1].end - transcript.sentences[i - 1].start
            for i in kept_indices
        )
        print(f"Resuming from existing sentences.json: {len(kept_indices)} kept, {kept_duration / 60:.1f} min")
    else:
        kept_indices = None
        kept_duration = total_duration

    for pass_num in range(1, max_passes + 1):
        if kept_duration / 60 <= target_minutes:
            print(f"\nTarget reached! {kept_duration / 60:.1f} min <= {target_minutes} min")
            break

        print(f"\n{'=' * 60}")
        print(f"LLM PASS {pass_num} — Current: {kept_duration / 60:.1f} min, Target: {target_minutes} min")
        print(f"{'=' * 60}")

        result = llm.get_stream_highlights(
            transcript,
            kept_indices=kept_indices,
            chunk_size=STREAM_CHUNK_SIZE,
            overlap=STREAM_CHUNK_OVERLAP,
            target_minutes=target_minutes,
            current_minutes=kept_duration / 60,
        )

        util.save_stream_sentences(name, result)

        kept_indices = {
            int(idx) for idx, sr in result.sentence_results.items() if sr.keep
        }
        kept_duration = sum(
            transcript.sentences[i - 1].end - transcript.sentences[i - 1].start
            for i in kept_indices
        )

        # Also save a backup of this pass
        backup_path = util.get_video_dir(name) / f"stream_sentences_pass{pass_num}.json"
        util.save_json(backup_path, result)
        print(f"Pass {pass_num} saved. Kept: {len(kept_indices)} sentences, {kept_duration / 60:.1f} min")

    if kept_duration / 60 > target_minutes:
        print(f"\nWarning: Still at {kept_duration / 60:.1f} min after {max_passes} passes (target: {target_minutes})")
    else:
        print(f"\nDone! Final duration estimate: {kept_duration / 60:.1f} min ({len(kept_indices)} sentences)")


def cmd_stream_silence_detect(args):
    name = args.video
    adj_path = util.stream_adjusted_path(name)
    if adj_path.exists() and not args.force:
        print(f"Stream adjusted sentences exist: {adj_path.name}")
        return

    transcript = util.load_stream_transcription(name)
    sentences_result = util.load_stream_sentences(name)
    a_path = util.stream_audio_path(name)

    from cc_wsp.src.services.video.video_service import VideoService
    svc = VideoService(VIDEOS_DIR)

    kept = [
        (int(idx), transcript.sentences[int(idx) - 1])
        for idx, sr in sentences_result.sentence_results.items()
        if sr.keep
    ]
    kept.sort(key=lambda x: x[0])

    if args.skip_silence:
        print(f"Generating adjusted sentences (no silence removal) for {len(kept)} sentences...")
        adjusted_list = []
        for idx, sentence in kept:
            adjusted_list.append(AdjustedSentence(
                original_start=sentence.start, original_end=sentence.end,
                adjusted_start=sentence.start, adjusted_end=sentence.end,
                text=sentence.sentence, index=str(idx),
                threshold_source="skipped",
                words=sentence.words,
            ))
    else:
        print(f"Running silence detection on {len(kept)} sentences...")
        adjusted_list = []
        for idx, sentence in kept:
            result = svc._get_adjusted_sentence(a_path, sentence, idx)
            adjusted_list.append(result)

    adj = AdjustedSentences(sentences=adjusted_list)
    util.save_stream_adjusted(name, adj)

    total_dur = sum(s.adjusted_end - s.adjusted_start for s in adj.sentences)
    print(f"Saved: {adj_path.name} ({len(adj.sentences)} sentences, {total_dur / 60:.1f} min)")


def cmd_stream_preview(args):
    name = args.video
    adjusted = util.load_stream_adjusted(name)
    out_path = util.stream_preview_path(name)

    from cc_wsp.src.services.video.video_service import VideoService
    svc = VideoService(VIDEOS_DIR)
    svc.create_edited_video(
        base_name=name, adjusted_sentences=adjusted,
        use_downsampled=False, force=True, output_path=out_path,
    )
    total_dur = sum(s.adjusted_end - s.adjusted_start for s in adjusted.sentences)
    print(f"Stream preview: {out_path} ({total_dur / 60:.1f} min)")


def cmd_stream_render(args):
    """Render final stream video at native resolution using ffmpeg concat."""
    name = args.video
    adjusted = util.load_stream_adjusted(name)
    input_path = util.get_input_video_path(name)
    out_path = util.stream_final_path(name)

    if out_path.exists() and not args.force:
        print(f"Stream final exists: {out_path.name}")
        return

    from cc_wsp.src.services.video.video_service import VideoService
    svc = VideoService(VIDEOS_DIR)
    svc.create_edited_video(
        base_name=name, adjusted_sentences=adjusted,
        use_downsampled=False, force=True, output_path=out_path,
    )
    total_dur = sum(s.adjusted_end - s.adjusted_start for s in adjusted.sentences)
    print(f"Stream final: {out_path} ({total_dur / 60:.1f} min)")


def cmd_stream_show(args):
    name, file = args.video, args.file
    path_map = {
        "transcription": util.stream_transcription_path,
        "sentences": util.stream_sentences_path,
        "adjusted": util.stream_adjusted_path,
    }

    if file not in path_map:
        print(f"Unknown file: {file}. Options: {list(path_map.keys())}")
        return

    p = path_map[file](name)
    if not p.exists():
        print(f"File not found: {p}")
        return

    if file == "transcription":
        t = util.load_stream_transcription(name)
        for i, s in enumerate(t.sentences, 1):
            print(f"  {i:3d}: [{s.start:7.2f}-{s.end:7.2f}] \"{s.sentence}\"")
        total = sum(s.end - s.start for s in t.sentences)
        print(f"\n  Total: {len(t.sentences)} sentences, {total / 60:.1f} min")
    elif file == "sentences":
        r = util.load_stream_sentences(name)
        kept_dur = 0
        removed_dur = 0
        t = util.load_stream_transcription(name)
        for idx, sr in sorted(r.sentence_results.items(), key=lambda x: int(x[0])):
            status = "KEEP" if sr.keep else "REMOVE"
            dur = t.sentences[int(idx) - 1].end - t.sentences[int(idx) - 1].start
            if sr.keep:
                kept_dur += dur
            else:
                removed_dur += dur
            print(f"  {idx:>4s}: [{status:6s}] ({dur:.1f}s) \"{sr.text[:70]}\"")
        kept = sum(1 for sr in r.sentence_results.values() if sr.keep)
        print(f"\n  Kept: {kept} ({kept_dur / 60:.1f} min), Removed: {len(r.sentence_results) - kept} ({removed_dur / 60:.1f} min)")
    elif file == "adjusted":
        a = util.load_stream_adjusted(name)
        for s in a.sentences:
            dur = s.adjusted_end - s.adjusted_start
            print(f"  {s.index:>4s}: [{s.adjusted_start:7.2f}-{s.adjusted_end:7.2f}] ({dur:.2f}s) \"{s.text[:60]}\"")
        total = sum(s.adjusted_end - s.adjusted_start for s in a.sentences)
        print(f"\n  Total: {len(a.sentences)} sentences, {total / 60:.1f} min")


def cmd_stream_status(args):
    name = args.video
    d = util.get_video_dir(name)
    print(f"Stream: {name} ({d})")
    print()

    checks = [
        ("Input video", util.get_input_video_path(name)),
        ("Stream audio", util.stream_audio_path(name)),
        ("Stream proxy", util.stream_downsampled_path(name)),
        ("Transcription", util.stream_transcription_path(name)),
        ("Sentences", util.stream_sentences_path(name)),
        ("Adjusted", util.stream_adjusted_path(name)),
        ("Preview", util.stream_preview_path(name)),
        ("Final", util.stream_final_path(name)),
    ]

    for label, path in checks:
        exists = "OK" if path.exists() else "--"
        size = ""
        if path.exists():
            sz = path.stat().st_size
            if sz > 1_000_000:
                size = f" ({sz / 1_000_000:.1f} MB)"
            elif sz > 1000:
                size = f" ({sz / 1000:.0f} KB)"
        print(f"  [{exists:>2s}] {label:<16s}  {path.name}{size}")


def main():
    parser = argparse.ArgumentParser(description="cc_wsp video editing tools")
    parser.add_argument("--force", "-f", action="store_true", help="Force regenerate")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("preprocess").add_argument("video")
    sub.add_parser("detect-face").add_argument("video")
    sub.add_parser("transcribe").add_argument("video")
    sub.add_parser("suggest-edits").add_argument("video")
    sub.add_parser("silence-detect").add_argument("video")

    al = sub.add_parser("audio-levels")
    al.add_argument("video")
    al.add_argument("start", type=float)
    al.add_argument("end", type=float)
    al.add_argument("--resolution", type=float, default=0.01)

    wt = sub.add_parser("word-timestamps")
    wt.add_argument("video")
    wt.add_argument("sentence", type=int)

    ss = sub.add_parser("split-sentence")
    ss.add_argument("video")
    ss.add_argument("index")
    ss.add_argument("time", type=float)

    sub.add_parser("parse-doc").add_argument("video")
    sub.add_parser("place-images").add_argument("video")
    sub.add_parser("preview").add_argument("video")
    sub.add_parser("render").add_argument("video")

    cap = sub.add_parser("captions")
    cap.add_argument("video")
    cap.add_argument("--title", type=str, default=None, help="Custom title card text")
    cap.add_argument("--no-title", action="store_true", help="Skip title card")
    cap.add_argument("--caption-y", type=float, default=None, help="Caption vertical position (0.0-1.0)")

    sz = sub.add_parser("set-zooms")
    sz.add_argument("video")
    sz.add_argument("--sentence", "-s", help="Sentence index to zoom")
    sz.add_argument("--zoom", "-z", type=float, default=1.3, help="Zoom factor (default 1.3)")
    sz.add_argument("--x-offset", "-x", type=float, default=0.0, help="Horizontal pan -1 to 1")
    sz.add_argument("--y-offset", "-y", type=float, default=0.0, help="Vertical pan -1 to 1")
    sz.add_argument("--clear", action="store_true", help="Clear all zooms")

    sh = sub.add_parser("show")
    sh.add_argument("video")
    sh.add_argument("file", choices=["transcription", "sentences", "adjusted", "images"])

    sub.add_parser("status").add_argument("video")

    # Stream commands
    sub.add_parser("stream-preprocess").add_argument("video")
    sub.add_parser("stream-transcribe").add_argument("video")

    se = sub.add_parser("stream-edit")
    se.add_argument("video")
    se.add_argument("--target-minutes", type=float, default=25)
    se.add_argument("--max-passes", type=int, default=5)

    ssd = sub.add_parser("stream-silence-detect")
    ssd.add_argument("video")
    ssd.add_argument("--skip-silence", action="store_true", help="Skip silence removal, use original timestamps")

    sub.add_parser("stream-preview").add_argument("video")
    sub.add_parser("stream-render").add_argument("video")

    ssh = sub.add_parser("stream-show")
    ssh.add_argument("video")
    ssh.add_argument("file", choices=["transcription", "sentences", "adjusted"])

    sub.add_parser("stream-status").add_argument("video")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {
        "preprocess": cmd_preprocess,
        "detect-face": cmd_detect_face,
        "transcribe": cmd_transcribe,
        "suggest-edits": cmd_suggest_edits,
        "silence-detect": cmd_silence_detect,
        "audio-levels": cmd_audio_levels,
        "word-timestamps": cmd_word_timestamps,
        "split-sentence": cmd_split_sentence,
        "parse-doc": cmd_parse_doc,
        "place-images": cmd_place_images,
        "preview": cmd_preview,
        "render": cmd_render,
        "captions": cmd_captions,
        "set-zooms": cmd_set_zooms,
        "show": cmd_show,
        "status": cmd_status,
        "stream-preprocess": cmd_stream_preprocess,
        "stream-transcribe": cmd_stream_transcribe,
        "stream-edit": cmd_stream_edit,
        "stream-silence-detect": cmd_stream_silence_detect,
        "stream-preview": cmd_stream_preview,
        "stream-render": cmd_stream_render,
        "stream-show": cmd_stream_show,
        "stream-status": cmd_stream_status,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
