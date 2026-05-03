#!/usr/bin/env python3
"""Process bcai_top_habits episode 22 through the full pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cc_wsp.src import util
from cc_wsp.src.models import AdjustedSentences

NAME = "bcai_top_habits_episodes/bcai_top_habits_22"

# Step 1: Preprocess (downsample + extract audio)
import subprocess

input_path = util.get_input_video_path(NAME)
print(f"Input video: {input_path}")

ds_path = util.downsampled_path(NAME)
if not ds_path.exists():
    print("Step 1a: Downsampling...")
    subprocess.run([
        "ffmpeg", "-i", str(input_path), "-vf", "scale=-2:480",
        "-c:v", "libx264", "-preset", "fast", "-crf", "28",
        "-c:a", "aac", "-b:a", "128k", "-y", str(ds_path),
    ], capture_output=True, check=True)
    print(f"  Created: {ds_path.name}")
else:
    print(f"Step 1a: Downsampled exists: {ds_path.name}")

a_path = util.audio_path(NAME)
if not a_path.exists():
    print("Step 1b: Extracting audio...")
    subprocess.run([
        "ffmpeg", "-i", str(input_path), "-vn", "-acodec", "mp3",
        "-ar", "16000", "-ac", "1", "-b:a", "64k", "-y", str(a_path),
    ], capture_output=True, check=True)
    print(f"  Created: {a_path.name}")
else:
    print(f"Step 1b: Audio exists: {a_path.name}")

# Step 2: Transcribe
t_path = util.transcription_path(NAME)
if not t_path.exists():
    print("Step 2: Transcribing...")
    from cc_wsp.src.services.stt.deepgram import DeepgramSTTService
    stt = DeepgramSTTService()
    transcript = stt.transcribe(a_path)
    util.save_transcription(NAME, transcript)
    print(f"  Saved: {t_path.name} ({len(transcript.sentences)} sentences)")
else:
    print(f"Step 2: Transcription exists: {t_path.name}")

# Step 3: Suggest edits
s_path = util.sentences_path(NAME)
if not s_path.exists():
    print("Step 3: Getting LLM edit suggestions...")
    transcript = util.load_transcription(NAME)
    from cc_wsp.src.services.llm.openrouter import OpenRouterLLMService
    llm = OpenRouterLLMService()
    decision = llm.get_edits(transcript)
    result = util.convert_decision_to_result(decision, transcript)
    util.save_sentences(NAME, result)
    kept = sum(1 for s in result.sentence_results.values() if s.keep)
    removed = sum(1 for s in result.sentence_results.values() if not s.keep)
    print(f"  Saved: {s_path.name} ({kept} kept, {removed} removed)")
    print(f"  LLM thoughts: {decision.thoughts[:200]}...")
else:
    print(f"Step 3: Sentences exist: {s_path.name}")

# Step 4: Silence detection
adj_path = util.adjusted_path(NAME)
if not adj_path.exists():
    print("Step 4: Running silence detection...")
    transcript = util.load_transcription(NAME)
    sentences_result = util.load_sentences(NAME)
    from cc_wsp.src.services.video.video_service import VideoService
    from cc_wsp.src.constants import VIDEOS_DIR
    svc = VideoService(VIDEOS_DIR)
    kept = [
        (i, s) for i, s in enumerate(transcript.sentences, 1)
        if sentences_result.sentence_results[str(i)].keep
    ]
    adjusted_list = []
    for idx, sentence in kept:
        result = svc._get_adjusted_sentence(a_path, sentence, idx)
        adjusted_list.append(result)
    adj = AdjustedSentences(sentences=adjusted_list)
    util.save_adjusted(NAME, adj)
    print(f"  Saved: {adj_path.name} ({len(adjusted_list)} sentences)")
else:
    print(f"Step 4: Adjusted exists: {adj_path.name}")

# Step 5: Generate preview
preview = util.preview_path(NAME)
if not preview.exists():
    print("Step 5: Generating preview...")
    from cc_wsp.src.services.video.video_service import VideoService
    from cc_wsp.src.constants import VIDEOS_DIR
    adj = util.load_adjusted(NAME)
    svc = VideoService(VIDEOS_DIR)
    svc.create_edited_video(
        base_name=NAME, adjusted_sentences=adj,
        use_downsampled=True, force=True, output_path=preview,
    )
    print(f"  Created: {preview.name}")
else:
    print(f"Step 5: Preview exists: {preview.name}")

# Step 6: Downsample to 1080p
ds_1080 = util.downsampled_1080p_path(NAME)
if not ds_1080.exists():
    print("Step 6: Downsampling to 1080p...")
    subprocess.run([
        "ffmpeg", "-i", str(input_path), "-vf", "scale=1080:1920",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k", "-y", str(ds_1080),
    ], capture_output=True, check=True)
    print(f"  Created: {ds_1080.name}")
else:
    print(f"Step 6: 1080p exists: {ds_1080.name}")

# Step 7: MLT render (cuts only)
final = util.final_path(NAME)
mlt_path = util.final_mlt_path(NAME)
if not final.exists():
    print("Step 7: Rendering final video via MLT...")
    from cc_wsp.src.services.video.mlt_video_service import MLTVideoService
    adj = util.load_adjusted(NAME)
    mlt_svc = MLTVideoService()
    mlt_svc._create_mlt_xml_for_cutting(ds_1080, adj, mlt_path)
    print(f"  MLT XML: {mlt_path.name}")

    cmd = [
        "melt", str(mlt_path),
        "-consumer", f"avformat:{final}",
        "vcodec=libx264", "acodec=aac", "crf=23",
        "preset=medium", "pix_fmt=yuv420p",
    ]
    print(f"  Running melt...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  melt stderr: {result.stderr[-500:]}")
        raise RuntimeError("melt failed")
    print(f"  Created: {final.name}")
else:
    print(f"Step 7: Final exists: {final.name}")

print("\nDone! Final video at:", final)
