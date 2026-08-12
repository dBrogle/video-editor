# cc_wsp — Claude Code Video Editing Workspace

This workspace provides tools for Claude Code to edit short-form vertical videos.

- **Phases 1-4** (preprocess → transcribe → suggest-edits → silence-detect) run through `tool.py`. They produce `adjusted.json`, the source of truth for which sentences and timestamps go in the final cut.
- **Phases 5-9** (cut → re-transcribe → place images → preview/render → captions) run through `pipeline_v2.py`. The v2 path bakes the cut in a single ffmpeg pass upstream of MLT (hard concat for both video and audio — see Phase 5 for why no crossfade), then has MLT only compose images on top. This avoids the per-clip A/V drift you get from MLT's `acrossfade` chain. The legacy single-pass MLT path (`tool.py render` / `tool.py captions`) is documented at the bottom of this file for reference.

## Setup

Place the source video (and optional Google Doc zip) in `cc_wsp/videos/{name}/`:
```
cc_wsp/videos/d216_18/
  d216_18.mov          # Original video (name must match folder)
  216 Literacy.zip     # Optional Google Doc with images
```

**Important:** The `.mov` file must be renamed to match the folder name (e.g., `IMG_4059.mov` → `b268_claude_code.mov` for folder `b268_claude_code/`). Preprocess will fail otherwise.

## Quick Reference

```bash
# Phases 1-4 (cut decisions)
python cc_wsp/tool.py <command> <video> [args...]

# Phases 5-9 (render). Use --force to regenerate a stage.
python cc_wsp/pipeline_v2.py <command> <video> [args...]
python cc_wsp/pipeline_v2.py --force <command> <video>

# Check what exists
python cc_wsp/tool.py status my_video

# View any step file
python cc_wsp/tool.py show my_video transcription
python cc_wsp/tool.py show my_video sentences
python cc_wsp/tool.py show my_video adjusted
```

## Workflow

### Phase 1: Preprocess
```bash
python cc_wsp/tool.py preprocess my_video
```
Creates `audio.mp3`, `downsampled.mp4`, extracts google doc, prescales images.

### Phase 2: Transcribe
```bash
python cc_wsp/tool.py transcribe my_video
```
Runs Deepgram STT → `transcription.json` with sentence-level and word-level timestamps.

### Phase 3: Sentence Selection
```bash
# LLM suggests which sentences to remove (retakes, filler)
python cc_wsp/tool.py suggest-edits my_video

# Review the result
python cc_wsp/tool.py show my_video sentences
```
Creates `sentences.json`. Claude Code should review this and edit the JSON directly if needed (change `"keep": true/false`).

**Cross-check kept sentences against the parsed Google Doc script** (`google_doc_script.json` after `parse-doc`, or the raw HTML before then). The LLM sometimes flags non-retake bridge/transition lines as "redundant" and removes them — those are usually intentional in the script and dropping them creates an awkward jump. Re-mark any kept-in-script line back to `"keep": true`.

**Do not pause for sentence-selection feedback here.** The user reviews the cut by watching `cut.mp4` (Phase 5), not by reading the JSON. Make your best keep/remove decisions (using suggest-edits + the script cross-check above), then proceed through silence detection and the crossfade cut. Gather the user's feedback on sentence selection at the Phase 5 confirmation point.

### Phase 4: Adjusted Sentences (silence removal)
```bash
# Run silence detection on kept sentences
python cc_wsp/tool.py silence-detect my_video

# Review the result
python cc_wsp/tool.py show my_video adjusted
```
Creates `adjusted.json` with silence-trimmed start/end times.

Claude Code should then review and fix issues using:
```bash
# Check audio around a boundary
python cc_wsp/tool.py audio-levels my_video 48.5 49.5

# Check word-level timing for a sentence
python cc_wsp/tool.py word-timestamps my_video 9

# Split a sentence (remove mid-sentence silence)
python cc_wsp/tool.py split-sentence my_video 10 64.0
```

The "preview" of `adjusted.json` is `cut.mp4` produced by Phase 5 — there is no separate v1-style preview step in the v2 flow. Iterate on `adjusted.json` and re-run `crossfade-cut --force` to re-bake the cut.

For fixes: edit `adjusted.json` directly — change `adjusted_start`/`adjusted_end` values, or use `split-sentence` to break apart sentences with internal gaps.

**Common issues to check for:**
- Cut-off sibilants (s, z) at sentence ends — extend `adjusted_end`
- Cut-off plosives (k, t, p) — extend `adjusted_end`
- Trailing silence/pause at end — tighten `adjusted_end`
- Retakes within a sentence — trim `adjusted_start` to skip the first take
- Large silence gaps mid-sentence — split the sentence
- **Adjacent kept sentences that are contiguous in the source** (sentence A's `original_end` == sentence B's `original_start`): `silence-detect` often nudges B's `adjusted_start` ~0.04s *before* the boundary, so 0.04s of source plays twice in the cut. Clamp B's `adjusted_start` up to A's `adjusted_end`.
- **Very short clips** (a one-line hook on its own, a short interjection): `silence-detect` computes a clip-level threshold from just that clip, so it tends to over-trim — frequently it eats into the *last word* (the trailing `-ing`, the `s`, etc.) or starts mid-first-word. Always re-check the first and last word of short clips with `word-timestamps` + `audio-levels` and hand-fix `adjusted_start`/`adjusted_end`.

**Sibilant review workflow:** After silence detection, check every sentence ending with a sibilant sound (s, z, x, sh, etc.) using `audio-levels` with `--resolution 0.01` or `0.02` zoomed in on the last ~0.5s. If the sibilant energy extends past the `adjusted_end`, extend it by ~0.05-0.10s. Conversely, if the user reports trailing silence, tighten by small amounts (0.02-0.05s).

**Partial retakes within a kept sentence:** Sometimes a sentence contains a false start/retake (flagged by `***` gaps in `word-timestamps`). Rather than removing the whole sentence, keep it and trim `adjusted_start` past the retake using the word timestamps as a guide. Check `audio-levels` to find clean speech onset.

**Single-word outros** (e.g. "Cheers", "Thanks"): don't leave them as their own splice. Merge the outro into the prior sentence's clip — extend that sentence's `adjusted_end` to cover the outro's `original_end`, append the outro text to the prior sentence's `text`, and delete the outro entry from `adjusted.json`. The natural pre-outro pause is preserved inside the merged clip.

**Awkward cuts between visually-similar retakes:** When `suggest-edits` keeps a short bridge clip from one take and a longer continuation from a later take (e.g. "And hi." from take 1 + "I'm Dean, an ex Amazon AI engineer..." from take 3), the join reads as a jump-cut glitch because both clips show the same speaker framing. Check if the source has a *contiguous* range that covers a natural phrase boundary — if sentence A's `original_end` ≈ sentence B's `original_start` and B's content would smooth the cut, extend A's `adjusted_end` to absorb B, and shift the next kept entry's `adjusted_start` to a comma/clause break. Cuts at commas land much more cleanly than cuts mid-thought. Combine with an image overlay and a 1.1–1.15x zoom on one side (`set-zooms -z 1.15`) to fully mask the join.

Note: this technique relies on `caption_service.build_word_timeline` scanning transcription words by timestamp overlap (not by `adj.index`). The v1 caption builder supports this — see the "v1 caption builder cross-sentence support" note in the legacy section below.

**Backup:** After finalizing `adjusted.json` edits, copy it to `adjusted_backup.json` — later pipeline steps (like re-running silence-detect) may overwrite it.

**User confirmation point**: After Phase 5 produces `cut.mp4`, watch end-to-end and iterate on `adjusted.json` if needed.

### Phase 5: Crossfade Cut
```bash
python cc_wsp/pipeline_v2.py crossfade-cut my_video [--draft]
python cc_wsp/pipeline_v2.py cut-downsample my_video
python cc_wsp/pipeline_v2.py extract-cut-audio my_video
```
`crossfade-cut` reads `adjusted.json` and produces `cut.mp4` (1080p, BT.709) in a **single ffmpeg pass**: hard concat for both video and audio (butt joins). An earlier version used a 100ms `acrossfade` chain — clean joins, but it accumulated ~1s of audio-vs-video drift over a 22-clip cut, so late clips' audio (e.g. a loud jump-scare scream) played under the wrong video. Hard concat eliminates the drift; the tiny clicks at boundaries aren't noticeable on speech. HDR sources are tone-mapped HLG → BT.709 in the same pass. The output's `r_frame_rate` is also snapped to a standard rate (29.97/30/60/etc. via `util_v2.probe_avg_fps`) because iPhone `.mov` advertises `120/1` as the wrapper rate while the actual recorded frames are ~30 fps — MLT downstream reads `r_frame_rate` to decide playback speed and slow-mos the output if you don't normalize.

`--draft` sources `downsampled.mp4` instead of the original `.mov` (skips tonemap+scale, fast encode). Rebuilds in seconds — iterate `adjusted.json` against `--draft` cuts, then do the final pass once you're happy.

`cut-downsample` produces `cut_downsampled.mp4` (240p) for fast preview iteration. `extract-cut-audio` produces `cut_audio.mp3` for re-transcription.

**User confirmation point**: Watch `cut.mp4` end-to-end. **This is where the user reviews both the sentence selection (which takes/lines made the cut) and the audio cuts** — they give feedback from the video, not from the JSON. If a wrong take was kept, a line was dropped, or audio cuts feel off (sibilants dragged, throat clearings leaking, mid-clip silences), go back to `sentences.json` / `adjusted.json` — those are the source of truth — then re-run the cut (`crossfade-cut --force`, plus `silence-detect` if you changed keep/remove decisions).

### Phase 6: Re-transcribe Cut
```bash
python cc_wsp/pipeline_v2.py transcribe-v2 my_video
```
Runs Deepgram on `cut_audio.mp3` → `transcription_v2.json`. Word/sentence timestamps now match the **post-cut** timeline, which is what every downstream step needs. Deepgram may merge or split sentences differently from the original (e.g. 33 cut clips → 24 v2 sentences) because the silence boundaries are gone.

**Re-running this stage can renumber sentences.** Even a ~200ms change to one clip in `adjusted.json` (e.g. tightening a scream's tail) can flip whether Deepgram detects a borderline utterance as its own sentence, or merges it into the next one. When that happens, every downstream reference to a v2 sentence index — `images_v2.json` placements, `zooms.json`, and any caption text patches you've made to `transcription_v2.json` — needs to be re-shifted. After re-running `transcribe-v2`, diff the new sentence list against the old, walk through `images_v2.json` and `zooms.json`, and remap each `sentence_index`. For sentences that got *merged* (e.g. "facial hair" + "AI learns" become one), restore per-image fractions so each image still lands on its intended phrase.

### Phase 7: Image Placement (if google doc exists)
```bash
python cc_wsp/tool.py parse-doc my_video        # parses google_doc_script.json
python cc_wsp/pipeline_v2.py place-images-v2 my_video

# Review
ls cc_wsp/videos/my_video/images_v2.json
```
The LLM placer matches doc images to v2 sentences (`images_v2.json`). Each placement has:
- `filepath`: path to image
- `sentence_index`: v2 sentence index (1-based)
- `start_fraction` / `end_fraction`: when during the sentence (0.0 = start, 1.0 = end)

**Manual review:** The LLM placement is a starting point. After `place-images-v2`, review by cross-referencing the HTML doc structure (text → images order) against the v2 sentence list:
1. Parse the HTML to see which images follow which text lines (the parser may miss images when multiple appear after one line)
2. Match each doc line to the corresponding v2 sentence(s)
3. For rapid-succession images (e.g., episode thumbnails), split them into small equal fractions on the same sentence
4. For section-spanning images, duplicate the placement across consecutive sentences
5. Same image can appear on multiple sentences by adding multiple entries with the same `filepath`

**Concurrent image tiling:** To show multiple images **at the same time** on one sentence (e.g., 4 hook images on the intro line), give them all the same `(sentence_index, start_fraction, end_fraction)`. The renderer detects matching keys, pre-composes a tiled PNG sized to the safe-zone aspect, and renders it as a single overlay. Grid auto-picks to match safe-zone aspect: 2 → 1×2, 3 → 1×3, 4 → 2×2, etc. Composites are cached at `videos/<name>/.composites/` keyed by sentence + fractions + resolution + image hash, so preview (240p) and final (1080p) get separate composites without manual cleanup.

This is a manual edit to `images_v2.json` after `place-images-v2` — the LLM does not produce concurrent groups itself.

### Phase 7b: Zoom Emphasis (optional)
```bash
python cc_wsp/tool.py set-zooms my_video -s 2 -z 1.1
python cc_wsp/tool.py set-zooms my_video -s 8 -z 1.1
```
Adds a subtle zoom on specific sentences for emphasis. Stored in `zooms.json`; picked up automatically by `render-v2` (and the legacy `tool.py render`). Options:
- `-s SENTENCE` — sentence index. **For the v2 render, use v2 sentence indices** (i.e. positions in `transcription_v2.json`), not the original `adjusted.json` indices. Run `transcribe-v2` first, then read off `transcription_v2.json` to pick the right numbers.
- `-z ZOOM` — zoom factor (default 1.3, use 1.1 for subtle emphasis)
- `-x X_OFFSET` / `-y Y_OFFSET` — pan offset (-1 to 1), default 0 (center)
- `--clear` — remove all zooms

How it renders in v2: `build_overlay_mlt` attaches one `affine` filter per zoom to the source-video chain, scoped to that v2 sentence's frame range. So the zoom only affects the **speaker** track — overlay images stay at their normal size/position. The zoom **pops** in at the sentence start and out at the end (it's a static crop per sentence, not an animated push), which is fine at 1.1x.

Good candidates for zoom emphasis:
- Opening hook / first body line — draw the viewer in, add weight to a credibility claim
- Emotional/personal moments — "you're not alone", vulnerability, direct callouts ("one of you asked…")
- Core thesis/message — the main takeaway
- Call to action — "follow for more", "up next", closing statements
- Enthusiastic moments — passion about the subject

Use 1.1x for subtle emphasis (viewer feels it but doesn't consciously notice). Reserve 1.2-1.3x for dramatic moments. Typically 3-5 zoomed sentences per video is enough — too many dilutes the effect.

Zooms live on the MLT render, which is **upstream of captions**, so changing them means re-running `render-v2 --force` then `captions-v2 --force` — and if you've already concatenated alternate hooks onto the front (see "Multi-hook A/B videos" below), rebuild those too.

### Phase 8: Preview & Render
```bash
python cc_wsp/pipeline_v2.py preview-v2 my_video    # 240p MLT overlay → preview_v2.mp4
python cc_wsp/pipeline_v2.py render-v2 my_video     # 1080p final → final_v2.mp4
```
MLT only composes images (and zooms) on top of the pre-built `cut.mp4`. The MLT consumer writes `color_space=gbr`, so both services internally remux with `-c copy` and BT.709 stream tags before producing the final file — no manual remux needed.

**User confirmation point**: Generate preview, ask user for feedback, iterate. Once preview looks good, run `render-v2` for 1080p.

### Phase 9: Captions
```bash
python cc_wsp/pipeline_v2.py captions-v2 my_video
python cc_wsp/pipeline_v2.py captions-v2 my_video --caption-y 0.75
python cc_wsp/pipeline_v2.py captions-v2 my_video --title "Custom Title Text"
python cc_wsp/pipeline_v2.py captions-v2 my_video --no-title
```
Burns word-level captions and an optional title card onto `final_v2.mp4` → `final_v2_captioned.mp4`. Internally re-applies the BT.709 remux after the captioning ffmpeg pass.

**Captions:** White text with black outline (Avenir Next Heavy, 60px). Words grouped into 2-4 word chunks synced to v2 word timestamps. Positioned at `--caption-y` (frame fraction from top, where 0.5 = vertical center).

**Title card:** Blue (#2563EB) rounded-rectangle with white text (Avenir Next Demi Bold). Auto-detected from the Google Doc script `[Title text: "..."]` bracket instruction. Shows during the matched sentence. Override with `--title` or skip with `--no-title`.

**Vertical positioning:** With images in the standard top zone (10-35%) and the speaker's face in the middle, `--caption-y 0.75` places captions in the lower-middle band, below the chin. Adjust ±0.03 (≈ 1 caption-height) if they sit on/above the mouth.

**Caption text fixes:** Captions come from `transcription_v2.json` words. If Deepgram misheard a word (common for "Anthropic" → "morphoanthropic", "in Russian" → "and Russian", missing words like "every"), edit `transcription_v2.json` directly — find the offending word(s) in the relevant sentence's `words` array, change the `word` field, and update the `sentence` text to match. To split one word into two, divide its time range proportionally and append a new word entry; to merge, do the inverse. Then re-run `captions-v2 --force`.

### Phase 10: Deliver to on-deck

**Only after the user has approved the captioned final.** Copy `final_v2_captioned.mp4` into the on-deck folder under a descriptive name:

```bash
cp cc_wsp/videos/my_video/final_v2_captioned.mp4 \
   /Users/brogle/workspace/brogle/videos_on_deck/shorts/<name>.mp4
```

**Naming:** use the project folder name when it already says what the video is (`b324_doctor_riddle` → `b324_doctor_riddle.mp4`). When the folder is just a series number, append the topic (`b216_14` → `b216_14_fundamentals.mp4`). Keep the `b`/`bcai` prefix — that's what distinguishes the two channels.

Don't copy anything here on your own initiative. This folder is his publishing queue, so a file landing in it means "ready to post" — an unapproved cut showing up there is worse than no file at all. Wait for explicit sign-off on `final_v2_captioned.mp4`, then copy.

## File Overview

Phases 1-4 (in `tool.py`):

| File | Description |
|---|---|
| `audio.mp3`, `downsampled.mp4` | Preprocess outputs |
| `google_doc/`, `google_doc_script.json` | Extracted/parsed doc + images |
| `transcription.json` | Deepgram output on the original audio |
| `sentences.json` | Keep/remove decisions per sentence |
| `adjusted.json` | Silence-trimmed timestamps for kept sentences (source of truth for the cut) |
| `adjusted_backup.json` | Manual backup before re-runs |

Phases 5-9 (in `pipeline_v2.py`):

| File | Description |
|---|---|
| `cut.mp4` | 1080p crossfaded cut (Phase 5) |
| `cut_downsampled.mp4` | 240p of cut.mp4 for fast preview |
| `cut_audio.mp3` | Audio extracted from cut.mp4 |
| `transcription_v2.json` | Deepgram output on the cut audio |
| `images_v2.json` | Image placements on the v2 timeline |
| `.composites/tile_*.png` | Auto-generated tiled images for concurrent placements |
| `preview_v2.mp4` | 240p preview with image overlays |
| `final_v2.mp4` | 1080p final render with images |
| `final_v2_captioned.mp4` | Final with captions and (optional) title card |

## Analysis Tools

### audio-levels
```bash
python cc_wsp/tool.py audio-levels my_video 48.0 49.5 --resolution 0.02
```
Shows RMS dB levels with visual bars. `#` = above speech threshold, `.` = below. Use to verify cut points.

### word-timestamps
```bash
python cc_wsp/tool.py word-timestamps my_video 9
```
Shows per-word start/end/duration for a transcription sentence (1-based index). Flags gaps >0.3s with `***`.

### split-sentence
```bash
python cc_wsp/tool.py split-sentence my_video 10 64.0
```
Splits sentence `10` in `adjusted.json` at timestamp `64.0s`. Creates two entries (`10` and `10b`). Use for removing mid-sentence silence.

## HDR / color handling (iPhone HLG footage)

iPhone `.mov` files are HLG (arib-std-b67) / BT.2020 / 10-bit. Without explicit handling the picture comes out pale/washed-out at every stage. The pipeline has several traps — all of them already fixed in `tool.py`, but keep these in mind when debugging a new codepath.

### 1. Downsample must tone-map HLG → BT.709
`_build_scale_vf` in `tool.py` detects HDR inputs via `ffprobe stream=color_transfer,color_primaries`. When the source is HLG or PQ it prepends a full zscale+tonemap chain:

```
zscale=t=linear:npl=100,format=gbrpf32le,
zscale=p=bt709,tonemap=tonemap=hable:desat=0,
zscale=t=bt709:m=bt709:r=tv,format=yuv420p,
<scale expression>
```

**Stock Homebrew ffmpeg does NOT ship `zscale`.** `_get_tonemap_ffmpeg` falls back to the `imageio-ffmpeg` bundled binary (installed as a dep of `moviepy`) because that build includes libzimg. If you ever switch pipelines that shell out to `ffmpeg`, route them through this helper or both `downsampled.mp4` and `1080p.mp4` will decode HLG as SDR and everything downstream washes out.

### 2. Preview (moviepy) drops color tags
`moviepy.write_videofile` doesn't emit `-color_*` flags by default. `video_service.create_edited_video` passes explicit `ffmpeg_params`:

```python
ffmpeg_params=[
    "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-colorspace", "bt709",
    "-color_range", "tv",
]
```

Both the container metadata AND the x264 SEI must be tagged — one or the other is not enough, players (especially Apple's AVFoundation and Instagram's transcoder) will pick the wrong matrix otherwise.

### 3. MLT render writes `color_space=gbr` — fix with a copy-remux
`melt`'s `avformat` consumer decodes the input into its internal RGB image format, and the x264 muxer writes the SEI as `gbr` with no primaries/transfer. **Do not** try to fix this by passing `colorspace=bt709` as a consumer arg — MLT interprets `colorspace=` as the profile's numeric colorspace field, which, when given a non-numeric value or the value `709`, silently falls back to its default 720×576 PAL profile and you get a landscape 4:3 render (discovered the hard way).

The working fix is a post-process remux with `-c copy` that rewrites only the stream metadata:

```bash
ffmpeg -y -i final_raw.mp4 \
  -c copy \
  -color_primaries bt709 -color_trc bt709 \
  -colorspace bt709 -color_range tv \
  final.mp4
```

The v2 services (`mlt_overlay_service.render_overlay`, `captions_v2_service.caption_v2`) already do this remux internally — their public outputs (`final_v2.mp4`, `final_v2_captioned.mp4`) come out tagged correctly. The same applies to `crossfade_service.create_crossfaded_cut` for `cut.mp4`. You only need to do the remux manually if you're using the legacy v1 path or driving `melt` directly.

**Verify** with `ffprobe -v error -show_streams <path> | grep -iE 'pix_fmt|color'` — you want `pix_fmt=yuv420p`, `color_range=tv`, and all three of `color_space/color_transfer/color_primaries=bt709`.

### 4. MLT overlay XML needs a `main_bin` playlist (or the last frame is garbage)

Shotcut XML carries `<mlt producer="main_bin">` — that attribute names the producer `melt` actually renders. `build_overlay_mlt` now emits a real `<playlist id="main_bin">` pointing at the timeline tractor (clipped to the true length). Without it, `melt` falls back to a behaviour that **over-renders ~0.4–0.5s past the timeline**, and in those trailing frames the source-video chain is past its declared length (returns black) while the last overlay image is *held* — so the final frame is the orphaned image floating on black. The same function also makes the source chain + black-background producers a hair longer than the timeline so the genuine last frame still has a real base-video frame under the overlays. If you ever hand-write or post-process one of these MLT files, keep both pieces.

## Title cards (bracket instructions)

Title cards are picked up from `[Title text: "..."]` bracket instructions in the Google Doc script. They render as blue rounded-rect overlays during the matched sentence.

**Sentence matching** (`extract_title_instructions`): the script line containing the instruction is matched against adjusted sentences by content-word overlap. Stop words (`a, the, of, in, on, to, for, is, it, …`) are stripped first — otherwise "Out of microsoft" scores a false 1-point overlap on every sentence containing "of" and maps to the wrong clip. If a new title card is routing to the wrong sentence, add the offending word to the stop-word set in `caption_service.py`.

**Multi-line titles**: Google Docs exports literal `\n` escape sequences inside the bracket text. The renderer replaces them with real newlines and then word-wraps within each paragraph. It also:
- Auto-shrinks the font by 6px per extra line beyond 2 (floor 36px) so 5-line cards don't take up a third of the frame
- Centers the **first** line (the header) and **left-aligns** the subsequent lines so list items ("Top Product: …", "Top Paper: …") stack neatly

**Missing closing quote**: Google Docs exports the opening `"` but often not the closing `"`. The regex is tolerant of this — don't tighten it.

**Vertical position**: `TITLE_CARD_Y_PERCENT` in `constants.py` (currently 0.05 — 5% from top). Raise to push down, lower to push up.

## Image placement on title-card sentences

When a sentence carries a title card, its image would otherwise sit behind the title. The render logic detects title-card sentences (from the same bracket-instruction extraction) and compresses their image safe zone into a lower band:

- **First image** on a title-card sentence → uses `title_band_frac=0.55` (bottom 45% of the safe zone)
- **Second-and-later images** on the same sentence → uses `title_band_frac=0.75` (bottom 25% of the safe zone)

So an S1 hero image sits just under the card, and a follow-up logo banner sits visibly below that. If the title card still overlaps an image, the knobs to turn are the `_compress(0.55)` / `_compress(0.75)` calls in `mlt_overlay_service.build_overlay_mlt` (v2) or `_create_mlt_xml_for_cutting_with_images` (v1), and `TITLE_CARD_Y_PERCENT` / `TITLE_CARD_FONT_SIZE` in `constants.py`.

## Face-derived safe zone

`_face_derived_safe_zone` in `util.py` picks the space above the face for images. The current tuning:

- `above_top = 0.10` (10% from top — reserves room for the title card without crowding the frame edge)
- `above_bottom = face.top_frac - FACE_IMAGE_GAP` with `FACE_IMAGE_GAP = -0.02` (slight overlap into the forehead area so images have more vertical room)

If a user complains that images sit too high and are covered by the title card, raise `above_top` or nudge `FACE_IMAGE_GAP` more negative.

## Multi-hook A/B videos

Sometimes a recording opens with **several alternate hook takes** (the creator films 2–3 hook variants to A/B-test as separate reels), then the shared script body. The deliverable is one final video per hook = `[hook_i + body]`. Editing the body three times would be wasteful; edit it once and reuse it.

1. Preprocess + transcribe the full source as the main project. In `transcription.json`, identify which leading sentences are the hook takes vs. where the body starts.
2. **Edit the body** in the main project the normal way, but in `sentences.json` mark every hook sentence `keep:false`. Run `silence-detect` → fix `adjusted.json` → `crossfade-cut` → `transcribe-v2` → `place-images-v2` → (zooms) → `render-v2` → `captions-v2`. Result: the body's `final_v2_captioned.mp4`.
3. **Edit each hook as a lightweight sub-project** — folder `b289_h1/`, `b289_h2/`, …:
   - symlink the source `.mov` (named to match the folder) and `audio.mp3` from the main project — the hooks come from the same recording, so the audio (and therefore `transcription.json`) is identical;
   - copy `transcription.json` and `face_data.json` from the main project (no need to re-preprocess or re-transcribe);
   - write `sentences.json` keeping only that hook's sentence(s);
   - if a hook carries a doc image (e.g. a "Sun Tzu" pic on the "teach a man to fish" line), symlink `google_doc/` and hand-write `images_v2.json`; for image-less hooks write `{"placements": []}` (`render-v2` needs the file to exist);
   - run `silence-detect` → **carefully fix `adjusted.json`** (short single-clip silence-detect over-trims — see Phase 4) → `crossfade-cut` → `cut-downsample` → `extract-cut-audio` → `transcribe-v2` → `render-v2` → `captions-v2`.
4. **Concatenate** each hook's `final_v2_captioned.mp4` onto the front of the body's `final_v2_captioned.mp4` → `final_hook{1,2,3}_full.mp4` in the main project folder.
   - Don't use `ffmpeg -f concat -c copy`: an MLT-rendered clip can come out 30000/1001 fps while the body is 30/1, and the concat-demuxer copy produces non-monotonic timestamps at the join. Re-encode with a `concat` filter that normalizes fps:
     ```bash
     ffmpeg -y -i hook.mp4 -i body.mp4 -filter_complex \
       "[0:v]fps=30,setsar=1[v0];[1:v]fps=30,setsar=1[v1];[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[v][a]" \
       -map "[v]" -map "[a]" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
       -c:a aac -b:a 192k -ar 48000 /tmp/joined.mp4
     ```
   - That re-encode loses the BT.709 stream tags (you get `smpte170m`/`bt470bg`). Fix without another encode:
     ```bash
     ffmpeg -y -i /tmp/joined.mp4 -c copy \
       -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
       -bsf:v h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0 \
       -movflags +faststart final_hook1_full.mp4
     ```
   - The hook→body join is a hard cut (separate renders), which reads fine — that's the normal hook handoff anyway.

Sub-project source `.mov`s are symlinks, so they don't cost disk; the per-hook `cut.mp4`/`final_v2*.mp4` are small (hooks are a few seconds each).

## Long-form base cut (OBS walkthroughs, course lectures)

For long-form YouTube/course recordings (`videos/youtube/*`, `videos/product/*` — OBS captures), the deliverable is a **first-pass base cut**, not a finished video: remove only (1) long silences and (2) retakes. The user does the tight final cut on top of it, so pad generously and never clip a word. He has said outright that he's fine with leftover silences and word gaps — get the major stuff out and stop; don't get surgical.

Two source profiles show up, and they need different attention:

| | agent-build walkthrough (`obs_prototype_*`) | straight-to-camera lecture (`product/d7/*`) |
|---|---|---|
| runtime | 2h+ | 5–25 min |
| bulk of the waste | dead air waiting on an AI agent (~40 min of it) | retakes, back to back |
| keep rate | ~20–35% | ~50% |
| what to watch for | broken transcription chunks, background audio | multi-take beats, intra-sentence retakes |

**Project layout for a multi-video shoot.** A course section is several recordings under one project (`product/d7/cybersec/`). Give each its own folder inside `vid_raws/` and keep every intermediate for that video in it — audio, transcript, drop spec, cut:

```
vid_raws/
  cybersec_both_raw.mov        # sources stay put
  both/
    both.mov -> ../cybersec_both_raw.mov   # symlink: leaf name must match the folder
    stream_audio.mp3  stream_transcription.json
    base_cut_drops.json  base_cut_clips.json  base_cut_script.txt
    both_cut.mp4                           # the base cut lands here
```

The base cut belongs in that per-video folder, **not** in `vid_cuts/` — `vid_cuts/` holds *his* final edits, made on top of what you hand him.

```bash
# 1. Folder + symlink so tool.py name resolution works
mkdir -p <folder> && ln -sf ../cybersec_both_raw.mov <folder>/both.mov

# 2. 16kHz mono audio + chunked Deepgram (caches each chunk, retries on reset)
ffmpeg -i <src>.mov -vn -acodec libmp3lame -ar 16000 -ac 1 -b:a 96k stream_audio.mp3
python cc_wsp/sandbox/longform_transcribe.py <path/to/video>

# 3. Hand-pick removals into base_cut_drops.json, then render
python cc_wsp/sandbox/longform_base_cut.py <path/to/video> <folder>/base_cut_drops.json --dry-run
python cc_wsp/sandbox/longform_base_cut.py <path/to/video> <folder>/base_cut_drops.json
rm -rf <folder>/_segments     # per-clip intermediates; 137 MB on a 24-min source
```

**Keep the LAST take, always.** The creator re-records until he's happy, so the final attempt of a beat is the keeper — *even when the earlier take is longer or contains content the last one drops*. Don't "rescue" a fuller earlier take; flag it in the summary instead and let him decide. Beats routinely get 3-4 takes, often with abandoned fragments and an expletive between them.

What the last take tends to lose, and what therefore belongs in the flag list: a joke that only landed in the discarded take, an outro line the re-record dropped ("So go out there and make some secure projects"), an example the tighter retake cut, and whole sections he started and abandoned without ever re-recording ("Next is input." — the input-validation section never came back). Cut all of it per the rule, then list it so he can pull any of it back.

**Re-recorded beats interleave — map beats, not sentences.** He doesn't always retake a line and move on; he'll re-record a whole *section* and shuffle the order of its beats between attempts. In `cybersec_both`, sentences 24–41 were five passes at one section, with the beats ("Claude Code read the network info" / "it wrote code that did the same thing" / "now the trades load automatically") reappearing in a different arrangement each pass. Sentence-by-sentence keep-last gives nonsense there. Instead label each sentence with the beat it belongs to, then keep the LAST take *of each beat* — those 18 sentences collapsed to four (35, 36, 40, 41). Contiguous survivors merge into one clip, so the result plays as a single unbroken take.

**Retakes hide *inside* sentences too.** When he restarts a line without pausing, Deepgram has no punctuation boundary to split on and merges every attempt into ONE sentence ("and for every single piece of information that... and for every piece of information that the mobile and web app with users can access"). Dropping whole sentences can't remove those, so `base_cut_drops.json` also takes a `trim` map keyed by sentence index:

```jsonc
"trim": {
  "22": {"from_word": 37},   // start the clip at word 37 — the last attempt
  "60": {"to_word": 12}      // end the clip after word 12 — drop the trailing attempts
}
```

Find the word index with the attempt boundaries (gaps between takes run 0.2–0.9s) and cut on the LAST one. Trim padding uses smaller floors (`TRIM_MIN_LEAD/TRAIL`) than sentence padding, because the adjacent audio is the discarded attempt (speech), not room tone. Every run writes `base_cut_script.txt` — exactly the words that survive, in order — so proofread the trims there instead of scrubbing the render. Skip the trim when Deepgram's word timestamps *overlap* at the boundary (its alignment is unreliable there and the cut will clip a consonant); leave the stutter for the manual pass.

Do **not** use `stream-edit` (LLM highlight/target-duration trimming) — it over-prunes. Keep all content; only remove silence + retakes. Also drop: abandoned fragments, and background audio (music/side conversations during a wait) — flag those for veto.

**Verify silence, don't assume it.** A chunk transcribing to zero sentences might be a broken chunk rather than real silence. Check with `ffmpeg -ss X -t Y -i stream_audio.mp3 -af volumedetect -f null -`: his speech peaks around -1 dB, so a stretch peaking at -20 dB is genuinely dead air (an unattended agent build). One 2h19m recording had ~40 min of it.

**Padding** (`longform_base_cut.py`): lead ≤0.30s, trail ≤0.55s, capped to the neighbouring gap so padding never crosses into an adjacent take. Floors of `MIN_LEAD=0.10` / `MIN_TRAIL=0.15` apply *even when that bleeds a sliver of a discarded take* — at a retake boundary the dropped take often butts straight against the kept one with zero gap, and Deepgram onsets run 100-200ms late, so cutting on the transcript boundary chops the first phoneme. Overlapping clips are merged into one continuous clip rather than split.

**Render frame-accurately.** Each clip is encoded with input `-ss`/`-to` (ffmpeg decodes from the preceding keyframe and discards) and the segments are then concatenated with `-c copy`. Do **not** use the concat demuxer's `inpoint`/`outpoint` on the source: it snaps to a keyframe, and on OBS's sparse-keyframe h264 that shifts cuts by up to a GOP — enough to drag a discarded retake back in. No crossfades (he adds his own edits on top).

**`tool.py stream-transcribe` has no retry** and saves nothing until every chunk lands, so one Deepgram connection reset throws away a 2-hour run. Use `cc_wsp/sandbox/longform_transcribe.py` instead — it caches each chunk's JSON under `_chunks/` and retries with backoff, so a rerun resumes where it left off. Delete `_chunks/` when the cut is done.

## Known limitations

- **Low-res source images**: Google Docs exports images at their embedded resolution, which is often tiny (e.g. a header badge may be 192×58). `_prescale_images` Lanczos-upscales to fit the safe zone, which is the best you can do without AI upscaling — blur in the final render on those images is inherent to the source, not a pipeline bug.
- **`adjusted.json` is fragile**: re-running `silence-detect` can overwrite it. Always `cp adjusted.json adjusted_backup.json` before iterating.
- **Doc parser misses some images**: when the Google Doc has multiple images directly under one text line, the HTML parser may associate only the first one or two with that line. Verify by inspecting the raw `google_doc/<name>.html` and add the missing image filenames manually to `images_v2.json`.

## Legacy v1 path (`tool.py render` / `tool.py captions`)

The original pipeline cuts AND overlays in one MLT pass via `tool.py`. It is the active workflow for shorts (see `feedback_editing_workflow` memory). Per-clip A/V drift was fixed in this session — see notes below.

```bash
python cc_wsp/tool.py parse-doc my_video
python cc_wsp/tool.py place-images my_video      # → images.json
python cc_wsp/tool.py render my_video             # → final.mp4 (needs manual BT.709 remux per HDR section)
python cc_wsp/tool.py captions my_video --caption-y 0.67   # → final_captioned.mp4
```

Notable differences vs v2:
- Uses `images.json` (not `images_v2.json`) and the original `adjusted.json` sentence indices
- Does **not** support concurrent image tiling
- Does **not** internally remux to BT.709; you must do the manual `-c copy` remux on `final.mp4` and `final_captioned.mp4`
- Default `--caption-y` is 0.38 (legacy face-derived) rather than 0.75

### v1 render/preview internals (fixed in May 2026)

The v1 render path has three pieces that all need to agree on the per-clip frame count:

1. **MLT cuts each clip to `int(d_i × fps)` frames** (rounded down per clip) — see `_create_mlt_xml_for_cutting_with_images` in `mlt_video_service.py`.
2. **MLT tractor total length must equal `sum(int(d_i × fps))`**, not `int(sum(d_i) × fps)`. The old formula was up to ~1/3s longer than the playlist, so MLT extended the video with a black tail past the last clip. All three `total_frames = …` sites in `mlt_video_service.py` now use the per-clip sum.
3. **Audio crossfade post-pass** (`_apply_audio_crossfade`, both `mlt_video_service.py` and `video_service.py` for preview) is a *separate ffmpeg pass* — it copies the MLT video stream and rebuilds audio from the original source with `acrossfade` at boundaries. It must receive `fps=…` so each clip's audio endpoint snaps to `start + int(d_i × fps) / fps` — otherwise audio drifts ~1/fps per clip and accumulates to ~0.3s on a 22-cut video. See `feedback_av_frame_quantize` memory.

Crossfade duration is **80ms** in both paths (was previously commented out / 0ms). Larger smears consonants; smaller is barely audible on jump cuts.

### v1 caption builder cross-sentence support

`build_word_timeline` in `caption_service.py` now scans **all** transcription words by timestamp overlap rather than looking up by `adj.index`. This is required when an `adjusted.json` entry extends across multiple original transcription sentences (see `feedback_combine_contiguous_takes` memory) — without it, the absorbed sentence's words are missing from captions. Don't revert to the old per-index lookup.
