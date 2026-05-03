# cc_wsp — Claude Code Video Editing Workspace

This workspace provides tools for Claude Code to edit short-form vertical videos. All commands go through `tool.py`.

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
# All commands
python cc_wsp/tool.py <command> <video> [args...]

# Check what exists
python cc_wsp/tool.py status my_video

# View any step file
python cc_wsp/tool.py show my_video transcription
python cc_wsp/tool.py show my_video sentences
python cc_wsp/tool.py show my_video adjusted
python cc_wsp/tool.py show my_video images
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

**User confirmation point**: Show the user the kept/removed sentences and ask if they want changes.

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

# Generate and review preview
python cc_wsp/tool.py preview my_video
```

For fixes: edit `adjusted.json` directly — change `adjusted_start`/`adjusted_end` values, or use `split-sentence` to break apart sentences with internal gaps.

**Common issues to check for:**
- Cut-off sibilants (s, z) at sentence ends — extend `adjusted_end`
- Cut-off plosives (k, t, p) — extend `adjusted_end`
- Trailing silence/pause at end — tighten `adjusted_end`
- Retakes within a sentence — trim `adjusted_start` to skip the first take
- Large silence gaps mid-sentence — split the sentence

**Sibilant review workflow:** After silence detection, check every sentence ending with a sibilant sound (s, z, x, sh, etc.) using `audio-levels` with `--resolution 0.01` or `0.02` zoomed in on the last ~0.5s. If the sibilant energy extends past the `adjusted_end`, extend it by ~0.05-0.10s. Conversely, if the user reports trailing silence, tighten by small amounts (0.02-0.05s).

**Partial retakes within a kept sentence:** Sometimes a sentence contains a false start/retake (flagged by `***` gaps in `word-timestamps`). Rather than removing the whole sentence, keep it and trim `adjusted_start` past the retake using the word timestamps as a guide. Check `audio-levels` to find clean speech onset.

**Backup:** After finalizing `adjusted.json` edits, copy it to `adjusted_backup.json` — later pipeline steps (like re-running silence-detect) may overwrite it.

**User confirmation point**: Generate preview, ask user for feedback, iterate.

### Phase 5: Image Placement (if google doc exists)
```bash
python cc_wsp/tool.py parse-doc my_video
python cc_wsp/tool.py place-images my_video

# Review
python cc_wsp/tool.py show my_video images
```
Edit `images.json` directly to adjust placements. Each placement has:
- `filepath`: path to image
- `sentence_index`: which sentence it appears on
- `start_fraction` / `end_fraction`: when during the sentence (0.0 = start, 1.0 = end)

**Manual review:** The LLM placement is a starting point. After `place-images`, review by cross-referencing the HTML doc structure (text → images order) against the actual video sentences:
1. Parse the HTML to see which images follow which text lines (the parser may miss images when multiple appear after one line)
2. Match each doc line to the corresponding video sentence(s)
3. For rapid-succession images (e.g., episode thumbnails), split them into small equal fractions on the same sentence
4. For section-spanning images, duplicate the placement across consecutive sentences
5. Same image can appear on multiple sentences by adding multiple entries with the same `filepath`

### Phase 5b: Zoom Emphasis (optional)
```bash
python cc_wsp/tool.py set-zooms my_video -s 2 -z 1.1
python cc_wsp/tool.py set-zooms my_video -s 8 -z 1.1
```
Adds a subtle zoom on specific sentences for emphasis. Options:
- `-s SENTENCE` — sentence index to zoom
- `-z ZOOM` — zoom factor (default 1.3, use 1.1 for subtle emphasis)
- `-x X_OFFSET` / `-y Y_OFFSET` — pan offset (-1 to 1), default 0 (center)
- `--clear` — remove all zooms

Good candidates for zoom emphasis:
- Opening hook — draw the viewer in immediately
- Emotional/personal moments — "you're not alone", vulnerability
- Core thesis/message — the main takeaway
- Call to action — "up next", closing statements
- Enthusiastic moments — passion about the subject

Use 1.1x for subtle emphasis (viewer feels it but doesn't consciously notice). Reserve 1.2-1.3x for dramatic moments. Typically 3-5 zoomed sentences per video is enough — too many dilutes the effect.

### Phase 6: Render
```bash
python cc_wsp/tool.py render my_video
```
Creates the final 1080p video with cuts, image overlays, and zoom filters.

### Phase 7: Captions (optional)
```bash
python cc_wsp/tool.py captions my_video
python cc_wsp/tool.py captions my_video --caption-y 0.38   # custom vertical position
python cc_wsp/tool.py captions my_video --title "Custom Title Text"
python cc_wsp/tool.py captions my_video --no-title
```
Post-renders word-level captions and an optional title card onto `final.mp4` → `final_captioned.mp4`.

**Captions:** White text with black outline (Avenir Next Heavy, 60px). Words grouped into 2-4 word chunks synced to Deepgram word timestamps. Positioned at `--caption-y` (default 0.38 = 38% from top).

**Title card:** Blue (#2563EB) rounded-rectangle with white text (Avenir Next Demi Bold). Auto-detected from the Google Doc script `Text: "..."` line. Shows during the first sentence. Override with `--title` or skip with `--no-title`.

**Vertical positioning tip:** For videos with images at 45-70% (like b216_0), use `--caption-y 0.38` to place captions just above. For standard image zone (10-40%), use `--caption-y 0.06` or omit for default.

## File Overview

| File | Description |
|---|---|
| `transcription.json` | Deepgram output: sentences with word timestamps |
| `sentences.json` | Keep/remove decisions per sentence |
| `adjusted.json` | Silence-trimmed timestamps for kept sentences |
| `images.json` | Google doc image placements on timeline |
| `adjusted_backup.json` | Manual backup of adjusted.json before re-runs |
| `preview.mp4` | Low-res preview from current adjusted.json |
| `final.mp4` | Final 1080p render |

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

The working fix is a post-process remux with `-c copy`:

```bash
ffmpeg -y -i final_raw.mp4 \
  -c copy \
  -color_primaries bt709 -color_trc bt709 \
  -colorspace bt709 -color_range tv \
  final.mp4
```

Rewrites only the stream metadata, no re-encode. The same remux is needed on `final_captioned.mp4` because the captions step re-encodes via ffmpeg overlay filters and also drops the color flags.

**Render workflow**:
```bash
python cc_wsp/tool.py render my_video
mv videos/my_video/final.mp4 videos/my_video/final_raw.mp4
ffmpeg -y -i videos/my_video/final_raw.mp4 -c copy \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
  videos/my_video/final.mp4
rm videos/my_video/final_raw.mp4

python cc_wsp/tool.py captions my_video --caption-y 0.67
mv videos/my_video/final_captioned.mp4 videos/my_video/final_captioned_raw.mp4
ffmpeg -y -i videos/my_video/final_captioned_raw.mp4 -c copy \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
  videos/my_video/final_captioned.mp4
rm videos/my_video/final_captioned_raw.mp4
```

**Verify** with `ffprobe -v error -show_streams <path> | grep -iE 'pix_fmt|color'` — you want `pix_fmt=yuv420p`, `color_range=tv`, and all three of `color_space/color_transfer/color_primaries=bt709`.

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

So an S1 hero image sits just under the card, and a follow-up logo banner sits visibly below that. If the title card still overlaps an image, the knobs to turn are `title_band_frac` in `mlt_video_service._create_mlt_xml_for_cutting_with_images` and `TITLE_CARD_Y_PERCENT` / `TITLE_CARD_FONT_SIZE` in `constants.py`.

## Face-derived safe zone

`_face_derived_safe_zone` in `util.py` picks the space above the face for images. The current tuning:

- `above_top = 0.10` (10% from top — reserves room for the title card without crowding the frame edge)
- `above_bottom = face.top_frac - FACE_IMAGE_GAP` with `FACE_IMAGE_GAP = -0.02` (slight overlap into the forehead area so images have more vertical room)

If a user complains that images sit too high and are covered by the title card, raise `above_top` or nudge `FACE_IMAGE_GAP` more negative.

## Known limitations

- **Low-res source images**: Google Docs exports images at their embedded resolution, which is often tiny (e.g. a header badge may be 192×58). `_prescale_images` Lanczos-upscales to fit the safe zone, which is the best you can do without AI upscaling — blur in the final render on those images is inherent to the source, not a pipeline bug.
- **`adjusted.json` is fragile**: re-running `silence-detect` or `place-images` can overwrite it. Always `cp adjusted.json adjusted.safe.json` before iterating.
