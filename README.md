# AI Video Editing Pipeline

Modular AI-assisted video editing pipeline with speech-to-text transcription.

## Architecture

- **LLMs never process raw media** — only structured data
- **All intermediate assets organized in `/assets/{video_name}/` folders**
- **All service APIs wrapped behind abstract base classes**
- **All service outputs converted to internal Pydantic models**

## File Structure

Assets are organized into folders by video name:

```
assets/
  IMG_2362/
    IMG_2362.MOV              # Original video
    s1_downsampled.mp4        # Step 1: Downsampled video (preprocessing)
    s1_audio.wav              # Step 1: Extracted audio (preprocessing)
    s2_transcription.json     # Step 2: Transcription
    s3_editing_decision.json  # Step 3: Initial LLM editing decision
    s3_editing_result.json    # Step 3: Human-editable format
    s5_adjusted_sentences.json # Step 5: Timestamps with silence removed
    s6_downsampled_edited.mp4 # Step 6: Preview video (iteration)
    google_doc/               # Step 7-8: Google Doc script and images
      IMG_2362.html           # Google Doc HTML export
      images/                 # Images from Google Doc
        image_001.png
        image_002.png
    s7_google_doc_script.json # Step 7: Parsed Google Doc script
    s8_google_doc_image_placements.json # Step 8: Image placement data
    s9_with_google_doc_images.mp4 # Step 9: Downsampled video with images
    s10_full_res_with_images.mp4 # Step 10: Final full-res video with images
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg  # macOS

cp .env.example .env
# Edit .env with your API keys
```

## Migration (Existing Users)

If you have existing assets in the old flat structure (`IMG_2362_s1_downsampled.mp4`), use the migration script to move them to the new folder structure:

```bash
# Preview what will be migrated (dry run)
python migrate_to_folders.py

# Actually perform the migration
python migrate_to_folders.py --execute
```

The script will:
- Create a folder for each video (e.g., `assets/IMG_2362/`)
- Move all stage files into their respective folders
- Move source videos into their folders (e.g., `IMG_2362.MOV` → `IMG_2362/IMG_2362.MOV`)

## Usage

```bash
python main.py
```

### Quick Examples

```python
from src.services.video import VideoService
from src.services.stt.deepgram import DeepgramSTTService
from src.services.local_saver import LocalSaverService
from src.util import prepare_transcript_for_prompt

# Process video
video_service = VideoService()
proxy_video, audio_file = video_service.process_video("assets/input.mp4")

# Transcribe (automatically generates sentences)
stt = DeepgramSTTService()
transcript = stt.transcribe(audio_file)

# Sentences are now available directly in transcript
for sentence in transcript.sentences:
    print(sentence)  # [{start}-{end}]-{sentence}

# Or use utility function (returns transcript.sentences if available)
sentences = prepare_transcript_for_prompt(transcript)
```

## Data Models

```python
class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float

class LLMTranscriptSentence(BaseModel):
    sentence: str
    start: float
    end: float
    words: list[WordTimestamp]

class Transcript(BaseModel):
    sentences: list[LLMTranscriptSentence]  # Sentences with word-level timestamps
    language: str | None
    duration: float | None

class EditingDecision(BaseModel):  # LLM response
    thoughts: str
    sentences_to_remove: list[int]

class SentenceResult(BaseModel):
    text: str
    keep: bool

class EditingResult(BaseModel):  # Human-editable format
    sentence_results: dict[str, SentenceResult]

class AdjustedSentence(BaseModel):
    original_start: float
    original_end: float
    adjusted_start: float  # After silence removal
    adjusted_end: float    # After silence removal
    text: str

class AdjustedSentences(BaseModel):
    sentences: list[AdjustedSentence]
```

## Pipeline Steps

The video editing pipeline consists of 10 main steps:

1. **Preprocess video** - Rotate if needed, downsample, and extract audio in one step
2. **Get transcription** - Transcribe audio with word-level timestamps
3. **Initial edit with LLM** - Get AI suggestions for which sentences to remove
4. **Iterate sentence selection** - Interactive AI agent for refining which sentences to keep/remove
5. **Generate adjusted sentences** - Analyze audio and remove silence from clip boundaries
6. **Iterate adjusted sentences** - Interactive AI agent for fine-tuning timestamps and pacing
7. **Parse Google Doc script** - Extract text lines and images from Google Doc HTML export
8. **Place Google Doc images** - Use LLM to match script images to video timeline
9. **Create downsampled video with images** - Generate preview video with image overlays
10. **Create full res video with images** - Generate final full-resolution video (single pass)

### Advanced Options

- **Step 11**: Cut full resolution video only (no images)
- **Step 12**: Add images to full resolution video (two-step approach)

## Editing Workflow

The pipeline creates three editable files:

1. **`_editing_decision.json`** - Raw LLM response with thoughts and sentence numbers
2. **`_editing_result.json`** - Human-editable format mapping each sentence to keep/remove
3. **`_adjusted_sentences.json`** - Sentence timestamps with silence trimmed from start/end

You can manually edit these files between steps:

**Edit `s3_editing_result.json` to change which sentences to keep:**
```json
{
  "sentence_results": {
    "1": {"text": "Sentence text here", "keep": true},
    "2": {"text": "Another sentence", "keep": false}
  }
}
```

**Edit `s5_adjusted_sentences.json` to fine-tune timing:**
```json
{
  "sentences": [
    {
      "original_start": 0.0,
      "original_end": 2.5,
      "adjusted_start": 0.1,
      "adjusted_end": 2.4,
      "text": "Sentence text here"
    }
  ]
}
```

## AI Feedback Agents (Steps 4 & 6)

The pipeline includes two interactive AI agents that help refine your video cut in separate steps.

### Step 4: Sentence Selection Iteration
1. Review the downsampled video to see which sentences are included
2. Provide feedback on which sentences to keep or remove (e.g., "Remove sentence 5", "Keep sentence 12")
3. The Sentence Selection Agent updates the editing result (s3_editing_result.json)
4. Video is regenerated with the new sentence selection
5. Loop continues until you approve the sentence selection

### Step 6: Timestamp Adjustment Iteration
1. Review the approved sentence selection with programmatically-generated timestamps
2. Provide feedback on timing and pacing (e.g., "Cut 2 seconds from the beginning", "Reduce pause between sentence 3 and 4")
3. The Timestamp Adjustment Agent updates the adjusted sentences (s5_adjusted_sentences.json)
4. Video is regenerated with the new timestamps
5. Loop continues until you approve the timestamps

### Sentence Selection Agent Actions

- **Keep sentence** - Mark a sentence to be kept in the final video
- **Remove sentence** - Mark a sentence to be removed from the final video
- **Approve** - Finalize sentence selection and move to timestamp adjustment stage

### Timestamp Adjustment Agent Actions

- **Adjust timestamps** - Modify start/end times of any sentence (uses word-level timestamps)
- **Approve** - Finalize the cut and proceed to the next step

### Example Feedback

#### Step 4 (Sentence Selection)
```
💬 Is the sentence selection good?
Your feedback: Remove sentences 6 and 7, they're filler

🤖 Agent thoughts: User wants to remove sentences 6 and 7 as they are filler content...
   Executing: remove_sentence with {'sentence_index': '6'}
   ✓ Marked sentence 6 to be REMOVED
   Executing: remove_sentence with {'sentence_index': '7'}
   ✓ Marked sentence 7 to be REMOVED

🎬 Generating video with current sentence selection...
```

#### Step 6 (Timestamp Adjustment)
```
💬 How do the timestamps look?
Your feedback: The pause between sentence 3 and 4 is too long

🤖 Agent thoughts: User wants to reduce the gap between sentences 3 and 4...
   Executing: adjust_timestamp with {'sentence_index': '4', 'field': 'adjusted_start', 'new_value': 9.2}
   ✓ Adjusted sentence 4 adjusted_start to 9.2s

🎬 Regenerating video with timestamp adjustments...
```

### Usage in Code

```python
from src.services.agents import SentenceSelectionAgent, TimestampAdjustmentAgent
from src.services.local_saver import LocalSaverService

# Step 4: Sentence Selection
sentence_agent = SentenceSelectionAgent()
saver = LocalSaverService()

editing_result = saver.load_editing_result("video_name")
user_feedback = "Remove sentence 5"
updated_result, is_approved = sentence_agent.process_feedback(
    editing_result=editing_result,
    user_feedback=user_feedback,
)

# Step 6: Timestamp Adjustment
timestamp_agent = TimestampAdjustmentAgent()
adjusted_sentences = saver.load_adjusted_sentences("video_name")
user_feedback = "Cut 1 second from the start"
updated_sentences, is_approved = timestamp_agent.process_feedback(
    adjusted_sentences=adjusted_sentences,
    user_feedback=user_feedback,
)
```

## MLT Video Service

The project includes an MLT-based video service that uses the MLT framework for efficient video editing. This is an alternative to the default MoviePy/ffmpeg approach.

### Installation

```bash
# macOS
brew install mlt

# Ubuntu/Debian
sudo apt-get install melt
```

### Usage

```python
from src.services.video.mlt_video_service import MLTVideoService
from src.services.local_saver import LocalSaver

# Initialize services
mlt_service = MLTVideoService()
saver = LocalSaver()

# Load adjusted sentences
adjusted_sentences = saver.load_adjusted_sentences("video_name")

# Create final cut using MLT
output_path = mlt_service.create_final_cut_with_mlt(
    base_name="video_name",
    adjusted_sentences=adjusted_sentences,
    force=False  # Set to True to regenerate
)
```

Or use the test script:

```bash
python test_mlt.py
```

### How it works in the Pipeline

Steps 9-10 of the pipeline use the MLT video service to create videos with image overlays:

1. Reads the adjusted sentences from step 5 (with silence-trimmed timestamps)
2. Reads the image placements from step 8 (with timing and positioning data)
3. Detects the original video's properties (resolution, framerate)
4. Generates an MLT XML file with video clips and image overlays
5. Runs `melt` command to render the final video
6. Cleans up temporary files

This approach is more efficient than loading the entire video into memory, especially for large high-resolution files.

### MLT XML Format

The service generates MLT XML files like this:

```xml
<?xml version="1.0"?>
<mlt LC_NUMERIC="C" version="7.0.1" root="/path/to/video/folder">
  <profile description="HD 1080p 30 fps" 
          width="1920" height="1080" 
          progressive="1" 
          frame_rate_num="30" frame_rate_den="1"/>
  
  <producer id="source_video">
    <property name="resource">/path/to/video.mp4</property>
    <property name="mlt_service">avformat</property>
  </producer>
  
  <playlist id="main_playlist">
    <entry producer="source_video" in="0" out="150"/>
    <entry producer="source_video" in="300" out="600"/>
  </playlist>
  
  <tractor id="main_tractor">
    <track producer="main_playlist"/>
  </tractor>
</mlt>
```

## Google Doc Image Overlays (Steps 7-9)

Steps 7-9 add images from a Google Doc script as overlays to your video.

### Features

- **Google Doc Integration**: Export your script as HTML with embedded images
- **LLM-Powered Placement**: An LLM analyzes your script and video to place images intelligently
- **Smart Positioning**: Images placed in a "safe zone" (20-40% height, 30-70% width)
- **Automatic Timing**: Images synced to specific sentences in the transcript
- **MLT Integration**: Efficient video compositing using MLT framework

### Workflow

1. **Step 7 - Parse Google Doc**: Extract text lines and images from HTML export
2. **Step 8 - Place Images**: LLM matches script images to video timeline
3. **Step 9 - Create Video**: MLT composites images onto downsampled video
4. **Step 10 - Final Output**: MLT creates full-resolution video with images

### Usage

```bash
python main.py
# Run steps 7, 8, 9, and 10 for complete workflow
```

### Google Doc Setup

1. Create your script in Google Docs with embedded images
2. Export as HTML: File → Download → Web Page (.html, zipped)
3. Extract the zip file
4. Place the HTML file and images folder in: `assets/{video_name}/google_doc/`
   - HTML file: `assets/{video_name}/google_doc/{video_name}.html`
   - Images: `assets/{video_name}/google_doc/images/`

### Configuration

Image safe zone can be adjusted in `src/constants.py`:

```python
IMAGE_SAFE_ZONE_TOP_PERCENT = 0.20    # Start at 20% from top
IMAGE_SAFE_ZONE_BOTTOM_PERCENT = 0.40  # End at 40% from top
IMAGE_SAFE_ZONE_LEFT_PERCENT = 0.30    # Start at 30% from left
IMAGE_SAFE_ZONE_RIGHT_PERCENT = 0.70   # End at 70% from left
```

## Requirements

- Python 3.10+
- ffmpeg
- Deepgram API key (for transcription) - **Recommended** for better sentence segmentation
  - Alternative: ElevenLabs API key (legacy support)
- OpenRouter API key (for LLM editing decisions and image generation)
- MLT framework (required for stages 7-8)
  - macOS: `brew install mlt`
  - Ubuntu: `sudo apt-get install melt`
