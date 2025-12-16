# AI Video Editing Pipeline

Modular AI-assisted video editing pipeline with speech-to-text transcription.

## Architecture

- **LLMs never process raw media** — only structured data
- **All intermediate assets live in `/assets`**
- **All service APIs wrapped behind abstract base classes**
- **All service outputs converted to internal Pydantic models**

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg  # macOS

cp .env.example .env
# Edit .env with your API keys
```

## Usage

```bash
python main.py
```

### Quick Examples

```python
from src.services.video import VideoService
from src.services.stt.elevenlabs import ElevenLabsSTTService
from src.services.local_saver import LocalSaverService
from src.util import prepare_transcript_for_prompt

# Process video
video_service = VideoService()
proxy_video, audio_file = video_service.process_video("assets/input.mp4")

# Transcribe (automatically generates sentences)
stt = ElevenLabsSTTService()
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

class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    words: list[WordTimestamp]

class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    sentences: list[LLMTranscriptSentence]  # Pre-computed sentences for editing
    language: str | None
    duration: float | None

class LLMTranscriptSentence(BaseModel):
    sentence: str
    start: float
    end: float

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

The video editing pipeline consists of 7 steps:

1. **Downsample video** - Create low-res proxy for faster processing
2. **Extract audio** - Extract audio track as WAV file
3. **Get transcription** - Transcribe audio with word-level timestamps
4. **Prompt LLM for editing** - Get AI suggestions for which sentences to remove
5. **Generate adjusted sentences** - Analyze audio and remove silence from clip boundaries
6. **Create edited video** - Generate edited video using adjusted timestamps (downsampled)
7. **Create final cut** - Generate full-resolution final cut using MLT framework, extract audio, downsample and transcribe

## Editing Workflow

The pipeline creates three editable files:

1. **`_editing_decision.json`** - Raw LLM response with thoughts and sentence numbers
2. **`_editing_result.json`** - Human-editable format mapping each sentence to keep/remove
3. **`_adjusted_sentences.json`** - Sentence timestamps with silence trimmed from start/end

You can manually edit these files between steps:

**Edit `_editing_result.json` to change which sentences to keep:**
```json
{
  "sentence_results": {
    "1": {"text": "Sentence text here", "keep": true},
    "2": {"text": "Another sentence", "keep": false}
  }
}
```

**Edit `_adjusted_sentences.json` to fine-tune timing:**
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

Step 7 of the pipeline now uses the MLT video service to create the final cut:

1. Reads the adjusted sentences from step 5 (with silence-trimmed timestamps)
2. Detects the original video's properties (resolution, framerate)
3. Generates a temporary MLT XML file with all the clip segments
4. Runs `melt` command to render the final high-resolution video
5. Cleans up temporary files
6. Then extracts audio, downsamples, and transcribes the final cut

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

## Requirements

- Python 3.10+
- ffmpeg
- ElevenLabs API key
- MLT framework (required for step 7 - final cut generation)
  - macOS: `brew install mlt`
  - Ubuntu: `sudo apt-get install melt`
