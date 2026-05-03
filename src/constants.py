"""
Static configuration constants for the video editing pipeline.
No logic allowed - only configuration values.
"""

from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

# Base paths - source of truth for all paths in the system
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ENV_FILE = PROJECT_ROOT / ".env"

# Load environment variables from .env file
# Try the project .env file first, then fall back to current working directory
load_dotenv(dotenv_path=ENV_FILE, verbose=False)
load_dotenv(verbose=False)  # Also check CWD

# Video processing settings
LOW_RES_HEIGHT = 240  # 240p resolution
HD_1080P_HEIGHT = 1920  # 1080p resolution for vertical video (1080x1920)

# Stage-based file names (without prefix)
# Format: s{stage_number}_{description}
# New pipeline structure:
# Step 1: Preprocess (rotate, downsample, extract audio)
STAGE_1_DOWNSAMPLED_NAME = "s1_downsampled"
STAGE_1_AUDIO_NAME = "s1_audio"
# Step 2: Get transcription
STAGE_2_TRANSCRIPTION_NAME = "s2_transcription"
# Step 3: Initial edit with LLM
STAGE_3_EDITING_DECISION_NAME = "s3_editing_decision"
STAGE_3_EDITING_RESULT_NAME = "s3_editing_result"
# Step 4: Iterate sentence selection
STAGE_4_SENTENCE_SELECTION_VIDEO_NAME = "s4_sentence_selection_video"
STAGE_4_FINAL_EDITING_RESULT_NAME = "s4_final_editing_result"
# Step 5: Generate adjusted sentences
STAGE_5_ADJUSTED_SENTENCES_NAME = "s5_adjusted_sentences"
# Step 6: Iterate adjusted sentences
STAGE_6_ADJUSTED_SENTENCES_VIDEO_NAME = "s6_adjusted_sentences_video"
STAGE_6_FINAL_ADJUSTED_SENTENCES_NAME = "s6_final_adjusted_sentences"
# Legacy names (kept for backward compatibility)
STAGE_6_EDITED_VIDEO_NAME = "s6_edited"
STAGE_6_DOWNSAMPLED_EDITED_NAME = "s6_downsampled_edited"
# Step 7: Parse Google Doc script
STAGE_7_GOOGLE_DOC_SCRIPT_NAME = "s7_google_doc_script"
# Step 8: Place Google Doc images
STAGE_8_GOOGLE_DOC_IMAGE_PLACEMENTS_NAME = "s8_google_doc_image_placements"
# Step 9: Create downsampled video with Google Doc images
STAGE_9_WITH_GOOGLE_DOC_IMAGES_NAME = "s9_with_google_doc_images"
STAGE_9_MLT_XML_NAME = "s9_with_google_doc_images_mlt"
# Step 10: Downsample full res video to 1080p
STAGE_10_1080P_DOWNSAMPLE_NAME = "s10_1080p_downsample"
# Step 11: Create 1080p video with images (single pass on downsampled)
STAGE_11_1080P_WITH_IMAGES_NAME = "s11_1080p_with_images"
STAGE_11_1080P_WITH_IMAGES_MLT_NAME = "s11_1080p_with_images_mlt"
# Step 12: Create full res video with images (single pass)
STAGE_12_FULL_RES_WITH_IMAGES_NAME = "s12_full_res_with_images"
STAGE_12_FULL_RES_WITH_IMAGES_MLT_NAME = "s12_full_res_with_images_mlt"
# Advanced: Two-step approach
STAGE_13_FULL_RES_CUT_NAME = "s13_full_res_cut"
STAGE_13_FULL_RES_CUT_MLT_NAME = "s13_full_res_cut_mlt"
STAGE_14_FULL_RES_WITH_IMAGES_NAME = "s14_full_res_with_images"
STAGE_14_FULL_RES_WITH_IMAGES_MLT_NAME = "s14_full_res_with_images_mlt"
# Stream pipeline stages
STREAM_STAGE_1_DOWNSAMPLED_NAME = "stream_s1_downsampled"
STREAM_STAGE_1_AUDIO_NAME = "stream_s1_audio"
STREAM_STAGE_2_TRANSCRIPTION_NAME = "stream_s2_transcription"
STREAM_STAGE_3_EDITING_RESULT_NAME = "stream_s3_editing_result"
STREAM_STAGE_4_SENTENCE_SELECTION_VIDEO_NAME = "stream_s4_sentence_selection_video"
STREAM_STAGE_4_FINAL_EDITING_RESULT_NAME = "stream_s4_final_editing_result"
STREAM_STAGE_5_ADJUSTED_SENTENCES_NAME = "stream_s5_adjusted_sentences"
STREAM_STAGE_6_ADJUSTED_SENTENCES_VIDEO_NAME = "stream_s6_adjusted_sentences_video"
STREAM_STAGE_6_FINAL_ADJUSTED_SENTENCES_NAME = "stream_s6_final_adjusted_sentences"
STREAM_STAGE_7_FINAL_VIDEO_NAME = "stream_s7_final"
STREAM_STAGE_7_MLT_NAME = "stream_s7_final_mlt"

# Stream processing settings
STREAM_LOW_RES_HEIGHT = 360  # Slightly higher than shorts proxy for readability
STREAM_CHUNK_SIZE = 100  # Sentences per LLM chunk
STREAM_CHUNK_OVERLAP = 5  # Overlap between chunks

# Legacy/shared resources
STAGE_7_IMAGES_FOLDER_NAME = "images"
STAGE_7_IMAGES_METADATA_NAME = "images_metadata"
STAGE_7_MLT_XML_NAME = "s7_with_images_mlt"

# Audio settings
AUDIO_SAMPLE_RATE = 16000  # 16kHz
AUDIO_CHANNELS = 1  # Mono
AUDIO_FORMAT = "mp3"
AUDIO_BITRATE = "32k"  # MP3 bitrate for transcription quality

# Video settings
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "fast"
AUDIO_CODEC = "libmp3lame"

# Silence detection
SILENCE_THRESHOLD_DB = -40
SILENCE_MIN_DURATION = 0.5
SILENCE_EXTENSION = 0.3  # Seconds to extend clips when silence was trimmed

# Transcription sentence splitting
TIME_BETWEEN_WORDS_THRESHOLD = (
    1.0  # Seconds - split sentences if gap between words exceeds this
)

# Image overlay settings
# Position: 10th-45th percentile height (upper portion)
# Position: 10th-90th percentile width (wide)
IMAGE_SAFE_ZONE_TOP_PERCENT = 0.10
IMAGE_SAFE_ZONE_BOTTOM_PERCENT = 0.40
IMAGE_SAFE_ZONE_LEFT_PERCENT = 0.15
IMAGE_SAFE_ZONE_RIGHT_PERCENT = 0.85
IMAGE_DEFAULT_WIDTH = 1024
IMAGE_DEFAULT_HEIGHT = 1024

# Environment variable names
ENV_ELEVENLABS_API_KEY = "ELEVENLABS_API_KEY"
ENV_DEEPGRAM_API_KEY = "DEEPGRAM_API_KEY"
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"

# API endpoints
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# OpenRouter Model Options
class OpenRouterModel(str, Enum):
    """Available models for OpenRouter API."""

    # OpenAI Models
    GPT_51 = "openai/gpt-5.1"
    GEMINI_3_FLASH = "google/gemini-3-flash-preview"
    CLAUDE_SONNET_45 = "anthropic/claude-sonnet-4.5"


# OpenRouter Image Generation Model Options
class OpenRouterImageModel(str, Enum):
    """Available image generation models for OpenRouter API."""

    GEMINI_25_FLASH_IMAGE = "google/gemini-2.5-flash-image"
    GEMINI_3_PRO_IMAGE_PREVIEW = "google/gemini-3-pro-image-preview"
    FLUX_2_PRO = "black-forest-labs/flux.2-pro"
