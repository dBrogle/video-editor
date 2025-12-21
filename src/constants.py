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

# Stage-based file names (without prefix)
# Format: s{stage_number}_{description}
STAGE_1_DOWNSAMPLED_NAME = "s1_downsampled"
STAGE_2_AUDIO_NAME = "s2_audio"
STAGE_3_TRANSCRIPTION_NAME = "s3_transcription"
STAGE_4_EDITING_DECISION_NAME = "s4_editing_decision"
STAGE_4_EDITING_RESULT_NAME = "s4_editing_result"
STAGE_5_ADJUSTED_SENTENCES_NAME = "s5_adjusted_sentences"
STAGE_6_EDITED_VIDEO_NAME = "s6_edited"
STAGE_6_DOWNSAMPLED_EDITED_NAME = "s6_downsampled_edited"
STAGE_7_WITH_IMAGES_DOWNSAMPLED_NAME = "s7_with_images_downsampled"
STAGE_7_IMAGES_FOLDER_NAME = "images"
STAGE_7_IMAGES_METADATA_NAME = "images_metadata"
STAGE_7_MLT_XML_NAME = "s7_with_images_mlt"
STAGE_8_GOOGLE_DOC_SCRIPT_NAME = "s8_google_doc_script"
STAGE_9_GOOGLE_DOC_IMAGE_PLACEMENTS_NAME = "s9_google_doc_image_placements"
STAGE_10_WITH_GOOGLE_DOC_IMAGES_NAME = "s10_with_google_doc_images"
STAGE_10_MLT_XML_NAME = "s10_with_google_doc_images_mlt"
# Full resolution stages
STAGE_11_FULL_RES_CUT_NAME = "s11_full_res_cut"
STAGE_11_FULL_RES_CUT_MLT_NAME = "s11_full_res_cut_mlt"
STAGE_12_FULL_RES_WITH_IMAGES_NAME = "s12_full_res_with_images"
STAGE_12_FULL_RES_WITH_IMAGES_MLT_NAME = "s12_full_res_with_images_mlt"

# Audio settings
AUDIO_SAMPLE_RATE = 16000  # 16kHz
AUDIO_CHANNELS = 1  # Mono
AUDIO_FORMAT = "wav"

# Video settings
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "fast"
AUDIO_CODEC = "pcm_s16le"

# Silence detection (future use)
SILENCE_THRESHOLD_DB = -40
SILENCE_MIN_DURATION = 0.5

# Image overlay settings
# Position: 20th-40th percentile height (centered vertically in upper portion, 1/3 from top)
# Position: 30th-70th percentile width (centered horizontally)
IMAGE_SAFE_ZONE_TOP_PERCENT = 0.20  # Start at 20% from top
IMAGE_SAFE_ZONE_BOTTOM_PERCENT = 0.40  # End at 40% from top
IMAGE_SAFE_ZONE_LEFT_PERCENT = 0.30  # Start at 30% from left
IMAGE_SAFE_ZONE_RIGHT_PERCENT = 0.70  # End at 70% from left
IMAGE_DEFAULT_WIDTH = 1024
IMAGE_DEFAULT_HEIGHT = 1024

# Environment variable names
ENV_ELEVENLABS_API_KEY = "ELEVENLABS_API_KEY"
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"

# API endpoints
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# OpenRouter Model Options
class OpenRouterModel(str, Enum):
    """Available models for OpenRouter API."""

    # OpenAI Models
    GPT_51 = "openai/gpt-5.1"
    GEMINI_25_FLASH = "google/gemini-2.5-flash"
    CLAUDE_SONNET_45 = "anthropic/claude-sonnet-4.5"


# OpenRouter Image Generation Model Options
class OpenRouterImageModel(str, Enum):
    """Available image generation models for OpenRouter API."""

    GEMINI_25_FLASH_IMAGE = "google/gemini-2.5-flash-image"
    GEMINI_3_PRO_IMAGE_PREVIEW = "google/gemini-3-pro-image-preview"
    FLUX_2_PRO = "black-forest-labs/flux.2-pro"
