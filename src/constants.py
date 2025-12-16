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

# Stage-based file suffixes
# Format: _s{stage_number}_{description}
STAGE_1_DOWNSAMPLED_SUFFIX = "_s1_downsampled"
STAGE_2_AUDIO_SUFFIX = "_s2_audio"
STAGE_3_TRANSCRIPTION_SUFFIX = "_s3_transcription"
STAGE_4_EDITING_DECISION_SUFFIX = "_s4_editing_decision"
STAGE_4_EDITING_RESULT_SUFFIX = "_s4_editing_result"
STAGE_5_ADJUSTED_SENTENCES_SUFFIX = "_s5_adjusted_sentences"
STAGE_6_EDITED_VIDEO_SUFFIX = "_s6_edited"
STAGE_6_DOWNSAMPLED_EDITED_SUFFIX = "_s6_downsampled_edited"
STAGE_7_FINAL_CUT_SUFFIX = "_s7_final_cut"
STAGE_7_AUDIO_SUFFIX = "_s7_audio"
STAGE_7_FINAL_CUT_DOWNSAMPLED_SUFFIX = "_s7_final_cut_downsampled"
STAGE_7_FINAL_CUT_TRANSCRIPTION_SUFFIX = "_s7_final_cut_transcription"

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
