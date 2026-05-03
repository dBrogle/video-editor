"""
Configuration constants for the cc_wsp video editing workspace.
"""

from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

# Base paths
WORKSPACE_ROOT = Path(__file__).parent.parent
VIDEOS_DIR = WORKSPACE_ROOT / "videos"

# Load environment variables from parent project
PROJECT_ROOT = WORKSPACE_ROOT.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, verbose=False)
load_dotenv(verbose=False)

# Video processing
LOW_RES_HEIGHT = 240
HD_1080P_HEIGHT = 1920

# Audio
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "mp3"
AUDIO_BITRATE = "32k"
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "fast"
AUDIO_CODEC = "libmp3lame"

# Silence detection (multiband_zcr)
SPEECH_LEVEL_PERCENTILE = 85
RMS_OFFSET_DB = 18
NOISE_FLOOR_MARGIN_DB = 8
HF_OFFSET_DB = 20
ZCR_CENTROID_MIN = 0.08
CENTROID_MIN = 1800
END_HOLD_FRAMES = 14
START_PADDING = 0.02
END_PADDING = 0.01
HF_DECLINE_LIMIT = 14
CLIP_DB_DIFFERENCE_THRESHOLD = 5

# Transcription
TIME_BETWEEN_WORDS_THRESHOLD = 1.0

# Image overlay safe zone (vertical video)
IMAGE_SAFE_ZONE_TOP_PERCENT = 0.10
IMAGE_SAFE_ZONE_BOTTOM_PERCENT = 0.35
IMAGE_SAFE_ZONE_LEFT_PERCENT = 0.15
IMAGE_SAFE_ZONE_RIGHT_PERCENT = 0.85

# Rendering
LOW_RES_CRF = 24
HIGH_RES_CRF = 19

# Captions
CAPTION_FONT_PATH = str(WORKSPACE_ROOT / "assets" / "fonts" / "Milliard-Bold.otf")
CAPTION_FONT_INDEX = 0
CAPTION_FONT_SIZE = 60
CAPTION_Y_PERCENT = 0.55
CAPTION_OUTLINE_WIDTH = 3
CAPTION_MAX_WORDS_PER_CHUNK = 4  # fallback, character limit takes priority
CAPTION_MAX_CHARS_PER_CHUNK = 20
CAPTION_COLOR = (255, 255, 255)  # white
CAPTION_OUTLINE_COLOR = (0, 0, 0)  # black

# Title card
TITLE_CARD_FONT_PATH = str(WORKSPACE_ROOT / "assets" / "fonts" / "Milliard-SemiBold.otf")
TITLE_CARD_FONT_INDEX = 0
TITLE_CARD_BG_COLOR = (37, 99, 235)  # #2563EB
TITLE_CARD_TEXT_COLOR = (255, 255, 255)  # white
TITLE_CARD_FONT_SIZE = 44
TITLE_CARD_Y_PERCENT = 0.15
TITLE_CARD_CORNER_RADIUS = 24
TITLE_CARD_PADDING_H = 44
TITLE_CARD_PADDING_V = 24
TITLE_CARD_MAX_WIDTH_PERCENT = 0.85  # max 85% of video width
TITLE_CARD_DURATION = 7.0  # seconds

# Step file names
STEP_AUDIO = "audio"
STEP_DOWNSAMPLED = "downsampled"
STEP_TRANSCRIPTION = "transcription"
STEP_SENTENCES = "sentences"
STEP_ADJUSTED = "adjusted"
STEP_IMAGES = "images"
STEP_PREVIEW = "preview"
STEP_FINAL = "final"

# API
ENV_DEEPGRAM_API_KEY = "DEEPGRAM_API_KEY"
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterModel(str, Enum):
    GPT_51 = "openai/gpt-5.1"
    GEMINI_3_FLASH = "google/gemini-3-flash-preview"
    CLAUDE_SONNET_45 = "anthropic/claude-sonnet-4.5"
