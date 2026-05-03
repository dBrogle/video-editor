"""Speech-to-Text service implementations."""

from cc_wsp.src.services.stt.base import SpeechToTextService
from cc_wsp.src.services.stt.deepgram import DeepgramSTTService

__all__ = ["SpeechToTextService", "DeepgramSTTService"]
