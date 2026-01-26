"""
Deepgram Speech-to-Text service implementation.
Converts Deepgram API responses to internal models using the official SDK.
"""

import os
from pathlib import Path
from typing import Any, Dict

from deepgram import DeepgramClient  # type: ignore

from src.services.stt.base import SpeechToTextService
from src.models import (
    Transcript,
    WordTimestamp,
    LLMTranscriptSentence,
)
from src.constants import ENV_DEEPGRAM_API_KEY
from src.util import validate_file_exists

# SAMPLE_RESPONSE = {
#     "metadata": {
#         "transaction_key": "deprecated",
#         "request_id": "decbf1a4-18ed-4082-8a60-53c022953d59",
#         "sha256": "50c3f710b324c94b45ac1d7b2759cd4bc4211e6e41197117bbd7fee2f7bbb334",
#         "created": "2026-01-20T16:23:40.588Z",
#         "duration": 244.58669,
#         "channels": 1,
#         "models": ["2187e11a-3532-4498-b076-81fa530bdd49"],
#         "model_info": {
#             "2187e11a-3532-4498-b076-81fa530bdd49": {
#                 "name": "general-nova-3",
#                 "version": "2025-07-31.0",
#                 "arch": "nova-3",
#             }
#         },
#     },
#     "results": {
#         "channels": [
#             {
#                 "alternatives": [
#                     {
#                         "transcript": "Why not? Google's AI famously once labeled black people as gorillas. So why did that happen, and how can you fix this? This is day 13 of AI fundamentals at forty four seconds and forty four days. Google's Photos AI once famously labeled black people as Google's AI once famously labeled black people as gorillas. So how did this happen and how can you fix this? Well, this is day 44. Fuck. Google's AI once famously labeled black people as gorillas. So how did this happen and how can you fix it? This is day 13 of AI fundamentals in forty four seconds and forty four days. Overfitting, underfitting, and bias are three common concepts in AI. Underfitting is when AI doesn't recognize enough patterns in the data, so it underfits to it. And over fitting is where AI kind of memorizes the data it was trained on and over fits to it. But how do you fix this? For underfitting, you can train the model more or increase its complexity so it can learn more from the data. And, for overfitting, you can train it more or ah, fuck. And for overfitting, you can train it less or reduce its complexity. And there's a few clever tricks we'll get into tomorrow. And bias is what happened to Google. This is where it does worse for some groups. And, it's usually caused by two things. First is bad data. If when training it on people, it's only ever given white people, then when you give it a black person, it'll be like, I've never seen this as a person before. And second is just having a dumb AI, which was also a problem in 2015. So, how did Google fix this? Well, as mentioned, in 2015, AI was pretty dumb. So they just remove gorillas as an option because you can't call people gorillas if you can't call anything a gorilla. But today, AI is a retake. But today, AI is big brain, so overfitting is a big problem. And how that can be fixed is up tomorrow. Cheers.",
#                         "confidence": 0.99742717,
#                         "words": [
#                             {
#                                 "word": "why",
#                                 "start": 12.16,
#                                 "end": 12.48,
#                                 "confidence": 0.39997545,
#                                 "punctuated_word": "Why",
#                             },
#                             {
#                                 "word": "not",
#                                 "start": 12.48,
#                                 "end": 12.799999,
#                                 "confidence": 0.6307834,
#                                 "punctuated_word": "not?",
#                             },
#                             {
#                                 "word": "google's",
#                                 "start": 43.855003,
#                                 "end": 44.335003,
#                                 "confidence": 0.9975623,
#                                 "punctuated_word": "Google's",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 44.335003,
#                                 "end": 44.575,
#                                 "confidence": 0.9940765,
#                                 "punctuated_word": "AI",
#                             },
#                             {
#                                 "word": "famously",
#                                 "start": 44.575,
#                                 "end": 45.055,
#                                 "confidence": 0.9990926,
#                                 "punctuated_word": "famously",
#                             },
#                             {
#                                 "word": "once",
#                                 "start": 45.055,
#                                 "end": 45.295002,
#                                 "confidence": 0.99454427,
#                                 "punctuated_word": "once",
#                             },
#                             {
#                                 "word": "labeled",
#                                 "start": 45.295002,
#                                 "end": 45.535004,
#                                 "confidence": 0.9516256,
#                                 "punctuated_word": "labeled",
#                             },
#                             {
#                                 "word": "black",
#                                 "start": 45.535004,
#                                 "end": 45.855003,
#                                 "confidence": 0.9838334,
#                                 "punctuated_word": "black",
#                             },
#                             {
#                                 "word": "people",
#                                 "start": 45.855003,
#                                 "end": 46.095,
#                                 "confidence": 0.9998155,
#                                 "punctuated_word": "people",
#                             },
#                             {
#                                 "word": "as",
#                                 "start": 46.095,
#                                 "end": 46.335003,
#                                 "confidence": 0.9992742,
#                                 "punctuated_word": "as",
#                             },
#                             {
#                                 "word": "gorillas",
#                                 "start": 46.335003,
#                                 "end": 46.975002,
#                                 "confidence": 0.95713127,
#                                 "punctuated_word": "gorillas.",
#                             },
#                             {
#                                 "word": "so",
#                                 "start": 46.975002,
#                                 "end": 47.055,
#                                 "confidence": 0.9984487,
#                                 "punctuated_word": "So",
#                             },
#                             {
#                                 "word": "why",
#                                 "start": 47.055,
#                                 "end": 47.375,
#                                 "confidence": 0.97938424,
#                                 "punctuated_word": "why",
#                             },
#                             {
#                                 "word": "did",
#                                 "start": 47.375,
#                                 "end": 47.535004,
#                                 "confidence": 0.9987386,
#                                 "punctuated_word": "did",
#                             },
#                             {
#                                 "word": "that",
#                                 "start": 47.535004,
#                                 "end": 47.695,
#                                 "confidence": 0.99948394,
#                                 "punctuated_word": "that",
#                             },
#                             {
#                                 "word": "happen",
#                                 "start": 47.695,
#                                 "end": 48.175003,
#                                 "confidence": 0.7335682,
#                                 "punctuated_word": "happen,",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 48.175003,
#                                 "end": 48.255,
#                                 "confidence": 0.9998584,
#                                 "punctuated_word": "and",
#                             },
#                             {
#                                 "word": "how",
#                                 "start": 48.255,
#                                 "end": 48.495003,
#                                 "confidence": 0.9997813,
#                                 "punctuated_word": "how",
#                             },
#                             {
#                                 "word": "can",
#                                 "start": 48.495003,
#                                 "end": 48.655003,
#                                 "confidence": 0.9993906,
#                                 "punctuated_word": "can",
#                             },
#                             {
#                                 "word": "you",
#                                 "start": 48.655003,
#                                 "end": 48.735,
#                                 "confidence": 0.976843,
#                                 "punctuated_word": "you",
#                             },
#                             {
#                                 "word": "fix",
#                                 "start": 48.735,
#                                 "end": 48.975002,
#                                 "confidence": 0.9997875,
#                                 "punctuated_word": "fix",
#                             },
#                             {
#                                 "word": "this",
#                                 "start": 48.975002,
#                                 "end": 49.535,
#                                 "confidence": 0.9988083,
#                                 "punctuated_word": "this?",
#                             },
#                             {
#                                 "word": "this",
#                                 "start": 54.655003,
#                                 "end": 54.895,
#                                 "confidence": 0.99974483,
#                                 "punctuated_word": "This",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 54.895,
#                                 "end": 55.055,
#                                 "confidence": 0.99948126,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "day",
#                                 "start": 55.055,
#                                 "end": 55.295002,
#                                 "confidence": 0.99281305,
#                                 "punctuated_word": "day",
#                             },
#                             {
#                                 "word": "13",
#                                 "start": 55.295002,
#                                 "end": 55.615,
#                                 "confidence": 0.9995827,
#                                 "punctuated_word": "13",
#                             },
#                             {
#                                 "word": "of",
#                                 "start": 55.615,
#                                 "end": 55.775,
#                                 "confidence": 0.999734,
#                                 "punctuated_word": "of",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 55.775,
#                                 "end": 56.015,
#                                 "confidence": 0.9953101,
#                                 "punctuated_word": "AI",
#                             },
#                             {
#                                 "word": "fundamentals",
#                                 "start": 56.015,
#                                 "end": 56.415,
#                                 "confidence": 0.92229897,
#                                 "punctuated_word": "fundamentals",
#                             },
#                             {
#                                 "word": "at",
#                                 "start": 56.415,
#                                 "end": 56.655003,
#                                 "confidence": 0.44414923,
#                                 "punctuated_word": "at",
#                             },
#                             {
#                                 "word": "forty",
#                                 "start": 56.655003,
#                                 "end": 56.895,
#                                 "confidence": 0.99844366,
#                                 "punctuated_word": "forty",
#                             },
#                             {
#                                 "word": "four",
#                                 "start": 56.895,
#                                 "end": 57.055,
#                                 "confidence": 0.9994947,
#                                 "punctuated_word": "four",
#                             },
#                             {
#                                 "word": "seconds",
#                                 "start": 57.055,
#                                 "end": 57.375,
#                                 "confidence": 0.9988451,
#                                 "punctuated_word": "seconds",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 57.375,
#                                 "end": 57.535004,
#                                 "confidence": 0.9406322,
#                                 "punctuated_word": "and",
#                             },
#                             {
#                                 "word": "forty",
#                                 "start": 57.535004,
#                                 "end": 57.775,
#                                 "confidence": 0.9997693,
#                                 "punctuated_word": "forty",
#                             },
#                             {
#                                 "word": "four",
#                                 "start": 57.775,
#                                 "end": 57.935,
#                                 "confidence": 0.9996612,
#                                 "punctuated_word": "four",
#                             },
#                             {
#                                 "word": "days",
#                                 "start": 57.935,
#                                 "end": 58.255,
#                                 "confidence": 0.9995063,
#                                 "punctuated_word": "days.",
#                             },
#                             {
#                                 "word": "google's",
#                                 "start": 63.129997,
#                                 "end": 63.61,
#                                 "confidence": 0.9954593,
#                                 "punctuated_word": "Google's",
#                             },
#                             {
#                                 "word": "photos",
#                                 "start": 63.61,
#                                 "end": 63.93,
#                                 "confidence": 0.7700598,
#                                 "punctuated_word": "Photos",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 63.93,
#                                 "end": 64.25,
#                                 "confidence": 0.9575833,
#                                 "punctuated_word": "AI",
#                             },
#                             {
#                                 "word": "once",
#                                 "start": 64.25,
#                                 "end": 64.57,
#                                 "confidence": 0.96850574,
#                                 "punctuated_word": "once",
#                             },
#                             {
#                                 "word": "famously",
#                                 "start": 64.57,
#                                 "end": 64.89,
#                                 "confidence": 0.9988995,
#                                 "punctuated_word": "famously",
#                             },
#                             {
#                                 "word": "labeled",
#                                 "start": 64.89,
#                                 "end": 65.21,
#                                 "confidence": 0.95826656,
#                                 "punctuated_word": "labeled",
#                             },
#                             {
#                                 "word": "black",
#                                 "start": 65.21,
#                                 "end": 65.45,
#                                 "confidence": 0.98308384,
#                                 "punctuated_word": "black",
#                             },
#                             {
#                                 "word": "people",
#                                 "start": 65.45,
#                                 "end": 65.69,
#                                 "confidence": 0.99940085,
#                                 "punctuated_word": "people",
#                             },
#                             {
#                                 "word": "as",
#                                 "start": 65.69,
#                                 "end": 66.01,
#                                 "confidence": 0.8759564,
#                                 "punctuated_word": "as",
#                             },
#                             {
#                                 "word": "google's",
#                                 "start": 66.25,
#                                 "end": 69.13,
#                                 "confidence": 0.916495,
#                                 "punctuated_word": "Google's",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 69.13,
#                                 "end": 69.369995,
#                                 "confidence": 0.9979095,
#                                 "punctuated_word": "AI",
#                             },
#                             {
#                                 "word": "once",
#                                 "start": 69.369995,
#                                 "end": 69.69,
#                                 "confidence": 0.99478346,
#                                 "punctuated_word": "once",
#                             },
#                             {
#                                 "word": "famously",
#                                 "start": 69.69,
#                                 "end": 70.09,
#                                 "confidence": 0.9997251,
#                                 "punctuated_word": "famously",
#                             },
#                             {
#                                 "word": "labeled",
#                                 "start": 70.09,
#                                 "end": 70.409996,
#                                 "confidence": 0.9965563,
#                                 "punctuated_word": "labeled",
#                             },
#                             {
#                                 "word": "black",
#                                 "start": 70.409996,
#                                 "end": 70.65,
#                                 "confidence": 0.99870396,
#                                 "punctuated_word": "black",
#                             },
#                             {
#                                 "word": "people",
#                                 "start": 70.65,
#                                 "end": 70.89,
#                                 "confidence": 0.99955696,
#                                 "punctuated_word": "people",
#                             },
#                             {
#                                 "word": "as",
#                                 "start": 70.89,
#                                 "end": 71.13,
#                                 "confidence": 0.9991672,
#                                 "punctuated_word": "as",
#                             },
#                             {
#                                 "word": "gorillas",
#                                 "start": 71.13,
#                                 "end": 71.77,
#                                 "confidence": 0.98630935,
#                                 "punctuated_word": "gorillas.",
#                             },
#                             {
#                                 "word": "so",
#                                 "start": 71.77,
#                                 "end": 71.93,
#                                 "confidence": 0.99903226,
#                                 "punctuated_word": "So",
#                             },
#                             {
#                                 "word": "how",
#                                 "start": 71.93,
#                                 "end": 72.25,
#                                 "confidence": 0.5357615,
#                                 "punctuated_word": "how",
#                             },
#                             {
#                                 "word": "did",
#                                 "start": 72.25,
#                                 "end": 72.409996,
#                                 "confidence": 0.99842024,
#                                 "punctuated_word": "did",
#                             },
#                             {
#                                 "word": "this",
#                                 "start": 72.409996,
#                                 "end": 72.57,
#                                 "confidence": 0.9990433,
#                                 "punctuated_word": "this",
#                             },
#                             {
#                                 "word": "happen",
#                                 "start": 72.57,
#                                 "end": 72.81,
#                                 "confidence": 0.99345434,
#                                 "punctuated_word": "happen",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 72.81,
#                                 "end": 73.049995,
#                                 "confidence": 0.54905945,
#                                 "punctuated_word": "and",
#                             },
#                             {
#                                 "word": "how",
#                                 "start": 73.049995,
#                                 "end": 73.369995,
#                                 "confidence": 0.9986217,
#                                 "punctuated_word": "how",
#                             },
#                             {
#                                 "word": "can",
#                                 "start": 73.369995,
#                                 "end": 73.45,
#                                 "confidence": 0.99765223,
#                                 "punctuated_word": "can",
#                             },
#                             {
#                                 "word": "you",
#                                 "start": 73.45,
#                                 "end": 73.61,
#                                 "confidence": 0.9980469,
#                                 "punctuated_word": "you",
#                             },
#                             {
#                                 "word": "fix",
#                                 "start": 73.61,
#                                 "end": 73.77,
#                                 "confidence": 0.99953485,
#                                 "punctuated_word": "fix",
#                             },
#                             {
#                                 "word": "this",
#                                 "start": 73.77,
#                                 "end": 74.17,
#                                 "confidence": 0.9971649,
#                                 "punctuated_word": "this?",
#                             },
#                             {
#                                 "word": "well",
#                                 "start": 74.17,
#                                 "end": 74.49,
#                                 "confidence": 0.9897163,
#                                 "punctuated_word": "Well,",
#                             },
#                             {
#                                 "word": "this",
#                                 "start": 74.49,
#                                 "end": 74.65,
#                                 "confidence": 0.99906164,
#                                 "punctuated_word": "this",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 74.65,
#                                 "end": 74.729996,
#                                 "confidence": 0.9962631,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "day",
#                                 "start": 74.729996,
#                                 "end": 74.97,
#                                 "confidence": 0.99549484,
#                                 "punctuated_word": "day",
#                             },
#                             {
#                                 "word": "44",
#                                 "start": 74.97,
#                                 "end": 75.505005,
#                                 "confidence": 0.81271994,
#                                 "punctuated_word": "44.",
#                             },
#                             {
#                                 "word": "fuck",
#                                 "start": 75.505005,
#                                 "end": 76.225,
#                                 "confidence": 0.9794826,
#                                 "punctuated_word": "Fuck.",
#                             },
#                             {
#                                 "word": "google's",
#                                 "start": 82.545,
#                                 "end": 83.025,
#                                 "confidence": 0.995272,
#                                 "punctuated_word": "Google's",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 83.025,
#                                 "end": 83.265,
#                                 "confidence": 0.9701957,
#                                 "punctuated_word": "AI",
#                             },
#                             {
#                                 "word": "once",
#                                 "start": 83.265,
#                                 "end": 83.505005,
#                                 "confidence": 0.98342055,
#                                 "punctuated_word": "once",
#                             },
#                             {
#                                 "word": "famously",
#                                 "start": 83.505005,
#                                 "end": 83.905,
#                                 "confidence": 0.9986992,
#                                 "punctuated_word": "famously",
#                             },
#                             {
#                                 "word": "labeled",
#                                 "start": 83.905,
#                                 "end": 84.225,
#                                 "confidence": 0.9663158,
#                                 "punctuated_word": "labeled",
#                             },
#                             {
#                                 "word": "black",
#                                 "start": 84.225,
#                                 "end": 84.465004,
#                                 "confidence": 0.9877759,
#                                 "punctuated_word": "black",
#                             },
#                             {
#                                 "word": "people",
#                                 "start": 84.465004,
#                                 "end": 84.705,
#                                 "confidence": 0.99971503,
#                                 "punctuated_word": "people",
#                             },
#                             {
#                                 "word": "as",
#                                 "start": 84.705,
#                                 "end": 84.945,
#                                 "confidence": 0.9993112,
#                                 "punctuated_word": "as",
#                             },
#                             {
#                                 "word": "gorillas",
#                                 "start": 84.945,
#                                 "end": 85.585,
#                                 "confidence": 0.98337907,
#                                 "punctuated_word": "gorillas.",
#                             },
#                             {
#                                 "word": "so",
#                                 "start": 85.585,
#                                 "end": 85.665,
#                                 "confidence": 0.99973696,
#                                 "punctuated_word": "So",
#                             },
#                             {
#                                 "word": "how",
#                                 "start": 85.665,
#                                 "end": 85.905,
#                                 "confidence": 0.95639473,
#                                 "punctuated_word": "how",
#                             },
#                             {
#                                 "word": "did",
#                                 "start": 85.905,
#                                 "end": 86.145004,
#                                 "confidence": 0.99819463,
#                                 "punctuated_word": "did",
#                             },
#                             {
#                                 "word": "this",
#                                 "start": 86.145004,
#                                 "end": 86.225,
#                                 "confidence": 0.99920744,
#                                 "punctuated_word": "this",
#                             },
#                             {
#                                 "word": "happen",
#                                 "start": 86.225,
#                                 "end": 86.465004,
#                                 "confidence": 0.993909,
#                                 "punctuated_word": "happen",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 86.465004,
#                                 "end": 86.785,
#                                 "confidence": 0.94261605,
#                                 "punctuated_word": "and",
#                             },
#                             {
#                                 "word": "how",
#                                 "start": 86.785,
#                                 "end": 87.025,
#                                 "confidence": 0.9996952,
#                                 "punctuated_word": "how",
#                             },
#                             {
#                                 "word": "can",
#                                 "start": 87.025,
#                                 "end": 87.105,
#                                 "confidence": 0.99862695,
#                                 "punctuated_word": "can",
#                             },
#                             {
#                                 "word": "you",
#                                 "start": 87.105,
#                                 "end": 87.265,
#                                 "confidence": 0.999081,
#                                 "punctuated_word": "you",
#                             },
#                             {
#                                 "word": "fix",
#                                 "start": 87.265,
#                                 "end": 87.505005,
#                                 "confidence": 0.99977845,
#                                 "punctuated_word": "fix",
#                             },
#                             {
#                                 "word": "it",
#                                 "start": 87.505005,
#                                 "end": 87.665,
#                                 "confidence": 0.99734974,
#                                 "punctuated_word": "it?",
#                             },
#                             {
#                                 "word": "this",
#                                 "start": 90.92,
#                                 "end": 91.159996,
#                                 "confidence": 0.99985826,
#                                 "punctuated_word": "This",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 91.159996,
#                                 "end": 91.32,
#                                 "confidence": 0.9989718,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "day",
#                                 "start": 91.32,
#                                 "end": 91.56,
#                                 "confidence": 0.9920476,
#                                 "punctuated_word": "day",
#                             },
#                             {
#                                 "word": "13",
#                                 "start": 91.56,
#                                 "end": 91.88,
#                                 "confidence": 0.9991398,
#                                 "punctuated_word": "13",
#                             },
#                             {
#                                 "word": "of",
#                                 "start": 91.88,
#                                 "end": 92.04,
#                                 "confidence": 0.99942964,
#                                 "punctuated_word": "of",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 92.04,
#                                 "end": 92.36,
#                                 "confidence": 0.9826904,
#                                 "punctuated_word": "AI",
#                             },
#                             {
#                                 "word": "fundamentals",
#                                 "start": 92.36,
#                                 "end": 92.76,
#                                 "confidence": 0.85259837,
#                                 "punctuated_word": "fundamentals",
#                             },
#                             {
#                                 "word": "in",
#                                 "start": 92.76,
#                                 "end": 93,
#                                 "confidence": 0.9472564,
#                                 "punctuated_word": "in",
#                             },
#                             {
#                                 "word": "forty",
#                                 "start": 93,
#                                 "end": 93.24,
#                                 "confidence": 0.93274295,
#                                 "punctuated_word": "forty",
#                             },
#                             {
#                                 "word": "four",
#                                 "start": 93.24,
#                                 "end": 93.4,
#                                 "confidence": 0.99908984,
#                                 "punctuated_word": "four",
#                             },
#                             {
#                                 "word": "seconds",
#                                 "start": 93.4,
#                                 "end": 93.72,
#                                 "confidence": 0.9992066,
#                                 "punctuated_word": "seconds",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 93.72,
#                                 "end": 93.88,
#                                 "confidence": 0.93865484,
#                                 "punctuated_word": "and",
#                             },
#                             {
#                                 "word": "forty",
#                                 "start": 93.88,
#                                 "end": 94.119995,
#                                 "confidence": 0.9996555,
#                                 "punctuated_word": "forty",
#                             },
#                             {
#                                 "word": "four",
#                                 "start": 94.119995,
#                                 "end": 94.36,
#                                 "confidence": 0.9990682,
#                                 "punctuated_word": "four",
#                             },
#                             {
#                                 "word": "days",
#                                 "start": 94.36,
#                                 "end": 94.92,
#                                 "confidence": 0.99951243,
#                                 "punctuated_word": "days.",
#                             },
#                             {
#                                 "word": "overfitting",
#                                 "start": 97.8,
#                                 "end": 98.52,
#                                 "confidence": 0.9787144,
#                                 "punctuated_word": "Overfitting,",
#                             },
#                             {
#                                 "word": "underfitting",
#                                 "start": 98.52,
#                                 "end": 99,
#                                 "confidence": 0.9264496,
#                                 "punctuated_word": "underfitting,",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 99,
#                                 "end": 99.08,
#                                 "confidence": 0.9994468,
#                                 "punctuated_word": "and",
#                             },
#                             {
#                                 "word": "bias",
#                                 "start": 99.08,
#                                 "end": 99.4,
#                                 "confidence": 0.96035,
#                                 "punctuated_word": "bias",
#                             },
#                             {
#                                 "word": "are",
#                                 "start": 99.4,
#                                 "end": 99.56,
#                                 "confidence": 0.9955848,
#                                 "punctuated_word": "are",
#                             },
#                             {
#                                 "word": "three",
#                                 "start": 99.56,
#                                 "end": 99.8,
#                                 "confidence": 0.99938405,
#                                 "punctuated_word": "three",
#                             },
#                             {
#                                 "word": "common",
#                                 "start": 99.8,
#                                 "end": 100.04,
#                                 "confidence": 0.99919194,
#                                 "punctuated_word": "common",
#                             },
#                             {
#                                 "word": "concepts",
#                                 "start": 100.04,
#                                 "end": 100.44,
#                                 "confidence": 0.9996302,
#                                 "punctuated_word": "concepts",
#                             },
#                             {
#                                 "word": "in",
#                                 "start": 100.44,
#                                 "end": 100.68,
#                                 "confidence": 0.99875975,
#                                 "punctuated_word": "in",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 100.68,
#                                 "end": 101,
#                                 "confidence": 0.99450684,
#                                 "punctuated_word": "AI.",
#                             },
#                             {
#                                 "word": "underfitting",
#                                 "start": 108.365,
#                                 "end": 109.005,
#                                 "confidence": 0.9791035,
#                                 "punctuated_word": "Underfitting",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 109.005,
#                                 "end": 109.165,
#                                 "confidence": 0.9988619,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "when",
#                                 "start": 109.165,
#                                 "end": 109.325,
#                                 "confidence": 0.9990922,
#                                 "punctuated_word": "when",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 109.325,
#                                 "end": 109.645004,
#                                 "confidence": 0.998131,
#                                 "punctuated_word": "AI",
#                             },
#                             {
#                                 "word": "doesn't",
#                                 "start": 109.645004,
#                                 "end": 109.885,
#                                 "confidence": 0.99917805,
#                                 "punctuated_word": "doesn't",
#                             },
#                             {
#                                 "word": "recognize",
#                                 "start": 109.885,
#                                 "end": 110.285,
#                                 "confidence": 0.99887174,
#                                 "punctuated_word": "recognize",
#                             },
#                             {
#                                 "word": "enough",
#                                 "start": 110.285,
#                                 "end": 110.525,
#                                 "confidence": 0.99993646,
#                                 "punctuated_word": "enough",
#                             },
#                             {
#                                 "word": "patterns",
#                                 "start": 110.525,
#                                 "end": 110.845,
#                                 "confidence": 0.99935263,
#                                 "punctuated_word": "patterns",
#                             },
#                             {
#                                 "word": "in",
#                                 "start": 110.845,
#                                 "end": 111.005,
#                                 "confidence": 0.99742717,
#                                 "punctuated_word": "in",
#                             },
#                             {
#                                 "word": "the",
#                                 "start": 111.005,
#                                 "end": 111.165,
#                                 "confidence": 0.9989384,
#                                 "punctuated_word": "the",
#                             },
#                             {
#                                 "word": "data",
#                                 "start": 111.165,
#                                 "end": 111.565,
#                                 "confidence": 0.86882913,
#                                 "punctuated_word": "data,",
#                             },
#                             {
#                                 "word": "so",
#                                 "start": 111.565,
#                                 "end": 111.645004,
#                                 "confidence": 0.99962735,
#                                 "punctuated_word": "so",
#                             },
#                             {
#                                 "word": "it",
#                                 "start": 111.645004,
#                                 "end": 112.045,
#                                 "confidence": 0.9984573,
#                                 "punctuated_word": "it",
#                             },
#                             {
#                                 "word": "underfits",
#                                 "start": 112.045,
#                                 "end": 112.605,
#                                 "confidence": 0.94316036,
#                                 "punctuated_word": "underfits",
#                             },
#                             {
#                                 "word": "to",
#                                 "start": 112.605,
#                                 "end": 112.845,
#                                 "confidence": 0.99932647,
#                                 "punctuated_word": "to",
#                             },
#                             {
#                                 "word": "it",
#                                 "start": 112.845,
#                                 "end": 113.005,
#                                 "confidence": 0.98079586,
#                                 "punctuated_word": "it.",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 116.38,
#                                 "end": 116.62,
#                                 "confidence": 0.8410498,
#                                 "punctuated_word": "And",
#                             },
#                             {
#                                 "word": "over",
#                                 "start": 116.62,
#                                 "end": 116.94,
#                                 "confidence": 0.9641975,
#                                 "punctuated_word": "over",
#                             },
#                             {
#                                 "word": "fitting",
#                                 "start": 116.94,
#                                 "end": 117.18,
#                                 "confidence": 0.6492947,
#                                 "punctuated_word": "fitting",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 117.18,
#                                 "end": 117.42,
#                                 "confidence": 0.99517524,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "where",
#                                 "start": 117.42,
#                                 "end": 117.66,
#                                 "confidence": 0.99833614,
#                                 "punctuated_word": "where",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 117.66,
#                                 "end": 118.06,
#                                 "confidence": 0.96223676,
#                                 "punctuated_word": "AI",
#                             },
#                             {
#                                 "word": "kind",
#                                 "start": 118.06,
#                                 "end": 118.22,
#                                 "confidence": 0.95977783,
#                                 "punctuated_word": "kind",
#                             },
#                             {
#                                 "word": "of",
#                                 "start": 118.22,
#                                 "end": 118.3,
#                                 "confidence": 0.9995598,
#                                 "punctuated_word": "of",
#                             },
#                             {
#                                 "word": "memorizes",
#                                 "start": 118.3,
#                                 "end": 118.78,
#                                 "confidence": 0.99195194,
#                                 "punctuated_word": "memorizes",
#                             },
#                             {
#                                 "word": "the",
#                                 "start": 118.78,
#                                 "end": 119.02,
#                                 "confidence": 0.99904925,
#                                 "punctuated_word": "the",
#                             },
#                             {
#                                 "word": "data",
#                                 "start": 119.02,
#                                 "end": 119.18,
#                                 "confidence": 0.99950635,
#                                 "punctuated_word": "data",
#                             },
#                             {
#                                 "word": "it",
#                                 "start": 119.18,
#                                 "end": 119.34,
#                                 "confidence": 0.98501724,
#                                 "punctuated_word": "it",
#                             },
#                             {
#                                 "word": "was",
#                                 "start": 119.34,
#                                 "end": 119.5,
#                                 "confidence": 0.9996269,
#                                 "punctuated_word": "was",
#                             },
#                             {
#                                 "word": "trained",
#                                 "start": 119.5,
#                                 "end": 119.74,
#                                 "confidence": 0.9994259,
#                                 "punctuated_word": "trained",
#                             },
#                             {
#                                 "word": "on",
#                                 "start": 119.74,
#                                 "end": 119.9,
#                                 "confidence": 0.99916947,
#                                 "punctuated_word": "on",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 119.9,
#                                 "end": 120.22,
#                                 "confidence": 0.8316325,
#                                 "punctuated_word": "and",
#                             },
#                             {
#                                 "word": "over",
#                                 "start": 120.22,
#                                 "end": 120.7,
#                                 "confidence": 0.9719954,
#                                 "punctuated_word": "over",
#                             },
#                             {
#                                 "word": "fits",
#                                 "start": 120.7,
#                                 "end": 121.02,
#                                 "confidence": 0.9780507,
#                                 "punctuated_word": "fits",
#                             },
#                             {
#                                 "word": "to",
#                                 "start": 121.02,
#                                 "end": 121.18,
#                                 "confidence": 0.99806184,
#                                 "punctuated_word": "to",
#                             },
#                             {
#                                 "word": "it",
#                                 "start": 121.18,
#                                 "end": 121.659996,
#                                 "confidence": 0.99524426,
#                                 "punctuated_word": "it.",
#                             },
#                             {
#                                 "word": "but",
#                                 "start": 125.979996,
#                                 "end": 126.22,
#                                 "confidence": 0.9990729,
#                                 "punctuated_word": "But",
#                             },
#                             {
#                                 "word": "how",
#                                 "start": 126.22,
#                                 "end": 126.46,
#                                 "confidence": 0.8764552,
#                                 "punctuated_word": "how",
#                             },
#                             {
#                                 "word": "do",
#                                 "start": 126.46,
#                                 "end": 126.54,
#                                 "confidence": 0.99958044,
#                                 "punctuated_word": "do",
#                             },
#                             {
#                                 "word": "you",
#                                 "start": 126.54,
#                                 "end": 126.7,
#                                 "confidence": 0.99984217,
#                                 "punctuated_word": "you",
#                             },
#                             {
#                                 "word": "fix",
#                                 "start": 126.7,
#                                 "end": 126.94,
#                                 "confidence": 0.99977595,
#                                 "punctuated_word": "fix",
#                             },
#                             {
#                                 "word": "this",
#                                 "start": 126.94,
#                                 "end": 127.18,
#                                 "confidence": 0.99879694,
#                                 "punctuated_word": "this?",
#                             },
#                             {
#                                 "word": "for",
#                                 "start": 130.725,
#                                 "end": 131.045,
#                                 "confidence": 0.99961567,
#                                 "punctuated_word": "For",
#                             },
#                             {
#                                 "word": "underfitting",
#                                 "start": 131.045,
#                                 "end": 131.605,
#                                 "confidence": 0.9167199,
#                                 "punctuated_word": "underfitting,",
#                             },
#                             {
#                                 "word": "you",
#                                 "start": 131.605,
#                                 "end": 131.685,
#                                 "confidence": 0.99982613,
#                                 "punctuated_word": "you",
#                             },
#                             {
#                                 "word": "can",
#                                 "start": 131.685,
#                                 "end": 131.845,
#                                 "confidence": 0.9992392,
#                                 "punctuated_word": "can",
#                             },
#                             {
#                                 "word": "train",
#                                 "start": 131.845,
#                                 "end": 132.005,
#                                 "confidence": 0.99934214,
#                                 "punctuated_word": "train",
#                             },
#                             {
#                                 "word": "the",
#                                 "start": 132.005,
#                                 "end": 132.16501,
#                                 "confidence": 0.99703443,
#                                 "punctuated_word": "the",
#                             },
#                             {
#                                 "word": "model",
#                                 "start": 132.16501,
#                                 "end": 132.325,
#                                 "confidence": 0.99977475,
#                                 "punctuated_word": "model",
#                             },
#                             {
#                                 "word": "more",
#                                 "start": 132.325,
#                                 "end": 132.645,
#                                 "confidence": 0.99860686,
#                                 "punctuated_word": "more",
#                             },
#                             {
#                                 "word": "or",
#                                 "start": 132.645,
#                                 "end": 132.965,
#                                 "confidence": 0.93813646,
#                                 "punctuated_word": "or",
#                             },
#                             {
#                                 "word": "increase",
#                                 "start": 132.965,
#                                 "end": 133.365,
#                                 "confidence": 0.9984022,
#                                 "punctuated_word": "increase",
#                             },
#                             {
#                                 "word": "its",
#                                 "start": 133.365,
#                                 "end": 133.605,
#                                 "confidence": 0.87072754,
#                                 "punctuated_word": "its",
#                             },
#                             {
#                                 "word": "complexity",
#                                 "start": 133.605,
#                                 "end": 134.085,
#                                 "confidence": 0.9963838,
#                                 "punctuated_word": "complexity",
#                             },
#                             {
#                                 "word": "so",
#                                 "start": 134.085,
#                                 "end": 134.245,
#                                 "confidence": 0.91034883,
#                                 "punctuated_word": "so",
#                             },
#                             {
#                                 "word": "it",
#                                 "start": 134.245,
#                                 "end": 134.405,
#                                 "confidence": 0.960389,
#                                 "punctuated_word": "it",
#                             },
#                             {
#                                 "word": "can",
#                                 "start": 134.405,
#                                 "end": 134.485,
#                                 "confidence": 0.9990846,
#                                 "punctuated_word": "can",
#                             },
#                             {
#                                 "word": "learn",
#                                 "start": 134.485,
#                                 "end": 134.725,
#                                 "confidence": 0.9993882,
#                                 "punctuated_word": "learn",
#                             },
#                             {
#                                 "word": "more",
#                                 "start": 134.725,
#                                 "end": 134.885,
#                                 "confidence": 0.99923456,
#                                 "punctuated_word": "more",
#                             },
#                             {
#                                 "word": "from",
#                                 "start": 134.885,
#                                 "end": 135.045,
#                                 "confidence": 0.9987937,
#                                 "punctuated_word": "from",
#                             },
#                             {
#                                 "word": "the",
#                                 "start": 135.045,
#                                 "end": 135.205,
#                                 "confidence": 0.99900097,
#                                 "punctuated_word": "the",
#                             },
#                             {
#                                 "word": "data",
#                                 "start": 135.205,
#                                 "end": 135.76501,
#                                 "confidence": 0.9932225,
#                                 "punctuated_word": "data.",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 139.525,
#                                 "end": 139.845,
#                                 "confidence": 0.80464256,
#                                 "punctuated_word": "And,",
#                             },
#                             {
#                                 "word": "for",
#                                 "start": 139.845,
#                                 "end": 139.925,
#                                 "confidence": 0.9997857,
#                                 "punctuated_word": "for",
#                             },
#                             {
#                                 "word": "overfitting",
#                                 "start": 139.925,
#                                 "end": 140.565,
#                                 "confidence": 0.98035496,
#                                 "punctuated_word": "overfitting,",
#                             },
#                             {
#                                 "word": "you",
#                                 "start": 140.565,
#                                 "end": 140.645,
#                                 "confidence": 0.9998883,
#                                 "punctuated_word": "you",
#                             },
#                             {
#                                 "word": "can",
#                                 "start": 140.645,
#                                 "end": 140.80501,
#                                 "confidence": 0.99936336,
#                                 "punctuated_word": "can",
#                             },
#                             {
#                                 "word": "train",
#                                 "start": 140.80501,
#                                 "end": 140.965,
#                                 "confidence": 0.99889404,
#                                 "punctuated_word": "train",
#                             },
#                             {
#                                 "word": "it",
#                                 "start": 140.965,
#                                 "end": 141.125,
#                                 "confidence": 0.99706334,
#                                 "punctuated_word": "it",
#                             },
#                             {
#                                 "word": "more",
#                                 "start": 141.125,
#                                 "end": 141.365,
#                                 "confidence": 0.9994892,
#                                 "punctuated_word": "more",
#                             },
#                             {
#                                 "word": "or",
#                                 "start": 141.365,
#                                 "end": 141.685,
#                                 "confidence": 0.7635503,
#                                 "punctuated_word": "or",
#                             },
#                             {
#                                 "word": "ah",
#                                 "start": 141.925,
#                                 "end": 142.16501,
#                                 "confidence": 0.61700916,
#                                 "punctuated_word": "ah,",
#                             },
#                             {
#                                 "word": "fuck",
#                                 "start": 142.16501,
#                                 "end": 142.405,
#                                 "confidence": 0.9967946,
#                                 "punctuated_word": "fuck.",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 151.65001,
#                                 "end": 151.89,
#                                 "confidence": 0.999627,
#                                 "punctuated_word": "And",
#                             },
#                             {
#                                 "word": "for",
#                                 "start": 151.89,
#                                 "end": 152.05,
#                                 "confidence": 0.9912804,
#                                 "punctuated_word": "for",
#                             },
#                             {
#                                 "word": "overfitting",
#                                 "start": 152.05,
#                                 "end": 152.69,
#                                 "confidence": 0.9963717,
#                                 "punctuated_word": "overfitting,",
#                             },
#                             {
#                                 "word": "you",
#                                 "start": 152.69,
#                                 "end": 152.77,
#                                 "confidence": 0.99994516,
#                                 "punctuated_word": "you",
#                             },
#                             {
#                                 "word": "can",
#                                 "start": 152.77,
#                                 "end": 152.93001,
#                                 "confidence": 0.99984956,
#                                 "punctuated_word": "can",
#                             },
#                             {
#                                 "word": "train",
#                                 "start": 152.93001,
#                                 "end": 153.09001,
#                                 "confidence": 0.9993767,
#                                 "punctuated_word": "train",
#                             },
#                             {
#                                 "word": "it",
#                                 "start": 153.09001,
#                                 "end": 153.25,
#                                 "confidence": 0.9952924,
#                                 "punctuated_word": "it",
#                             },
#                             {
#                                 "word": "less",
#                                 "start": 153.25,
#                                 "end": 153.49,
#                                 "confidence": 0.9998166,
#                                 "punctuated_word": "less",
#                             },
#                             {
#                                 "word": "or",
#                                 "start": 153.49,
#                                 "end": 153.81,
#                                 "confidence": 0.9921325,
#                                 "punctuated_word": "or",
#                             },
#                             {
#                                 "word": "reduce",
#                                 "start": 153.81,
#                                 "end": 154.21,
#                                 "confidence": 0.9971433,
#                                 "punctuated_word": "reduce",
#                             },
#                             {
#                                 "word": "its",
#                                 "start": 154.21,
#                                 "end": 154.53,
#                                 "confidence": 0.98221934,
#                                 "punctuated_word": "its",
#                             },
#                             {
#                                 "word": "complexity",
#                                 "start": 154.53,
#                                 "end": 155.09001,
#                                 "confidence": 0.99286413,
#                                 "punctuated_word": "complexity.",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 157.405,
#                                 "end": 157.645,
#                                 "confidence": 0.9973489,
#                                 "punctuated_word": "And",
#                             },
#                             {
#                                 "word": "there's",
#                                 "start": 157.645,
#                                 "end": 157.965,
#                                 "confidence": 0.7778107,
#                                 "punctuated_word": "there's",
#                             },
#                             {
#                                 "word": "a",
#                                 "start": 157.965,
#                                 "end": 158.045,
#                                 "confidence": 0.9989273,
#                                 "punctuated_word": "a",
#                             },
#                             {
#                                 "word": "few",
#                                 "start": 158.045,
#                                 "end": 158.205,
#                                 "confidence": 0.9998287,
#                                 "punctuated_word": "few",
#                             },
#                             {
#                                 "word": "clever",
#                                 "start": 158.205,
#                                 "end": 158.44499,
#                                 "confidence": 0.99914384,
#                                 "punctuated_word": "clever",
#                             },
#                             {
#                                 "word": "tricks",
#                                 "start": 158.44499,
#                                 "end": 158.765,
#                                 "confidence": 0.99954575,
#                                 "punctuated_word": "tricks",
#                             },
#                             {
#                                 "word": "we'll",
#                                 "start": 158.765,
#                                 "end": 159.085,
#                                 "confidence": 0.9906485,
#                                 "punctuated_word": "we'll",
#                             },
#                             {
#                                 "word": "get",
#                                 "start": 159.085,
#                                 "end": 159.165,
#                                 "confidence": 0.9991748,
#                                 "punctuated_word": "get",
#                             },
#                             {
#                                 "word": "into",
#                                 "start": 159.165,
#                                 "end": 159.405,
#                                 "confidence": 0.99304515,
#                                 "punctuated_word": "into",
#                             },
#                             {
#                                 "word": "tomorrow",
#                                 "start": 159.405,
#                                 "end": 160.205,
#                                 "confidence": 0.97520095,
#                                 "punctuated_word": "tomorrow.",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 161.965,
#                                 "end": 162.205,
#                                 "confidence": 0.9936085,
#                                 "punctuated_word": "And",
#                             },
#                             {
#                                 "word": "bias",
#                                 "start": 162.205,
#                                 "end": 162.605,
#                                 "confidence": 0.81563455,
#                                 "punctuated_word": "bias",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 162.605,
#                                 "end": 162.845,
#                                 "confidence": 0.99735993,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "what",
#                                 "start": 162.845,
#                                 "end": 163.005,
#                                 "confidence": 0.99330467,
#                                 "punctuated_word": "what",
#                             },
#                             {
#                                 "word": "happened",
#                                 "start": 163.005,
#                                 "end": 163.245,
#                                 "confidence": 0.8967317,
#                                 "punctuated_word": "happened",
#                             },
#                             {
#                                 "word": "to",
#                                 "start": 163.245,
#                                 "end": 163.405,
#                                 "confidence": 0.9992447,
#                                 "punctuated_word": "to",
#                             },
#                             {
#                                 "word": "google",
#                                 "start": 163.405,
#                                 "end": 164.125,
#                                 "confidence": 0.99756765,
#                                 "punctuated_word": "Google.",
#                             },
#                             {
#                                 "word": "this",
#                                 "start": 166.365,
#                                 "end": 166.605,
#                                 "confidence": 0.9998165,
#                                 "punctuated_word": "This",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 166.605,
#                                 "end": 166.765,
#                                 "confidence": 0.99932873,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "where",
#                                 "start": 166.765,
#                                 "end": 166.925,
#                                 "confidence": 0.99911684,
#                                 "punctuated_word": "where",
#                             },
#                             {
#                                 "word": "it",
#                                 "start": 166.925,
#                                 "end": 167.08499,
#                                 "confidence": 0.9992895,
#                                 "punctuated_word": "it",
#                             },
#                             {
#                                 "word": "does",
#                                 "start": 167.08499,
#                                 "end": 167.245,
#                                 "confidence": 0.99659556,
#                                 "punctuated_word": "does",
#                             },
#                             {
#                                 "word": "worse",
#                                 "start": 167.245,
#                                 "end": 167.485,
#                                 "confidence": 0.9885184,
#                                 "punctuated_word": "worse",
#                             },
#                             {
#                                 "word": "for",
#                                 "start": 167.485,
#                                 "end": 167.645,
#                                 "confidence": 0.99965656,
#                                 "punctuated_word": "for",
#                             },
#                             {
#                                 "word": "some",
#                                 "start": 167.645,
#                                 "end": 167.885,
#                                 "confidence": 0.99987876,
#                                 "punctuated_word": "some",
#                             },
#                             {
#                                 "word": "groups",
#                                 "start": 167.885,
#                                 "end": 168.285,
#                                 "confidence": 0.8125683,
#                                 "punctuated_word": "groups.",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 171.58,
#                                 "end": 171.9,
#                                 "confidence": 0.9594344,
#                                 "punctuated_word": "And,",
#                             },
#                             {
#                                 "word": "it's",
#                                 "start": 171.9,
#                                 "end": 172.14,
#                                 "confidence": 0.9979924,
#                                 "punctuated_word": "it's",
#                             },
#                             {
#                                 "word": "usually",
#                                 "start": 172.14,
#                                 "end": 172.45999,
#                                 "confidence": 0.9993476,
#                                 "punctuated_word": "usually",
#                             },
#                             {
#                                 "word": "caused",
#                                 "start": 172.45999,
#                                 "end": 172.78,
#                                 "confidence": 0.99867624,
#                                 "punctuated_word": "caused",
#                             },
#                             {
#                                 "word": "by",
#                                 "start": 172.78,
#                                 "end": 173.02,
#                                 "confidence": 0.99976367,
#                                 "punctuated_word": "by",
#                             },
#                             {
#                                 "word": "two",
#                                 "start": 173.02,
#                                 "end": 173.34,
#                                 "confidence": 0.9995846,
#                                 "punctuated_word": "two",
#                             },
#                             {
#                                 "word": "things",
#                                 "start": 173.34,
#                                 "end": 173.66,
#                                 "confidence": 0.9756458,
#                                 "punctuated_word": "things.",
#                             },
#                             {
#                                 "word": "first",
#                                 "start": 183.095,
#                                 "end": 183.575,
#                                 "confidence": 0.98271793,
#                                 "punctuated_word": "First",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 183.575,
#                                 "end": 183.895,
#                                 "confidence": 0.7609724,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "bad",
#                                 "start": 183.895,
#                                 "end": 184.13501,
#                                 "confidence": 0.9540817,
#                                 "punctuated_word": "bad",
#                             },
#                             {
#                                 "word": "data",
#                                 "start": 184.13501,
#                                 "end": 184.535,
#                                 "confidence": 0.9870979,
#                                 "punctuated_word": "data.",
#                             },
#                             {
#                                 "word": "if",
#                                 "start": 184.535,
#                                 "end": 184.695,
#                                 "confidence": 0.9971487,
#                                 "punctuated_word": "If",
#                             },
#                             {
#                                 "word": "when",
#                                 "start": 184.695,
#                                 "end": 184.935,
#                                 "confidence": 0.57906276,
#                                 "punctuated_word": "when",
#                             },
#                             {
#                                 "word": "training",
#                                 "start": 184.935,
#                                 "end": 185.175,
#                                 "confidence": 0.9986375,
#                                 "punctuated_word": "training",
#                             },
#                             {
#                                 "word": "it",
#                                 "start": 185.175,
#                                 "end": 185.335,
#                                 "confidence": 0.9802877,
#                                 "punctuated_word": "it",
#                             },
#                             {
#                                 "word": "on",
#                                 "start": 185.335,
#                                 "end": 185.495,
#                                 "confidence": 0.99796116,
#                                 "punctuated_word": "on",
#                             },
#                             {
#                                 "word": "people",
#                                 "start": 185.495,
#                                 "end": 185.815,
#                                 "confidence": 0.8643048,
#                                 "punctuated_word": "people,",
#                             },
#                             {
#                                 "word": "it's",
#                                 "start": 185.815,
#                                 "end": 185.975,
#                                 "confidence": 0.99826956,
#                                 "punctuated_word": "it's",
#                             },
#                             {
#                                 "word": "only",
#                                 "start": 185.975,
#                                 "end": 186.13501,
#                                 "confidence": 0.99951553,
#                                 "punctuated_word": "only",
#                             },
#                             {
#                                 "word": "ever",
#                                 "start": 186.13501,
#                                 "end": 186.295,
#                                 "confidence": 0.99105,
#                                 "punctuated_word": "ever",
#                             },
#                             {
#                                 "word": "given",
#                                 "start": 186.295,
#                                 "end": 186.535,
#                                 "confidence": 0.9941156,
#                                 "punctuated_word": "given",
#                             },
#                             {
#                                 "word": "white",
#                                 "start": 186.535,
#                                 "end": 186.77501,
#                                 "confidence": 0.8670402,
#                                 "punctuated_word": "white",
#                             },
#                             {
#                                 "word": "people",
#                                 "start": 186.77501,
#                                 "end": 187.175,
#                                 "confidence": 0.9325635,
#                                 "punctuated_word": "people,",
#                             },
#                             {
#                                 "word": "then",
#                                 "start": 187.175,
#                                 "end": 187.255,
#                                 "confidence": 0.995572,
#                                 "punctuated_word": "then",
#                             },
#                             {
#                                 "word": "when",
#                                 "start": 187.255,
#                                 "end": 187.41501,
#                                 "confidence": 0.94182867,
#                                 "punctuated_word": "when",
#                             },
#                             {
#                                 "word": "you",
#                                 "start": 187.41501,
#                                 "end": 187.495,
#                                 "confidence": 0.99872917,
#                                 "punctuated_word": "you",
#                             },
#                             {
#                                 "word": "give",
#                                 "start": 187.495,
#                                 "end": 187.655,
#                                 "confidence": 0.99830663,
#                                 "punctuated_word": "give",
#                             },
#                             {
#                                 "word": "it",
#                                 "start": 187.655,
#                                 "end": 187.735,
#                                 "confidence": 0.9966012,
#                                 "punctuated_word": "it",
#                             },
#                             {
#                                 "word": "a",
#                                 "start": 187.735,
#                                 "end": 187.895,
#                                 "confidence": 0.8880037,
#                                 "punctuated_word": "a",
#                             },
#                             {
#                                 "word": "black",
#                                 "start": 187.895,
#                                 "end": 188.05501,
#                                 "confidence": 0.9466706,
#                                 "punctuated_word": "black",
#                             },
#                             {
#                                 "word": "person",
#                                 "start": 188.05501,
#                                 "end": 188.375,
#                                 "confidence": 0.94573414,
#                                 "punctuated_word": "person,",
#                             },
#                             {
#                                 "word": "it'll",
#                                 "start": 188.375,
#                                 "end": 188.615,
#                                 "confidence": 0.98271465,
#                                 "punctuated_word": "it'll",
#                             },
#                             {
#                                 "word": "be",
#                                 "start": 188.615,
#                                 "end": 188.695,
#                                 "confidence": 0.9984365,
#                                 "punctuated_word": "be",
#                             },
#                             {
#                                 "word": "like",
#                                 "start": 188.695,
#                                 "end": 189.015,
#                                 "confidence": 0.8972057,
#                                 "punctuated_word": "like,",
#                             },
#                             {
#                                 "word": "i've",
#                                 "start": 189.015,
#                                 "end": 189.175,
#                                 "confidence": 0.9943862,
#                                 "punctuated_word": "I've",
#                             },
#                             {
#                                 "word": "never",
#                                 "start": 189.175,
#                                 "end": 189.41501,
#                                 "confidence": 0.99977714,
#                                 "punctuated_word": "never",
#                             },
#                             {
#                                 "word": "seen",
#                                 "start": 189.41501,
#                                 "end": 189.575,
#                                 "confidence": 0.99967897,
#                                 "punctuated_word": "seen",
#                             },
#                             {
#                                 "word": "this",
#                                 "start": 189.575,
#                                 "end": 189.735,
#                                 "confidence": 0.99943155,
#                                 "punctuated_word": "this",
#                             },
#                             {
#                                 "word": "as",
#                                 "start": 189.735,
#                                 "end": 189.895,
#                                 "confidence": 0.99479747,
#                                 "punctuated_word": "as",
#                             },
#                             {
#                                 "word": "a",
#                                 "start": 189.895,
#                                 "end": 189.975,
#                                 "confidence": 0.9987451,
#                                 "punctuated_word": "a",
#                             },
#                             {
#                                 "word": "person",
#                                 "start": 189.975,
#                                 "end": 190.215,
#                                 "confidence": 0.9998696,
#                                 "punctuated_word": "person",
#                             },
#                             {
#                                 "word": "before",
#                                 "start": 190.215,
#                                 "end": 190.93501,
#                                 "confidence": 0.9965433,
#                                 "punctuated_word": "before.",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 192.375,
#                                 "end": 192.615,
#                                 "confidence": 0.9983804,
#                                 "punctuated_word": "And",
#                             },
#                             {
#                                 "word": "second",
#                                 "start": 192.615,
#                                 "end": 192.935,
#                                 "confidence": 0.8803209,
#                                 "punctuated_word": "second",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 192.935,
#                                 "end": 193.175,
#                                 "confidence": 0.6546616,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "just",
#                                 "start": 193.175,
#                                 "end": 193.41501,
#                                 "confidence": 0.988289,
#                                 "punctuated_word": "just",
#                             },
#                             {
#                                 "word": "having",
#                                 "start": 193.41501,
#                                 "end": 193.655,
#                                 "confidence": 0.9993643,
#                                 "punctuated_word": "having",
#                             },
#                             {
#                                 "word": "a",
#                                 "start": 193.655,
#                                 "end": 193.815,
#                                 "confidence": 0.9989784,
#                                 "punctuated_word": "a",
#                             },
#                             {
#                                 "word": "dumb",
#                                 "start": 193.815,
#                                 "end": 194.05501,
#                                 "confidence": 0.9993305,
#                                 "punctuated_word": "dumb",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 194.05501,
#                                 "end": 194.615,
#                                 "confidence": 0.8609268,
#                                 "punctuated_word": "AI,",
#                             },
#                             {
#                                 "word": "which",
#                                 "start": 194.615,
#                                 "end": 194.77501,
#                                 "confidence": 0.9998331,
#                                 "punctuated_word": "which",
#                             },
#                             {
#                                 "word": "was",
#                                 "start": 194.77501,
#                                 "end": 194.935,
#                                 "confidence": 0.9990809,
#                                 "punctuated_word": "was",
#                             },
#                             {
#                                 "word": "also",
#                                 "start": 194.935,
#                                 "end": 195.255,
#                                 "confidence": 0.99803203,
#                                 "punctuated_word": "also",
#                             },
#                             {
#                                 "word": "a",
#                                 "start": 195.255,
#                                 "end": 195.41501,
#                                 "confidence": 0.9992818,
#                                 "punctuated_word": "a",
#                             },
#                             {
#                                 "word": "problem",
#                                 "start": 195.41501,
#                                 "end": 195.655,
#                                 "confidence": 0.9997038,
#                                 "punctuated_word": "problem",
#                             },
#                             {
#                                 "word": "in",
#                                 "start": 195.655,
#                                 "end": 195.895,
#                                 "confidence": 0.9985018,
#                                 "punctuated_word": "in",
#                             },
#                             {
#                                 "word": "2015",
#                                 "start": 195.895,
#                                 "end": 196.455,
#                                 "confidence": 0.996579,
#                                 "punctuated_word": "2015.",
#                             },
#                             {
#                                 "word": "so",
#                                 "start": 201.48,
#                                 "end": 202.04,
#                                 "confidence": 0.8357139,
#                                 "punctuated_word": "So,",
#                             },
#                             {
#                                 "word": "how",
#                                 "start": 202.04,
#                                 "end": 202.12,
#                                 "confidence": 0.9996997,
#                                 "punctuated_word": "how",
#                             },
#                             {
#                                 "word": "did",
#                                 "start": 202.12,
#                                 "end": 202.36,
#                                 "confidence": 0.9996166,
#                                 "punctuated_word": "did",
#                             },
#                             {
#                                 "word": "google",
#                                 "start": 202.36,
#                                 "end": 202.76,
#                                 "confidence": 0.99906474,
#                                 "punctuated_word": "Google",
#                             },
#                             {
#                                 "word": "fix",
#                                 "start": 202.76,
#                                 "end": 203,
#                                 "confidence": 0.9979906,
#                                 "punctuated_word": "fix",
#                             },
#                             {
#                                 "word": "this",
#                                 "start": 203,
#                                 "end": 203.56,
#                                 "confidence": 0.9992672,
#                                 "punctuated_word": "this?",
#                             },
#                             {
#                                 "word": "well",
#                                 "start": 204.04,
#                                 "end": 204.43999,
#                                 "confidence": 0.99467766,
#                                 "punctuated_word": "Well,",
#                             },
#                             {
#                                 "word": "as",
#                                 "start": 204.43999,
#                                 "end": 204.59999,
#                                 "confidence": 0.99948186,
#                                 "punctuated_word": "as",
#                             },
#                             {
#                                 "word": "mentioned",
#                                 "start": 204.59999,
#                                 "end": 205.07999,
#                                 "confidence": 0.8957324,
#                                 "punctuated_word": "mentioned,",
#                             },
#                             {
#                                 "word": "in",
#                                 "start": 205.07999,
#                                 "end": 205.15999,
#                                 "confidence": 0.8795341,
#                                 "punctuated_word": "in",
#                             },
#                             {
#                                 "word": "2015",
#                                 "start": 205.15999,
#                                 "end": 205.79999,
#                                 "confidence": 0.94402456,
#                                 "punctuated_word": "2015,",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 205.79999,
#                                 "end": 206.04,
#                                 "confidence": 0.99782014,
#                                 "punctuated_word": "AI",
#                             },
#                             {
#                                 "word": "was",
#                                 "start": 206.04,
#                                 "end": 206.2,
#                                 "confidence": 0.9994702,
#                                 "punctuated_word": "was",
#                             },
#                             {
#                                 "word": "pretty",
#                                 "start": 206.2,
#                                 "end": 206.43999,
#                                 "confidence": 0.99977463,
#                                 "punctuated_word": "pretty",
#                             },
#                             {
#                                 "word": "dumb",
#                                 "start": 206.43999,
#                                 "end": 206.68,
#                                 "confidence": 0.966584,
#                                 "punctuated_word": "dumb.",
#                             },
#                             {
#                                 "word": "so",
#                                 "start": 210.585,
#                                 "end": 210.90501,
#                                 "confidence": 0.80729187,
#                                 "punctuated_word": "So",
#                             },
#                             {
#                                 "word": "they",
#                                 "start": 210.90501,
#                                 "end": 211.065,
#                                 "confidence": 0.9588324,
#                                 "punctuated_word": "they",
#                             },
#                             {
#                                 "word": "just",
#                                 "start": 211.065,
#                                 "end": 211.30501,
#                                 "confidence": 0.99917346,
#                                 "punctuated_word": "just",
#                             },
#                             {
#                                 "word": "remove",
#                                 "start": 211.30501,
#                                 "end": 211.54501,
#                                 "confidence": 0.62038034,
#                                 "punctuated_word": "remove",
#                             },
#                             {
#                                 "word": "gorillas",
#                                 "start": 211.54501,
#                                 "end": 212.02501,
#                                 "confidence": 0.99328184,
#                                 "punctuated_word": "gorillas",
#                             },
#                             {
#                                 "word": "as",
#                                 "start": 212.02501,
#                                 "end": 212.18501,
#                                 "confidence": 0.99729925,
#                                 "punctuated_word": "as",
#                             },
#                             {
#                                 "word": "an",
#                                 "start": 212.18501,
#                                 "end": 212.265,
#                                 "confidence": 0.997993,
#                                 "punctuated_word": "an",
#                             },
#                             {
#                                 "word": "option",
#                                 "start": 212.265,
#                                 "end": 212.425,
#                                 "confidence": 0.9997571,
#                                 "punctuated_word": "option",
#                             },
#                             {
#                                 "word": "because",
#                                 "start": 212.425,
#                                 "end": 212.74501,
#                                 "confidence": 0.87439954,
#                                 "punctuated_word": "because",
#                             },
#                             {
#                                 "word": "you",
#                                 "start": 212.74501,
#                                 "end": 212.90501,
#                                 "confidence": 0.99453545,
#                                 "punctuated_word": "you",
#                             },
#                             {
#                                 "word": "can't",
#                                 "start": 212.90501,
#                                 "end": 213.145,
#                                 "confidence": 0.99835765,
#                                 "punctuated_word": "can't",
#                             },
#                             {
#                                 "word": "call",
#                                 "start": 213.145,
#                                 "end": 213.30501,
#                                 "confidence": 0.99951434,
#                                 "punctuated_word": "call",
#                             },
#                             {
#                                 "word": "people",
#                                 "start": 213.30501,
#                                 "end": 213.54501,
#                                 "confidence": 0.99972254,
#                                 "punctuated_word": "people",
#                             },
#                             {
#                                 "word": "gorillas",
#                                 "start": 213.54501,
#                                 "end": 214.02501,
#                                 "confidence": 0.9981026,
#                                 "punctuated_word": "gorillas",
#                             },
#                             {
#                                 "word": "if",
#                                 "start": 214.02501,
#                                 "end": 214.18501,
#                                 "confidence": 0.9774131,
#                                 "punctuated_word": "if",
#                             },
#                             {
#                                 "word": "you",
#                                 "start": 214.18501,
#                                 "end": 214.265,
#                                 "confidence": 0.9995555,
#                                 "punctuated_word": "you",
#                             },
#                             {
#                                 "word": "can't",
#                                 "start": 214.265,
#                                 "end": 214.505,
#                                 "confidence": 0.99797654,
#                                 "punctuated_word": "can't",
#                             },
#                             {
#                                 "word": "call",
#                                 "start": 214.505,
#                                 "end": 214.66501,
#                                 "confidence": 0.99700063,
#                                 "punctuated_word": "call",
#                             },
#                             {
#                                 "word": "anything",
#                                 "start": 214.66501,
#                                 "end": 214.985,
#                                 "confidence": 0.9997155,
#                                 "punctuated_word": "anything",
#                             },
#                             {
#                                 "word": "a",
#                                 "start": 214.985,
#                                 "end": 215.145,
#                                 "confidence": 0.9240809,
#                                 "punctuated_word": "a",
#                             },
#                             {
#                                 "word": "gorilla",
#                                 "start": 215.145,
#                                 "end": 215.86502,
#                                 "confidence": 0.99764717,
#                                 "punctuated_word": "gorilla.",
#                             },
#                             {
#                                 "word": "but",
#                                 "start": 218.905,
#                                 "end": 219.145,
#                                 "confidence": 0.9976876,
#                                 "punctuated_word": "But",
#                             },
#                             {
#                                 "word": "today",
#                                 "start": 219.145,
#                                 "end": 219.625,
#                                 "confidence": 0.8203881,
#                                 "punctuated_word": "today,",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 219.625,
#                                 "end": 219.945,
#                                 "confidence": 0.99049664,
#                                 "punctuated_word": "AI",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 219.945,
#                                 "end": 220.10501,
#                                 "confidence": 0.98553586,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "a",
#                                 "start": 220.10501,
#                                 "end": 220.265,
#                                 "confidence": 0.5911662,
#                                 "punctuated_word": "a",
#                             },
#                             {
#                                 "word": "retake",
#                                 "start": 223.865,
#                                 "end": 224.425,
#                                 "confidence": 0.9767554,
#                                 "punctuated_word": "retake.",
#                             },
#                             {
#                                 "word": "but",
#                                 "start": 227.65,
#                                 "end": 227.89,
#                                 "confidence": 0.99972254,
#                                 "punctuated_word": "But",
#                             },
#                             {
#                                 "word": "today",
#                                 "start": 227.89,
#                                 "end": 228.37,
#                                 "confidence": 0.88603336,
#                                 "punctuated_word": "today,",
#                             },
#                             {
#                                 "word": "ai",
#                                 "start": 228.37,
#                                 "end": 228.61,
#                                 "confidence": 0.99553776,
#                                 "punctuated_word": "AI",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 228.61,
#                                 "end": 228.77,
#                                 "confidence": 0.9929831,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "big",
#                                 "start": 228.77,
#                                 "end": 228.93,
#                                 "confidence": 0.9670036,
#                                 "punctuated_word": "big",
#                             },
#                             {
#                                 "word": "brain",
#                                 "start": 228.93,
#                                 "end": 229.33,
#                                 "confidence": 0.7435652,
#                                 "punctuated_word": "brain,",
#                             },
#                             {
#                                 "word": "so",
#                                 "start": 229.33,
#                                 "end": 229.41,
#                                 "confidence": 0.999476,
#                                 "punctuated_word": "so",
#                             },
#                             {
#                                 "word": "overfitting",
#                                 "start": 229.41,
#                                 "end": 229.97,
#                                 "confidence": 0.9178285,
#                                 "punctuated_word": "overfitting",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 229.97,
#                                 "end": 230.20999,
#                                 "confidence": 0.9997335,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "a",
#                                 "start": 230.20999,
#                                 "end": 230.37,
#                                 "confidence": 0.976314,
#                                 "punctuated_word": "a",
#                             },
#                             {
#                                 "word": "big",
#                                 "start": 230.37,
#                                 "end": 230.53,
#                                 "confidence": 0.9997733,
#                                 "punctuated_word": "big",
#                             },
#                             {
#                                 "word": "problem",
#                                 "start": 230.53,
#                                 "end": 231.17,
#                                 "confidence": 0.9625913,
#                                 "punctuated_word": "problem.",
#                             },
#                             {
#                                 "word": "and",
#                                 "start": 233.17,
#                                 "end": 233.33,
#                                 "confidence": 0.99757296,
#                                 "punctuated_word": "And",
#                             },
#                             {
#                                 "word": "how",
#                                 "start": 233.33,
#                                 "end": 233.48999,
#                                 "confidence": 0.93225133,
#                                 "punctuated_word": "how",
#                             },
#                             {
#                                 "word": "that",
#                                 "start": 233.48999,
#                                 "end": 233.65,
#                                 "confidence": 0.9971016,
#                                 "punctuated_word": "that",
#                             },
#                             {
#                                 "word": "can",
#                                 "start": 233.65,
#                                 "end": 233.81,
#                                 "confidence": 0.9993168,
#                                 "punctuated_word": "can",
#                             },
#                             {
#                                 "word": "be",
#                                 "start": 233.81,
#                                 "end": 233.89,
#                                 "confidence": 0.9979317,
#                                 "punctuated_word": "be",
#                             },
#                             {
#                                 "word": "fixed",
#                                 "start": 233.89,
#                                 "end": 234.13,
#                                 "confidence": 0.99739885,
#                                 "punctuated_word": "fixed",
#                             },
#                             {
#                                 "word": "is",
#                                 "start": 234.13,
#                                 "end": 234.29,
#                                 "confidence": 0.91345114,
#                                 "punctuated_word": "is",
#                             },
#                             {
#                                 "word": "up",
#                                 "start": 234.29,
#                                 "end": 234.45,
#                                 "confidence": 0.993634,
#                                 "punctuated_word": "up",
#                             },
#                             {
#                                 "word": "tomorrow",
#                                 "start": 234.45,
#                                 "end": 235.17,
#                                 "confidence": 0.98923755,
#                                 "punctuated_word": "tomorrow.",
#                             },
#                             {
#                                 "word": "cheers",
#                                 "start": 235.17,
#                                 "end": 235.56999,
#                                 "confidence": 0.98617065,
#                                 "punctuated_word": "Cheers.",
#                             },
#                         ],
#                         "paragraphs": {
#                             "transcript": "\nWhy not? Google's AI famously once labeled black people as gorillas. So why did that happen, and how can you fix this? This is day 13 of AI fundamentals at forty four seconds and forty four days. Google's Photos AI once famously labeled black people as Google's AI once famously labeled black people as gorillas.\n\nSo how did this happen and how can you fix this? Well, this is day 44. Fuck. Google's AI once famously labeled black people as gorillas. So how did this happen and how can you fix it?\n\nThis is day 13 of AI fundamentals in forty four seconds and forty four days. Overfitting, underfitting, and bias are three common concepts in AI. Underfitting is when AI doesn't recognize enough patterns in the data, so it underfits to it. And over fitting is where AI kind of memorizes the data it was trained on and over fits to it. But how do you fix this?\n\nFor underfitting, you can train the model more or increase its complexity so it can learn more from the data. And, for overfitting, you can train it more or ah, fuck. And for overfitting, you can train it less or reduce its complexity. And there's a few clever tricks we'll get into tomorrow. And bias is what happened to Google.\n\nThis is where it does worse for some groups. And, it's usually caused by two things. First is bad data. If when training it on people, it's only ever given white people, then when you give it a black person, it'll be like, I've never seen this as a person before. And second is just having a dumb AI, which was also a problem in 2015.\n\nSo, how did Google fix this? Well, as mentioned, in 2015, AI was pretty dumb. So they just remove gorillas as an option because you can't call people gorillas if you can't call anything a gorilla. But today, AI is a retake. But today, AI is big brain, so overfitting is a big problem.\n\nAnd how that can be fixed is up tomorrow. Cheers.",
#                             "paragraphs": [
#                                 {
#                                     "sentences": [
#                                         {
#                                             "text": "Why not?",
#                                             "start": 12.16,
#                                             "end": 12.799999,
#                                         },
#                                         {
#                                             "text": "Google's AI famously once labeled black people as gorillas.",
#                                             "start": 43.855003,
#                                             "end": 46.975002,
#                                         },
#                                         {
#                                             "text": "So why did that happen, and how can you fix this?",
#                                             "start": 46.975002,
#                                             "end": 49.535,
#                                         },
#                                         {
#                                             "text": "This is day 13 of AI fundamentals at forty four seconds and forty four days.",
#                                             "start": 54.655003,
#                                             "end": 58.255,
#                                         },
#                                         {
#                                             "text": "Google's Photos AI once famously labeled black people as Google's AI once famously labeled black people as gorillas.",
#                                             "start": 63.129997,
#                                             "end": 71.77,
#                                         },
#                                     ],
#                                     "num_words": 55,
#                                     "start": 12.16,
#                                     "end": 71.77,
#                                 },
#                                 {
#                                     "sentences": [
#                                         {
#                                             "text": "So how did this happen and how can you fix this?",
#                                             "start": 71.77,
#                                             "end": 74.17,
#                                         },
#                                         {
#                                             "text": "Well, this is day 44.",
#                                             "start": 74.17,
#                                             "end": 75.505005,
#                                         },
#                                         {
#                                             "text": "Fuck.",
#                                             "start": 75.505005,
#                                             "end": 76.225,
#                                         },
#                                         {
#                                             "text": "Google's AI once famously labeled black people as gorillas.",
#                                             "start": 82.545,
#                                             "end": 85.585,
#                                         },
#                                         {
#                                             "text": "So how did this happen and how can you fix it?",
#                                             "start": 85.585,
#                                             "end": 87.665,
#                                         },
#                                     ],
#                                     "num_words": 37,
#                                     "start": 71.77,
#                                     "end": 87.665,
#                                 },
#                                 {
#                                     "sentences": [
#                                         {
#                                             "text": "This is day 13 of AI fundamentals in forty four seconds and forty four days.",
#                                             "start": 90.92,
#                                             "end": 94.92,
#                                         },
#                                         {
#                                             "text": "Overfitting, underfitting, and bias are three common concepts in AI.",
#                                             "start": 97.8,
#                                             "end": 101,
#                                         },
#                                         {
#                                             "text": "Underfitting is when AI doesn't recognize enough patterns in the data, so it underfits to it.",
#                                             "start": 108.365,
#                                             "end": 113.005,
#                                         },
#                                         {
#                                             "text": "And over fitting is where AI kind of memorizes the data it was trained on and over fits to it.",
#                                             "start": 116.38,
#                                             "end": 121.659996,
#                                         },
#                                         {
#                                             "text": "But how do you fix this?",
#                                             "start": 125.979996,
#                                             "end": 127.18,
#                                         },
#                                     ],
#                                     "num_words": 67,
#                                     "start": 90.92,
#                                     "end": 127.18,
#                                 },
#                                 {
#                                     "sentences": [
#                                         {
#                                             "text": "For underfitting, you can train the model more or increase its complexity so it can learn more from the data.",
#                                             "start": 130.725,
#                                             "end": 135.76501,
#                                         },
#                                         {
#                                             "text": "And, for overfitting, you can train it more or ah, fuck.",
#                                             "start": 139.525,
#                                             "end": 142.405,
#                                         },
#                                         {
#                                             "text": "And for overfitting, you can train it less or reduce its complexity.",
#                                             "start": 151.65001,
#                                             "end": 155.09001,
#                                         },
#                                         {
#                                             "text": "And there's a few clever tricks we'll get into tomorrow.",
#                                             "start": 157.405,
#                                             "end": 160.205,
#                                         },
#                                         {
#                                             "text": "And bias is what happened to Google.",
#                                             "start": 161.965,
#                                             "end": 164.125,
#                                         },
#                                     ],
#                                     "num_words": 60,
#                                     "start": 130.725,
#                                     "end": 164.125,
#                                 },
#                                 {
#                                     "sentences": [
#                                         {
#                                             "text": "This is where it does worse for some groups.",
#                                             "start": 166.365,
#                                             "end": 168.285,
#                                         },
#                                         {
#                                             "text": "And, it's usually caused by two things.",
#                                             "start": 171.58,
#                                             "end": 173.66,
#                                         },
#                                         {
#                                             "text": "First is bad data.",
#                                             "start": 183.095,
#                                             "end": 184.535,
#                                         },
#                                         {
#                                             "text": "If when training it on people, it's only ever given white people, then when you give it a black person, it'll be like, I've never seen this as a person before.",
#                                             "start": 184.535,
#                                             "end": 190.93501,
#                                         },
#                                         {
#                                             "text": "And second is just having a dumb AI, which was also a problem in 2015.",
#                                             "start": 192.375,
#                                             "end": 196.455,
#                                         },
#                                     ],
#                                     "num_words": 66,
#                                     "start": 166.365,
#                                     "end": 196.455,
#                                 },
#                                 {
#                                     "sentences": [
#                                         {
#                                             "text": "So, how did Google fix this?",
#                                             "start": 201.48,
#                                             "end": 203.56,
#                                         },
#                                         {
#                                             "text": "Well, as mentioned, in 2015, AI was pretty dumb.",
#                                             "start": 204.04,
#                                             "end": 206.68,
#                                         },
#                                         {
#                                             "text": "So they just remove gorillas as an option because you can't call people gorillas if you can't call anything a gorilla.",
#                                             "start": 210.585,
#                                             "end": 215.86502,
#                                         },
#                                         {
#                                             "text": "But today, AI is a retake.",
#                                             "start": 218.905,
#                                             "end": 224.425,
#                                         },
#                                         {
#                                             "text": "But today, AI is big brain, so overfitting is a big problem.",
#                                             "start": 227.65,
#                                             "end": 231.17,
#                                         },
#                                     ],
#                                     "num_words": 54,
#                                     "start": 201.48,
#                                     "end": 231.17,
#                                 },
#                                 {
#                                     "sentences": [
#                                         {
#                                             "text": "And how that can be fixed is up tomorrow.",
#                                             "start": 233.17,
#                                             "end": 235.17,
#                                         },
#                                         {
#                                             "text": "Cheers.",
#                                             "start": 235.17,
#                                             "end": 235.56999,
#                                         },
#                                     ],
#                                     "num_words": 10,
#                                     "start": 233.17,
#                                     "end": 235.56999,
#                                 },
#                             ],
#                         },
#                     }
#                 ]
#             }
#         ]
#     },
# }


class DeepgramSTTService(SpeechToTextService):
    """
    Deepgram implementation of Speech-to-Text service using the official SDK.
    Requests word-level timestamps and normalizes to internal format.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize Deepgram STT service.

        Args:
            api_key: Deepgram API key. If None, reads from environment.

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv(ENV_DEEPGRAM_API_KEY)
        if not self.api_key:
            raise ValueError(
                f"Deepgram API key not found. "
                f"Provide via constructor or {ENV_DEEPGRAM_API_KEY} env var."
            )

        # Initialize Deepgram client
        self.client = DeepgramClient(api_key=self.api_key)

    def transcribe(self, audio_path: str | Path) -> Transcript:
        """
        Transcribe audio using Deepgram STT API via official SDK.

        Args:
            audio_path: Path to audio file

        Returns:
            Internal Transcript model

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If API call fails
        """
        audio_path = Path(audio_path)
        validate_file_exists(audio_path)

        try:
            # Open audio file and call Deepgram API using SDK
            with open(audio_path, "rb") as audio_file:
                # Read the file content
                audio_data = audio_file.read()

                # Call Deepgram API with transcription options
                response = self.client.listen.v1.media.transcribe_file(
                    request=audio_data,
                    model="nova-3",
                    language="en",
                    smart_format=True,
                    punctuate=True,
                )

            # Convert SDK response to internal model with sentences
            transcript = self._convert_response(response)

            # Split sentences based on word gaps before returning
            transcript = self._split_sentences_by_word_gaps(transcript)

            return transcript

        except Exception as e:
            raise RuntimeError(f"Deepgram transcription failed: {str(e)}") from e

    def _convert_response(self, response: Any) -> Transcript:
        """
        Convert Deepgram SDK response to internal Transcript model.

        The Deepgram response structure:
        {
            "metadata": {...},
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "...",
                                "confidence": 0.99,
                                "words": [
                                    {
                                        "word": "why",
                                        "start": 10.719999,
                                        "end": 10.96,
                                        "confidence": 0.99,
                                        "punctuated_word": "Why"
                                    },
                                    ...
                                ]
                            }
                        ]
                    }
                ]
            }
        }

        Args:
            response: Deepgram SDK response object

        Returns:
            Internal Transcript model
        """
        # Convert response to dict if it's a Pydantic model or has to_dict method
        if hasattr(response, "to_dict"):
            response_dict = response.to_dict()
        elif hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        elif hasattr(response, "dict"):
            response_dict = response.dict()
        else:
            response_dict = response

        # Navigate to the results
        results = response_dict.get("results", {})
        channels = results.get("channels", [])

        if not channels:
            return Transcript(sentences=[], language=None, duration=None)

        # Get the first channel (usually there's only one)
        channel = channels[0]
        alternatives = channel.get("alternatives", [])

        if not alternatives:
            return Transcript(sentences=[], language=None, duration=None)

        # Get the best alternative (first one, highest confidence)
        alternative = alternatives[0]
        words_data = alternative.get("words", [])

        # Convert words to internal format and generate sentences
        words = self._extract_words_from_api(words_data)
        sentences = self._create_sentences_from_words(words)

        # Calculate duration
        duration = None
        if sentences and sentences[-1].words:
            duration = sentences[-1].words[-1].end

        # Extract language from metadata if available
        metadata = response_dict.get("metadata", {})
        language = metadata.get("language", "en")

        return Transcript(sentences=sentences, language=language, duration=duration)

    def _extract_words_from_api(
        self, words_data: list[Dict[str, Any]]
    ) -> list[WordTimestamp]:
        """
        Extract word-level timestamps from Deepgram API words array.

        According to Deepgram API spec, each word object has:
        - word: string (original word)
        - punctuated_word: string (word with punctuation)
        - start: float (seconds)
        - end: float (seconds)
        - confidence: float (0-1)

        Args:
            words_data: List of word objects from API

        Returns:
            List of WordTimestamp objects
        """
        words: list[WordTimestamp] = []

        for word_obj in words_data:
            # Convert to dict if it's a Pydantic model
            if hasattr(word_obj, "model_dump"):
                word_obj = word_obj.model_dump()
            elif hasattr(word_obj, "dict"):
                word_obj = word_obj.dict()

            # Use punctuated_word if available, otherwise fall back to word
            text = word_obj.get("punctuated_word") or word_obj.get("word", "")
            start = word_obj.get("start")
            end = word_obj.get("end")

            # Skip if no timing information
            if start is None or end is None:
                continue

            words.append(WordTimestamp(word=text, start=float(start), end=float(end)))

        return words

    def _create_sentences_from_words(
        self, words: list[WordTimestamp]
    ) -> list[LLMTranscriptSentence]:
        """
        Create sentences from words by detecting sentence-ending punctuation.

        A sentence ends when a word's last character is sentence-ending punctuation
        (., ?, or !). This uses Deepgram's punctuated_word field which includes
        proper punctuation.

        Args:
            words: List of WordTimestamp objects with punctuated words

        Returns:
            List of LLMTranscriptSentence objects
        """
        if not words:
            return []

        # Sentence-ending punctuation characters
        sentence_endings = {".", "!", "?"}

        sentences: list[LLMTranscriptSentence] = []
        current_words: list[WordTimestamp] = []
        current_start: float | None = None

        for word in words:
            # Set start time if this is the first word in the sentence
            if current_start is None:
                current_start = word.start

            # Add the word to current sentence
            current_words.append(word)

            # Check if the word ends with sentence-ending punctuation
            word_text = word.word.rstrip()  # Remove trailing whitespace
            if word_text and word_text[-1] in sentence_endings:
                # Complete the current sentence
                if current_words and current_start is not None:
                    sentence_text = " ".join(w.word for w in current_words)
                    sentences.append(
                        LLMTranscriptSentence(
                            sentence=sentence_text,
                            start=current_start,
                            end=word.end,
                            words=current_words.copy(),
                        )
                    )

                # Reset for next sentence
                current_words = []
                current_start = None

        # Handle any remaining words that didn't end with punctuation
        if current_words and current_start is not None:
            sentence_text = " ".join(w.word for w in current_words)
            sentences.append(
                LLMTranscriptSentence(
                    sentence=sentence_text,
                    start=current_start,
                    end=current_words[-1].end,
                    words=current_words,
                )
            )

        return sentences
