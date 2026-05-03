"""
Face detection for automatic caption/image placement.

Samples a small number of frames from a video, runs OpenCV's Haar cascade
frontal-face detector, and returns the median face bbox as fractions of
the frame dimensions.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from cc_wsp.src.models import FaceData


def detect_face_position(
    video_path: Path,
    num_samples: int = 20,
) -> Optional[FaceData]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if total_frames <= 0 or frame_h <= 0 or frame_w <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions: {video_path}")

    cascades = []
    for name in (
        "haarcascade_frontalface_default.xml",
        "haarcascade_profileface.xml",
    ):
        c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if not c.empty():
            cascades.append(c)
    if not cascades:
        cap.release()
        raise RuntimeError("Could not load any Haar cascade")

    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    min_side = max(20, min(frame_w, frame_h) // 12)

    tops, bottoms, lefts, rights = [], [], [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_faces = []
        for c in cascades:
            found = c.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(min_side, min_side),
            )
            if len(found) > 0:
                all_faces.extend(found.tolist())
        if not all_faces:
            continue
        x, y, w, h = max(all_faces, key=lambda f: f[2] * f[3])
        tops.append(y / frame_h)
        bottoms.append((y + h) / frame_h)
        lefts.append(x / frame_w)
        rights.append((x + w) / frame_w)

    cap.release()

    if not tops:
        return None

    return FaceData(
        top_frac=float(np.median(tops)),
        bottom_frac=float(np.median(bottoms)),
        left_frac=float(np.median(lefts)),
        right_frac=float(np.median(rights)),
        samples_with_face=len(tops),
        total_samples=num_samples,
    )
