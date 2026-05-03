#!/usr/bin/env python3
"""
Create a portrait B-roll video with face-aligned crops.

For each portrait:
1. Detect face and eyes using OpenCV Haar cascades
2. Compute a 9:16 crop that frames the face with consistent padding
3. Place eyes at a consistent vertical position (~1/3 from top)
4. Track zoom level (original height / crop height)
5. Sort by zoom level (least zoomed first)
6. Render to video at 0.25s per image
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

BASE = Path("cc_wsp/sandbox/portraits")
OUT = BASE / "top_performers_broll.mp4"
TARGET_W, TARGET_H = 1080, 1920
ASPECT = TARGET_W / TARGET_H  # 9:16 = 0.5625
FRAME_DURATION = 0.25
FPS = 30

# How much padding above the face as a multiple of face height
# This controls framing: higher = more space above head
HEAD_PADDING_RATIO = 0.6
# Minimum crop height as fraction of image height (don't zoom in too much)
MIN_CROP_FRAC = 0.4


@dataclass
class PortraitInfo:
    path: Path
    name: str
    eye_x_frac: float
    eye_y_frac: float
    face_h_frac: float  # face height as fraction of image height
    zoom_factor: float
    crop_params: tuple  # (x, y, w, h) in original image pixels


def detect_face_and_eyes(img_path: Path):
    """
    Detect face and eyes. Returns (eye_x_frac, eye_y_frac, face_x, face_y, face_w, face_h)
    in pixel coords, or None if detection fails.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
    if len(faces) == 0:
        return None

    # Use largest face
    face = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = face

    # Detect eyes within face region
    face_roi = gray[fy : fy + fh, fx : fx + fw]
    eyes = eye_cascade.detectMultiScale(
        face_roi, 1.1, 5, minSize=(int(fw * 0.1), int(fh * 0.05))
    )

    if len(eyes) >= 2:
        eyes_sorted = sorted(eyes, key=lambda e: e[1])[:2]
        eye_centers = [(ex + ew // 2, ey + eh // 2) for ex, ey, ew, eh in eyes_sorted]
        avg_eye_x = (eye_centers[0][0] + eye_centers[1][0]) / 2 + fx
        avg_eye_y = (eye_centers[0][1] + eye_centers[1][1]) / 2 + fy
    else:
        avg_eye_x = fx + fw / 2
        avg_eye_y = fy + fh * 0.35

    return (avg_eye_x / w, avg_eye_y / h, fx, fy, fw, fh)


def compute_crop(img_w, img_h, eye_x_frac, eye_y_frac, face_h, target_eye_y_frac):
    """
    Compute a 9:16 crop that places eyes at target_eye_y_frac from top.
    Uses face height to determine crop size (tighter face = more zoom).
    """
    eye_y_px = eye_y_frac * img_h
    eye_x_px = eye_x_frac * img_w

    # Crop height: determined by placing eyes at target position
    # We want: eye_y_px - crop_y = target_eye_y_frac * crop_h
    # And we want the face to be well-framed, so use face height to set scale.
    # A face that takes up ~25-35% of the crop height looks natural.
    target_face_frac = 0.30  # face should be ~30% of crop height
    crop_h = int(face_h / target_face_frac)

    # Clamp: don't crop smaller than MIN_CROP_FRAC of image
    min_crop_h = int(img_h * MIN_CROP_FRAC)
    crop_h = max(crop_h, min_crop_h)
    # Don't exceed image height
    crop_h = min(crop_h, img_h)

    crop_w = int(crop_h * ASPECT)
    # If crop_w exceeds image width, constrain
    if crop_w > img_w:
        crop_w = img_w
        crop_h = int(crop_w / ASPECT)

    # Position vertically: eyes at target_eye_y_frac
    crop_y = int(eye_y_px - target_eye_y_frac * crop_h)
    crop_y = max(0, min(crop_y, img_h - crop_h))

    # Position horizontally: center on eyes
    crop_x = int(eye_x_px - crop_w / 2)
    crop_x = max(0, min(crop_x, img_w - crop_w))

    zoom_factor = img_h / crop_h

    return crop_x, crop_y, crop_w, crop_h, zoom_factor


def process_portraits():
    image_paths = sorted(BASE.rglob("*.jpg"))
    print(f"Found {len(image_paths)} portraits")

    # Pass 1: detect faces/eyes
    detections = []
    for path in image_paths:
        name = path.stem.replace("_", " ")
        result = detect_face_and_eyes(path)
        img = cv2.imread(str(path))
        h, w = img.shape[:2]

        if result is None:
            print(f"  FAILED: {name} — using defaults")
            eye_x_frac, eye_y_frac = 0.5, 0.33
            face_h = h * 0.4  # assume face is 40% of image
        else:
            eye_x_frac, eye_y_frac, fx, fy, fw, fh = result
            face_h = fh
            print(f"  OK: {name} (eyes={eye_y_frac:.0%}, face_h={fh}px/{h}px={fh/h:.0%})")

        detections.append(PortraitInfo(
            path=path, name=name,
            eye_x_frac=eye_x_frac, eye_y_frac=eye_y_frac,
            face_h_frac=face_h / h,
            zoom_factor=1.0, crop_params=(0, 0, 0, 0),
        ))

    # Target eye position: mean of detected, capped around 1/3 from top
    eye_ys = [d.eye_y_frac for d in detections]
    mean_eye_y = float(np.mean(eye_ys))
    target_eye_y = min(mean_eye_y, 0.38)
    print(f"\nMean eye position: {mean_eye_y:.1%} from top")
    print(f"Target eye position: {target_eye_y:.1%} from top ({1-target_eye_y:.1%} from bottom)")

    # Pass 2: compute crops
    for d in detections:
        img = cv2.imread(str(d.path))
        h, w = img.shape[:2]
        face_h_px = d.face_h_frac * h

        cx, cy, cw, ch, zoom = compute_crop(
            w, h, d.eye_x_frac, d.eye_y_frac, face_h_px, target_eye_y
        )
        d.crop_params = (cx, cy, cw, ch)
        d.zoom_factor = zoom

    # Sort by zoom factor
    detections.sort(key=lambda d: d.zoom_factor)

    print(f"\nZoom range: {detections[0].zoom_factor:.2f}x ({detections[0].name}) — {detections[-1].zoom_factor:.2f}x ({detections[-1].name})")

    # Create video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_out = BASE / "top_performers_broll_temp.mp4"
    writer = cv2.VideoWriter(str(temp_out), fourcc, FPS, (TARGET_W, TARGET_H))
    frames_per_image = max(1, int(FRAME_DURATION * FPS))

    for d in detections:
        img = cv2.imread(str(d.path))
        cx, cy, cw, ch = d.crop_params
        cropped = img[cy : cy + ch, cx : cx + cw]
        resized = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
        for _ in range(frames_per_image):
            writer.write(resized)

    writer.release()

    # Re-encode with ffmpeg
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", str(temp_out),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        str(OUT),
    ], capture_output=True, check=True)
    temp_out.unlink()

    print(f"\nCreated: {OUT}")
    print(f"Duration: {len(detections) * FRAME_DURATION:.1f}s")
    print(f"\nSort order (least to most zoomed):")
    for i, d in enumerate(detections, 1):
        print(f"  {i:2d}. {d.name:<25s} zoom={d.zoom_factor:.2f}x  face={d.face_h_frac:.0%}")


if __name__ == "__main__":
    process_portraits()
