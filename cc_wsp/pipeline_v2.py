#!/usr/bin/env python3
"""cc_wsp v2 pipeline: crossfaded cut → MLT overlay-only → captions.

Standalone CLI; does not modify the v1 tool.py. Run after the v1 cut is
approved (adjusted.json finalized).

Usage:
  python cc_wsp/pipeline_v2.py crossfade-cut <video> [--xfade 0.1]
  python cc_wsp/pipeline_v2.py cut-downsample <video>
  python cc_wsp/pipeline_v2.py transcribe-v2 <video>
  python cc_wsp/pipeline_v2.py place-images-v2 <video>
  python cc_wsp/pipeline_v2.py preview-v2 <video>
  python cc_wsp/pipeline_v2.py render-v2 <video>
  python cc_wsp/pipeline_v2.py captions-v2 <video>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cc_wsp.src.services import (
    crossfade_service, transcribe_v2_service,
    place_images_v2_service, mlt_overlay_service, captions_v2_service,
)


def cmd_crossfade_cut(args):
    crossfade_service.create_crossfaded_cut(
        args.video,
        draft=args.draft,
        force=args.force,
    )


def cmd_cut_downsample(args):
    crossfade_service.downsample_cut(args.video, force=args.force)


def cmd_extract_cut_audio(args):
    crossfade_service.extract_cut_audio(args.video, force=args.force)


def cmd_transcribe_v2(args):
    transcribe_v2_service.transcribe_cut(args.video, force=args.force)


def cmd_place_images_v2(args):
    place_images_v2_service.place_images_v2(args.video, force=args.force)


def cmd_preview_v2(args):
    mlt_overlay_service.preview_v2(args.video, force=args.force)


def cmd_render_v2(args):
    mlt_overlay_service.render_v2(args.video, force=args.force)


def cmd_captions_v2(args):
    captions_v2_service.caption_v2(
        args.video,
        title=args.title,
        no_title=args.no_title,
        caption_y=args.caption_y,
        force=args.force,
    )


def main():
    p = argparse.ArgumentParser(description="cc_wsp v2 pipeline")
    p.add_argument("--force", "-f", action="store_true")
    sub = p.add_subparsers(dest="command")

    xc = sub.add_parser("crossfade-cut", help="Render cut.mp4 from adjusted.json")
    xc.add_argument("video")
    xc.add_argument("--draft", action="store_true",
                    help="use downsampled.mp4 as source for fast iteration")

    sub.add_parser("cut-downsample", help="Downsample cut.mp4 to 540p").add_argument("video")
    sub.add_parser("extract-cut-audio", help="Extract MP3 from cut.mp4").add_argument("video")
    sub.add_parser("transcribe-v2", help="Deepgram on cut.mp4 → transcription_v2.json").add_argument("video")
    sub.add_parser("place-images-v2", help="LLM places images using v2 sentences").add_argument("video")
    sub.add_parser("preview-v2", help="MLT overlay on cut_downsampled.mp4 → preview_v2.mp4").add_argument("video")
    sub.add_parser("render-v2", help="MLT overlay on cut.mp4 → final_v2.mp4").add_argument("video")

    cap = sub.add_parser("captions-v2", help="Burn captions on final_v2.mp4")
    cap.add_argument("video")
    cap.add_argument("--title", type=str, default=None)
    cap.add_argument("--no-title", action="store_true")
    cap.add_argument("--caption-y", type=float, default=None)

    args = p.parse_args()
    if not args.command:
        p.print_help()
        return

    cmds = {
        "crossfade-cut": cmd_crossfade_cut,
        "cut-downsample": cmd_cut_downsample,
        "extract-cut-audio": cmd_extract_cut_audio,
        "transcribe-v2": cmd_transcribe_v2,
        "place-images-v2": cmd_place_images_v2,
        "preview-v2": cmd_preview_v2,
        "render-v2": cmd_render_v2,
        "captions-v2": cmd_captions_v2,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
