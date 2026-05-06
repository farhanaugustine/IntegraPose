from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2


@dataclass
class Segment:
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    label: str
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay CSV annotations on the source video to visually audit the "
            "behavior predictions."
        )
    )
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--annotations_csv", type=Path, required=True)
    parser.add_argument(
        "--output_video",
        type=Path,
        default=None,
        help="Optional path to save an annotated video (e.g., review.mp4).",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Skip the interactive preview window (useful when only saving video).",
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=0.7,
        help="OpenCV font scale for overlay text.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Text/border thickness for overlays.",
    )
    parser.add_argument(
        "--wait_ms",
        type=int,
        default=1,
        help="Delay (ms) for cv2.waitKey. Increase to slow playback.",
    )
    return parser.parse_args()


def load_segments(csv_path: Path) -> List[Segment]:
    segments: List[Segment] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                segments.append(
                    Segment(
                        start_time=float(row["start_time"]),
                        end_time=float(row["end_time"]),
                        start_frame=int(row["start_frame"]),
                        end_frame=int(row["end_frame"]),
                        label=row["label"],
                        confidence=float(row["confidence"]),
                    )
                )
            except Exception as exc:  # pragma: no cover
                print(f"Warning: skipping malformed row {row} ({exc})")
    segments.sort(key=lambda seg: seg.start_frame)
    return segments


def annotate_frame(
    frame,
    segment: Optional[Segment],
    frame_idx: int,
    fps: float,
    total_frames: int,
    font_scale: float,
    thickness: int,
):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    panel_height = 90
    cv2.rectangle(
        overlay,
        (0, 0),
        (w, panel_height),
        (0, 0, 0),
        thickness=cv2.FILLED,
    )

    if segment:
        label = segment.label
        conf = segment.confidence
        time_text = f"{segment.start_time:.2f}s - {segment.end_time:.2f}s"
    else:
        label = "No Annotation"
        conf = 0.0
        time_text = "N/A"

    lines = [
        f"Frame {frame_idx}  Time {frame_idx / max(fps, 1e-3):.2f}s",
        f"Label: {label}  Confidence: {conf:.2f}",
        f"Segment: {time_text}",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            overlay,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
        y += int(30 * font_scale)

    progress = frame_idx / max(total_frames - 1, 1)
    bar_y = panel_height - 10
    cv2.rectangle(
        overlay,
        (0, bar_y),
        (int(w * progress), bar_y + 5),
        (0, 255, 0) if segment else (0, 128, 255),
        thickness=cv2.FILLED,
    )
    return overlay


def main():
    args = parse_args()
    video_path = args.video_path
    csv_path = args.annotations_csv

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    segments = load_segments(csv_path)
    if not segments:
        print(
            "No annotations were loaded. Check that the CSV has the expected header."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(args.output_video), fourcc, fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {args.output_video}")

    current_idx = 0
    frame_idx = 0
    window_name = "Behavior Review"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        while (
            current_idx < len(segments)
            and frame_idx > segments[current_idx].end_frame
        ):
            current_idx += 1

        if (
            current_idx < len(segments)
            and segments[current_idx].start_frame <= frame_idx <= segments[current_idx].end_frame
        ):
            active = segments[current_idx]
        else:
            active = None

        annotated = annotate_frame(
            frame,
            active,
            frame_idx,
            fps,
            total_frames,
            args.font_scale,
            args.thickness,
        )

        if writer:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(args.wait_ms) & 0xFF
            if key == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
