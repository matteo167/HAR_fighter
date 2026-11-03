#!/usr/bin/env python
"""Reprocessa os vídeos brutos (data/raw_videos) extraindo keypoints para
data/keypoints. Útil para regenerar o dataset de keypoints após trocar ou
melhorar o PoseEstimator, sem precisar regravar os vídeos originais.

Uso:
    python scripts/extract_keypoints.py
    python scripts/extract_keypoints.py --movement punch_light
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from eldorado_fighter.capture.video_file_source import VideoFileSource
from eldorado_fighter.dataset.schema import KeypointSample, save_sample
from eldorado_fighter.pose.yolo_estimator import YoloPoseEstimator
from eldorado_fighter.utils.config import load_settings
from eldorado_fighter.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def extract_from_video(video_path: Path, pose_estimator: YoloPoseEstimator, fps: int) -> KeypointSample:
    keypoints_frames: list[np.ndarray] = []
    visibility_frames: list[np.ndarray] = []

    with VideoFileSource(video_path) as source:
        while True:
            frame = source.read()
            if frame is None:
                break
            pose = pose_estimator.estimate(frame)
            if pose is not None:
                keypoints_frames.append(pose.landmarks)
                visibility_frames.append(pose.visibility)

    return KeypointSample(
        keypoints=np.stack(keypoints_frames, axis=0),
        visibility=np.stack(visibility_frames, axis=0),
        label=video_path.parent.name,
        fps=fps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movement", default=None, help="Reprocessar apenas este movimento")
    args = parser.parse_args()

    settings = load_settings()
    setup_logging(settings["logging"]["level"])

    raw_videos_dir = Path(settings["dataset"]["raw_videos_dir"])
    keypoints_dir = Path(settings["dataset"]["keypoints_dir"])
    fps = settings["capture"]["fps"]

    movement_dirs = (
        [raw_videos_dir / args.movement] if args.movement else sorted(raw_videos_dir.iterdir())
    )

    pose_estimator = YoloPoseEstimator(
        model_path=settings["pose"]["model_path"],
        confidence_threshold=settings["pose"]["confidence_threshold"],
        device=settings["pose"]["device"],
    )

    try:
        for movement_dir in movement_dirs:
            if not movement_dir.is_dir():
                continue
            for video_path in sorted(movement_dir.glob("*.mp4")):
                sample = extract_from_video(video_path, pose_estimator, fps)
                output_path = keypoints_dir / movement_dir.name / f"{video_path.stem}.npz"
                save_sample(output_path, sample)
                logger.info("%s -> %s", video_path, output_path)
    finally:
        pose_estimator.close()


if __name__ == "__main__":
    main()
