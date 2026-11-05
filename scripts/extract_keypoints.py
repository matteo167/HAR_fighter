#!/usr/bin/env python
"""Reprocessa os vídeos brutos (data/raw_videos/<movement>__<action>/) extraindo
keypoints para data/keypoints. Útil para regenerar o dataset de keypoints
após trocar ou melhorar o PoseEstimator, sem precisar regravar os vídeos
originais. Os dois labels de cada clipe são lidos do nome da pasta.

Uso:
    python scripts/extract_keypoints.py
    python scripts/extract_keypoints.py --combo idle__punch_light
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

    movement_label, action_label = video_path.parent.name.split("__", 1)
    return KeypointSample(
        keypoints=np.stack(keypoints_frames, axis=0),
        visibility=np.stack(visibility_frames, axis=0),
        movement_label=movement_label,
        action_label=action_label,
        fps=fps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combo", default=None, help="Reprocessar apenas esta pasta (<movement>__<action>)"
    )
    args = parser.parse_args()

    settings = load_settings()
    setup_logging(settings["logging"]["level"])

    raw_videos_dir = Path(settings["dataset"]["raw_videos_dir"])
    keypoints_dir = Path(settings["dataset"]["keypoints_dir"])
    fps = settings["capture"]["fps"]

    movement_dirs = (
        [raw_videos_dir / args.combo] if args.combo else sorted(raw_videos_dir.iterdir())
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
