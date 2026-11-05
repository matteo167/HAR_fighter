#!/usr/bin/env python
"""Migração one-off do dataset legado (schema de um label só, arquivado em
data/obsoleto/v1/) para o novo schema de dois labels (movement + action).

Mapa fixo: o label antigo vira o label do eixo a que pertencia; o outro
eixo recebe "idle" (nenhum desses clipes combinava os dois eixos).

Uso:
    python scripts/migrate_legacy_dataset.py
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from eldorado_fighter.dataset.schema import KeypointSample, save_sample
from eldorado_fighter.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_KEYPOINTS_DIR = REPO_ROOT / "data" / "obsoleto" / "v1" / "keypoints"
LEGACY_RAW_VIDEOS_DIR = REPO_ROOT / "data" / "obsoleto" / "v1" / "raw_videos"
NEW_KEYPOINTS_DIR = REPO_ROOT / "data" / "keypoints"
NEW_RAW_VIDEOS_DIR = REPO_ROOT / "data" / "raw_videos"

# old_label -> (movement_label, action_label)
LEGACY_LABEL_MAP: dict[str, tuple[str, str]] = {
    "idle": ("idle", "idle"),
    "walk_left": ("walk_left", "idle"),
    "walk_right": ("walk_right", "idle"),
    "punch_light": ("idle", "punch_light"),
    "kick_light": ("idle", "kick_light"),
}


def migrate_keypoints() -> int:
    count = 0
    for old_label, (movement_label, action_label) in LEGACY_LABEL_MAP.items():
        label_dir = LEGACY_KEYPOINTS_DIR / old_label
        if not label_dir.exists():
            continue

        combo_dir_name = f"{movement_label}__{action_label}"
        for sample_path in sorted(label_dir.glob("*.npz")):
            with np.load(sample_path, allow_pickle=False) as data:
                sample = KeypointSample(
                    keypoints=data["keypoints"],
                    visibility=data["visibility"],
                    movement_label=movement_label,
                    action_label=action_label,
                    fps=int(data["fps"]),
                )
            output_path = NEW_KEYPOINTS_DIR / combo_dir_name / sample_path.name
            save_sample(output_path, sample)
            count += 1
    return count


def migrate_raw_videos() -> int:
    count = 0
    for old_label, (movement_label, action_label) in LEGACY_LABEL_MAP.items():
        video_dir = LEGACY_RAW_VIDEOS_DIR / old_label
        if not video_dir.exists():
            continue

        combo_dir_name = f"{movement_label}__{action_label}"
        output_dir = NEW_RAW_VIDEOS_DIR / combo_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for video_path in sorted(video_dir.glob("*.mp4")):
            shutil.copy2(video_path, output_dir / video_path.name)
            count += 1
    return count


def main() -> None:
    setup_logging("INFO")
    keypoints_count = migrate_keypoints()
    logger.info("Migrados %d arquivos de keypoints para %s", keypoints_count, NEW_KEYPOINTS_DIR)
    videos_count = migrate_raw_videos()
    logger.info("Migrados %d vídeos brutos para %s", videos_count, NEW_RAW_VIDEOS_DIR)


if __name__ == "__main__":
    main()
