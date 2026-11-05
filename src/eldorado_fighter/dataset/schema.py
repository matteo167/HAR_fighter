"""Formato em disco de uma amostra do dataset de keypoints.

Guardamos landmarks BRUTOS (sem normalização) em .npz: a normalização é uma
decisão da camada de features (eldorado_fighter.features.normalizer) e pode
evoluir sem exigir que o dataset seja regravado — só re-extraído, se vier a
mudar o próprio estimador de pose.

Cada amostra guarda DOIS labels (um por eixo — ver
classification.movements), já que um mesmo clipe pode combinar os dois
eixos (ex.: andar para a esquerda enquanto soca):

  keypoints.npz
    keypoints:      float32 (T, num_landmarks, 3)
    visibility:     float32 (T, num_landmarks)
    movement_label: str
    action_label:   str
    fps:            int
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class KeypointSample:
    keypoints: np.ndarray
    visibility: np.ndarray
    movement_label: str
    action_label: str
    fps: int


def save_sample(path: Path, sample: KeypointSample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=sample.keypoints,
        visibility=sample.visibility,
        movement_label=sample.movement_label,
        action_label=sample.action_label,
        fps=sample.fps,
    )


def load_sample(path: Path) -> KeypointSample:
    with np.load(path, allow_pickle=False) as data:
        return KeypointSample(
            keypoints=data["keypoints"],
            visibility=data["visibility"],
            movement_label=str(data["movement_label"]),
            action_label=str(data["action_label"]),
            fps=int(data["fps"]),
        )
