"""Interface (Strategy) para estimadores de pose.

YOLOv8-Pose é o backend inicial, mas qualquer outro estimador de keypoints
(ex.: MediaPipe, MoveNet, OpenPose) pode implementar PoseEstimator sem exigir
mudanças nas camadas de features/classificação, que trabalham apenas com
PoseResult.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseResult:
    """Resultado da estimativa de pose para um único frame.

    landmarks: array (N, 3) com (x, y, z) normalizados em [0, 1] por landmark.
    visibility: array (N,) com a confiança de cada landmark.
    """

    landmarks: np.ndarray
    visibility: np.ndarray

    @property
    def num_landmarks(self) -> int:
        return self.landmarks.shape[0]


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray) -> PoseResult | None:
        """Extrai keypoints de um frame BGR. Retorna None se nada for detectado."""

    def close(self) -> None:
        """Libera recursos do estimador (override se necessário)."""
