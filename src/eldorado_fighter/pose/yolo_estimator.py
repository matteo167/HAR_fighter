"""Implementação de PoseEstimator usando YOLOv8-Pose (Ultralytics).

Esquema de keypoints: COCO-17 (ver features/normalizer.py para os índices
usados). YOLOv8-Pose não estima profundidade (z); o terceiro componente de
cada landmark é sempre 0.0 para manter o formato (N, 3) exigido por
PoseResult/SequenceBuffer.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from eldorado_fighter.pose.base import PoseEstimator, PoseResult


class YoloPoseEstimator(PoseEstimator):
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self._model = YOLO(model_path)
        self._confidence_threshold = confidence_threshold
        self._device = device

    def estimate(self, frame: np.ndarray) -> PoseResult | None:
        results = self._model.predict(
            frame,
            device=self._device,
            conf=self._confidence_threshold,
            verbose=False,
        )
        keypoints = results[0].keypoints
        if keypoints is None or keypoints.conf is None or len(keypoints.conf) == 0:
            return None

        # Várias pessoas podem ser detectadas; assume-se que o jogador é
        # quem tem a maior confiança média de pose (ex.: mais próximo/central).
        mean_conf = keypoints.conf.mean(dim=1)
        best_idx = int(mean_conf.argmax())

        xy = keypoints.xyn[best_idx].cpu().numpy()
        conf = keypoints.conf[best_idx].cpu().numpy()

        landmarks = np.zeros((xy.shape[0], 3), dtype=np.float32)
        landmarks[:, :2] = xy
        visibility = conf.astype(np.float32)

        return PoseResult(landmarks=landmarks, visibility=visibility)

    def close(self) -> None:
        pass
