"""Implementação de FrameSource que lê de um arquivo de vídeo em vez da
webcam. Usada por scripts/extract_keypoints.py para reprocessar os vídeos
brutos gravados em data/raw_videos (ex.: após trocar/melhorar o
PoseEstimator), sem precisar regravar o dataset.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from har_fighter.capture.base import FrameSource


class VideoFileSource(FrameSource):
    def __init__(self, video_path: Path) -> None:
        self._video_path = Path(video_path)
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(str(self._video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo '{self._video_path}'.")

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            raise RuntimeError("Chame start() antes de read().")
        ok, frame = self._cap.read()
        return frame if ok else None

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0
