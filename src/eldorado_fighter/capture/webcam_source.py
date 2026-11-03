"""Implementação de FrameSource usando a webcam via OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from eldorado_fighter.capture.base import FrameSource


class WebcamSource(FrameSource):
    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.device_index = device_index
        self._width = width
        self._height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a câmera (device_index={self.device_index})."
            )

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            raise RuntimeError("Chame start() antes de read().")
        ok, frame = self._cap.read()
        return frame if ok else None

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
