"""Orquestração do pipeline em tempo real (padrão Pipe-and-Filter):

  FrameSource -> PoseEstimator -> normalize_landmarks -> SequenceBuffer
      -> { MovementClassifier("movement"), MovementClassifier("action") }
      -> EventBus("movement_detected" / "action_detected")

Captura, pose, normalização e o buffer de sequência são ÚNICOS e
compartilhados — a parte cara do pipeline não duplica. A cada janela
pronta, um classificador por eixo roda sobre a MESMA sequência e publica
seu próprio evento, permitindo que os dois eixos (ex.: andar e socar)
sejam detectados e disparados simultaneamente (ver actions/action_mapper.py).

Cada estágio é uma dependência injetada (Dependency Injection), construída
pelas fábricas em scripts/run_live.py a partir da config. O pipeline em si
não conhece YOLOv8-Pose, TensorFlow ou pynput — apenas as interfaces.
"""

from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from eldorado_fighter.capture.base import FrameSource
from eldorado_fighter.classification.base import MovementClassifier, PredictionResult
from eldorado_fighter.features.normalizer import normalize_landmarks
from eldorado_fighter.features.sequence_buffer import SequenceBuffer
from eldorado_fighter.pipeline.events import EventBus
from eldorado_fighter.pose.base import PoseEstimator
from eldorado_fighter.pose.visualization import draw_pose, draw_prediction_labels

logger = logging.getLogger(__name__)

_PREVIEW_WINDOW = "Eldorado Fighter - live"


class RealtimePipeline:
    def __init__(
        self,
        frame_source: FrameSource,
        pose_estimator: PoseEstimator,
        sequence_buffer: SequenceBuffer,
        classifiers: dict[str, MovementClassifier],
        event_bus: EventBus,
        confidence_thresholds: dict[str, float],
        feature_fn: Callable[[np.ndarray], np.ndarray] = normalize_landmarks,
        window_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        show_preview: bool = True,
    ) -> None:
        self._frame_source = frame_source
        self._pose_estimator = pose_estimator
        self._sequence_buffer = sequence_buffer
        self._classifiers = classifiers
        self._event_bus = event_bus
        self._confidence_thresholds = confidence_thresholds
        self._feature_fn = feature_fn
        self._window_fn = window_fn
        self._show_preview = show_preview
        self._running = False

    def run(self) -> None:
        self._running = True
        with self._frame_source:
            try:
                while self._running:
                    frame = self._frame_source.read()
                    if frame is None:
                        break
                    self._process_frame(frame)
            finally:
                if self._show_preview:
                    cv2.destroyWindow(_PREVIEW_WINDOW)

    def stop(self) -> None:
        self._running = False

    def _process_frame(self, frame) -> None:
        pose = self._pose_estimator.estimate(frame)

        predictions: dict[str, PredictionResult | None] = {axis: None for axis in self._classifiers}
        if pose is not None:
            self._sequence_buffer.push(self._feature_fn(pose.landmarks))
            if self._sequence_buffer.is_ready():
                sequence = self._sequence_buffer.as_array()
                if self._window_fn is not None:
                    sequence = self._window_fn(sequence)
                predictions = {axis: clf.predict(sequence) for axis, clf in self._classifiers.items()}

        if self._show_preview:
            self._show_frame(frame, pose, predictions)

        for axis, prediction in predictions.items():
            if prediction is None or prediction.confidence < self._confidence_thresholds[axis]:
                continue
            logger.debug(
                "%s detectado: %s (%.2f)", axis, prediction.movement.name, prediction.confidence
            )
            self._event_bus.publish(f"{axis}_detected", prediction)

    def _show_frame(self, frame, pose, predictions: dict[str, PredictionResult | None]) -> None:
        annotated = draw_pose(frame, pose)
        rows = [
            (
                axis,
                prediction.movement.name if prediction is not None else None,
                prediction.confidence if prediction is not None else 0.0,
                self._confidence_thresholds[axis],
            )
            for axis, prediction in predictions.items()
        ]
        draw_prediction_labels(annotated, rows)
        cv2.imshow(_PREVIEW_WINDOW, annotated)
        cv2.waitKey(1)
