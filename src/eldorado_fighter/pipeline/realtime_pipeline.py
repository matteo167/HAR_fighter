"""Orquestração do pipeline em tempo real (padrão Pipe-and-Filter):

  FrameSource -> PoseEstimator -> normalize_landmarks -> SequenceBuffer
      -> MovementClassifier -> EventBus("movement_detected")

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
from eldorado_fighter.pose.visualization import draw_pose, draw_prediction_label

logger = logging.getLogger(__name__)

_PREVIEW_WINDOW = "Eldorado Fighter - live"


class RealtimePipeline:
    def __init__(
        self,
        frame_source: FrameSource,
        pose_estimator: PoseEstimator,
        sequence_buffer: SequenceBuffer,
        classifier: MovementClassifier,
        event_bus: EventBus,
        confidence_threshold: float = 0.6,
        feature_fn: Callable[[np.ndarray], np.ndarray] = normalize_landmarks,
        window_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        show_preview: bool = True,
    ) -> None:
        self._frame_source = frame_source
        self._pose_estimator = pose_estimator
        self._sequence_buffer = sequence_buffer
        self._classifier = classifier
        self._event_bus = event_bus
        self._confidence_threshold = confidence_threshold
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

        prediction: PredictionResult | None = None
        if pose is not None:
            self._sequence_buffer.push(self._feature_fn(pose.landmarks))
            if self._sequence_buffer.is_ready():
                sequence = self._sequence_buffer.as_array()
                if self._window_fn is not None:
                    sequence = self._window_fn(sequence)
                prediction = self._classifier.predict(sequence)

        if self._show_preview:
            self._show_frame(frame, pose, prediction)

        if prediction is None or prediction.confidence < self._confidence_threshold:
            return

        logger.debug(
            "Movimento detectado: %s (%.2f)", prediction.movement.name, prediction.confidence
        )
        self._event_bus.publish("movement_detected", prediction)

    def _show_frame(self, frame, pose, prediction: PredictionResult | None) -> None:
        annotated = draw_pose(frame, pose)
        movement_name = prediction.movement.name if prediction is not None else None
        confidence = prediction.confidence if prediction is not None else 0.0
        draw_prediction_label(annotated, movement_name, confidence, self._confidence_threshold)
        cv2.imshow(_PREVIEW_WINDOW, annotated)
        cv2.waitKey(1)
