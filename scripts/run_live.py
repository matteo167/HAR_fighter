#!/usr/bin/env python
"""Executa o pipeline em tempo real: câmera -> keypoints -> classificação
(dois eixos simultâneos, movement + action) -> input no jogo.

O jogo deve estar com a janela em foco e configurado para aceitar as teclas
definidas em config/movements.yaml (game_keys).

Uma janela mostra a câmera com os keypoints e as duas classes detectadas
(uma linha por eixo: movimentação e ação) em tempo real; use --no-preview
para desativar.

Uso:
    python scripts/run_live.py
    python scripts/run_live.py --no-preview
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from har_fighter.actions.action_mapper import ActionMapper
from har_fighter.actions.keyboard_backend import KeyboardInputBackend
from har_fighter.capture.webcam_source import WebcamSource
from har_fighter.classification.movements import MovementRegistry
from har_fighter.classification.tf_classifier import TensorFlowMovementClassifier
from har_fighter.features.motion import append_velocity
from har_fighter.features.sequence_buffer import SequenceBuffer
from har_fighter.features.tilt import build_feature_fn
from har_fighter.pipeline.events import EventBus
from har_fighter.pipeline.realtime_pipeline import RealtimePipeline
from har_fighter.pose.yolo_estimator import YoloPoseEstimator
from har_fighter.utils.config import load_settings
from har_fighter.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-preview", action="store_true", help="Não abrir a janela com vídeo + keypoints em tempo real"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    setup_logging(settings["logging"]["level"])

    axes = ("movement", "action")
    registries = {axis: MovementRegistry.from_config(axis) for axis in axes}

    frame_source = WebcamSource(
        device_index=settings["capture"]["device_index"],
        width=settings["capture"]["width"],
        height=settings["capture"]["height"],
        fps=settings["capture"]["fps"],
    )
    pose_estimator = YoloPoseEstimator(
        model_path=settings["pose"]["model_path"],
        confidence_threshold=settings["pose"]["confidence_threshold"],
        device=settings["pose"]["device"],
    )
    sequence_buffer = SequenceBuffer(sequence_length=settings["features"]["sequence_length"])
    classifiers = {
        axis: TensorFlowMovementClassifier(
            model_path=Path(settings["classification"][axis]["model_path"]),
            registry=registries[axis],
        )
        for axis in axes
    }
    confidence_thresholds = {
        axis: settings["classification"][axis]["confidence_threshold"] for axis in axes
    }

    input_backend = KeyboardInputBackend()
    action_mapper = ActionMapper(
        input_backend=input_backend,
        cooldown_seconds=settings["actions"]["cooldown_seconds"],
    )

    event_bus = EventBus()
    for axis in axes:
        event_bus.subscribe(f"{axis}_detected", lambda prediction, axis=axis: action_mapper.on_label_detected(axis, prediction))

    pipeline = RealtimePipeline(
        frame_source=frame_source,
        pose_estimator=pose_estimator,
        sequence_buffer=sequence_buffer,
        classifiers=classifiers,
        event_bus=event_bus,
        confidence_thresholds=confidence_thresholds,
        feature_fn=build_feature_fn(settings["features"].get("feature_mode", "full")),
        window_fn=append_velocity if settings["features"].get("use_velocity", False) else None,
        show_preview=not args.no_preview,
    )

    logger.info("Iniciando pipeline em tempo real. Ctrl+C para parar.")
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()
    finally:
        pose_estimator.close()


if __name__ == "__main__":
    main()
