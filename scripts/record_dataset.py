#!/usr/bin/env python
"""Grava amostras rotuladas do dataset de treino. Cada clipe tem uma classe
por eixo (ver config/movements.yaml): --movement e --action, ambos
opcionais (default "idle"). Grave um eixo só (o outro fica "idle") ou os
dois juntos, para ensinar o modelo a reconhecer combos (ex.: andar
socando).

Uso:
    python scripts/record_dataset.py --action punch_light --samples 40
    python scripts/record_dataset.py --movement walk_left --samples 20 --clip-seconds 2.5
    python scripts/record_dataset.py --movement walk_left --action punch_light --samples 10
    python scripts/record_dataset.py --action punch_light --samples 10 --no-countdown

Para cada amostra, uma janela mostra a câmera com os keypoints desenhados;
aperte ESPAÇO (com a janela em foco) para iniciar a contagem regressiva (ou
a gravação direto, se --no-countdown), que dá tempo de se posicionar antes
de executar o movimento.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eldorado_fighter.capture.webcam_source import WebcamSource
from eldorado_fighter.classification.movements import MovementRegistry
from eldorado_fighter.dataset.recorder import DatasetRecorder
from eldorado_fighter.pose.yolo_estimator import YoloPoseEstimator
from eldorado_fighter.utils.config import load_settings
from eldorado_fighter.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--movement", default="idle", help="Classe do eixo movement (ver config/movements.yaml)"
    )
    parser.add_argument(
        "--action", default="idle", help="Classe do eixo action (ver config/movements.yaml)"
    )
    parser.add_argument("--samples", type=int, default=None, help="Quantidade de clipes a gravar")
    parser.add_argument("--clip-seconds", type=float, default=None, help="Duração de cada clipe")
    parser.add_argument("--countdown", type=float, default=3.0, help="Contagem regressiva (s) antes de cada clipe")
    parser.add_argument(
        "--no-preview", action="store_true", help="Não abrir a janela com vídeo + keypoints em tempo real"
    )
    parser.add_argument(
        "--no-countdown",
        action="store_true",
        help="Pular a contagem regressiva e gravar imediatamente após apertar espaço",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    setup_logging(settings["logging"]["level"])

    movement_registry = MovementRegistry.from_config("movement")
    action_registry = MovementRegistry.from_config("action")
    # KeyError com nome inválido é intencional e claro
    movement = movement_registry.get(args.movement)
    action = action_registry.get(args.action)

    samples = args.samples or settings["dataset"]["samples_per_movement"]
    clip_seconds = args.clip_seconds or settings["dataset"]["clip_seconds"]

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
    recorder = DatasetRecorder(
        frame_source=frame_source,
        pose_estimator=pose_estimator,
        keypoints_dir=Path(settings["dataset"]["keypoints_dir"]),
        raw_videos_dir=Path(settings["dataset"]["raw_videos_dir"]),
        fps=settings["capture"]["fps"],
        show_preview=not args.no_preview,
    )

    combo_label = f"{movement.name}+{action.name}"
    logger.info("Movimento: %s — %s", movement.name, movement.description)
    logger.info("Ação: %s — %s", action.name, action.description)
    logger.info("Gravando %d amostras de %.1fs cada.", samples, clip_seconds)

    try:
        with recorder:
            for i in range(1, samples + 1):
                print(f"\nAmostra {i}/{samples} — prepare-se para: {combo_label}")
                print("  clique na janela de vídeo e aperte ESPAÇO para iniciar...")
                recorder.wait_for_space(f"Amostra {i}/{samples}: {combo_label} - ESPACO para iniciar")

                if not args.no_countdown:
                    for remaining in range(int(args.countdown), 0, -1):
                        print(f"  gravando em {remaining}...")
                        time.sleep(1.0)

                output_path = recorder.record_clip(movement.name, action.name, clip_seconds)
                print(f"  salvo em {output_path}")
    finally:
        pose_estimator.close()

    logger.info("Concluído.")


if __name__ == "__main__":
    main()
