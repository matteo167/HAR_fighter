#!/usr/bin/env python
"""Treina o classificador de movimentos a partir do dataset de keypoints.

Uso:
    python scripts/train_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eldorado_fighter.classification.movements import MovementRegistry
from eldorado_fighter.training.train import run_training
from eldorado_fighter.utils.config import load_settings
from eldorado_fighter.utils.logging_setup import setup_logging


def main() -> None:
    settings = load_settings()
    setup_logging(settings["logging"]["level"])

    registry = MovementRegistry.from_config()
    run_training(settings, registry)


if __name__ == "__main__":
    main()
