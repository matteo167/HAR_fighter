#!/usr/bin/env python
"""Treina o(s) classificador(es) a partir do dataset de keypoints. Sem
--axis, treina os dois eixos (movement e action) em sequência.

Uso:
    python scripts/train_model.py
    python scripts/train_model.py --axis movement
    python scripts/train_model.py --axis action
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eldorado_fighter.classification.movements import MovementRegistry
from eldorado_fighter.training.train import run_training
from eldorado_fighter.utils.config import load_settings
from eldorado_fighter.utils.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis", choices=["movement", "action"], default=None, help="Treinar só um dos eixos"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    setup_logging(settings["logging"]["level"])

    axes = [args.axis] if args.axis else ["movement", "action"]
    for axis in axes:
        registry = MovementRegistry.from_config(axis)
        run_training(settings, registry, axis)


if __name__ == "__main__":
    main()
