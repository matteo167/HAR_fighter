#!/usr/bin/env python
"""Plota a matriz de confusão (confusion_matrix.json) de um run de treino.

Uso:
    python scripts/plot_confusion_matrix.py
    python scripts/plot_confusion_matrix.py --run-dir data/models/20261103T194302
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "data" / "models" / "latest",
        help="Pasta do run de treino (contém confusion_matrix.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads((args.run_dir / "confusion_matrix.json").read_text())
    labels = data["labels"]
    matrix = np.array(data["matrix"])

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusão")

    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > threshold else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontweight=weight)

    fig.colorbar(im, ax=ax, label="amostras")
    fig.tight_layout()

    output_path = args.run_dir / "confusion_matrix.png"
    fig.savefig(output_path, dpi=150)
    print(f"Salvo em {output_path}")


if __name__ == "__main__":
    main()
