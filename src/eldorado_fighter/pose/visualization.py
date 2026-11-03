"""Desenho de keypoints/esqueleto sobre um frame, para feedback visual em
tempo real durante gravação ou execução ao vivo.

Conexões seguem o esquema COCO-17 (ver features/normalizer.py).
"""

from __future__ import annotations

import cv2
import numpy as np

from eldorado_fighter.pose.base import PoseResult

_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # cabeça
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # ombros/braços
    (5, 11), (6, 12), (11, 12),                # tronco
    (11, 13), (13, 15), (12, 14), (14, 16),    # quadril/pernas
]

_POINT_COLOR = (0, 255, 0)
_LINE_COLOR = (255, 128, 0)
_MIN_VISIBILITY = 0.3

_DIRECTION_LABELS = {
    "walk_left": "<< ESQUERDA",
    "walk_right": "DIREITA >>",
}
_DETECTED_COLOR = (0, 255, 0)
_LOW_CONFIDENCE_COLOR = (160, 160, 160)


def draw_pose(frame: np.ndarray, pose: PoseResult | None) -> np.ndarray:
    """Retorna uma cópia do frame com o esqueleto desenhado (se pose detectada)."""
    annotated = frame.copy()
    if pose is None:
        return annotated

    height, width = annotated.shape[:2]
    points = pose.landmarks[:, :2] * np.array([width, height])

    for start, end in _SKELETON_EDGES:
        if pose.visibility[start] < _MIN_VISIBILITY or pose.visibility[end] < _MIN_VISIBILITY:
            continue
        cv2.line(annotated, tuple(points[start].astype(int)), tuple(points[end].astype(int)), _LINE_COLOR, 2)

    for idx, (x, y) in enumerate(points):
        if pose.visibility[idx] < _MIN_VISIBILITY:
            continue
        cv2.circle(annotated, (int(x), int(y)), 4, _POINT_COLOR, -1)

    return annotated


def draw_prediction_label(
    frame: np.ndarray,
    movement_name: str | None,
    confidence: float,
    confidence_threshold: float,
) -> np.ndarray:
    """Escreve o movimento detectado no canto superior do frame — traduzido
    para "ESQUERDA"/"DIREITA" quando é um dos movimentos de caminhada, ou o
    nome do movimento em caixa alta para os demais.

    Em cinza quando não há pessoa detectada ou a confiança está abaixo do
    limiar (mesmo caso em que o pipeline não dispara a ação no jogo).
    """
    if movement_name is None:
        text, color = "...", _LOW_CONFIDENCE_COLOR
    else:
        label = _DIRECTION_LABELS.get(movement_name, movement_name.upper())
        color = _DETECTED_COLOR if confidence >= confidence_threshold else _LOW_CONFIDENCE_COLOR
        text = f"{label} ({confidence:.0%})"

    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return frame
