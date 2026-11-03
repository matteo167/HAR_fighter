"""Normalização de landmarks para torná-los invariantes à posição do corpo
no quadro e à distância até a câmera.

Sem isso, o modelo aprenderia "onde a pessoa está na imagem" em vez de
"qual movimento ela está fazendo", o que não generaliza entre sessões de
gravação diferentes.

Índices seguem o esquema COCO-17 (YOLOv8-Pose):
5 = ombro esquerdo, 6 = ombro direito, 11 = quadril esquerdo, 12 = quadril direito.
"""

from __future__ import annotations

import numpy as np

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
_EPSILON = 1e-6


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Centraliza no meio do quadril e normaliza pela escala do tronco.

    landmarks: array (N, 3) em coordenadas (x, y, z).
    """
    hip_center = (landmarks[LEFT_HIP] + landmarks[RIGHT_HIP]) / 2.0
    shoulder_center = (landmarks[LEFT_SHOULDER] + landmarks[RIGHT_SHOULDER]) / 2.0

    torso_size = float(np.linalg.norm(shoulder_center - hip_center))
    scale = max(torso_size, _EPSILON)

    return (landmarks - hip_center) / scale
