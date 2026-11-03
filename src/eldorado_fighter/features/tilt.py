"""Features explícitas de inclinação de tronco e cabeça.

Andar para a esquerda/direita costuma vir acompanhado de uma leve
inclinação do tronco e da cabeça na direção do movimento. Em vez de confiar
no modelo para descobrir esse sinal sozinho entre os ~51 valores de posição
dos 17 keypoints, ele é calculado explicitamente aqui e anexado como dois
pseudo-landmarks extras — assim o restante do pipeline (SequenceBuffer,
Reshape em model_factory) não precisa saber que são features derivadas, só
vê landmarks com (N+2, 3) em vez de (N, 3).

Espera landmarks já normalizados (ver features.normalizer.normalize_landmarks),
onde o centro do quadril está na origem — por isso a posição do tronco em x
já representa diretamente a inclinação lateral.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from eldorado_fighter.features.normalizer import normalize_landmarks

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12

NUM_TILT_LANDMARKS = 2


def append_tilt_landmarks(normalized_landmarks: np.ndarray) -> np.ndarray:
    """Anexa 2 pseudo-landmarks (x, y, 0) com as inclinações calculadas:

    pseudo-landmark 1: (inclinação do tronco, inclinação da cabeça, 0)
    pseudo-landmark 2: (rotação dos ombros, rotação do quadril, 0)
    """
    shoulder_center = (normalized_landmarks[LEFT_SHOULDER] + normalized_landmarks[RIGHT_SHOULDER]) / 2.0

    torso_lean_x = shoulder_center[0]
    head_lean_x = normalized_landmarks[NOSE][0] - shoulder_center[0]
    shoulder_roll = normalized_landmarks[RIGHT_SHOULDER][1] - normalized_landmarks[LEFT_SHOULDER][1]
    hip_roll = normalized_landmarks[RIGHT_HIP][1] - normalized_landmarks[LEFT_HIP][1]

    tilt_rows = np.array(
        [
            [torso_lean_x, head_lean_x, 0.0],
            [shoulder_roll, hip_roll, 0.0],
        ],
        dtype=np.float32,
    )
    return np.concatenate([normalized_landmarks, tilt_rows], axis=0)


_FEATURE_MODES = ("full", "full_plus_tilt", "tilt_only")


def build_feature_fn(mode: str = "full") -> Callable[[np.ndarray], np.ndarray]:
    """Monta a função de features usada tanto no treino (dataset_loader)
    quanto na inferência (realtime_pipeline) — devem sempre bater.

    mode:
      "full"            — só normaliza (17 landmarks completos).
      "full_plus_tilt"  — normaliza e anexa os 2 pseudo-landmarks de inclinação.
      "tilt_only"       — descarta o grid completo, usa só os 2 pseudo-landmarks
                          (4 escalares úteis). Modelo bem menor; aposta que a
                          inclinação sozinha já basta para o movimento em questão.
    """
    if mode == "full":
        return normalize_landmarks
    if mode == "full_plus_tilt":
        return lambda landmarks: append_tilt_landmarks(normalize_landmarks(landmarks))
    if mode == "tilt_only":
        return lambda landmarks: append_tilt_landmarks(normalize_landmarks(landmarks))[-NUM_TILT_LANDMARKS:]

    raise ValueError(f"feature_mode '{mode}' desconhecido. Opções: {_FEATURE_MODES}")
