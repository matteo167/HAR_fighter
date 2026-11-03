"""Carrega o dataset de keypoints (data/keypoints/<label>/*.npz) e gera
janelas de sequência idênticas às usadas em tempo real (SequenceBuffer),
para que a distribuição de treino bata com a de inferência.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from eldorado_fighter.classification.movements import MovementRegistry
from eldorado_fighter.dataset.schema import load_sample
from eldorado_fighter.features.normalizer import normalize_landmarks

FeatureFn = Callable[[np.ndarray], np.ndarray]
WindowFn = Callable[[np.ndarray], np.ndarray]


def _windows_from_clip(
    keypoints: np.ndarray,
    sequence_length: int,
    stride: int,
    feature_fn: FeatureFn,
    window_fn: WindowFn | None = None,
) -> np.ndarray:
    """Extrai janelas (sequence_length, N, 3) de um clipe (T, N, 3).

    `window_fn`, se dado, roda em cima de cada janela já montada (não por
    frame) — usado por features que precisam do frame anterior, como
    velocidade (ver features.motion.append_velocity).
    """
    normalized = np.stack([feature_fn(frame) for frame in keypoints], axis=0)

    num_frames = normalized.shape[0]
    if num_frames < sequence_length:
        pad = np.repeat(normalized[-1:], sequence_length - num_frames, axis=0)
        windows = [np.concatenate([normalized, pad], axis=0)]
    else:
        windows = [
            normalized[start : start + sequence_length]
            for start in range(0, num_frames - sequence_length + 1, stride)
        ]

    if window_fn is not None:
        windows = [window_fn(w) for w in windows]

    return np.stack(windows, axis=0)


def load_dataset(
    keypoints_dir: Path,
    registry: MovementRegistry,
    sequence_length: int,
    stride: int = 1,
    feature_fn: FeatureFn = normalize_landmarks,
    window_fn: WindowFn | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (X, y, clip_ids): X com shape (N, sequence_length, num_landmarks, 3),
    y com shape (N,) contendo o índice do movimento em `registry`, clip_ids
    com shape (N,) identificando de qual clipe cada janela veio.

    `clip_ids` existe para permitir split treino/validação por CLIPE em vez
    de por janela: janelas do mesmo clipe se sobrepõem bastante (sliding
    window), então um split aleatório por janela deixa treino e validação
    quase idênticos — acurácia de validação artificialmente inflada/instável.

    `feature_fn` transforma cada frame de landmarks brutos (N, 3) antes da
    janela deslizante — por padrão só normaliza (ver features.normalizer),
    mas pode ser trocado por algo que também anexe features derivadas (ver
    features.tilt.build_feature_fn), desde que a mesma função seja usada na
    inferência (ver pipeline/realtime_pipeline.py).
    """
    all_windows: list[np.ndarray] = []
    all_labels: list[int] = []
    all_clip_ids: list[int] = []

    clip_id = 0
    for movement in registry:
        label_dir = Path(keypoints_dir) / movement.name
        if not label_dir.exists():
            continue

        label_index = registry.index_of(movement.name)
        for sample_path in sorted(label_dir.glob("*.npz")):
            sample = load_sample(sample_path)
            windows = _windows_from_clip(sample.keypoints, sequence_length, stride, feature_fn, window_fn)
            all_windows.append(windows)
            all_labels.extend([label_index] * len(windows))
            all_clip_ids.extend([clip_id] * len(windows))
            clip_id += 1

    if not all_windows:
        raise RuntimeError(
            f"Nenhuma amostra encontrada em '{keypoints_dir}'. "
            "Grave o dataset com scripts/record_dataset.py antes de treinar."
        )

    X = np.concatenate(all_windows, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)
    clip_ids = np.array(all_clip_ids, dtype=np.int64)
    return X, y, clip_ids
