"""Buffer de janela deslizante (sliding window) de keypoints normalizados.

Movimentos como soco, chute e caminhada são temporais: um único frame não
basta para distinguir "parado" de "no meio de um chute". O classificador
sempre recebe uma sequência de `sequence_length` frames.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SequenceBuffer:
    def __init__(self, sequence_length: int, num_landmarks: int = 17) -> None:
        self.sequence_length = sequence_length
        self.num_landmarks = num_landmarks
        self._frames: deque[np.ndarray] = deque(maxlen=sequence_length)

    def push(self, normalized_landmarks: np.ndarray) -> None:
        self._frames.append(normalized_landmarks)

    def is_ready(self) -> bool:
        return len(self._frames) == self.sequence_length

    def as_array(self) -> np.ndarray:
        """Retorna a sequência como array (sequence_length, num_landmarks, 3)."""
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer incompleto: {len(self._frames)}/{self.sequence_length} frames."
            )
        return np.stack(self._frames, axis=0)

    def clear(self) -> None:
        self._frames.clear()
