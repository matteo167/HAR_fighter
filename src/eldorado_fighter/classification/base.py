"""Interface (Strategy) para classificadores de movimento.

Permite trocar a implementação (ex.: TensorFlow hoje, outro framework ou um
classificador heurístico de fallback no futuro) sem alterar o pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from eldorado_fighter.classification.movements import Movement


@dataclass(frozen=True)
class PredictionResult:
    movement: Movement
    confidence: float


class MovementClassifier(ABC):
    @abstractmethod
    def predict(self, sequence: np.ndarray) -> PredictionResult:
        """Classifica uma sequência (sequence_length, num_landmarks, 3)."""
