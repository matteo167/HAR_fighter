"""Implementação de MovementClassifier usando um modelo Keras/TensorFlow
treinado a partir do dataset de keypoints (ver har_fighter.training).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from har_fighter.classification.base import MovementClassifier, PredictionResult
from har_fighter.classification.movements import MovementRegistry

# Import exigido para que camadas customizadas (ex.: _MLPMixerLayer) se
# registrem via @keras.saving.register_keras_serializable antes do
# load_model abaixo — sem isso, carregar um modelo treinado com essas
# camadas falha com "Could not locate class".
import har_fighter.training.model_factory  # noqa: F401


class TensorFlowMovementClassifier(MovementClassifier):
    def __init__(self, model_path: Path, registry: MovementRegistry) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em '{model_path}'. "
                "Grave o dataset (scripts/record_dataset.py) e treine um "
                "modelo (scripts/train_model.py) antes de usar este backend."
            )
        self._model = tf.keras.models.load_model(model_path)
        self._registry = registry

    def predict(self, sequence: np.ndarray) -> PredictionResult:
        batch = np.expand_dims(sequence, axis=0)
        probabilities = self._model.predict(batch, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        return PredictionResult(
            movement=self._registry.by_index(best_index),
            confidence=float(probabilities[best_index]),
        )
