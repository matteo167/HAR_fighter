"""Fábrica de arquiteturas de modelo (Factory + Strategy).

Trocar a arquitetura usada no treino é uma mudança de config
(training.architecture), não de código — útil para comparar "lstm" vs um
baseline "dense" sem duplicar o script de treino.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from keras_nlp.layers import FNetEncoder, SinePositionEncoding, TransformerEncoder
from tensorflow.keras import layers


@keras.saving.register_keras_serializable(package="eldorado_fighter")
class _MLPMixerLayer(layers.Layer):
    """Mistura tokens (frames) e canais (features) via MLPs, sem atenção
    nem FFT — ver rascunho/arquiteturas_exemplo. Substitui tfa.layers.GELU()
    (tensorflow_addons está descontinuado) pela ativação "gelu" nativa do
    Keras 3.

    Precisa de @register_keras_serializable + get_config: sem isso o Keras
    salva o modelo sem erro, mas não consegue recarregar (tf.keras.models.
    load_model) porque não sabe reconstruir essa camada customizada — só
    aparece na hora de usar o modelo treinado (ex.: scripts/run_live.py).
    """

    def __init__(self, num_patches: int, intermediate_dim: int, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_patches = num_patches
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate

        self.mlp1 = tf.keras.Sequential(
            [
                layers.Dense(units=num_patches),
                layers.Activation("gelu"),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = tf.keras.Sequential(
            [
                layers.Dense(units=num_patches),
                layers.Activation("gelu"),
                layers.Dense(units=intermediate_dim),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.normalize(inputs)
        x_channels = tf.linalg.matrix_transpose(x)
        mlp1_outputs = self.mlp1(x_channels)
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        x = mlp1_outputs + inputs

        x_patches = self.normalize(x)
        mlp2_outputs = self.mlp2(x_patches)
        return x + mlp2_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "intermediate_dim": self.intermediate_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


def _build_lstm(seq_len: int, num_landmarks: int, num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(seq_len, num_landmarks, 3))
    x = layers.Reshape((seq_len, num_landmarks * 3))(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="eldorado_fighter_lstm")


def _build_dense_baseline(seq_len: int, num_landmarks: int, num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(seq_len, num_landmarks, 3))
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="eldorado_fighter_dense_baseline")


def _build_fnet(seq_len: int, num_landmarks: int, num_classes: int) -> tf.keras.Model:
    """Encoder FNet (ver rascunho/arquiteturas_exemplo): mistura tokens via
    FFT em vez de atenção, mais leve que um Transformer para sequências curtas.
    """
    inputs = layers.Input(shape=(seq_len, num_landmarks, 3))
    x = layers.Reshape((seq_len, num_landmarks * 3))(inputs)
    x = x + SinePositionEncoding()(x)
    x = FNetEncoder(intermediate_dim=64)(x)
    x = layers.Permute((2, 1))(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(seq_len // 2, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="eldorado_fighter_fnet")


def _build_mlp_mixer(seq_len: int, num_landmarks: int, num_classes: int) -> tf.keras.Model:
    """MLP-Mixer (ver rascunho/arquiteturas_exemplo): mistura frames e
    features só com MLPs — bem mais leve que Transformer/FNet, bom
    candidato quando o dataset é pequeno.

    Pooling final é average+max concatenados, não só average: um soco/chute
    é um pico breve dentro de uma janela majoritariamente parada (idle) —
    average sozinho dilui esse pico até ficar indistinguível de idle; max
    preserva o valor de pico, average preserva o padrão sustentado (bom pra
    caminhar). Ver features/motion.py para o motivo completo.
    """
    feature_dim = num_landmarks * 3
    inputs = layers.Input(shape=(seq_len, num_landmarks, 3))
    x = layers.Reshape((seq_len, feature_dim))(inputs)
    x = _MLPMixerLayer(num_patches=seq_len, intermediate_dim=feature_dim, dropout_rate=0.3)(x)
    x = layers.Permute((2, 1))(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(seq_len // 2, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="eldorado_fighter_mlp_mixer")


def _build_transformer(seq_len: int, num_landmarks: int, num_classes: int) -> tf.keras.Model:
    """TransformerEncoder (ver rascunho/arquiteturas_exemplo): atenção
    completa entre frames, mais expressivo porém com mais parâmetros —
    mais fácil de overfitar em datasets pequenos que fnet/mlp_mixer.
    """
    inputs = layers.Input(shape=(seq_len, num_landmarks, 3))
    x = layers.Reshape((seq_len, num_landmarks * 3))(inputs)
    x = x + SinePositionEncoding()(x)
    x = TransformerEncoder(intermediate_dim=64, num_heads=4, dropout=0.2)(x)
    x = layers.Permute((2, 1))(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(seq_len // 2, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="eldorado_fighter_transformer")


_ARCHITECTURES = {
    "lstm": _build_lstm,
    "dense": _build_dense_baseline,
    "fnet": _build_fnet,
    "mlp_mixer": _build_mlp_mixer,
    "transformer": _build_transformer,
}


def build_model(
    architecture: str, seq_len: int, num_landmarks: int, num_classes: int
) -> tf.keras.Model:
    try:
        builder = _ARCHITECTURES[architecture]
    except KeyError as exc:
        raise ValueError(
            f"Arquitetura '{architecture}' desconhecida. Opções: {list(_ARCHITECTURES)}"
        ) from exc

    model = builder(seq_len, num_landmarks, num_classes)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
