"""Feature de velocidade (diferença quadro-a-quadro).

Soco e chute são ações instantâneas (hold: false em movements.yaml) dentro
de um clipe de 2s / janela de 30 frames — a maior parte da janela mostra a
pessoa parada antes/depois do golpe, então a posição estática por frame
sozinha não basta (o modelo aprende "majoritariamente parado" = idle).

Velocidade dá um sinal direto de "quanto está se movendo" a cada instante:
~0 em idle, moderada e sustentada em caminhar, um pico breve e alto em
soco/chute — justamente o que falta à representação puramente posicional.

É aplicada em cima da JANELA já montada (não por frame como normalize/tilt)
porque depende do frame anterior; por isso entra como `window_fn` em
dataset_loader.load_dataset e deve ser espelhada na inferência (ver
pipeline/realtime_pipeline.py) para treino e inferência baterem.
"""

from __future__ import annotations

import numpy as np


def append_velocity(window: np.ndarray) -> np.ndarray:
    """window: (seq_len, N, 3) já normalizado. Retorna (seq_len, 2N, 3) com
    a velocidade (diferença entre frames consecutivos) anexada como
    landmarks extras — o primeiro frame usa velocidade zero (sem frame anterior).
    """
    velocity = np.diff(window, axis=0, prepend=window[:1])
    return np.concatenate([window, velocity], axis=1).astype(np.float32)
