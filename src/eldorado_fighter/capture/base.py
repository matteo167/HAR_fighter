"""Interface (Strategy) para qualquer fonte de frames de vídeo.

Hoje a única implementação é a webcam (WebcamSource), mas a interface já
permite plugar uma fonte de arquivo de vídeo (para reprocessar gravações)
sem alterar o restante do pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def start(self) -> None:
        """Abre a fonte (ex.: inicia a captura da câmera)."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Retorna o próximo frame (BGR) ou None se não houver mais frames."""

    @abstractmethod
    def stop(self) -> None:
        """Libera os recursos da fonte."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Largura do frame, em pixels (necessário para gravar vídeo)."""

    @property
    @abstractmethod
    def height(self) -> int:
        """Altura do frame, em pixels (necessário para gravar vídeo)."""

    def __enter__(self) -> "FrameSource":
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()
