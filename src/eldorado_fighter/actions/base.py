"""Interface (Strategy) para backends de input.

O jogo não é modificado: o controle acontece simulando eventos de teclado
(ou, no futuro, um gamepad virtual) no nível do sistema operacional. Trocar
KeyboardInputBackend por um GamepadInputBackend não exige mudanças no
ActionMapper nem no pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class InputBackend(ABC):
    @abstractmethod
    def press(self, key: str) -> None:
        """Mantém uma tecla pressionada."""

    @abstractmethod
    def release(self, key: str) -> None:
        """Solta uma tecla previamente pressionada."""

    @abstractmethod
    def tap(self, key: str, duration: float = 0.05) -> None:
        """Pressiona e solta uma tecla rapidamente (ação instantânea)."""
