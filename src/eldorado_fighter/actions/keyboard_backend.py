"""Implementação de InputBackend que emula teclado via pynput.

No macOS, requer que o terminal/processo Python tenha permissão de
"Acessibilidade" em Preferências do Sistema > Privacidade e Segurança,
caso contrário os eventos de teclado são ignorados pelo sistema.
"""

from __future__ import annotations

import time

from pynput.keyboard import Controller, Key, KeyCode

from eldorado_fighter.actions.base import InputBackend

_SPECIAL_KEYS = {
    "up": Key.up,
    "down": Key.down,
    "left": Key.left,
    "right": Key.right,
}


def _to_pynput_key(key: str) -> Key | KeyCode:
    return _SPECIAL_KEYS.get(key, KeyCode.from_char(key))


class KeyboardInputBackend(InputBackend):
    def __init__(self) -> None:
        self._controller = Controller()

    def press(self, key: str) -> None:
        self._controller.press(_to_pynput_key(key))

    def release(self, key: str) -> None:
        self._controller.release(_to_pynput_key(key))

    def tap(self, key: str, duration: float = 0.05) -> None:
        pynput_key = _to_pynput_key(key)
        self._controller.press(pynput_key)
        time.sleep(duration)
        self._controller.release(pynput_key)
