"""Traduz um Movement classificado em eventos de input (Observer pattern:
assina eventos "movement_detected" publicados pelo pipeline).

Duas estratégias de disparo, conforme Movement.hold:
  - hold=True  (direção/postura): mantém a tecla pressionada enquanto o
    mesmo movimento continuar sendo detectado; solta ao mudar.
  - hold=False (ataques): dá um "tap" por detecção, respeitando um
    cooldown para não disparar múltiplos golpes a partir de frames
    consecutivos do mesmo soco/chute.
"""

from __future__ import annotations

import time

from eldorado_fighter.actions.base import InputBackend
from eldorado_fighter.classification.base import PredictionResult
from eldorado_fighter.classification.movements import Movement


class ActionMapper:
    def __init__(self, input_backend: InputBackend, cooldown_seconds: float = 0.08) -> None:
        self._backend = input_backend
        self._cooldown_seconds = cooldown_seconds
        self._held_movement: Movement | None = None
        self._last_instant_fire: dict[str, float] = {}

    def on_movement_detected(self, prediction: PredictionResult) -> None:
        movement = prediction.movement
        if movement.hold:
            self._handle_hold_movement(movement)
        else:
            self._release_held_movement()
            self._handle_instant_movement(movement)

    def _handle_hold_movement(self, movement: Movement) -> None:
        if self._held_movement is not None and self._held_movement.name == movement.name:
            return
        self._release_held_movement()
        for key in movement.game_keys:
            self._backend.press(key)
        self._held_movement = movement

    def _release_held_movement(self) -> None:
        if self._held_movement is None:
            return
        for key in self._held_movement.game_keys:
            self._backend.release(key)
        self._held_movement = None

    def _handle_instant_movement(self, movement: Movement) -> None:
        if not movement.game_keys:
            return
        now = time.monotonic()
        last_fire = self._last_instant_fire.get(movement.name, 0.0)
        if now - last_fire < self._cooldown_seconds:
            return
        self._last_instant_fire[movement.name] = now
        for key in movement.game_keys:
            self._backend.tap(key)
