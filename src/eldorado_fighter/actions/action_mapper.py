"""Traduz um Movement classificado em eventos de input (Observer pattern:
assina eventos "movement_detected"/"action_detected" publicados pelo
pipeline, um por eixo).

Duas estratégias de disparo, conforme Movement.hold:
  - hold=True  (direção/postura): mantém a tecla pressionada enquanto o
    mesmo movimento continuar sendo detectado; solta ao mudar.
  - hold=False (ataques): dá um "tap" por detecção, respeitando um
    cooldown para não disparar múltiplos golpes a partir de frames
    consecutivos do mesmo soco/chute.

Estado de tecla segurada e cooldown de tap são indexados por EIXO
("movement"/"action"): os dois eixos rodam em paralelo, então pode haver
duas teclas seguradas ao mesmo tempo (ex.: "left" do eixo movimento + "down"
do eixo action) — um único estado global apagaria uma delas. Indexar por
eixo também evita colisão entre a classe "idle" do eixo movimento e a
classe "idle" do eixo action, que têm o mesmo nome mas são independentes.
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
        self._held_by_axis: dict[str, Movement | None] = {}
        self._last_instant_fire: dict[tuple[str, str], float] = {}

    def on_label_detected(self, axis: str, prediction: PredictionResult) -> None:
        movement = prediction.movement
        if movement.hold:
            self._handle_hold_movement(axis, movement)
        else:
            self._release_held_movement(axis)
            self._handle_instant_movement(axis, movement)

    def _handle_hold_movement(self, axis: str, movement: Movement) -> None:
        held = self._held_by_axis.get(axis)
        if held is not None and held.name == movement.name:
            return
        self._release_held_movement(axis)
        for key in movement.game_keys:
            self._backend.press(key)
        self._held_by_axis[axis] = movement

    def _release_held_movement(self, axis: str) -> None:
        held = self._held_by_axis.get(axis)
        if held is None:
            return
        for key in held.game_keys:
            self._backend.release(key)
        self._held_by_axis[axis] = None

    def _handle_instant_movement(self, axis: str, movement: Movement) -> None:
        if not movement.game_keys:
            return
        now = time.monotonic()
        fire_key = (axis, movement.name)
        last_fire = self._last_instant_fire.get(fire_key, 0.0)
        if now - last_fire < self._cooldown_seconds:
            return
        self._last_instant_fire[fire_key] = now
        for key in movement.game_keys:
            self._backend.tap(key)
