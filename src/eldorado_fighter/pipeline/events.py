"""EventBus simples (Observer pattern).

Desacopla a detecção de movimento (pipeline) de quem reage a ela
(ActionMapper hoje; logger, overlay de debug ou métricas no futuro) —
múltiplos assinantes podem reagir ao mesmo evento sem que o pipeline saiba
que eles existem.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[Any], None]) -> None:
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, payload: Any) -> None:
        for handler in self._subscribers.get(event_type, []):
            handler(payload)
