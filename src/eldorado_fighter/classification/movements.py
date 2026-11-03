"""Registro de movimentos (Registry pattern), construído a partir de
config/movements.yaml.

Esta é a peça central da extensibilidade do projeto: adicionar um novo
movimento no futuro (ex.: "defesa", "soco especial") é só editar o YAML.
Nenhum outro módulo referencia nomes de movimento como strings soltas —
todos usam `Movement`, obtido via `MovementRegistry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eldorado_fighter.utils.config import load_movements


@dataclass(frozen=True)
class Movement:
    name: str
    category: str
    hold: bool
    game_keys: tuple[str, ...]
    description: str


class MovementRegistry:
    def __init__(self, movements: list[Movement]) -> None:
        if not movements:
            raise ValueError("A lista de movimentos não pode ser vazia.")
        self._by_name = {m.name: m for m in movements}
        self._ordered = tuple(movements)

    @classmethod
    def from_config(cls, path: Path | None = None) -> "MovementRegistry":
        raw = load_movements(path)
        movements = [
            Movement(
                name=entry["name"],
                category=entry["category"],
                hold=bool(entry.get("hold", False)),
                game_keys=tuple(entry.get("game_keys", [])),
                description=entry.get("description", ""),
            )
            for entry in raw
        ]
        return cls(movements)

    def get(self, name: str) -> Movement:
        return self._by_name[name]

    def index_of(self, name: str) -> int:
        return self._ordered.index(self._by_name[name])

    def by_index(self, index: int) -> Movement:
        return self._ordered[index]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self._ordered)

    def __len__(self) -> int:
        return len(self._ordered)

    def __iter__(self):
        return iter(self._ordered)
