"""Carregamento de configuração a partir dos arquivos YAML em /config.

Centralizar o acesso à config aqui permite que todos os componentes do
pipeline sejam construídos por fábricas (factories) que leem seções deste
dicionário, em vez de terem valores fixos no código (facilita trocar
implementações — padrão Strategy — e testar com configs alternativas).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Carrega config/settings.yaml (ou um caminho alternativo)."""
    return load_yaml(path or CONFIG_DIR / "settings.yaml")


def load_axis(axis: str, path: Path | None = None) -> list[dict[str, Any]]:
    """Carrega a lista de classes de um eixo ("movement" ou "action")
    definida em config/movements.yaml.
    """
    data = load_yaml(path or CONFIG_DIR / "movements.yaml")
    axes = data.get("axes", {})
    if axis not in axes:
        raise KeyError(f"Eixo '{axis}' não encontrado em movements.yaml. Opções: {list(axes)}")
    return axes[axis]
