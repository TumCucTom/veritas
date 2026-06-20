"""Load and validate signals.yaml into typed config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Theme(BaseModel):
    weight: float = 1.0
    description: str = ""
    phrases: list[str] = Field(default_factory=list)


class Problem(BaseModel):
    name: str
    statement: str = ""


class Config(BaseModel):
    problem: Problem
    themes: dict[str, Theme]
    core_terms: list[str] = Field(default_factory=list)
    min_score: float = 3.0
    queries: list[str] = Field(default_factory=list)
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def source_cfg(self, name: str) -> dict[str, Any]:
        return self.sources.get(name, {}) or {}

    def source_enabled(self, name: str) -> bool:
        cfg = self.source_cfg(name)
        return bool(cfg.get("enabled", False))


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "signals.yaml"


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"signals config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)
