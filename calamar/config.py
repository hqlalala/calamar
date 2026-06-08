"""Agent engine configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    name: str = ""
    provider: str = "openai"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.0
    context_window: int = 200_000


@dataclass
class RoutingRule:
    match: dict[str, Any] = field(default_factory=dict)
    model: str = ""


@dataclass
class RoutingConfig:
    enabled: bool = False
    rules: list[RoutingRule] = field(default_factory=list)
    default_model: str = ""


@dataclass
class BudgetConfig:
    daily_limit_usd: float = 0.0
    session_limit_usd: float = 0.0
    warn_threshold: float = 0.8


@dataclass
class PermissionMode:
    mode: str = "normal"  # strict / normal / auto


@dataclass
class CompactionConfig:
    threshold: float = 0.85
    max_levels: int = 4


@dataclass
class Config:
    model: str = "claude-sonnet-4-6-20250514"
    provider: str = "openai"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.0
    context_window: int = 200_000
    max_turns: int = 100

    permission: PermissionMode = field(default_factory=PermissionMode)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)

    system_prompt: str = ""
    project_root: str = ""

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            name=self.model,
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            context_window=self.context_window,
        )
