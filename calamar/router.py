"""Intelligent model routing — select optimal model per task complexity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from calamar.config import Config, RoutingRule


@dataclass
class TaskProfile:
    complexity: str = "medium"  # simple / medium / complex
    task_type: str = "general"  # qa / code_edit / code_review / bug_fix / general


class ModelRouter:
    def __init__(self, config: Config) -> None:
        self._config = config

    def classify(self, user_message: str, has_code_context: bool = False) -> TaskProfile:
        msg_len = len(user_message)

        if msg_len < 100 and not has_code_context:
            return TaskProfile(complexity="simple", task_type="qa")

        keywords_complex = [
            "refactor", "implement", "build", "create", "design",
            "migrate", "重构", "实现", "构建", "设计", "迁移",
        ]
        keywords_fix = ["fix", "bug", "error", "broken", "修复", "报错", "异常"]
        keywords_review = ["review", "check", "审查", "检查"]

        lower = user_message.lower()
        if any(k in lower for k in keywords_complex):
            return TaskProfile(complexity="complex", task_type="code_edit")
        if any(k in lower for k in keywords_fix):
            return TaskProfile(complexity="complex", task_type="bug_fix")
        if any(k in lower for k in keywords_review):
            return TaskProfile(complexity="medium", task_type="code_review")
        if has_code_context:
            return TaskProfile(complexity="medium", task_type="code_edit")

        return TaskProfile(complexity="medium", task_type="general")

    def select_model(self, profile: TaskProfile) -> str:
        if not self._config.routing.enabled:
            return self._config.model

        for rule in self._config.routing.rules:
            if self._matches(rule, profile):
                return rule.model

        return self._config.routing.default_model or self._config.model

    def route(self, user_message: str, has_code_context: bool = False) -> str:
        profile = self.classify(user_message, has_code_context)
        return self.select_model(profile)

    @staticmethod
    def _matches(rule: RoutingRule, profile: TaskProfile) -> bool:
        for key, value in rule.match.items():
            actual = getattr(profile, key, None)
            if isinstance(value, list):
                if actual not in value:
                    return False
            elif actual != value:
                return False
        return True
