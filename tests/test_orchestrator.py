"""Tests for the multi-agent orchestrator."""

from __future__ import annotations

import pytest

from calamar.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    TaskComplexity,
    classify_complexity,
)


class TestComplexityClassifier:
    def test_simple_greeting(self):
        assert classify_complexity("hello") == TaskComplexity.SIMPLE

    def test_simple_question(self):
        assert classify_complexity("what does this function do") == TaskComplexity.SIMPLE

    def test_moderate_fix(self):
        assert classify_complexity("fix the login bug") == TaskComplexity.MODERATE

    def test_moderate_add(self):
        assert classify_complexity("add a new endpoint") == TaskComplexity.MODERATE

    def test_moderate_implement(self):
        assert classify_complexity("implement caching") == TaskComplexity.MODERATE

    def test_moderate_chinese_fix(self):
        assert classify_complexity("修复认证问题") == TaskComplexity.MODERATE

    def test_moderate_chinese_add(self):
        assert classify_complexity("添加新功能") == TaskComplexity.MODERATE

    def test_complex_refactor(self):
        assert classify_complexity("refactor the auth module") == TaskComplexity.COMPLEX

    def test_complex_migrate(self):
        assert classify_complexity("migrate from REST to gRPC") == TaskComplexity.COMPLEX

    def test_complex_rewrite(self):
        assert classify_complexity("rewrite the parser") == TaskComplexity.COMPLEX

    def test_complex_chinese_refactor(self):
        assert classify_complexity("重构认证模块") == TaskComplexity.COMPLEX

    def test_complex_chinese_migrate(self):
        assert classify_complexity("迁移数据库") == TaskComplexity.COMPLEX

    def test_complex_many_files(self):
        assert (
            classify_complexity("update foo.py, bar.py, baz.py")
            == TaskComplexity.COMPLEX
        )

    def test_complex_architecture(self):
        assert classify_complexity("redesign the architecture") == TaskComplexity.COMPLEX

    def test_case_insensitive(self):
        assert classify_complexity("REFACTOR everything") == TaskComplexity.COMPLEX
        assert classify_complexity("Fix the Bug") == TaskComplexity.MODERATE


class TestIsPass:
    def test_pass_signals(self):
        assert Orchestrator._is_pass("All tests pass. Looks good.")
        assert Orchestrator._is_pass("Verified. No issues found.")
        assert Orchestrator._is_pass("LGTM, working correctly.")
        assert Orchestrator._is_pass("通过，没有问题。")

    def test_fail_signals(self):
        assert not Orchestrator._is_pass("Test failed. There is a bug.")
        assert not Orchestrator._is_pass("Error in the implementation.")
        assert not Orchestrator._is_pass("Missing edge case handling.")
        assert not Orchestrator._is_pass("有错误，需要修复。")

    def test_mixed_signals_favor_pass(self):
        assert Orchestrator._is_pass(
            "There was an error initially but it was fixed. All tests pass now."
        )

    def test_mixed_signals_favor_fail(self):
        assert not Orchestrator._is_pass(
            "Some tests pass but there are two errors and a missing feature."
        )

    def test_empty_text(self):
        assert Orchestrator._is_pass("")


class TestOrchestratorConfig:
    def test_default_config(self):
        cfg = OrchestratorConfig()
        assert cfg.max_retries == 2
        assert cfg.auto_upgrade is True
        assert cfg.force_plan is False

    def test_force_plan(self):
        cfg = OrchestratorConfig(force_plan=True)
        assert cfg.force_plan is True

    def test_no_auto_upgrade(self):
        cfg = OrchestratorConfig(auto_upgrade=False)
        assert cfg.auto_upgrade is False
