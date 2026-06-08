"""Tests for git workflow — auto-commit, branch, undo."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from calamar.git.workflow import AGENT_PREFIX, BRANCH_PREFIX, GitWorkflow


async def _init_repo(path: str) -> None:
    """Create a git repo with an initial commit."""
    for args in [
        ["git", "init"],
        ["git", "config", "user.email", "test@test.com"],
        ["git", "config", "user.name", "Test User"],
    ]:
        proc = await asyncio.create_subprocess_exec(*args, cwd=path)
        await proc.wait()

    readme = os.path.join(path, "README.md")
    with open(readme, "w") as f:
        f.write("# Test\n")

    for args in [
        ["git", "add", "."],
        ["git", "commit", "-m", "initial commit"],
    ]:
        proc = await asyncio.create_subprocess_exec(*args, cwd=path)
        await proc.wait()


@pytest.fixture
def git_repo():
    with tempfile.TemporaryDirectory() as d:
        asyncio.run(_init_repo(d))
        yield d


@pytest.fixture
def non_repo():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestIsRepo:
    def test_git_repo(self, git_repo):
        gw = GitWorkflow(git_repo)
        assert asyncio.run(gw.is_repo())

    def test_non_repo(self, non_repo):
        gw = GitWorkflow(non_repo)
        assert not asyncio.run(gw.is_repo())


class TestStatus:
    def test_clean_repo(self, git_repo):
        gw = GitWorkflow(git_repo)
        st = asyncio.run(gw.status())
        assert st.is_repo
        assert st.branch == "main" or st.branch == "master"
        assert not st.has_changes

    def test_modified_file(self, git_repo):
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("more\n")
        gw = GitWorkflow(git_repo)
        st = asyncio.run(gw.status())
        assert st.has_changes
        assert "README.md" in st.modified

    def test_untracked_file(self, git_repo):
        with open(os.path.join(git_repo, "new.txt"), "w") as f:
            f.write("hello\n")
        gw = GitWorkflow(git_repo)
        st = asyncio.run(gw.status())
        assert st.has_changes
        assert "new.txt" in st.untracked

    def test_non_repo_status(self, non_repo):
        gw = GitWorkflow(non_repo)
        st = asyncio.run(gw.status())
        assert not st.is_repo


class TestAutoCommit:
    def test_commits_changes(self, git_repo):
        with open(os.path.join(git_repo, "app.py"), "w") as f:
            f.write("def hello(): pass\n")
        gw = GitWorkflow(git_repo)
        info = asyncio.run(gw.auto_commit("add hello function"))
        assert info is not None
        assert info.is_agent
        assert AGENT_PREFIX in info.message

    def test_no_changes(self, git_repo):
        gw = GitWorkflow(git_repo)
        info = asyncio.run(gw.auto_commit("nothing to commit"))
        assert info is None

    def test_tracks_session(self, git_repo):
        with open(os.path.join(git_repo, "a.py"), "w") as f:
            f.write("x = 1\n")
        gw = GitWorkflow(git_repo)
        info = asyncio.run(gw.auto_commit("first"))
        assert len(gw.session_commits) == 1
        assert gw.session_commits[0] == info.sha

        with open(os.path.join(git_repo, "b.py"), "w") as f:
            f.write("y = 2\n")
        asyncio.run(gw.auto_commit("second"))
        assert len(gw.session_commits) == 2

    def test_non_repo(self, non_repo):
        gw = GitWorkflow(non_repo)
        info = asyncio.run(gw.auto_commit("test"))
        assert info is None


class TestBranch:
    def test_create_branch(self, git_repo):
        gw = GitWorkflow(git_repo)
        branch = asyncio.run(gw.create_branch("refactor-auth"))
        assert branch == f"{BRANCH_PREFIX}refactor-auth"
        current = asyncio.run(gw.current_branch())
        assert current == branch

    def test_switch_branch(self, git_repo):
        gw = GitWorkflow(git_repo)
        asyncio.run(gw.create_branch("feature-x"))
        original = "main"
        asyncio.run(gw.switch_branch(original))
        current = asyncio.run(gw.current_branch())
        assert current == original

    def test_non_repo_branch(self, non_repo):
        gw = GitWorkflow(non_repo)
        branch = asyncio.run(gw.create_branch("test"))
        assert branch is None


class TestUndo:
    def test_undo_last(self, git_repo):
        with open(os.path.join(git_repo, "app.py"), "w") as f:
            f.write("x = 1\n")
        gw = GitWorkflow(git_repo)
        asyncio.run(gw.auto_commit("add app"))

        reverted = asyncio.run(gw.undo_last())
        assert reverted is not None
        assert reverted.is_agent
        assert not os.path.exists(os.path.join(git_repo, "app.py"))
        assert len(gw.session_commits) == 0

    def test_undo_all(self, git_repo):
        gw = GitWorkflow(git_repo)

        with open(os.path.join(git_repo, "a.py"), "w") as f:
            f.write("a = 1\n")
        asyncio.run(gw.auto_commit("add a"))

        with open(os.path.join(git_repo, "b.py"), "w") as f:
            f.write("b = 2\n")
        asyncio.run(gw.auto_commit("add b"))

        assert len(gw.session_commits) == 2

        reverted = asyncio.run(gw.undo_all())
        assert len(reverted) == 2
        assert not os.path.exists(os.path.join(git_repo, "a.py"))
        assert not os.path.exists(os.path.join(git_repo, "b.py"))
        assert len(gw.session_commits) == 0

    def test_undo_nothing(self, git_repo):
        gw = GitWorkflow(git_repo)
        reverted = asyncio.run(gw.undo_last())
        assert reverted is None


class TestDiff:
    def test_shows_diff(self, git_repo):
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("new line\n")
        gw = GitWorkflow(git_repo)
        diff = asyncio.run(gw.diff())
        assert "new line" in diff

    def test_no_diff(self, git_repo):
        gw = GitWorkflow(git_repo)
        diff = asyncio.run(gw.diff())
        assert diff == ""


class TestLog:
    def test_lists_commits(self, git_repo):
        gw = GitWorkflow(git_repo)
        commits = asyncio.run(gw.log())
        assert len(commits) >= 1
        assert commits[0].message == "initial commit"

    def test_agent_filter(self, git_repo):
        with open(os.path.join(git_repo, "app.py"), "w") as f:
            f.write("x = 1\n")
        gw = GitWorkflow(git_repo)
        asyncio.run(gw.auto_commit("add app"))

        all_commits = asyncio.run(gw.log())
        agent_commits = asyncio.run(gw.log(agent_only=True))

        assert len(all_commits) >= 2
        assert len(agent_commits) == 1
        assert agent_commits[0].is_agent

    def test_empty_repo_log(self, non_repo):
        gw = GitWorkflow(non_repo)
        commits = asyncio.run(gw.log())
        assert commits == []
