"""Git workflow — auto-commit, branch isolation, and safe rollback."""

from __future__ import annotations

import asyncio
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

AGENT_PREFIX = "[calamar]"
BRANCH_PREFIX = "calamar/"

SENSITIVE_PATTERNS = (
    ".env", ".env.local", ".env.production",
    "credentials.json", "secrets.json", "service-account.json",
    "*.pem", "*.key", "id_rsa", "id_ed25519",
)


@dataclass
class GitStatus:
    """Summary of the working tree state."""

    is_repo: bool = False
    branch: str = ""
    has_changes: bool = False
    staged: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)


@dataclass
class CommitInfo:
    """A single commit record."""

    sha: str
    message: str
    is_agent: bool = False


class GitWorkflow:
    """Git operations for agent-driven development."""

    def __init__(self, repo_dir: str | Path) -> None:
        self._repo_dir = str(Path(repo_dir).resolve())
        self._session_commits: list[str] = []

    @property
    def session_commits(self) -> list[str]:
        return list(self._session_commits)

    async def is_repo(self) -> bool:
        code, _ = await self._run("git", "rev-parse", "--is-inside-work-tree")
        return code == 0

    async def status(self) -> GitStatus:
        if not await self.is_repo():
            return GitStatus(is_repo=False)

        result = GitStatus(is_repo=True)

        code, branch = await self._run("git", "branch", "--show-current")
        if code == 0:
            result.branch = branch.strip()

        code, porcelain = await self._run("git", "status", "--porcelain")
        if code == 0 and porcelain.strip():
            result.has_changes = True
            for line in porcelain.rstrip("\n").split("\n"):
                if not line or len(line) < 4:
                    continue
                status = line[:2]
                path = line[3:].strip()
                if status[0] in ("A", "M", "R"):
                    result.staged.append(path)
                if status[1] == "M":
                    result.modified.append(path)
                if status == "??":
                    result.untracked.append(path)

        return result

    async def auto_commit(
        self, message: str, files: list[str] | None = None,
    ) -> CommitInfo | None:
        """Stage specified files and commit with the agent prefix.

        If files is provided, only those paths are staged.
        If files is None or empty, stages all changes (legacy fallback).
        Returns None if there are no changes to commit.
        """
        if not await self.is_repo():
            return None

        if files:
            for f in files:
                await self._run("git", "add", "--", f)
        else:
            st = await self.status()
            if not st.has_changes:
                return None
            await self._run("git", "add", "-A")

        await self._unstage_sensitive()

        # Check if anything is actually staged
        code, staged = await self._run("git", "diff", "--cached", "--name-only")
        if code != 0 or not staged.strip():
            return None

        full_msg = f"{AGENT_PREFIX} {message}"
        code, output = await self._run("git", "commit", "-m", full_msg)
        if code != 0:
            return None

        code, sha = await self._run("git", "rev-parse", "--short", "HEAD")
        if code != 0:
            return None

        sha = sha.strip()
        self._session_commits.append(sha)
        return CommitInfo(sha=sha, message=full_msg, is_agent=True)

    async def create_branch(self, name: str) -> str | None:
        """Create and switch to an agent branch.

        Returns the full branch name, or None on failure.
        """
        if not await self.is_repo():
            return None

        branch = f"{BRANCH_PREFIX}{name}"

        code, _ = await self._run("git", "checkout", "-b", branch)
        if code != 0:
            return None
        return branch

    async def switch_branch(self, branch: str) -> bool:
        code, _ = await self._run("git", "checkout", branch)
        return code == 0

    async def current_branch(self) -> str:
        code, branch = await self._run("git", "branch", "--show-current")
        if code == 0:
            return branch.strip()
        return ""

    async def undo_last(self) -> CommitInfo | None:
        """Revert the most recent agent commit."""
        if not self._session_commits:
            last = await self._find_last_agent_commit()
            if not last:
                return None
        else:
            last = self._session_commits[-1]

        code, msg = await self._run("git", "log", "-1", "--format=%s", last)
        if code != 0:
            return None

        code, _ = await self._run("git", "revert", "--no-edit", last)
        if code != 0:
            return None

        if last in self._session_commits:
            self._session_commits.remove(last)

        return CommitInfo(sha=last, message=msg.strip(), is_agent=True)

    async def undo_all(self) -> list[CommitInfo]:
        """Revert all agent commits from this session (newest first)."""
        reverted: list[CommitInfo] = []
        commits = list(reversed(self._session_commits))

        for sha in commits:
            code, msg = await self._run("git", "log", "-1", "--format=%s", sha)
            if code != 0:
                continue

            code, _ = await self._run("git", "revert", "--no-edit", sha)
            if code == 0:
                reverted.append(CommitInfo(sha=sha, message=msg.strip(), is_agent=True))

        self._session_commits.clear()
        return reverted

    async def diff(self) -> str:
        """Show unstaged + staged diff."""
        parts: list[str] = []

        code, staged = await self._run("git", "diff", "--cached")
        if code == 0 and staged.strip():
            parts.append(staged)

        code, unstaged = await self._run("git", "diff")
        if code == 0 and unstaged.strip():
            parts.append(unstaged)

        return "\n".join(parts)

    async def log(self, count: int = 10, agent_only: bool = False) -> list[CommitInfo]:
        """List recent commits."""
        args = ["git", "log", f"-{count}", "--format=%h|%s"]
        code, output = await self._run(*args)
        if code != 0:
            return []

        commits: list[CommitInfo] = []
        for line in output.strip().split("\n"):
            if not line or "|" not in line:
                continue
            sha, msg = line.split("|", 1)
            is_agent = msg.strip().startswith(AGENT_PREFIX)
            if agent_only and not is_agent:
                continue
            commits.append(CommitInfo(sha=sha.strip(), message=msg.strip(), is_agent=is_agent))
        return commits

    async def _unstage_sensitive(self) -> None:
        """Remove files matching SENSITIVE_PATTERNS from the staging area."""
        code, output = await self._run("git", "diff", "--cached", "--name-only")
        if code != 0 or not output.strip():
            return

        for staged_file in output.strip().split("\n"):
            name = Path(staged_file).name
            if any(fnmatch.fnmatch(name, pat) for pat in SENSITIVE_PATTERNS):
                await self._run("git", "reset", "HEAD", "--", staged_file)

    async def _find_last_agent_commit(self) -> str | None:
        commits = await self.log(count=20, agent_only=True)
        return commits[0].sha if commits else None

    async def _run(self, *args: str) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=self._repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode(errors="replace")
        if proc.returncode != 0:
            output = stderr.decode(errors="replace") + output
        return proc.returncode or 0, output
