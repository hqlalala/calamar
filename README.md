# Calamar

A modular AI coding agent. Combines the best design patterns from Claude Code, OpenAI Codex, Cursor, Windsurf, Aider, and top SWE-Bench agents.

## Features

- **Streaming output** — typewriter-style response with markdown re-rendering
- **Multi-provider** — native Anthropic (with prompt caching), OpenAI, Ducky/Aone (with auto-detected native tool calling)
- **MCP tools** — connect external tool servers via Model Context Protocol
- **Built-in tools** — file read/write/edit, terminal, code search, directory listing, web fetch
- **Code understanding** — tree-sitter repo map for multi-language symbol extraction
- **Three-agent collaboration** — Planner / Executor / Verifier with generate-test-fix cycle
- **Intelligent model routing** — auto-selects the best model per task complexity
- **Permission system** — asks confirmation before dangerous operations (rm, git push, sudo)
- **Middleware pipeline** — permission → guardrail → cost → timing → output redaction
- **Context management** — progressive compaction with LLM-powered summarization, manual /compact
- **Git-native workflow** — auto-commit, undo, diff, branch management
- **Config files** — `~/.calamar/config.json` for global, `.calamar/config.json` per project
- **Interactive REPL** — history, model switching, paste support, retry, cost tracking, cache stats
- **Rich tool display** — colored diffs for edits, terminal output preview, thinking indicator

## Install

```bash
pip install calamar
```

With optional features:

```bash
pip install calamar[anthropic]     # Native Anthropic API
pip install calamar[mcp]           # MCP tool integration
pip install calamar[code-index]    # Tree-sitter code understanding
pip install calamar[dev]           # Development tools
```

## Quick Start

### CLI

```bash
# Interactive REPL (auto-detects provider from API key)
export ANTHROPIC_API_KEY="sk-ant-..."
calamar

# One-shot mode
calamar "Fix the bug in auth.py"

# Specify model and provider
calamar -m gpt-4o -p openai "Explain this code"
```

### Python API

```python
from calamar import AgentLoop, Config

config = Config(model="claude-sonnet-4-6-20250514", api_key="...")
agent = AgentLoop(config)

async for event in agent.run("Fix the bug in auth.py"):
    print(event)
```

## Configuration

Create `~/.calamar/config.json` for global defaults:

```json
{
  "model": "claude-sonnet-4-6-20250514",
  "provider": "anthropic",
  "api_key": "sk-ant-...",
  "max_tokens": 8192,
  "temperature": 0.0
}
```

Project-level `.calamar/config.json` overrides global settings. CLI args and environment variables override both.

### MCP Tools

Create `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    }
  }
}
```

### Project Instructions

Create `AGENT.md`, `.calamar.md`, or `CLAUDE.md` in your project root with project-specific instructions for the agent.

## Architecture

```
User Input
    ↓
AgentLoop (single-threaded turn loop)
    ↓
ContextBuilder (static prefix + dynamic suffix, maximize cache hits)
    ↓
ModelRouter (task complexity → optimal model)
    ↓
Provider.stream() → StreamDelta (with retry + exponential backoff)
    ↓
MiddlewarePipeline (guardrail → cost → timing → output redaction)
    ↓
ToolRegistry (built-in + MCP tools)
    ↓
ContextCompactor (truncation → LLM summarization → emergency trim)
```

## REPL Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/clear` | Clear conversation history |
| `/retry` | Retry last message |
| `/compact` | Compact context to free tokens |
| `/context` | Show context usage |
| `/model` | Show / switch / list models |
| `/cost` | Show session cost |
| `/config` | Show current configuration |
| `/verbose` | Toggle verbose mode |
| `/undo` | Undo last agent commit |
| `/diff` | Show uncommitted changes |
| `/branch` | Show current branch |
| `/init` | Initialize project config |

## License

MIT
