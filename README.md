# Calamar

A modular AI agent engine built for coding tasks. Combines the best design patterns from Claude Code, OpenAI Codex, Cursor, Windsurf, Aider, and top SWE-Bench agents.

## Core Design

- **Simple loop, complex periphery** — single-threaded agent loop with a layered middleware pipeline
- **Three-agent collaboration** — Planner / Executor / Verifier with generate-test-fix cycle
- **Intelligent model routing** — auto-selects the best model per task complexity
- **Middleware pipeline** — Guardrail → Permission → Cost → Trace → Cache
- **Code understanding** — tree-sitter repo map + vector index for precise code localization
- **Git-native workflow** — auto-commit, branch isolation, clean rollback
- **Protocol-first** — MCP for tools, A2A for agent collaboration

## Install

```bash
pip install calamar
```

For development:

```bash
git clone https://github.com/hqlalala/calamar.git
cd calamar
pip install -e ".[dev]"
```

## Quick Start

```python
from calamar import AgentLoop, Config

config = Config(model="claude-sonnet-4-6-20250514")
agent = AgentLoop(config)

async for event in agent.run("Fix the bug in auth.py"):
    print(event)
```

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
LLM Response
    ↓
MiddlewarePipeline (guardrail → permission → cost → execute → trace)
    ↓
ToolRegistry (built-in + MCP tools)
    ↓
ContextCompactor (5-level progressive compression)
    ↓
Checkpointer (save state, enable branching)
```

## License

MIT
