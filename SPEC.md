# Calamar Specification

Calamar 是一个开箱即用的通用 AI agent。安装即可在终端使用，无需额外配置。支持编码、数据分析、文件处理、信息检索等多种任务场景。

## 定位

```bash
pip install calamar
calamar                              # 交互式 REPL
calamar "重构 auth 模块"               # 编码任务
calamar "分析 sales.csv 的趋势"        # 数据分析
calamar "帮我整理这个目录的文件"         # 文件处理
```

对标产品：Claude Code / Aider / Hermes，而非 SDK 或框架。用户是开发者和高级用户，不是框架消费者。

**差异化不在功能多，在核心机制强**：智能模型路由省钱、中间件管线保安全、代码索引提精度、三代理协作提准确率。这些不是 feature list，是用户切实感知到的"更便宜、更安全、更准"。

## 设计哲学

### 1. 少即是多

每个模块的公开 API 不超过 5 个方法。宁可让用户组合简单的原语，不提供一个复杂的万能接口。

```python
agent = AgentLoop(config)
async for event in agent.run("fix the bug"):
    handle(event)
```

### 2. 中间件优于继承

所有横切关注点（权限、成本、追踪、安全）通过中间件管线组合，而非通过类继承扩展。一个 agent 的行为 = 主循环 + 中间件栈。

```python
pipeline = (
    MiddlewarePipeline()
    .use(InputGuardrail())
    .use(PermissionGate())
    .use(CostTracker())
    .use(OutputGuardrail())
)
```

### 3. 事件流驱动

Agent 的所有输出通过 `AsyncIterator[Event]` 流式返回。终端 REPL、Web UI、API server 都是同一个事件流的不同消费者。

### 4. 模型无关

不绑定任何特定模型或提供商。任何兼容 OpenAI Chat Completions API 的端点都可以直接使用。Anthropic、Gemini 等原生 API 通过可选 adapter 接入。

### 5. 可测试性第一

每个组件都可以独立实例化和测试，不需要启动整个 agent、不需要真实 API key、不需要网络。

## 核心概念

### AgentLoop

单线程、回合制、flat 消息历史。

```
用户输入
  → 组装上下文（静态前缀 + 动态后缀）
  → 检查 steering queue（中途指令注入）
  → 选择模型（ModelRouter）
  → 调用 LLM
  → 解析响应
    → 纯文本 → 输出 TextEvent，结束 turn
    → 工具调用 → 走中间件管线 → 执行 → 输出 ToolEvent → 继续循环
  → 检查上下文压缩
  → 保存检查点
```

设计约束：
- 主循环不超过 100 行有效代码
- 单线程，无锁，无竞态
- 消息历史是唯一的状态源

### Tool

工具是 agent 影响外部世界的唯一方式。

```python
class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...
    async def execute(self, **kwargs) -> ToolResult: ...
```

工具通过 ToolRegistry 管理，支持运行时动态注册/注销。MCP 工具和内置工具在 registry 层统一，对 agent 完全透明。

### Middleware

洋葱模型：

```
请求 → [Guard] → [Permission] → [Cost] → [Execute] → [Trace] → 结果
结果 ← [Guard] ← [Permission] ← [Cost] ← [Execute] ← [Trace] ← 结果
```

每个中间件可以拦截、修改输入/输出、记录副作用、或直接放行。

### Context

**静态前缀 + 动态后缀**，最大化 prompt cache 命中率。4 级渐进压缩处理长会话。

### ModelRouter

规则引擎，根据任务画像自动选择最优模型。分类器是轻量级的（不调 LLM）。

### AgentRole

可热插拔的 system prompt + toolset 组合。内置四个角色：default、planner、executor、verifier。

## 技术约束

| 约束 | 要求 |
|------|------|
| 语言 | Python 3.12+ |
| 核心依赖 | openai, httpx, pydantic（仅 3 个） |
| 异步 | 全 async/await，基于 asyncio |
| 类型 | 100% 类型注解，通过 ruff 检查 |
| 测试 | 每个模块都有单元测试，不依赖网络 |
| 零配置启动 | `Config()` + 一个环境变量即可运行 |

## 模块边界

```
calamar/
├── cli.py           # CLI 入口 + 交互式 REPL
├── loop.py          # AgentLoop — 主循环
├── events.py        # Event 类型 — 输出协议
├── config.py        # Config — 输入协议
├── context.py       # ContextBuilder + Compactor
├── middleware.py     # MiddlewarePipeline
├── router.py        # ModelRouter
├── roles.py         # AgentRole
├── tools/           # 工具系统
│   ├── base.py      # Tool 协议 + ToolResult
│   ├── registry.py  # 工具注册和分发
│   ├── terminal.py  # 终端命令执行
│   ├── file.py      # 文件读写编辑
│   └── search.py    # 代码/文件搜索
├── code_index/      # 代码理解
│   ├── repo_map.py  # tree-sitter 符号图
│   └── locator.py   # 层级式代码定位
└── git/             # Git 工作流
    └── workflow.py  # 自动 commit、分支、回滚
```

## 质量标准

- 每个 PR 必须包含测试，无测试不合并
- ruff 零警告，CI 强制检查
- 公开 API 必须有类型注解
- commit message 遵循 conventional commits
- 不引入非必要依赖，每个新依赖需要说明理由

## 路线图

### v0.1 — 核心骨架（当前）

- [x] AgentLoop 主循环 + 事件流
- [x] ToolRegistry + Tool 协议
- [x] ContextBuilder + 4 级压缩
- [x] MiddlewarePipeline（guardrail、cost、timing）
- [x] ModelRouter 规则路由
- [x] AgentRole 角色系统
- [x] 18 个单元测试

### v0.2 — 可交互的 Agent

- [x] 内置工具：terminal、file_read、file_edit、file_write、search
- [x] CLI 交互式 REPL
- [x] 流式终端输出（rich/prompt_toolkit）
- [ ] 端到端真实 API 调用

### v0.3 — 代码理解

- [ ] tree-sitter Repo Map
- [ ] 层级式代码定位
- [ ] sqlite-vec 向量索引

### v0.4 — Git 原生

- [ ] 自动 commit + 分支隔离
- [ ] 安全回滚

### v0.5 — 多 Agent

- [ ] 三代理协作（Planner → Executor → Verifier）
- [ ] Subagent 委托 + 并行扇出

### v1.0 — 生产就绪

- [ ] MCP 工具集成
- [ ] 全链路追踪
- [ ] Session 持久化 + 分叉
- [ ] 完整文档
