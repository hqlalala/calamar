# Calamar Specification

Calamar 是一个可嵌入的 AI agent 引擎，为构建编码代理和自动化工具提供核心运行时。它不是一个完整的产品，而是产品的引擎——就像 V8 之于 Chrome，SQLite 之于 iOS。

## 定位

```
                    完整产品
                 (Cursor, Devin)
                       ↑
                  产品框架层
              (Claude Code, Aider)
                       ↑
              ┌────────────────┐
              │    Calamar     │  ← 我们在这里
              │  Agent Engine  │
              └────────────────┘
                       ↑
                  模型 API 层
           (OpenAI SDK, Anthropic SDK)
```

**不做什么**：不做 IDE 插件、不做 Web UI、不做平台集成、不做用户账户系统。这些是上层产品的事。

**只做什么**：agent loop、工具系统、上下文管理、模型路由、中间件管线、多 agent 协作。做好这一层，做到极致。

## 设计哲学

### 1. 少即是多

每个模块的公开 API 不超过 5 个方法。宁可让用户组合简单的原语，不提供一个复杂的万能接口。

```python
# 好：三行启动一个 agent
agent = AgentLoop(config)
async for event in agent.run("fix the bug"):
    handle(event)

# 坏：需要理解 15 个概念才能开始
orchestrator = Orchestrator(
    planner=Planner(strategy=TreeOfThought()),
    executor=Executor(sandbox=DockerSandbox()),
    verifier=Verifier(criteria=TestSuite()),
    memory=VectorMemory(embedder=OpenAIEmbedder()),
    ...
)
```

### 2. 中间件优于继承

所有横切关注点（权限、成本、追踪、安全）通过中间件管线组合，而非通过类继承扩展。一个 agent 的行为 = 主循环 + 中间件栈。

```python
pipeline = (
    MiddlewarePipeline()
    .use(InputGuardrail())      # 安全检查
    .use(PermissionGate())      # 权限控制
    .use(CostTracker())         # 成本追踪
    .use(RateLimiter())         # 速率限制
    .use(OutputGuardrail())     # 输出过滤
)
```

### 3. 事件流，不是回调

Agent 的所有输出通过 `AsyncIterator[Event]` 流式返回。消费方自行决定如何处理——打印到终端、发送到 WebSocket、写入日志、触发 webhook。引擎不关心。

### 4. 模型无关

引擎不绑定任何特定模型或提供商。任何兼容 OpenAI Chat Completions API 的端点都可以直接使用。Anthropic、Gemini 等原生 API 通过可选 adapter 接入。

### 5. 可测试性第一

每个组件都可以独立实例化和测试，不需要启动整个 agent、不需要真实 API key、不需要网络。Mock 一个 Tool 只需要实现两个方法。

## 核心概念

### AgentLoop

引擎的心脏。单线程、回合制、flat 消息历史。

一个 turn 的生命周期：

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

两个方法，没有第三个。`spec` 描述工具的能力（自动转换为 OpenAI function schema），`execute` 执行操作。ToolResult 只有 `output` 和 `error` 两个字段。

工具通过 ToolRegistry 管理，支持运行时动态注册/注销。MCP 工具和内置工具在 registry 层统一，对 agent 完全透明。

### Middleware

中间件是一个洋葱模型：

```
请求 → [Guard] → [Permission] → [Cost] → [Execute] → [Trace] → 结果
结果 ← [Guard] ← [Permission] ← [Cost] ← [Execute] ← [Trace] ← 结果
```

每个中间件可以：
- 拦截并阻止执行（guardrail 发现危险操作）
- 修改输入（参数转换、默认值注入）
- 修改输出（密钥脱敏、结果截断）
- 记录副作用（成本累计、追踪日志）
- 直接放行到下一层

### Context

上下文管理的核心思想：**静态前缀 + 动态后缀**。

```
┌────────────────────────────┐
│ System Prompt              │ ← 静态前缀（跨 turn 不变）
│ Tool Definitions           │    命中 prompt cache
│ Project Context Files      │
├────────────────────────────┤
│ Conversation History       │ ← 动态后缀（每 turn 变化）
│ (older turns may be        │
│  summarized or removed)    │
└────────────────────────────┘
```

当上下文逼近窗口上限时，4 级渐进压缩自动介入：
1. 截断冗长的工具输出
2. 摘要早期对话轮次
3. 精简系统上下文
4. 紧急压缩（只保留最近 2-3 轮）

### ModelRouter

根据任务画像自动选择最优模型。不是 magic，是规则引擎：

```yaml
routing:
  rules:
    - match: { complexity: simple }
      model: haiku
    - match: { task_type: bug_fix }
      model: opus
  default: sonnet
```

分类器是轻量级的（关键词 + 长度，不调 LLM），误判的成本是用了一个稍贵/稍便宜的模型，不是灾难。

### AgentRole

角色是可热插拔的 system prompt + toolset 组合。同一个 AgentLoop 可以在 turn 之间切换角色，实现多 agent 协作而不需要多个进程。

内置四个角色：
- **default**：通用 agent
- **planner**：只读分析，不做修改
- **executor**：精确执行计划
- **verifier**：测试和验证

## 技术约束

| 约束 | 要求 |
|------|------|
| 语言 | Python 3.12+ |
| 核心依赖 | openai, httpx, pydantic（仅 3 个） |
| 异步 | 全 async/await，基于 asyncio |
| 类型 | 100% 类型注解，通过 ruff 检查 |
| 测试 | 每个模块都有单元测试，不依赖网络 |
| 包体积 | 核心引擎 < 2000 行代码 |
| 零配置启动 | `Config()` + 一个环境变量即可运行 |

## 模块边界

```
calamar/
├── loop.py          # AgentLoop — 主循环，引擎的心脏
├── events.py        # Event 类型 — 引擎的输出协议
├── config.py        # Config — 引擎的输入协议
├── context.py       # ContextBuilder + Compactor — 上下文管理
├── middleware.py     # MiddlewarePipeline — 横切关注点
├── router.py        # ModelRouter — 模型选择策略
├── roles.py         # AgentRole — 角色模板
├── tools/           # 工具系统
│   ├── base.py      # Tool 协议 + ToolResult
│   └── registry.py  # ToolRegistry — 工具注册和分发
├── code_index/      # 代码理解（Phase 2）
│   ├── repo_map.py  # tree-sitter 符号图
│   └── locator.py   # 层级式代码定位
└── git/             # Git 工作流（Phase 2）
    └── workflow.py  # 自动 commit、分支、回滚
```

每个文件的职责单一、边界清晰。任何两个文件之间的依赖关系都应该是单向的。循环依赖是 bug。

## 质量标准

- **每个 PR 必须包含测试**，无测试不合并
- **ruff 零警告**，CI 强制检查
- **公开 API 必须有类型注解**，内部函数可省略
- **commit message 遵循 conventional commits**：`feat:` / `fix:` / `refactor:` / `test:` / `docs:`
- **不引入非必要依赖**，每个新依赖需要说明理由

## 路线图

### v0.1 — 引擎核心（当前）

- [x] AgentLoop 主循环
- [x] Event 流式输出
- [x] ToolRegistry + Tool 协议
- [x] ContextBuilder + 4 级压缩
- [x] MiddlewarePipeline（guardrail、cost、timing）
- [x] ModelRouter 规则路由
- [x] AgentRole 角色系统
- [x] 18 个单元测试

### v0.2 — 可用的 Agent

- [ ] 内置工具：terminal、file_read、file_edit、file_write、search_code
- [ ] 端到端测试：真实 LLM API 调用
- [ ] CLI 交互模式（REPL）
- [ ] 流式输出到终端

### v0.3 — 代码理解

- [ ] tree-sitter Repo Map（符号图）
- [ ] 层级式代码定位（仓库 → 文件 → 函数）
- [ ] sqlite-vec 向量索引（可选增强）

### v0.4 — Git 原生

- [ ] 自动 commit（agent 编辑后自动提交）
- [ ] 分支隔离（复杂任务自动创建 feature branch）
- [ ] 安全回滚（一键 revert agent 的修改）

### v0.5 — 多 Agent 协作

- [ ] 三代理模式（Planner → Executor → Verifier）
- [ ] Subagent 委托（独立上下文的子 agent）
- [ ] 并行扇出（多个子 agent 同时工作）

### v1.0 — 生产就绪

- [ ] MCP 工具集成
- [ ] 全链路追踪（OpenTelemetry）
- [ ] Session 持久化和恢复
- [ ] Session 分叉（fork 对话尝试不同方案）
- [ ] 完整文档和示例
