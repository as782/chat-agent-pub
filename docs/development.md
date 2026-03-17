# 开发规范

本文档说明当前仓库的开发流程、Git 规范、测试规范和中文注释规范。

## 开发环境

- Python 版本：`3.11`
- 包管理：`uv`
- 数据库：PostgreSQL
- 缓存/短期状态：Redis

初始化依赖：

```bash
uv sync
uv run pre-commit install
```

本地启动：

```bash
docker compose up -d postgres redis
uv run uvicorn app.main:app --reload
```

## 自检要求

每次提交前至少执行：

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

如需手动执行 pre-commit：

```bash
uv run pre-commit run --all-files
```

## 测试分层

- `tests/unit`
  - 纯逻辑测试
  - schema、repository、router、tool registry、配置与注释规范检查
- `tests/integration`
  - API、service、graph、client 协作测试
- `tests/e2e`
  - 从会话创建到消息查询的完整业务链路

新增功能时，优先同步补对应层级测试，不允许最后集中补测试。

## 分支规范

- `main`
  - 稳定分支
  - 禁止直接推送未经评审的业务改动
- `develop`
  - 集成分支
  - 用于汇总待发布功能
- `feature/*`
  - 功能开发分支
- `fix/*`
  - 缺陷修复分支

## Commit 规范

推荐使用以下前缀：

- `feat:` 新功能
- `fix:` 缺陷修复
- `refactor:` 重构
- `test:` 测试相关
- `docs:` 文档相关
- `chore:` 工程杂项

提交信息要求：

- 第一行简洁描述改动意图
- 避免无语义描述，例如 `update`、`modify`
- 一个提交尽量只表达一个清晰目标

## PR 规范

- 禁止直接提交到 `main`
- PR 描述必须说明：
  - 背景
  - 改动点
  - 测试结果
  - 风险说明
- 如果变更涉及接口、配置或数据结构，需要明确说明兼容性影响

## 代码规范

- 所有 Python 代码必须带类型标注
- 路由层不能写复杂业务逻辑
- repository 层不能写业务决策
- 第三方调用必须统一封装到 client 层
- 配置必须从环境变量读取，禁止硬编码密钥
- 布尔变量使用 `is_`、`has_`、`need_` 前缀
- 关键异常必须显式处理

## 中文注释规范

- 所有注释使用中文
- 核心文件开头必须有中文文件说明
- 核心类必须有中文类说明
- 以下函数必须有中文 docstring：
  - API handler
  - service 层函数
  - repository 方法
  - graph 节点函数
  - RAGFlow client 方法
  - 工具执行函数
  - 上下文构建函数
- 注释重点说明“为什么这样设计”，避免无意义注释

当前仓库已增加自动化基线测试：

- [tests/unit/test_chinese_comment_conventions.py](/c:/Users/wengkaibin/DATA/WorkSpace/Test spaces/chat-agent/tests/unit/test_chinese_comment_conventions.py)

该测试会检查核心模块文件说明和一批关键函数的中文 docstring。行内注释质量仍需要在代码评审中人工确认。
