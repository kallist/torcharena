# TorchArena recruiting and portfolio copy

Positioning: AI engineering / ML engineering / Python infrastructure roles.
These are project descriptions, not claims of employment, sole authorship or
commercial use. Implementation was AI-assisted; describe your actual decisions,
verification and review honestly. See [the claim/evidence map](PUBLIC_EVIDENCE.md).

## 中文简历（3 条）

**TorchArena｜PyTorch 可复现训练与实验管理工具**

- 构建配置驱动的 PyTorch 训练工作台，集成隔离式 Preflight、NaN/Inf 训练守卫、SQLite 实验记录与可比条件下的模型对比，让训练失败和恢复过程可检查。
- 实现完整训练状态 Checkpoint/Resume，保存模型、优化器、调度器、GradScaler、RNG 与批次游标；在自有确定性 CPU 数据路径的 6 个中断位置验证连续训练与恢复训练等价。
- 交付 Model Race、Crash → Resume、Failure Graveyard 三类真实 CLI Showcase；GitHub Actions 的 Python 3.11/3.12 CPU 作业各通过 67 项测试，Docker 作业通过构建与训练 smoke 验证。

## English resume (3 bullets)

**TorchArena | Reproducible PyTorch Training Workbench**

- Built a config-driven PyTorch workbench with isolated preflight validation, finite-value training guards, SQLite experiment history and comparable model runs.
- Implemented full-state checkpoint/resume for model, optimizer, scheduler, GradScaler, RNG and batch cursor; verified uninterrupted/resumed equivalence at six interruption positions on the owned deterministic CPU data path.
- Delivered three CLI showcases with real training and failure evidence; GitHub Actions passed 67 tests per Python 3.11/3.12 CPU job, plus Docker build and training smoke checks.

## BOSS / 招聘平台短版

我做了一个 AI 辅助开发的 PyTorch 训练工作台 TorchArena，支持模型对比、完整状态恢复和 SQLite 实验管理。确定性 CPU 路径在 6 个中断位置验证恢复等价，Python 3.11/3.12 各通过 67 项测试，并完成 Docker 训练验证。

## Portfolio / 作品集

**Title:** TorchArena — Train. Race. Break. Resume.

**One-line summary:** A local PyTorch training workbench that makes model comparison,
interrupted training and failed experiments visible and explainable.

- **Race:** two real models, comparable conditions, measured categories and preserved ties.
- **Resume:** fresh training objects restore full state and continue from the owned batch cursor.
- **Inspect:** failures retain diagnostics and the last healthy checkpoint, with terminal and static HTML evidence.

**Tech stack:** Python · PyTorch · Pydantic · SQLite · Typer · Rich · pytest · GitHub Actions · Docker.

**GitHub:** [kallist/torcharena](https://github.com/kallist/torcharena)

**Visuals:** [GIF](assets/torcharena-demo.gif) · [Race](assets/model-race.png) ·
[Recovery](assets/crash-resume.png) · [Failure](assets/failure-graveyard.png).

## Copy audit / 面试边界

| 表述 | 证据 | 边界与可追问点 |
| --- | --- | --- |
| 配置驱动、Preflight、训练守卫、SQLite | [架构](ARCHITECTURE.md)、配置/安全/存储模块与行为测试 | 能解释隔离检查、状态转换和失败事务；不声称实际商业部署 |
| 完整状态恢复与 6 个中断位置 | [恢复契约](RECOVERY.md)、恢复等价测试 | 数据集/顺序由工具拥有；不能外推到任意 DataLoader 或跨硬件 |
| Python 3.11/3.12 各 67 项测试 | [main Hosted CI](https://github.com/kallist/torcharena/actions/runs/37044363967) | 两个 CPU 作业各跑完整套件；Docker 仅构建/doctor/训练 smoke |
| 真实 showcase | [证据表](PUBLIC_EVIDENCE.md)、TXT/JSON/HTML 与素材哈希 | NaN 是明确标注的演示注入，GIF 是真实输出回放 |

Avoid “independently hand-coded”, “production-grade”, “fault-tolerant platform”,
performance uplift, commercial adoption and universal bitwise reproducibility.
Do not imply that Docker executed the 67-test suite. No provider/CUDA results are
established by these CPU checks. Project capabilities alone do not prove personal
mastery; use only bullets whose design and tests you can explain under questioning.
