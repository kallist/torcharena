# TorchArena｜90 秒演示脚本

目标：先让观众看到真实训练，再用一个恢复过程说明工程价值。开场标注
“本地确定性 CPU 演示 / 自有合成数据”；不要把准确率和运行时间讲成 benchmark。
若使用 [README GIF](assets/torcharena-demo.gif)，标注“真实 CLI 输出回放”，不要假装实时录屏。

| 时间 | 画面 / 操作 | 自然口播 |
| --- | --- | --- |
| 0–5s | 项目名和中断画面；停在 `Safe step: 3` | “PyTorch 训练跑到一半被打断，还能带着原来的训练状态接着跑吗？” |
| 5–20s | `torcharena showcase race`，展示两行模型和比较表 | “这是 TorchArena。先让两个小模型在相同数据、种子和训练预算下顺序训练。它比较实际测量值，准确率相同就保留并列，不硬选一个冠军。” |
| 20–40s | `torcharena showcase crash-resume`，依次突出 INTERRUPTED、RESTORED、COMPLETED | “这里在第三步安全中断。原来的训练对象释放后，重新创建模型和优化器，从文件恢复状态，再完成训练。这次演示还和不中断的基线比较，参数误差是零；这是这条确定性 CPU 路径的结果。” |
| 40–55s | `torcharena showcase graveyard`；保持“demo-only NaN injection”可见 | “这次我明确注入一个 NaN 演示故障。训练守卫会记录失败原因，留下上一个健康 checkpoint，失败实验不会只剩一行报错。” |
| 55–70s | 架构图或 RECOVERY.md；突出状态列表和六个中断位置 | “恢复保存的不只是权重，还有优化器、调度器、scaler、随机数状态和批次游标。SQLite 留下实验历史，写入和恢复边界都有说明，也有六个中断位置的行为测试。” |
| 70–90s | GitHub README、真实 Actions 作业与仓库链接 | “Python 3.11 和 3.12 的 CPU 作业各通过 67 项测试，Docker 也验证了构建和小型训练。代码和证据都在 GitHub。CUDA、任意 DataLoader 恢复和分布式训练不在这次验证范围里，欢迎看实现和测试。” |

## 录制准备

激活已安装 TorchArena 的开发环境，打开干净的终端，关闭显示私人路径的 prompt。
在项目根目录用隔离目录录制，避免覆盖 checked-in showcase：

```bash
export TORCHARENA_HOME=.torcharena/demo-recording
export TORCHARENA_SHOWCASE_DIR=.torcharena/demo-recording/exports
torcharena showcase race
torcharena showcase crash-resume
torcharena showcase graveyard
```

PowerShell 的环境变量语法：

```powershell
$env:TORCHARENA_HOME = '.torcharena/demo-recording'
$env:TORCHARENA_SHOWCASE_DIR = '.torcharena/demo-recording/exports'
```

脚本中的路径不会出现在干净截图中；录制前检查实际输出。不要展示 doctor
中的机器环境信息、文件管理器、账号邮箱、凭据、桌面通知或私人仓库。
字幕可以解释步骤，终端数值必须来自录制本身，不能用手写数字替换。

## 恢复表述

这里展示的是安全边界中断，不是把进程强杀后保证第三步永远不丢。
硬终止只能从已落盘的健康 snapshot 恢复，未保存的工作可能重做。
NaN 是演示注入，不是伪装成自然模型故障；scaler 的保存不等于 CUDA/AMP 已测试。
本项目为 AI 辅助开发；讲解时明确自己实际理解和验证的设计，不包装为未经辅助的独立手写。
