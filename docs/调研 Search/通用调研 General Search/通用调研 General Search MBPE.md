# 通用调研 General Search MBPE

## 要解决的工程问题

2026-11-12 核查 MBPE 当前工作区、远端 Workspace 接入提交及本机 RPipe。目标是让核心实现、运行流程和研究代码各有明确位置，并让测试与证据能跟随模块迁移。文献机制与论文证据见[文献调研](<../文献调研 Literature Search/文献调研 Literature Search MBPE.md>)。

## 现状与成因

Workspace 提供研发协作与文档职责约定。远端 `2e93307` 已合入模板接入。本机模板 `d58cadf` 与当时 `AGENTS.md` 的比较只有仓库入口差异。远端 README 仍指向旧的根级 tuple、pipeline 及示例，本地已把它们迁入 `src/legacy`，设立了 `studies` 并完成两柱职责归位。依据为 [远端接入 PR](https://github.com/DreamSoul-AI/mbpe/pull/5)和[本地迁移记录](<../../开发记录 Development Record/2026-11-12 两柱架构与调研整理.md>)。

核心文件最初留在包根，因为第一次迁移保留原实现的相对导入，只在旁边添加配置、合成数据和 CLI。现有 `flow/run.py` 从 `structure/bpe` 装配 Tokenizer、词表保存和指标。合成对照的流程与算法也分开归位，继续支撑 `plan`、`run`、`controls` 三个入口。依据为 [CLI](../../../src/mbpe/flow/cli.py)、[合成运行](../../../src/mbpe/flow/run.py)、[对照运行](../../../src/mbpe/flow/tile_controls.py)和[对照算法](../../../src/mbpe/structure/bpe/controls.py)。

## RPipe 的可借用边界

本机 RPipe `915c52f` 的包根保留 `__init__.py` 与 `__main__.py` 作为 Python 导入和模块执行入口。科学部件、条件和产物定义在 structure，生命周期动作在 flow。独立课题在 studies，核心测试和课题测试分开组织。本项目借用这三条边界，把根级业务模块按职责归位。当前实验只需 CPU BPE，运行阶段沿用现有调用，执行权、后台监管和完整调度没有对应的需求。依据为 [RPipe README](https://github.com/DreamSoul-AI/RPipe/blob/915c52feb47eacc9bd8400794a743224792ae9a5/README.md)及 [RPipe 测试目录](https://github.com/DreamSoul-AI/RPipe/tree/915c52feb47eacc9bd8400794a743224792ae9a5/tests)。

| 方案 | 代价与判断 |
| --- | --- |
| 继续平铺 | 原导入稳定，算法、持久化和实验代码继续共享同一个命名空间。后续候选会加重混杂。 |
| 两柱归位并镜像测试 | 能分别约束科学部件、运行编排和课题复现。必须同步调整导入、包数据、测试发现和历史调用入口。采用为目标方案。 |
| 直接引入 RPipe 全流程 | 可提供较完整的研究调度，但会额外引入工作目录、调度和生命周期约定，超出当前 CPU BPE 的需求。 |

## 实施边界与证据分工

包根的正式目标只有两个入口文件与两柱目录。已有实验接口的兼容需求要按真实调用方判断。仓内调用全部随迁移更新，确有需要保留的外部导入用显式兼容方案讨论，避免留下永久的根级实现。移动前冻结规则、输出、持久化及 CLI 合同，先做职责迁移，再讨论算法与新文件格式。测试层分别看纯部件、流程、研究候选和安装入口。逐项映射和门槛见[代码结构设计](<../../代码结构设计 Code Structure Design MBPE.md>)。

Workspace 的固定文档角色里有研究计划、文献调研、开发记录和运行记录。原先混放的批次结果、统计和 JSON 证据已归入运行记录，设计推导、接续分析和交接已归入开发记录。论文和机制判断分别由文献、通用调研维护，待检验假设由研究计划维护。入口统一见[文档导航](../../README.md)。
