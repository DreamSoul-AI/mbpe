# MBPE

多维 Byte Pair Encoding 研究代码。工程入口 `src/mbpe` 提供张量实现、离线配置预览和词表 JSON 持久化，合成整数 CPU 张量可逐值往返。独立文件的压缩对照由研究脚本提供。 [研究脚本](studies/README.md)

## 当前研究

研究范围是 CPU 上的整数格点 BPE，包括规则学习、编码、逐格还原和压缩评估。研究脚本覆盖原像素、掩码、官方基线和任意形状候选。 [研究统合报告](<docs/研究报告 Research Report/2026-11-14 MBPE 研究统合报告.md>) · [研究计划](<docs/研究计划 Research Plan MBPE.md>)

官方公开实现的碰撞、方向位修复和同输入对照见[运行记录](<docs/运行记录 Run Record/2026-11-11 碰撞复现与同输入对比.md>)。带孔域、未知整数、标量和空域的支持情况见[无损候选设计](<docs/开发记录 Development Record/2026-11-11 无损候选设计与验证.md>)。

## 开始使用

使用独立的 Python 3.10–3.12 环境安装。依赖及版本范围由 `pyproject.toml` 定义，安装可能下载依赖。基础检查使用离线合成数据。

```powershell
git clone https://github.com/DreamSoul-AI/mbpe.git
cd mbpe
python -m pip install -e .
python -m mbpe plan
python tests/run.py
```

`plan` 只输出配置，不下载数据、不训练、不写结果，也不导入 torch。未安装包时可从源码使用：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m mbpe plan --patch-size 1 2 2 --batch-size 4
```

安装元数据见 pyproject.toml，核心依赖沿用旧清单的固定版本。requirements.txt 保留为历史环境快照，不建议按旧快照整体重建环境。images 可选依赖用于图像工作，历史 pipeline 的 Transformers 环境尚未整理。

## 结构与迁移

- `src/mbpe/`：可安装包，根层保留导入和命令入口
- `src/mbpe/structure/`：张量 BPE、词表、持久化、指标和配置
- `src/mbpe/flow/`：配置预览、合成运行和合并对照的命令入口
- `tests/`：按核心部件、流程和研究脚本组织的离线 CPU 回归
- [`studies/`](studies/README.md)：CPU 研究脚本、官方基线包装和候选原型
- `src/legacy/`：旧 tuple 实现、下游训练流程与张量示例
- `data/` 和 `results/` 是旧批次约定的本机路径。原件已备份，复跑前按 [研究脚本说明](studies/README.md)恢复所需文件。

Python 调用使用新模块路径，例如 `from mbpe.structure.bpe.tokenizer import Tokenizer`。研究旧 tuple 版本时在 `src/legacy/tuple` 下单独使用。模块映射见[代码结构设计](<docs/代码结构设计 Code Structure Design MBPE.md>)。

## 文档

[项目约定](REPO.md) · [Agent 入口](AGENTS.md) · [文档导航](docs/README.md)

## 许可

沿用 [LICENSE](LICENSE)。
