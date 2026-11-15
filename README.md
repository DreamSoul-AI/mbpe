# MBPE

多维 Byte Pair Encoding 研究代码。工程入口 `src/mbpe` 提供张量实现、离线配置预览和词表 JSON 持久化，合成整数 CPU 张量可逐值往返。独立文件的压缩对照由研究脚本提供。 [研究脚本](studies/README.md)

## 当前研究

最新[颜色与解析 CPU 结果](<docs/研究记录 Research Record/2026-11-15 颜色坐标与多尺度词表的输入收益.md>)：固定 150,000 规则，可逆颜色差分加同词表最短解析，第二批独立测试每集 1000 图，CIFAR-10/SVHN 输入位置平均减少 70.33%/79.43%，全部逐图及独立还原。对齐方块模板未胜过强一维；这是固定 RGB 条件下的基线证据，方法新颖性和 GPU 效果未验证。

最新[大词表CPU结果](<docs/研究记录 Research Record/2026-11-15 大词表一维BPE的长度收益.md>)：同一RGB学习/测试池，差分按列一维从256扩至150,000规则后，CIFAR-10/SVHN序列均值缩短56.79%/69.30%；逐图减半覆盖为93.7%/99.8%，全部无损。固定192维BF16，每个token一个位置；词表大小单列，GPU任务效果未测。

研究范围是 CPU 上的整数格点 BPE，包括规则学习、编码、逐格还原和压缩评估。研究脚本覆盖原像素、掩码、官方基线和任意形状候选。 [研究统合报告](<docs/研究报告 Research Report/2026-11-14 MBPE 研究统合报告.md>) · [研究计划](<docs/计划 Plan/研究计划 Research Plan MBPE.md>)

当前主比较为一个内容 token 占一个同宽 embedding 位置。已测 MNIST 按行一维序列缩短 81.85%，WikiText-2 测试集 Qwen3 BPE 缩短 76.79%；256规则基线下，原始 RGB 经可逆左差分与按列一维 BPE 后，CIFAR-10、SVHN 分别缩短 23.13%、37.13%，大预算结果见上方最新曲线。文件字节和多个编号打包进向量的额外收益另列。模型任务效果与新颖性未验证。 [长度对照及条件](<docs/研究记录 Research Record/2026-11-15 文本与图像的序列长度对照.md>) · [组批与强一维对照](<docs/研究记录 Research Record/2026-11-15 有限窗口收益与二维独立作用.md>)

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

Python 调用使用新模块路径，例如 `from mbpe.structure.bpe.tokenizer import Tokenizer`。研究旧 tuple 版本时在 `src/legacy/tuple` 下单独使用。模块映射见[代码结构设计](<docs/设计 Design/代码结构设计 Code Structure Design MBPE.md>)。

## 文档

[项目约定](REPO.md) · [Agent 入口](AGENTS.md) · [文档导航](docs/README.md)

## 许可

沿用 [LICENSE](LICENSE)。
