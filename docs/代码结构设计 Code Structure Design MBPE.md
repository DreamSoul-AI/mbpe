# 代码结构设计 Code Structure Design MBPE

## 当前目录

`src/mbpe` 的 BPE 实现、指标、持久化和实验对照部件已归入 `structure/bpe`。CLI 从 `flow/cli.py` 进入，配置与合成清单位于 structure。

```text
src/
  mbpe/
    __init__.py
    __main__.py
    structure/
      bpe/
      config.py
      synthetic.py
      synthetic_manifest_v1.json
      synthetic_manifest_v2.json
    flow/
      cli.py
      run.py
      tile_controls.py
  legacy/
studies/
tests/
  run.py
  mbpe/
    structure/
    flow/
  studies/
```

`__init__.py` 保留包入口，`__main__.py` 只转发 CLI。其他 Python 实现均放进 structure 或 flow。沿用 [RPipe](https://github.com/DreamSoul-AI/RPipe) 的职责分离，目录深度由实际依赖决定。

## 模块迁移映射

| 迁移前 `src/mbpe` 下的模块 | 当前位置 | 合同及注意点 |
| --- | --- | --- |
| `tokenizer.py`、`state.py`、`vocab.py`、`frequency_counter.py` | `structure/bpe/` 同名 | 学习、合并和往返合同。保持并列次序、未知对停止、频次冻结和符号上限。 |
| `patch.py`、`utils.py`、`tensor_shuffle.py` | `structure/bpe/` 同名 | 先原样迁移张量操作。重复 shuffle 实现和通用工具职责另做审计。 |
| `persist.py`、`compression_metrics.py` | `structure/bpe/` 同名 | v1 词表 JSON 的读写合同不变。既有指标保持兼容，完整传输比特计量属于独立课题。 |
| `controls.py` | `structure/bpe/controls.py` | 由 `flow/tile_controls.py` 和核心单测调用，支撑现有 `controls` 命令。 |
| `structure/config.py`、`structure/synthetic.py` 与两个 JSON | 沿用现有位置 | 预览轻依赖，继续按清单版本和种子得到同一格点。 |
| `flow/cli.py`、`run.py`、`tile_controls.py` | 沿用现有位置 | 继续负责参数、装配与输出。改为依赖新结构路径。 |
| `studies/*.py` 与 `src/legacy/` | 沿用现有位置 | 候选仍通过独立研究入口运行，历史脚本供复现和追溯。 |
| 第 33 批完整文件对照 | `studies/batch33_celeba_files.py`、`studies/bpe_block_file.py` | 实验入口负责样本冻结、配置选择、测量及取证。文件模块负责字节布局和独立解码，复用候选的覆盖还原。 |
| 第 34 批编码对照 | `studies/batch34_token_coding.py`、`studies/canonical_huffman.py` | 从冻结文件复用规则与 token，用验证流构造码表。文件模块负责 v1 与 v2 的分块读取和还原。 |
| 第 35 批形状对照 | `studies/batch35_shape_layouts.py` | 生成配对整数格子，复用规则学习与独立解码，保存等预算文件和逐对差值。 |
| 第 36 批扫描对照 | `studies/batch36_scan_orders.py` | 冻结原样本和规则，保存验证选择及全组、方向子组完整文件。 |
| 第 37 批自动选向 | `studies/batch37_adaptive_scan.py`、`studies/batch37_shared_budget.py`、`studies/adaptive_bpe_file.py` | 实验入口冻结逐图选择规则并比较完整文件。文件模块存两套词表、码表和方向位，复用既有分块编码与网格还原。 |
| 第 38 至 40 批整数格子对照 | `studies/batch38_native_grids.py`、`studies/batch39_common_width.py`、`studies/batch40_shaped_grids.py` | 分别测真实像素、冻结 token 的位宽对照和任意形状文件。 |
| 整数格子研究文件 | `studies/grid_bpe_file.py`、`studies/grid_bpe_study.py` | 文件模块负责形状、域、词表和码流；研究助手负责复用既有规则学习与独立往返检查。 |

旧子模块导入如 `from mbpe.tokenizer import Tokenizer` 需改成 `from mbpe.structure.bpe.tokenizer import Tokenizer`，持久化等模块同理。包入口保留原有的 `from mbpe import tokenizer` 等惰性属性。console script `mbpe` 和 `python -m mbpe` 沿用原入口。

## 测试归属与验收顺序

| 测试范围 | 当前位置 | 验收点 |
| --- | --- | --- |
| tokenizer、tensor、persist、compression、symbol cap、synthetic | `tests/mbpe/structure/` | 确定性、旧 JSON、逐格 shape 与 dtype、推断不修改状态、非法输入失败 |
| setup、run | `tests/mbpe/flow/` | 预览无 Torch，合成运行的参数、退出码和 JSON 输出 |
| merge controls 的纯算法用例 | `tests/mbpe/structure/` | 现有次数、空间、邻接和重叠对照语义 |
| Elsner、Zhang、ND 原型 | `tests/studies/` | 包装行为、反例和候选往返。第三方实跑仍由实验计划单独安排 |

`tests/run.py` 从仓库根解析测试包名，例如 `tests.mbpe.flow.test_setup`，避免遮蔽正式包。迁移前后的 unittest 清单逐项对应，45 项通过。包发现已覆盖新子包。本机环境缺 pytest，支持版本的打包安装仍待单独核对。见[迁移验证](<运行记录 Run Record/2026-11-12 两柱迁移离线验证.md>)。
