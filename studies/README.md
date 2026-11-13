# MBPE 研究脚本

这里存放一维与多维 BPE 对照、官方基线包装和研究候选。各批次共用同一层的助手，便于按原文件名复现和追查依赖。可安装的核心包位于 [src/mbpe](../src/mbpe/)。

| 文件 | 职责 |
| --- | --- |
| `batch27_*.py` 至 `batch32_*.py` | 历史 token 长度与估算比特对照 |
| `batch33_celeba_files.py`、`bpe_block_file.py` | 按验证集选配置，比较完整文件并独立解码 |
| `batch34_token_coding.py`、`canonical_huffman.py` | 从冻结文件提取 token，比较 Huffman 与 zlib 的实际字节 |
| `batch35_shape_layouts.py` | 生成等直方图的方块与长条配对，比较等预算的二维和 Hilbert 文件 |
| `batch36_scan_orders.py` | 复用配对格子，在验证集选一维扫描，测全组与方向子组 |
| `batch37_adaptive_scan.py`、`batch37_shared_budget.py`、`adaptive_bpe_file.py` | 冻结逐图按行或按列的规则，比较完整文件，独立解码混合方向文件 |
| `batch38_native_grids.py`、`batch39_common_width.py`、`batch40_shaped_grids.py` | 分别测原始整数像素、冻结 token 的位宽对照、矩形及带孔和三维格子 |
| `grid_bpe_file.py`、`grid_bpe_study.py` | 整数格子文件的读写与逐格验收，支持形状、有效域和未知整数 |
| `elsner_*.py`、`zhang_*.py` | 官方运行包装、规则解码和诊断 |
| `nd_bpe_*.py`、`lossless_nd_*.py` | 多维原型、计数对照和无损候选 |
| `bpe_pair_inputs.py`、`bpe_round2_summary.py` | 冻结对照输入和汇总运行证据 |

从仓库根调用研究脚本。历史命令里的 `src/<脚本名>.py` 改用 `studies/<脚本名>.py`，原记录中的输入、参数和产物路径仍按对应批次读取。当前实验边界、已有数据和结果从 [CPU 交接](<../docs/开发记录 Development Record/2026-11-12 MBPE-CPU交接给Codex.md>) 定位。搬迁后的离线检查使用 `python tests/run.py`。

本机历史 `data/` 和 `results/` 已收至临时备份。复跑依赖这些路径的批次前，从备份复制回所需的输入或词表，位置和清单见[目录与运行产物整理](<../docs/开发记录 Development Record/2026-11-12 目录与运行产物整理.md>)。

第 28–30 批在运行 SVHN 或 CelebA 对照时，分别用 `--svhn-test-grids` 和 `--celeba-test-grids` 传入相应的 `transcribed_data_og` 目录。第 29 批仅运行 `--only p13` 时可省略两项，第 30 批只需指定本次 `--jobs` 用到的格子目录。早期原件位置见相应运行记录。

扩大灰度测量可用 `--output` 选择新的输出目录，默认在本仓库 `tmp/elsner-expanded-evaluation`。可在启动前设置 `TMP`、`TEMP`、`NUMBA_CACHE_DIR` 和 `MPLCONFIGDIR`。两个 Elsner 包装在未指定 `MPLCONFIGDIR` 时使用本仓库的 `tmp/matplotlib`。

第 33 批读取已解出的 CelebA JPEG。用 `--source` 指定图像目录，用 `--out` 指定尚不存在的输出目录。每份 `.mbpe` 文件可由 `bpe_block_file.py <archive> <raw_out>` 独立解为按行存储的 uint8 格子，标准输出包含尺寸、张数和字节分项。格式使用共享词表和独立 zlib 块，适用于本批的稠密二值与四级方格。

第 34 批的 `--source` 指向完整的第 33 批输出目录，`--out` 指定新目录。独立解码入口支持 zlib v1 与 Huffman v2。 [第 34 批命令与输入要求](<../docs/运行记录 Run Record/2026-11-13 第34批 固定切词编码对照.md>)

第 35 批按固定种子生成配对格子。用 `--out` 指定新目录，运行回执含两档等规则预算、零规则基线和逐对差值。 [第 35 批命令与设置](<../docs/运行记录 Run Record/2026-11-13 第35批 紧凑块与长条对照.md>)

第 36 批的 `--source` 指向第 35 批完整输出，`--out` 指定新目录。测试前的扫描选择和横竖子组独立文件与全组结果一并保存。 [第 36 批运行记录](<../docs/运行记录 Run Record/2026-11-13 第36批 等预算扫描顺序对照.md>)

第 37 批的 `--source35`、`--source36` 指向两批完整输出，`--out` 指定新目录。`batch37_shared_budget.py` 用 `--primary` 读取主对照，另测总规则数相同的文件。独立解码入口为 `adaptive_bpe_file.py <archive.mbpa> <raw_out>`，输出按行排列的 uint8 格子。 [第 37 批运行记录](<../docs/运行记录 Run Record/2026-11-13 第37批 自动选向完整文件对照.md>)

第 38 至 40 批使用 `MBPG` 研究格式。运行命令及备份输入见各批记录。独立解码入口为 `grid_bpe_file.py <archive.mbpg> <out>`，每行输出一个格子的形状、有效域和按坐标排序的整数值。 [第 38 批](<../docs/运行记录 Run Record/2026-11-13 第38批 原始像素完整文件对照.md>) · [第 39 批](<../docs/运行记录 Run Record/2026-11-14 第39批 固定切词与统一位宽.md>) · [第 40 批](<../docs/运行记录 Run Record/2026-11-14 第40批 任意形状文件与线性对照.md>)
