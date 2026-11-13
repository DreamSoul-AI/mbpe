# WikiText-2 BPE Huffman 对照

**同一批 BPE token 接 Huffman 后，完整文件实省 67.17%。** 比自带词表的定宽文件少 2,108,372 字节（32.43%），比发布版 ZIP 少 329,334 字节（6.97%）。独立解码逐字节还原了训练、验证、测试三份文本。

## 同输入的实际文件大小

| 存法 | 完整文件字节 | 相比原始 UTF-8 节省 |
| --- | ---: | ---: |
| 原始 UTF-8 | 13,378,183 | 0% |
| BPE 16 位定宽，带词表 | 6,500,683 | 51.41% |
| BPE 加 Huffman，带词表与码表 | 4,392,311 | 67.17% |
| WikiText-2 发布版 ZIP | 4,721,645 | 64.71% |
| 同三份原文重新打包，ZIP Deflate 级别 9 | 4,710,055 | 64.79% |

Huffman 比定宽多省 15.76 个百分点，比重打的 ZIP 小 317,744 字节（6.75%）。两个 ZIP 都使用 Deflate 压缩文本。重打版用 Python 3.13.9 的 `zipfile` 和 zlib 1.3.1，将同样的三个文本作为三个文件项写入，读回逐字节核对。发布版另有一个目录项。

## 固定 token 与计费

从上次实际落盘的 `wikitext-2.bpe` 读取 40,127 项解码词表和三个编号流。复用文件中已确定的编号，保持全部 3,079,839 个 token 的内容与顺序。解码词表原样复制。按三份待压缩文本中的 token 总频率生成一份规范 Huffman 码表，码长按编号写入，每项一字节，解码时从码长重建码字。这是整包压缩，压缩器可以统计整包数据。 [定宽运行记录](<2026-11-14 WikiText BPE 定宽落盘.md>)

这批 token 由千问 Qwen3-0.6B 自带分词器生成。后续在同一测试集补做了 WikiText-2 训练集学习的 BPE 词表对照。 [千问与本地 BPE 对照](<2026-11-14 WikiText 千问与本地 BPE 对照.md>)

| 部分 | 字节 |
| --- | ---: |
| 解码词表（含词条数和长度字段） | 340,990 |
| Huffman 码表 | 40,127 |
| 格式标识和三个块头 | 28 |
| 编码数据（含 11 位补齐） | 4,011,166 |
| 完整文件 | 4,392,311 |

定宽编号为 16 位。Huffman 数据部分平均为 10.42 位/token，另计上表的词表、码表与文件头。三份文本共用一套表，数字适用于整批打包及这里实测的两个 ZIP。

## 运行与验收

2026-11-14 01:09（Asia/Shanghai），沿用仓库的 Python 3.13.9。编码和独立解码用一个逻辑核、Idle 优先级和零数据加载 worker。编码和进程内验收用时 7.30 秒。程序回读 Huffman 文件，核对词表、逐项 token 及原文 SHA-256。随后启动独立进程，仅用 Huffman 文件还原 10,940,747、1,146,846、1,290,590 字节，分别和发布版 ZIP 的原始训练、验证、测试文本逐字节一致。现有编码文件回归测试 8 项通过。

从仓库根运行，`--out` 和解码输出使用尚不存在的目录。 [实验脚本](../../studies/wikitext_huffman_file.py)

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
$env:TMP = (Resolve-Path -LiteralPath 'tmp').Path
$env:TEMP = $env:TMP
$python = './third_party/venv_mdbpe/Scripts/python.exe'
$out = './tmp/2026-11-14-wikitext-huffman'
& $python -B -X utf8 './studies/wikitext_huffman_file.py' measure --source './tmp/2026-11-14-wikitext-fixed-bpe' --archive './tmp/wikitext-2-raw-v1.zip' --out $out
& $python -B -X utf8 './studies/wikitext_huffman_file.py' decode "$out/wikitext-2.huffman" "$out/decoded"
```

运行原件在 `tmp/2026-11-14-wikitext-huffman/`。`report.json` 与 `zip-comparison.json` 分别记录 Huffman 和 ZIP 的字节分项及校验值。Huffman 文件 SHA-256 为 `661a443c6b66951ccfd36b5886af2716e8cdb3eb919a822821f51c4cfa9f14fe`。
