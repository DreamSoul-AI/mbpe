# WikiText-2 BPE 定宽落盘

**WikiText-2 原始文本用 BPE 定宽存储，全量实省 48.20%。** 三份编号文件合计从 13,378,183 字节降到 6,929,654 字节，前提是读写双方共享同一份词表。把这批实际用到的词条放进文件并重新编号，完整文件实省 51.41%。两种方式都逐字节还原了训练、验证和测试三份文本。

## 数据与测法

采用 Salesforce 发布的 `wikitext-2-raw-v1.zip`，共 4,721,645 字节。运行输入取自 ggml-org 的镜像，实测 SHA-256 为 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`，与 Salesforce 归档值相同。逐份读取 ZIP 中的 `wiki.train.raw`、`wiki.valid.raw` 和 `wiki.test.raw`，保留 UTF-8 字节和换行。 [Salesforce 数据校验值](https://huggingface.co/datasets/Salesforce/wikitext/blob/c8a1c90b2b6554c017d92ca208fc0c03460685be/dataset_infos.json#L299) · [归档镜像](https://huggingface.co/datasets/ggml-org/ci/blob/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip)

沿用 [中文 Markdown 实测](<2026-11-14 MNIST BPE 定宽落盘对照.md#文本定宽存储示例>) 中的 Qwen3-0.6B 分词器。词表固定，每份原文整体切词，生成 18 位定宽编号。每份文件含 5 字节头和字节对齐补位。对每份编号文件，读回后用相同词表解码，再逐字节比对输入。

| WikiText-2 划分 | 原始 UTF-8 字节 | BPE token 数 | 18 位编号文件字节 | 节省 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 10,940,747 | 2,518,423 | 5,666,457 | 48.21% |
| 验证 | 1,146,846 | 262,338 | 590,266 | 48.53% |
| 测试 | 1,290,590 | 299,078 | 672,931 | 47.86% |
| 合计 | 13,378,183 | 3,079,839 | 6,929,654 | 48.20% |

每 100 个原始字节平均切成约 23 个 token。每个编号占 18 位，因此编号文件实际占原文的 51.80%。

## 把词表也算进文件

- **完整分词器原样附上。** 原样 `tokenizer.json` 为 11,422,654 字节，与三份编号文件相加为 18,352,308 字节，比原文大 37.18%。
- **附上用于解码的词条。** 同一批 token 共用了 40,127 种编号。把它们按原编号排序并从零重编，编号用 16 位。共享解码表含长度字段共 340,990 字节，三份编号含文件头共 6,159,693 字节，合计 6,500,683 字节，比原文小 51.41%。独立进程只读这份 `wikitext-2.bpe` 文件就还原了三份文本。
- **同一份数据的发布 ZIP。** ZIP 为 4,721,645 字节，比三份原始文本共节省 64.71%。

定宽文件只用编号和解码表。这些数字适用于这三份 WikiText-2 原始文本和这份 Qwen 词表。解码表按三份文件合并计算，每份单独封装的成本仍待测。

同一份定宽文件也作为 Huffman 对照的输入。 [Huffman 对照记录](<2026-11-14 WikiText BPE Huffman 对照.md>)

## 运行与复现

2026-11-14 01:02（Asia/Shanghai），Python 3.13.9、`tokenizers` 0.22.2。用一个 CPU 逻辑核、Idle 优先级和零数据加载 worker，测量用时 12.58 秒。初次在进程启动时设置 `PYTHONPATH`，解释器退出码为 1，未写输出。正式运行在解释器启动后追加已有库目录，执行完成。

从仓库根运行。 `--out` 和解码输出用尚不存在的目录。 [实测脚本](../../studies/wikitext_fixed_file.py)

```powershell
$env:TOKENIZERS_PARALLELISM = 'false'
$env:RAYON_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$env:TMP = (Resolve-Path -LiteralPath 'tmp').Path
$env:TEMP = $env:TMP
$python = './third_party/venv_mdbpe/Scripts/python.exe'
$bootstrap = "import runpy, sys; sys.argv.pop(0); sys.path.insert(0, 'studies'); sys.path.append('D:/anaconda3/Lib/site-packages'); runpy.run_path(sys.argv[0], run_name='__main__')"
$out = './tmp/2026-11-14-wikitext-fixed-bpe'
& $python -B -X utf8 -c $bootstrap ./studies/wikitext_fixed_file.py measure --archive ./tmp/wikitext-2-raw-v1.zip --tokenizer 'D:/models/Qwen3-0.6B/tokenizer.json' --out $out
& $python -B -X utf8 ./studies/wikitext_fixed_file.py decode "$out/wikitext-2.bpe" "$out/decoded"
```

原件保存在仓库本地 `tmp/2026-11-14-wikitext-fixed-bpe/`。`report.json` 保存各份文本与编号文件的校验值。独立解码写出的三个文件逐字节等于原始 ZIP 中对应文件。 `wikitext-2.bpe` SHA-256 为 `a281faf48e65aeb7c40d1cc9256b8ae60e41619e51ca8339c60e0b8f98748b96`。
