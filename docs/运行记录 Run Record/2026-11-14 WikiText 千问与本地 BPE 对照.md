# WikiText-2 千问与本地 BPE 对照

**千问在这次测试里没有压得最小。** WikiText-2 训练集学出的 32,768 项词表，在同一份测试原文上用 Huffman 落盘，完整文件比千问小 17,614 字节（3.14%）。本地大词表的 token 最少，完整文件却比千问多 3,229 字节。

## 词表与输入

前一次整包节省 67.17% 的实验已用千问 Qwen3-0.6B 自带的 BPE。此次沿用同一份分词器，文件 SHA-256 为 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`，与千问发布文件相同。千问的测试序列逐项对回上次落盘的 `test.ids`。 [千问发布文件](https://huggingface.co/Qwen/Qwen3-0.6B/blob/167b810/tokenizer.json#L234) · [上次整包 Huffman 实测](<2026-11-14 WikiText BPE Huffman 对照.md>)

两套本地 BPE 只从 WikiText-2 raw 的训练划分（10,940,747 字节）学习合并规则。沿用千问的 NFC、预切分、字节映射和解码设置，起始字母表覆盖全部 256 个字节，不加特殊 token。最低合并频次设为 1。目标词表大小分别为 32,768 和 151,643，后者可合并对用尽时实际得到 106,222 项，因此实际等规模的对照未完成。

三套分词器在同一份 1,290,590 字节的测试文本上比较。每套按测试流统计 token 频率，生成该文件自己的 Huffman 码表。每个文件只写入实际用到的 token 对应的字节、码长表、12 字节文件头和编码数据，可独立解码。验证集未参与，两档词表之间的结果属于探索性比较。

## 测试结果

| 词表 | 实际基础词表项 | 测试 token 数 | 词表与码表字节 | Huffman 数据字节 | 完整文件字节 | 比测试原文省 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 千问 Qwen3-0.6B | 151,643 | 299,078 | 181,882 | 378,711 | 560,605 | 56.56% |
| WikiText-2 本地 BPE | 32,768 | 295,811 | 167,826 | 375,153 | 542,991 | 57.93% |
| WikiText-2 本地 BPE，大词表 | 106,222 | 282,198 | 208,745 | 355,077 | 563,834 | 56.31% |

本地大词表比千问少 16,880 个 token，编码数据小 23,634 字节，词表与码表大 26,863 字节。32,768 项词表的完整文件在这三种设置中最小。此处每个文件单独压缩测试集，开销只在一个划分内分摊。上次 67.17% 对应训练、验证、测试三份共用一套词表和码表。跨语料表现和千问预训练语料与 WikiText 的重叠情况未验证。

## 复现与验收

2026-11-14 01:16（Asia/Shanghai），Python 3.13.9、`tokenizers` 0.22.2。使用一个 CPU 逻辑核、Idle 优先级和零数据加载 worker。测量总计 11.69 秒。三个输出文件用独立进程解码，均逐字节对回测试原文。原整包定宽及 Huffman 文件回读通过，已有编码文件测试 8 项通过。

从仓库根运行，输出目录需尚不存在。 [对照脚本](../../studies/wikitext_tokenizer_comparison.py)

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
$out = './tmp/2026-11-14-wikitext-tokenizers'
& $python -B -X utf8 -c $bootstrap './studies/wikitext_tokenizer_comparison.py' --archive './tmp/wikitext-2-raw-v1.zip' --qwen 'D:/models/Qwen3-0.6B/tokenizer.json' --source './tmp/2026-11-14-wikitext-fixed-bpe' --out $out
foreach ($name in @('qwen3','wikitext-32768','wikitext-151643')) {
    & $python -B -X utf8 './studies/wikitext_huffman_file.py' decode "$out/$name.huffman" "$out/$name-decoded"
}
```

输出目录保存两个本地分词器、三个 Huffman 文件、独立解码原件和含逐项字节数及校验值的 `report.json`。文件名 `wikitext-151643` 中的数字表示训练目标。
