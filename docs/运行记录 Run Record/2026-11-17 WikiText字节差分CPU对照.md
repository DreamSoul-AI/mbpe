# WikiText 字节差分 CPU 对照

复用此前核对的 WikiText-2 raw 归档和 Qwen3-0.6B tokenizer。只用训练划分的 10,940,747 字节拟合本地词表。测试划分为 1,290,590 字节、1,288,556 个 Unicode 码点，此前已曝光，本次属于配对对照。

原字节与一阶差分两组各学到 151,643 项词表，再取 32,768、100,000 和 151,643 三档前缀，256 个字节原子计入词表。原文行边界相同。字节差分连续跨行，解码只需要原始 token 顺序和固定表示约定。两组均测普通 BPE 与同词表的单位成本最短切分。千问直接使用发布 tokenizer，关闭额外特殊 token。

CPU 使用单逻辑核、Idle 优先级和各库单线程，tokenizers 0.22.2。拟合、切词与写回约 48 秒。另起进程读取 ID 与词表，核对摘要并重新解码。13 个条件全部逐字节复原，完整 84 项回归通过。

[研究入口](../../studies/wikitext_delta_bpe.py)分 `run` 和 `decode` 两阶段，使用相同的 `--archive tmp/wikitext-2-raw-v1.zip`、`--qwen` 本机已核对的分词器路径与 `--out tmp/2026-11-17-wikitext-delta/results`。本机输出目录保留源码快照、日志、词表封存、ID 流、`report.json` 和 `independent-decode.json`。评估口径与完整长度见研究结论。[研究结论](<../研究记录 Research Record/2026-11-17 WikiText字节差分与最短切分.md>)
