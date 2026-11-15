# 论文索引 Paper Index MBPE

OpenAlex 引用量查询于 2026-11-12。数量仅对应链接所指记录，预印本和会议版本可能分开登记。

| 角色 | 论文与阅读范围 | 引用量 |
| --- | --- | --- |
| 主基线 | Tim Elsner 等. [Multidimensional Byte Pair Encoding: Shortened Sequences for Improved Visual Data Generation](https://arxiv.org/html/2411.10281v1). ICCV 2025。阅读 arXiv v1 算法 1、§4.2–4.3、§5.2。公开基线 [MDBPE_TF](https://github.com/DaiDaiLoh/MDBPE_TF) 已有本地运行记录。 | [2](https://openalex.org/W4404569698)，arXiv 记录 |
| 直接相邻方法 | Wanpeng Zhang 等. [From Pixels to Tokens: Byte-Pair Encoding on Quantized Visual Modalities](https://arxiv.org/html/2410.02155v3). ICLR 2025。读取 v3 命题 2、§4.1、附录 A.1。公开仓库[迁移说明](https://github.com/BeingBeyond/Being-VL-0)指向后续快照，旧论文对应固定源码待核。 | [9](https://openalex.org/W4403885200)，arXiv 记录 |
| 一维文法基础 | N. Jesper Larsson、Alistair Moffat. [Offline Dictionary-Based Compression](https://avadeaux.net/larsson.dogma.net/dcc99.pdf). DCC 1999，§2–5。覆盖词表、序列及提前停止的代价。 | [232](https://openalex.org/W1976149088)，DCC 记录 |
| 图文法近邻 | Sebastian Maneth、Fabian Peternek. [Grammar-Based Graph Compression](https://arxiv.org/html/1704.05254v1). arXiv 2017，§4.2–4.3、§4.6。覆盖超边、出现位置索引及语法编码。 | [42](https://openalex.org/W2606952915)，arXiv 对应记录 |
| 空间打分近邻 | Wanpeng Zhang 等. [Unified Multimodal Understanding via Byte-Pair Visual Encoding](https://arxiv.org/html/2506.23639v1). ICCV 2025，§3.4、附录算法 2。覆盖方向偏移的一致性得分和多样性过滤。 | 本轮未查询 |
| 压缩目标近邻 | Mete Erdogan、Abhiram Gorle、Shubham Chandak、Mert Pilanci、Tsachy Weissman. [An Information-Theoretic Perspective on LLM Tokenizers](https://arxiv.org/html/2601.09039v1). arXiv 2026，v1 §V-B。覆盖用验证流 gzip 大小选对及其时间代价。 | 本轮未查询 |
| 差分与文法近邻 | Francisco Claude、Antonio Fariña、Gonzalo Navarro. [Re-Pair Compression of Inverted Lists](https://arxiv.org/abs/0911.3318v1). arXiv 2009；[v1 PDF](https://arxiv.org/pdf/0911.3318v1) 内页日期为 2018-07-18，与 arXiv 提交日期分开保留。2026-11-14 由摘要筛选推进到定向阅读 §2.2–2.3、§3.1–3.2，核对差分、边界和展开；实验图表、代码和后续进展未核验。 | 本轮未查询 |
| 多尺度字节建模近邻 | Lili Yu 等. [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185v2). arXiv 2023 v2。2026-11-14 仅核对摘要中的块内局部、块间全局模型，未复现。 | 本轮未查询 |
| 动态字节 patch 近邻 | Artidoro Pagnoni 等. [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/html/2412.09871v1). arXiv v1；ACL 2025 [出版记录](https://aclanthology.org/2025.acl-long.453/)。2026-11-15 定向读取 §2.1–2.4、§3.1–3.3，不混用会议版实测数字，未复现。 | 未查询；ACL 记录标注 Outstanding Paper |
| 补齐与序列打包基线 | Mario Michael Krell、Matej Kosec、Sergio P. Perez、Andrew Fitzgibbon. [Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance](https://arxiv.org/html/2107.02027v2). arXiv v2，2022。2026-11-15 定向读取 §3、附录C中排序/packing 区分，未复现模型。 | 本轮未查询 |

2026-11-15 Elsner v1 补核 §4.3–4.4 的覆盖定位及删最长样本/补齐条件。MEGABYTE 阅读推进到 v2 §2.1–2.4、§4.1、表7，以上原行保留初次阅读时间。BLT v1 HTML 页内日期与 arXiv 标识时间不一致，按固定 URL 记录阅读对象；出版元数据由 ACL 记录独立核对，不据 HTML 页内日期更改发表年份。

工程来源另见 [PNG 第三版过滤规范](https://www.w3.org/TR/2025/REC-png-3-20250624/#filtering)，本次定向读取过滤章节，不作为 BPE 新颖性或模型效果证据。上述新增条目未下载 PDF 入库。

2026-11-15 定向新增阅读：Brittain、El-Sakka的[JVCIR2007二维字典压缩](https://www.csd.uwo.ca/~melsakka/publications/journals/pdfs/2007_jvcir_Nathanael.pdf)（摘要与§2.1–2.4，未复现）；Kudo的[ACL2018 Subword Regularization](https://aclanthology.org/P18-1007.pdf)（§3.1–3.2，未复现概率训练/翻译实验）。引用量未查询。CharLS的[HP1代码](https://raw.githubusercontent.com/team-charls/charls/main/src/color_transform.hpp)作为工程机制来源，main访问于同日，未核定提交，不称为JPEG-LS规范本身。作用和限制见综述最新小节。
