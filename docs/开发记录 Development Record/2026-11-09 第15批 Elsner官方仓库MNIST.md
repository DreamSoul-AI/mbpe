# 2026-11-09 第15批 Elsner 官方仓库 MNIST

Enmao 2026-11-09 23:27 批准下载 Elsner 官方仓库并先跑 MNIST 示例，也批准之后用家里的 GPU。本批只做了 Elsner 这一步。Zhang 的代码、量化模型、底模和图片没有下载。没有改官方代码，没有提交。

## 仓库

克隆到 `third_party/MDBPE_TF`。commit 是 `4905016226d71b909b3d3fd4f6521ca36a26d4e8`，时间 2025-07-12 20:24 +0200，说明是 `final`，分支 `main`。`.gitignore` 增加了 `third_party/` 和 `checkpoints/`。`data` 本来就忽略。

官方仓库没有 `requirements.txt`。README 只说把 `generator.ipynb` 复制进去执行。笔记本用的是 PyTorch，不是 TensorFlow。读代码时看到的依赖是 torch、torchvision、numpy、matplotlib、tqdm。C++ 预处理在 `MDBPE/`，要另外编译，README 说超过 12×12 再用它。本批没有编译。

## 环境

Python 用本机 Anaconda 3.13.9 建了 `third_party/venv_mdbpe`。装的是 torch 2.14.1+cu130、torchvision 0.29.1+cu130。pip 缓存、临时目录放在 `D:\Grok Bot\tmp`。下载前 D 盘剩余约 650 GB。

启动前 `nvidia-smi`：RTX 5090 D v2，显存 24455 MiB，已用约 2631 MiB，占用者都是桌面程序，没有计算任务，利用率 0%。因此用了 GPU。`torch.cuda.is_available()` 为 True。训练时利用率大约 30% 到 62%，功耗大约 200 W 到 330 W。不是 TensorFlow，所以不存在“Windows 上 TensorFlow 不能用 GPU”这一路。

图用 `MPLBACKEND=Agg`，没有弹窗。官方笔记本没有改。

MNIST 由笔记本自己下载。路径写死是 `./data`，工作目录必须是官方仓库，否则 `tokensequences/` 会写到别的地方。文件在 `third_party/MDBPE_TF/data`，约 66.5 MB、8 个文件，仍在本仓库里。没有改官方代码把它挪到仓库根的 `data/`。

## 示例实际在跑什么

`generator.ipynb` 开头写明：这是少量样本、少量压缩的演示，为了跑得快，不是最优设定。

| 项 | 笔记本 | 论文表 2 的 MNIST 行 |
| --- | --- | --- |
| 分辨率 | 缩到 12×12 | 28×28 灰度 |
| 额外 token | 0（`256+0`） | 256 |
| 图片数量 | 先转写 1000 张；训练数据集里写死最多大约 129 张 | 论文写 MNIST 每类 50 张用于 FID，训练集规模在附录，本批没有逐项抄 |
| 训练 | 1000 个 epoch，每个 epoch 把这约 129 张再重复 100 遍，batch 16，AdamW，学习率 0.0001，8 层、宽度 512 | 论文用 SFAdam，学习率另一套，附录 F 本批没有抄全 |
| 分数 | 没有 FID，只在每一格结束时打印训练损失，并且不存模型 | FID 89.07（不加额外 token）对 4.23（加 256 个） |

表 2 另外三行更不在这个笔记本里。CIFAR-10 要 32×32 再经 VQ-VAE 到 16×16，额外 1024 个 token。SVHN 要 32×32 再经 VQ-VAE 到 8×8，额外 512 个。CelebA 要 128×128 再经 VQ-VAE 到 16×16，额外 512 个。仓库里没有这些数据的下载脚本，也没有 VQ-VAE 训练和 FID。C++ 示例 `MDBPE/src/main.cc` 仍是 12×12，额外 32 个 token，输入目录要自己准备。

## 跑出来的数字

预处理保留了 1000 个序列文件。长度最小、最大、平均都是 144，也就是 12×12 的 100%。因为额外 token 是 0，合并循环不会新增符号，序列不会变短。论文表 2 的 MNIST 压缩率在图 6 那一类里，不是这一个 100%。

训练从 2026-11-09 23:56 跑到大约 2026-11-10 01:56。执行器把这一格的最长时间设成 7200 秒。到点后这一格被中断，笔记本还没执行到打印损失，模型也没有写入文件，所以没有损失，没有 FID，也没有生成图可以留下。不能和 89.07 或 4.23 相减。差多少：整行都缺，不是差一个数。

可能原因：官方演示和表 2 不是同一次实验；官方代码把 1000 个 epoch 放在一格里，中途不保存；本批按两小时上限停了。缺的是表 2 的分辨率、256 个额外 token、完整训练设定、FID 脚本，以及一份能续跑的模型文件。

## 怎么续跑

官方代码没有存档点。续跑就是再执行一次 `generator.ipynb`，工作目录 `third_party/MDBPE_TF`，解释器 `third_party/venv_mdbpe`，同样设置 `MPLBACKEND=Agg`，缓存仍放 `D:\Grok Bot\tmp`。已经写下的 `transcribed_data/`、`tokensequences/` 会被笔记本重新生成，不能当成训练到一半的模型。若不改官方笔记本，就得让最后那一格一次跑完；按这次的速度，两小时不够。
