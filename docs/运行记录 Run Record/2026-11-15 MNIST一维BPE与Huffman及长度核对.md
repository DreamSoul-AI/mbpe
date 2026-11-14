# 2026-11-15 MNIST一维BPE与Huffman及长度核对

## 输入、规则与协议

按用户要求补 MNIST 按行一维 BPE+Huffman，并以 sequence 长度对照已有文本与图像结果。使用本机官方 IDX 原始8位灰度，不缩放、不二值化、不量化。学习使用训练源每类前50张、共500张，与旧Elsner二维相同选样规则。验证为排除学习索引及重复内容后的每类100张，共1,000张；测试为完整官方10,000张。学习图像与测试图像内容摘要交集为0。

按行一维使用256个基本符号、256条合并规则，复用既有精确相邻对循环；不搜索扫描方式。Huffman码长只由验证序列加一平滑统计，在处理测试指标前封存。文件使用现有MBPG格式，保存全部字母表、规则、码表、逐图外框与块头；不省略必要解码信息。原始测试数组在启动时加载与校验，规则/码表未读取其内容选择，封存后才编码和计算测试指标。

## 实际结果

| 项目 | 实测 |
| --- | ---: |
| 原始测试像素/位置 | 7,840,000 |
| 一维BPE token总数 | 1,422,786 |
| 每图平均token | 142.2786 |
| 剩余长度比例 | 18.1478% |
| 长度减少 | 81.8522% |
| Huffman完整文件字节 | 1,603,027 |
| 完整文件节省 | 79.5532% |
| 独立逐格还原 | 10,000 / 10,000 |

Huffman接收和读回相同token序列，不改变上述token总数。字母表450字节、规则578字节、码表513字节，其余头/形状/数据分项保存在报告。已有同10,000图竖拼PNG为1,701,948字节，原IDX gzip为1,648,877字节。这里各实际文件编码后端不同，不能用文件差单独解释一维相对二维的几何作用；长度比较另列。

## 环境、命令与验证

CPU Idle、CPU0、每库线程1、worker0；Python3.13.9、NumPy2.5.4。源IDX及全部研究代码快照SHA256已登记，官方与legacy只读，没有GPU或下载。本次耗时21.52秒，主进程峰值RSS 216.60 MiB。测试编码后再次按规则逆展开逐像素比对，再由独立进程仅从完整Huffman文件恢复形状、域、原始灰度，逐字节核对完整网格序列。

```powershell
$out = './tmp/2026-11-15-mnist-linear-huffman'
& './third_party/venv_mdbpe/Scripts/python.exe' -B -X utf8 "$out/source/mnist_linear_huffman_study.py" --raw './third_party/MDBPE_TF/data/MNIST/raw' --out $out
```

命令采用已冻结源码，输出拒绝覆盖。首次运行前建立新输出/source并复制studies源文件，report.json登记快照哈希。入口为[MNIST研究脚本](../../studies/mnist_linear_huffman_study.py)，复用现有相邻对与MBPG编解码，没有改公共codec。本轮另执行2项MBPG定向回归，均通过，日志在输出目录；没有重新跑整个离线套件，不冒称新增完整回归或CI结果。

## 长度口径与附加核对

WikiText-2沿用历史测试token：Qwen299,078，本地大词表282,198。本次读取原ZIP测试文本，核对字节摘要并计得1,288,556个Unicode字符；不是用UTF-8字节冒充字符。其他划分的字符数也保存在text-lengths.json，没有重新训练或切词。

最新RGB序列沿用E2原件，对差分按列两套各1,000张再核对规则展开长度累计到1,024时的通道边界；不读取计数头也能推回同样的三段token数，全部相同。原token帧此前已独立逐格无损还原，这次没有重学RGB规则。该核对只覆盖已知3×32×32与固定RGB顺序，不是变形状元信息免费。

原件在 `tmp/2026-11-15-mnist-linear-huffman/`：protocol.json、selected-before-test.json、report.json、mnist-row-256.huffman.mbpg及独立decoded、text-lengths.json、rgb-implicit-boundaries.json、codec-tests.log和源码快照。新结果按[长度结论](<../研究记录 Research Record/2026-11-15 文本与图像的序列长度对照.md>)解释，不是GPU实测。未提交推送。
