# MBPE 文档导航

最新主口径见[文本与图像的序列长度对照](<研究记录 Research Record/2026-11-15 文本与图像的序列长度对照.md>)：一个token一个embedding，文件字节单列。

先读[研究统合报告](<研究报告 Research Report/2026-11-14 MBPE 研究统合报告.md>)了解已知结果，再从[研究计划](<计划 Plan/研究计划 Research Plan MBPE.md>)看下一步。最新关键结论是：[有限窗口能降低累计输入，最大单批仍未改善](<研究记录 Research Record/2026-11-15 有限窗口收益与二维独立作用.md>)；强一维对照以同页的实际完成状态为准。

## 按问题查资料

| 要找什么 | 入口与职责 |
| --- | --- |
| 项目与工作约定 | [项目说明](../README.md) · [REPO](../REPO.md) · [研究脚本](../studies/README.md) |
| 机制、工程与前人研究 | [调研 Search](<调研 Search/README.md>)：通用调研与文献调研各有正文 |
| 系统职责与源码组织 | [设计 Design](<设计 Design/README.md>)：架构设计、代码结构设计 |
| 接下来验证什么、实现什么 | [计划 Plan](<计划 Plan/README.md>)：研究计划、开发计划及历史存档 |
| 实现是否符合合同、还有什么问题 | [开发验证 Development Validation](<开发验证 Development Validation/README.md>)：实际测试/CI 入口及活动缺陷 |
| 证据支持哪些判断 | [研究记录 Research Record](<研究记录 Research Record/README.md>)：结论、条件和未验证项 |
| 跨实验收获 | [研究报告 Research Report](<研究报告 Research Report/>)：按日期统合结论，链接研究记录 |
| 改了哪些代码或文档结构 | [开发记录 Development Record](<开发记录 Development Record/>)：按日期追溯实现与演进 |
| 实际跑过什么 | [运行记录 Run Record](<运行记录 Run Record/>)：环境、命令、结果、失败及原始回执 |
| 可复用做法 | [经验 Learnings](<经验 Learnings/>)：条件、依据与 workspace 采纳状态 |

## 最近的接续入口

- [有限窗口与强一维运行证据](<运行记录 Run Record/2026-11-15 有限窗口组批与强一维CPU对照.md>) · [对应实现](<开发记录 Development Record/2026-11-15 有限窗口组批与强一维接口.md>)

- [2026-11-15 文档分区与研究计划重写](<开发记录 Development Record/2026-11-15 文档分区与研究计划重写.md>)
- [RGB 等容量输入实现与 CPU 验证](<开发记录 Development Record/2026-11-14 RGB等容量输入实现与CPU验证.md>)
- [独立 RGB 原始运行证据](<运行记录 Run Record/2026-11-14 RGB独立样本等容量输入对照.md>)
- [旧 CPU 交接](<开发记录 Development Record/2026-11-12 MBPE-CPU交接给Codex.md>)：保留当时状态，当前安排以研究计划为准
- [历史方案和旧计划](<计划 Plan/计划存档 Plan Archive/README.md>)：不作为当前待办或执行授权

目录与日期记录保留历史事实；导航只展示当前入口和最近接续，不再重复铺满所有批次。运行证据、研究结论、综合报告保持各自分工。
