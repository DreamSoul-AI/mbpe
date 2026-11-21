# MBPE 项目约定

## 范围与入口

MBPE 接续既有多维 BPE 研究。当前工程入口为 `src/mbpe`，这是本轮整理采用的工程基线，不表示已选定最终算法或已完成科学验证。先读 README 和 docs 中的设计、开发计划与活动缺陷。

参考 [RPipe](https://github.com/DreamSoul-AI/RPipe) 的核心包、studies 分区及 structure / flow 职责分离。可安装实现放在 `src/mbpe`，批次脚本、官方基线包装和候选原型放在 `studies/`。RPipe 仅作为结构参考。

`src/legacy/` 保存历史实现与示例，迁移只调整位置。旧源码迁移映射及批次入口的路径对应见开发记录。

## 资料与执行

2026年11月11日 Enmao 最新范围覆盖下文旧 setup 边界：当前线程继续 CPU BPE 研究，Cursor 批次停止，不开启、不催促。只做原始像素/已有整数格子切词，完全不用 GPU，不做量化模型编码、神经网络训练、FID 或下游生成。官方代码和 legacy 只读，包装脚本可负责数据与格式。实验最低优先级，CPU线程≤8、数据worker≤4。Enmao随后明确取消“必须使用Grok临时目录”的要求，临时及实验输出按项目实际配置，已启动试验保留现有路径，不据此移动或清理原件。不写 C 盘或 DreamSoul 资料库，不在 D 盘根目录新建目录，不结束他人进程。提交、推送、超大下载或 GPU 使用需本人决定。

从会话附加目录定位 DreamSoul 资料库，按任务读其研发规范及 `研发/研究/BPE/` 的现有成果。从附加 RPipe 目录阅读结构参考。workspace 研发模板是共用环境约定来源，无需作为本项目运行目录或依赖。

两篇相关论文是用户此前找到的相近工作，链接及实际阅读深度见文献调研。历史 Soft-Count 材料不代表本项目已采用该路线。

默认验证为离线配置与小型 CPU 单元检查。本轮不启动训练、下载模型或数据、提交推送或发布。下一阶段真实实验先明确目标、输入、环境、资源预算及输出位置。数据、模型、缓存和运行结果放源码树外，路径由实际配置提供。

## 分支与持续集成

`main` 保存已验收阶段，`dev` 汇集日常开发。任务从最新 `dev` 建立 `codex/<主题>` 分支，通过 PR 合入 `dev`，阶段验收后再用 PR 合入 `main`。合并完成后同步长期分支，确认所有提交已被包含再清理任务分支。保留用户指定的未合并工作。

GitHub Actions 的 `CPU checks` 在目标为这两条长期分支的 PR 及两条分支更新时运行。采用 Python 3.11、项目固定依赖、Windows CPU runner，执行安装依赖检查、配置入口与离线回归。CI 原件从 GitHub Actions 和 PR 检查页定位。当前实验仍按用户确认的 CPU 资源范围执行。

## 文档

从[文档导航](docs/README.md)定位项目资料。通用和文献调研放在 `docs/调研 Search/`，当前研究问题、假设和实验方案归研究计划。

docs 采用 workspace 研发模板的文档角色与命名。工程设计、活动缺陷与研究问题分别维护。研究尚处于接续阶段，研究计划只记录待明确的问题及形成实验协议的条件，不能把候选方向写成已确认假设。

- `docs/研究报告 Research Report/` 按日期汇总跨批次的研究收获、证据边界和论文方向，引用具体研究记录及运行证据。
- `docs/研究记录 Research Record/` 按日期记录证据支持的研究结论、适用条件和未决问题。按主题从[研究记录索引](<docs/研究记录 Research Record/README.md>)查找。旧版研究计划存入其中的 `计划存档 Plan Archive/`。
- `docs/运行记录 Run Record/` 保存实验的实际输入、配置、命令、执行经过、失败和原始输出。
- `docs/开发记录 Development Record/` 保存代码实现、重构和工程验证的经过。
- `docs/经验 Learnings/` 归纳尚待采用的做法和适用条件，已进入正式约定的内容从队列清理。

研究结论的目录与归档已由 [workspace 研发模板](https://github.com/DreamSoul-AI/workspace/blob/main/templates/software-engineering/AGENTS.template.md) 采用。本项目的阶段结论先用两到三个核心研究问题总览，明确比较对象和已测条件，先回答结果或尚未验证，再说明证据与适用范围。跨批次报告用核心问题串联收获，关键数字链接回对应研究记录和运行证据。
- 单项更新：[Workspace 65657c5](https://github.com/DreamSoul-AI/workspace/tree/65657c50cf6d98ac1898ae0183cb9ca98498e0a9)。采用旧对话重读、新对话定位、文档主动补链接、浏览器收尾、飞书专节及 CLI 任务临时目录、共享资源主动检查；保留项目专有约定和原整版基线。共享 data/models 直接复用，仅 artifacts 按仓库隔离，接入示例见 [Workspace 教程](https://github.com/DreamSoul-AI/workspace/blob/main/templates/software-engineering/examples/shared-resources/README.md)。
- 单项更新：[Workspace f01fab8](https://github.com/DreamSoul-AI/workspace/tree/f01fab8598f570a45eca8320e546d636268b700a)。采用通用文档规范及飞书/GitHub 平台章节入口、首次接入默认值和已有选择保留、代码架构设计与代码结构设计并列命名；图表简洁要求随通用规范读取。保留本项目专有约定、既有文档路径及原整版基线。[规范分发副本](https://github.com/DreamSoul-AI/workspace/blob/f01fab8598f570a45eca8320e546d636268b700a/%E6%96%87%E6%A1%A3/%E8%A7%84%E8%8C%83/%E9%80%9A%E7%94%A8/%E9%80%9A%E7%94%A8%E6%96%87%E6%A1%A3%E8%A7%84%E8%8C%83.md)。
