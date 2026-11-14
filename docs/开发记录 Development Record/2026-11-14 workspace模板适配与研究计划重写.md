# workspace 模板适配与研究计划重写

## 同步范围与来源

2026-11-14 按用户要求先同步 MBPE 远端，再核对 workspace 最新研发模板，适配本项目约定并制定新研究计划。本次只改文档，没有启动实验、下载数据或模型、修改算法、提交或推送。

MBPE 执行 `git fetch origin --prune`，随后在 `development` 执行 `git merge --ff-only origin/development`，返回 Already up to date。本地与远端 main/development 基线为 `a3d6e0d7a697b7e9aa856ced210857a3b9f49b04`。从 development 建立 `codex/research-plan-update`。开始时已有 `.gitignore` 的 `/.local/` 修改，保留原状，不纳入本次文档更新。

workspace 通过正确仓库目录下的 `git ls-remote origin refs/heads/main` 核对远端为 `5e4f7a3b5bada5c74f04b89854e05e402a685df8`，与本地已存在的 HEAD 对象一致。消费该已提交快照，没有在 workspace 修改文件、pull、提交或推送。研发模板与本地文件规范化换行后一致，首次字节比较不一致来自换行差异。读取内容包括根约定、项目维护入口、模板使用方案、采用进度、研发 AGENTS 和 REPO 模板，以及本任务相关研究、文献与 GitHub 文档规范。

直接来源：[研发 AGENTS 模板](https://github.com/DreamSoul-AI/workspace/blob/5e4f7a3b5bada5c74f04b89854e05e402a685df8/templates/software-engineering/AGENTS.template.md) · [模板采用进度](https://github.com/DreamSoul-AI/workspace/blob/5e4f7a3b5bada5c74f04b89854e05e402a685df8/docs/%E6%A8%A1%E6%9D%BF%E9%87%87%E7%94%A8%E8%BF%9B%E5%BA%A6.md)

## 实际采用与保留

| 来源变化 | MBPE 适配 |
| --- | --- |
| 长时后台作业保存任务标识、日志、检查点并在恢复时核对进度 | 加入 AGENTS 执行边界；本次不建立作业或调度系统 |
| 主动核对经验，沿用 Experience 或等价入口，避免重复建议 | AGENTS 同步正文，REPO 明确沿用已有 Learnings 主题文件和导航，不额外建重复清单 |
| 研究记录保存结论，研究报告综合判断，运行与实现另有职责 | AGENTS 第八章同步；既有目录已符合，标注本项目归档经验已被 workspace 部分采纳，保留本地补充 |
| 模板原文使用多个 H1 | 本项目继续单 H1、正文 H2/H3，遵守已有 GitHub 适配，不机械覆盖标题 |

CPU 限制、官方代码和 legacy 只读、RPipe 结构参照、development 集成流程及现有 CI 均保留。REPO 移除过时的“研究只记录待明确问题”阶段描述，明确当前可制定假设与协议，执行仍以本次授权为准。输入位置及输入张量作为研究主指标，文件大小和模型效果分开判断。

## 研究计划的实质调整

使用 research-plan 技能，沿已有结果和当前约束重写[研究计划](<../计划 Plan/研究计划 Research Plan MBPE.md>)。旧计划保存在[计划存档](<../计划 Plan/计划存档 Plan Archive/2026-11-14 输入位置研究计划重写前版本.md>)，内部链接按新位置调整。已完成批次不重跑。

新计划用三问组织：等容量下是否稳定多省位置；二维是否超出差分和强一维；任意形状如何保持完整可逆。主比较用满 192 维，固定分组计入 RGB 原字节，BPE 计入 9 位编号及逐通道计数。下一步拟用独立 RGB 清单，而非把反复观察的旧测试集冒充新验证；补上批次填充、逐图扩张及失效条件。

定向补查差分 Re-Pair、PNG 过滤和 MEGABYTE，登记实际阅读深度，不把已有组合称作新颖方法。全文比较和模型效果仍未验证。文献正文、大纲、论文索引相应更新；开发计划退出历史队列，单列下一阶段的实现合同，旧版由 Git 固定版本追溯。架构和代码结构仅补当前残差/打包脚本及待实现边界，未移动源码。

## 验证与未执行项

本机静态检查覆盖本次 14 份新增/修改 Markdown，UTF-8 可解码、每篇单 H1、标题不跳级、142 条仓内链接均可定位，`git diff --check` 通过。检查入口为 Git 忽略的 `tmp/2026-11-14-plan-sync/check_docs.py`，仅供本机复核；链接检查验证路径存在，不冒充 GitHub 页面视觉检查或外链权限验证。对模板正文按 GitHub 标题层级适配后重新比较，不引入其他共用条目差异。

文档改动不新增算法测试，不重跑历史实验。新 E1–E3 均为待执行；本次分支没有远端 CI 运行，也没有发布。
