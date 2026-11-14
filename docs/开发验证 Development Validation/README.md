# MBPE 开发验证入口

## 合同与活动问题

[缺陷记录](<缺陷记录 Bug Report MBPE.md>)保存尚未验收的问题；官方实现的观察不因外部包装成功而自动关闭。当前设计从[设计入口](<../设计 Design/README.md>)查找，工程待办从[开发计划](<../计划 Plan/开发计划 Development Plan MBPE.md>)查找。

## 真实验证入口

| 入口 | 已有证据与范围 |
| --- | --- |
| [离线回归](../../tests/run.py) | 覆盖核心 structure/flow 与 studies。2026-11-14 本机 63 项通过，环境和首次临时目录失败见[实现记录](<../开发记录 Development Record/2026-11-14 RGB等容量输入实现与CPU验证.md>)。2026-11-15 新增组批与一维适配后，本机 69 项离线回归通过，日志保存在 `tmp/2026-11-15-autonomous/offline-tests.log` |
| [CPU CI 工作流](../../.github/workflows/cpu-tests.yml) | Windows、Python 3.11 与项目固定依赖。当前未提交改动没有新的远端 CI 结果，不能把本机 3.13 环境结果当作支持版本安装验收 |
| [配置预览](../../src/mbpe/flow/cli.py) | `python -m mbpe plan` 只预览，不能代替实际实验 |
| [RGB 独立输入检查](../../studies/rgb_capacity_study.py) | 固定 RGB 编解码与保存输入的独立还原；[已执行证据](<../运行记录 Run Record/2026-11-14 RGB独立样本等容量输入对照.md>) |

## 怎样分清通过状态

2026-11-15 阶段提交前再次运行完整离线回归，70 项通过，包括“无 SciPy 时仍能导入 RGB 选样模块”的新增检查。SciPy 仅在实际读取 SVHN MAT 时加载；研究环境结果与 Python 3.11 固定依赖 CI 分开验收。提交前原件在 `tmp/2026-11-15-wrapup/offline-tests.log`。

代码回归检验实现合同；运行记录保存实际执行；研究记录判断假设。单元测试通过不代表论文主张成立，CPU BF16 往返不代表 GPU 峰值或模型任务有效。研究结论见[结论索引](<../研究记录 Research Record/README.md>)。
