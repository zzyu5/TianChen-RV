# Trellis 卫生纪律

本文件规定 spec、task、issue、报告、代码和实验工件的住址。Trellis 是工具，不是开工许可系统。

## 一、spec 是活契约

- 更新原文件，不创建同义副本。
- layer index 持有版本号；用户裁决导致稳定条文变化时递增版本。
- spec 写稳定契约，不写短期排期、当前覆盖数字或临时战役状态。
- 当前事实放在代码、测试、主表、ledger 或 issue 中，并由 spec 链接。
- 跨文件引用使用相对链接、标题锚或稳定条目编号，不使用行号。
- 历史教训压缩成当前规则；事故原件放 .trellis/事故档案。

## 二、task 是可选计划

推荐 task 的条件：

- 多阶段、跨多日；
- 多 agent/worktree；
- 跨层接口或大范围迁移；
- 用户要求任务树；
- 正式实验 campaign。

不强制 task 的条件：

- 用户直接要求的单阶段修改；
- 小型、可逆、范围清楚的实现；
- 文档/spec 同步；
- 只读诊断；
- 当前对话中的计划已经足够。

task 不授权修改，不覆盖用户范围，也不成为代码或实验事实。

## 三、实验预注册

正式性能或硬件实验仍需在测量前固定：

- 研究问题；
- 行键和 capability instance；
- 候选与对手；
- 正确性判据；
- 重复数和统计口径；
- 允许的写入目的地；
- 失败和 VOID 处理。

预注册可以住：

- Trellis task 的 prd；
- experiments 下具名 campaign plan；
- runner 可校验的 manifest。

是否使用 task 不影响预注册要求；关键是预注册早于测量且可追溯。

## 四、报告与数据

- 设计规则住 spec。
- 问题住 issues。
- 当前项目报告可以作为 task 附件，也可以住已有 ledger/result-table 体系。
- 不在仓内 docs 新建第二套治理正本。
- 测量数据只按 measurement 的目的地契约写入 experiments/master、experiments/runs 和 experiments/runs.log。
- 工具、runner 和 harness 住 tools，不把脚本混入数据目录。

## 五、归档

归档前依次判断：

1. 是否仍是代码、测试、spec、issue、主表、runner、gate 或现役证据可达对象；
2. 是否属于事故档案；
3. 是否只是已被取代的历史材料。

现役 authority 不进 _attic。_attic 是不可作为克隆后权威锚的历史区。

守护对象已经消失的 gate 应退役，不把旧门改造为新门继续存活。新契约需要新测试时，测试应直接保护新 authority。

## 六、并行与暂存

- 有并行线时精确暂存路径，禁止笼统吸入他人修改。
- 无并行线时仍建议精确暂存，便于审计提交范围。
- worktree 不自动保证逻辑隔离；公共 schema、接口和 spec 正本由单一整合者修改。

## 七、登记簿计数

issue 数、状态分布、缺号和重号由 .trellis/scripts/issues_census.py 机算。

- 禁凭记忆手改总数。
- 更新 issue 状态后先运行 census，再更新索引摘要。
- 若索引与脚本不一致，以脚本输出为事实并修索引。

## 八、删除入口文件

根 README 与 .trellis/spec/index.md 已足够提供上岗入口，因此不需要再维护一个重复的根 AGENTS.md。

删除 AGENTS.md 不删除项目规范；它消除的是重复、容易漂移的代理专用入口。任何工具若需要说明，应直接链接 README 或 spec 根地图。
