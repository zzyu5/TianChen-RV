# Weft-RV 第一性原理重构交接

本文写给下一位接手 Weft-RV 的 agent。它记录最近一次方向纠偏：发生了什么、A/B 两条线最初要解决什么、当前已经裁决的第一性原理、哪些旧材料仍有价值，以及接下来怎样修正代码与 spec。

本文不是 Trellis task，不建立新流程，也不要求先创建计划、任务树、角色或工作流状态。`.trellis/spec/` 继续作为项目更新文档与历史参考库；`.trellis/tasks/` 只保留历史记录，不再驱动工作。接手者读完本文后即可检查当前代码并直接行动。

## 一、发生了什么

项目原本提出 A/B 两条并行主线：

- A 线处理架构：收拢散落在 construction、front door、schedule、selector 和 emitter 中的格式事实 `g`、目标能力 `c` 与代码生成公式，使知识集中、模块化、可阅读、可扩展。
- B 线处理真实性能：沿 official runner 持续推进 deployed ggml、代表性 strong opponent 和端到端三类互补证据。

A 线随后发生了系统性偏航。工作重心从“把算法知识放到一个人能看懂和修改的地方”收缩成：

```text
给现有路径增加 selected stamp
  -> 增加 materializer
  -> 增加 reader
  -> 重新运行公式并逐字段比对
  -> 增加 missing / partial / forged / stale 门禁
  -> 为这些状态增加大量负例和 fixture
```

这些工作部分修复了真实的重复决策和 silent default，但它们不是项目要追求的核心重构。尤其是把内部编译流水线当成不可信序列化边界，为每个 mechanism 建立独立属性协议，导致代码状态越来越多、接口越来越碎、新格式或新能力反而更难接入。

已确认的代表性事实：

- `RVVFormulaDecision.h` 集中了若干函数，但主要仍是多套 mechanism-specific 类型和返回对象，没有形成一个清楚的公式层代码结构。
- Codebook 路径把 plan 展开为七字段 optional stamp，并配套逐字段写入、解析和重算比对。
- A4a 修复了 unrealizable candidate、silent default 和 reason override，但主要交付形态仍被 reader/verifier/materializer 主导。
- 未合入主线的 Grid 提交 `680681d0b` 为一个当前只有单一实现、没有真实 `c` 分叉的决策增加了三字段 stamp、219 行专用 materializer、公式重跑和 forged/stale 门禁。该提交不得合入。

因此，不能继续按 A4b 把相同模式复制到 Grid、KQuant 和 Ternary，也不能再以“完成几个 stamp slice”衡量架构进度。

## 二、当前第一性原理

### 1. 从项目真正要表达的知识出发

Weft-RV 的核心问题是：格式、机制、目标能力和性能知识怎样显式决定最终实现。

当前重构应从最直接的数据流出发：

```text
格式与机制事实 g
  + 目标能力 c
  + 真实需要的静态上下文
  -> 集中的公式与算法决策
  -> 最终 Plan / typed body
  -> emitter
```

打开公式层代码时，应当能够直接看到：

- 输入用了哪些 `g`；
- 输入用了哪些 `c`；
- 候选或参数怎样构造；
- 公式为什么选择某个实现；
- 最终哪些参数进入 Plan/body；
- emitter 怎样机械实现这个结果。

这些内容不能继续散落在 pass、verifier、materializer 和 emitter 中，也不能靠 ledger 才能拼出真实算法。

### 2. 当前阶段不从 legality 和 verification 出发

本轮明确裁决：先把正确的代码结构和公式层做出来，不以 legality framework、verifier、forged/stale 防御、完整负例矩阵或 fallback 设计作为前置条件。

如果实现走不通，再依据真实失败补必要条件。不要预先为所有可能错误建立框架，也不要把“可能被伪造”当作内部 IR 的默认威胁模型。

这不表示论文数字可以不做正确性验证，也不表示硬件测量可以绕过 correctness。它只表示：**架构重构的出发点是算法和代码结构，不是门禁。** B 线产生正式性能结论时，仍然必须使用真实正确性和测量证据。

### 3. 只保留一个生产状态

不接受以下多状态叠加：

- 未选择 / 半选择 / fully stamped / verified；
- 新路径失败后回旧路径；
- missing 属性表示默认候选；
- emitter 为上游补值或重新选择；
- 新旧 helper、overload、reader、writer 长期共存；
- reason/provenance 参与 compute；
- compatibility、legacy、deprecated alias 只为迁移期存在。

最终生产链中只应存在已经决定好的 Plan/body。旧路径直接删除；做错则向前修复，不保留回退世界。

### 4. 大胆重构和删除

当前不要求“最小 diff”“渐进兼容”或“每一步维持所有旧 fixture 输入形态”。可以改变内部 API、目录、typed body 和调用链，可以删除过渡测试和旧属性。

判断重构质量时优先看：

- 是否减少代码和状态；
- 是否能在一个清楚的位置读到公式；
- 是否消除重复 owner；
- 新增格式、能力或候选是否只需修改少数相关模块；
- emitter 是否真正退化为实现 Plan，而不是另一层决策器。

### 5. 不把弱目标当贡献

“小型 plugin-local decision contract”和“让第二 extension family 复用一次”都不是当前目标。它们最多是未来观察架构质量的现象，不能反过来限制重构，也不能作为完成标准。

同样，不需要预先设计通用 Formula IR、DSL 或跨插件最低公分母接口。先让项目中的真实公式集中、清楚、可扩展；共性应从真实代码中出现，而不是由 task 名称预设。

## 三、A 线现在应该做什么

A 线仍然存在，但含义已经恢复为最初的架构主线，而不是 stamp 主线。

### A 线目标

把当前所有影响代码生成的 `g`、`c` 和公式消费集中起来，形成一个清楚、模块化、可阅读的算法层；然后让 typed body 与 emitter 直接消费最终结果。

### A 线应推进的工作

- 从当前代码事实重新识别真正影响代码生成的格式事实和能力事实。
- 找出同一决策在 front door、schedule、selector、verifier、conversion 和 emitter 中的重复实现。
- 把公式、候选和参数推导集中到少数清楚的模块中；模块边界按真实知识组织，不按历史 task 切片组织。
- 让 `g` 与 `c` 在公式处直接相遇并生成最终 Plan/body。
- 让 emitter 只读取最终 Plan/body，不读取 format、board、march、measurement、reason 或旧 metadata 重新决定代码。
- 删除旧 provider、旧 selector、旧 materializer、旧 reader、默认值路径、属性镜像、过渡 overload 和只服务于旧协议的测试。
- 对只有一个确定实现的机制直接构造 Plan，不虚构 candidates、selection 或 selected stamp。
- 当真实代码证明某个条件不可缺少时，再把条件放到公式附近；不要先建设独立 legality/verification 体系。

### A 线不再做的事

- 不继续 A4b 的 Grid/KQuant/Ternary complete-stamp 扩散。
- 不以 forged/stale/missing 门禁数作为进度。
- 不为每个 plan 字段建立一套属性生命周期。
- 不维护迁移期双轨和兼容 fallback。
- 不要求先完成第二 family 演示。
- 不为了 task 可验收而把一个连续算法拆成多个 owner/reader/materializer 子项目。

## 四、B 线现在应该做什么

B 线保留，并与 A 线独立推进。它负责回答“生成出来的真实代码是否正确、是否有竞争力、收益能否进入部署和端到端”，而不是替 A 线规定内部架构。

### B 线已有基础

- official runner：`tools/bench/bench`；
- 持久结果：`experiments/master/`、`experiments/runs/<run-id>/`、`experiments/runs.log`；
- 已有 cell：dequantize_row、vec_dot、gemm_tile、product_reduce、scalar_vec_dot；
- deployed ggml、代表性 strong opponent、e2e 三类证据资产均已存在；
- 原 B1 已完成 measurement control plane 的主要工作；
- 原 B2 已补 K-quant 双板 correctness/route/parser 等覆盖，但没有因此产生新性能结论；
- 原 B3 已退役 ISSUE-109 中证伪的 K-quant 策略，相关负结果应保留，不得重新开启相同实验。

### B 线下一阶段的真实问题

- grid/codebook dequant 在真实板上的性能与边界，区分算法问题、compiler wall 和 gather/硬件行为；
- q4_0 等代表性 GEMM 从 request、实际生成到 deployed ggml symbol 的真实路径与强对手质量；
- kernel 改进是否传导到 deployed 与 e2e，收益和 wash 同权；
- 新鲜 official run、qualification 和 master lineage，而不是再建一套 runner/schema；
- 少量高价值结构杠杆，而不是追求一次翻完所有格式格子。

### A/B 关系

- B 线不等待 A 线完成，可以继续测量现有稳定路径。
- A 线大胆重构期间，不为保持历史 benchmark fixture 输入形态牺牲架构。
- A 线形成新的稳定生产路径后，B 线再做同输入、同对手、同板的 paired regression。
- B 线的测量可以发现公式缺陷，但不能把某个历史 winner 反向写成散落的 board/format 特判。

不再为 B4/B5/B6 建 Trellis task。需要实验时直接按 measurement 文档和 runner 执行，结果进入既有三目的地。

## 五、对当前代码和提交的裁决

### 明确停止

- 不合入 Grid 分支提交 `680681d0b`。
- 不继续 KQuant/Ternary 的同类 stamp 化。
- 当前已归档的 A4b/A5/A6/A8 与 B4/B5/B6 task 不再恢复为队列。
- 不从旧 task 的 dependency、acceptance criteria 或 current wave 恢复工作。

### 应回退或删除的语义

- per-mechanism complete selected stamp；
- per-mechanism pre-emission materializer；
- reader 在 emission 附近重新运行公式；
- partial/forged/stale 属性协议；
- reason、provider、capability mirror 与 code-affecting plan 的强绑定；
- 因迁移而产生的 optional/default/compatibility 状态；
- 只验证上述协议、但不验证真实算法或产物的 fixtures。

A4a implementation `09efd477c`、A3 implementation `33421d40f`、A2 implementation `3db0bbf95` 以及更早的五类 plan 包装提交，是下一轮审查这类语义的主要历史入口。不要机械地假设整笔提交都毫无价值：其中真实的 capability collector、`g` 提取、候选公式和旧路径删除可能值得保留。应从当前树做语义删除和结构重建，而不是为了保留提交边界留下坏架构。

### 应保留的方向

- canonical format/mechanism facts；
- canonical target capability facts；
- 真正改变候选或参数的解析公式；
- 已经证明有真实 body 的实现；
- A7/B3 这类确实删除旧路径或证伪策略的成果；
- B1/B2 的 runner、measurement control 与真实 correctness/route coverage；
- 正负硬件证据和 run lineage。

## 六、怎样修正现有 spec

当前 `.trellis/spec/` 不是要删除的工作流系统，而是项目的更新文档库。但其中关于公式层的若干内容已经落后于最新裁决，下一位 agent 应主动改写，而不是服从旧文档。

需要重点修正：

- `README.md`：删除“小型 plugin-local decision contract”“第二 extension family 使用同一最小 contract”作为主要剩余目标的表述；把主目标改为集中、可阅读的公式与 capability 消费层，以及单一最终 Plan/body。
- `.trellis/spec/index.md`：删除 legality-first、typed stamping、fallback/stale 等作为软件主链中心的表述；明确 Trellis 是参考文档库，不是任务或 agent 工作流。
- `.trellis/spec/architecture/index.md`：重写当前主改造方向和 pre-development 提示，避免旧提示再次把实现吸向 verifier、atomic migration protocol 和第二 family demo。
- `.trellis/spec/architecture/变体流水线.md`：这是最需要重写的文件。把核心改为 `g + c + real context -> readable formula -> final Plan/body -> emitter`；移除 complete stamp、pre-emission recomputation、forged/stale、fallback 分类和每个 slice 的迁移门。
- `.trellis/spec/architecture/退役与原子合入.md`：保留“最终只存在一条生产路径”和大胆删除旧路的精神，删除为渐进迁移、兼容桥和 verifier 矩阵服务的程序化负担。
- `.trellis/spec/canon/暂定-科研主张.md`：突出显式算法知识、集中能力消费、模块化公式和真实高性能；不要把 typed stamping 或第二 family demo写成贡献核心。
- `.trellis/spec/governance/思维准则.md`：保留“抵抗枝节、删除死镜像”的判断；去掉会强制 agent 进入 Trellis/task、legality/verifier 或 gate-first 路径的内容。
- `.trellis/spec/issues/发射器与架构.md`：将 ISSUE-122/125 的旧解法标为被本次第一性原理裁决取代；问题事实可保留，complete-stamp/verifier 解法不得继续作为待施工项。

更新原则：先以当前用户裁决和重构后的代码形态为准，再让 spec 描述稳定事实。不要为了保持旧文档自洽而保留旧代码。

## 七、旧 task 和 Trellis 的地位

2026-07-21，本轮 A/B active task 已全部移动到 `.trellis/tasks/archive/2026-07/`，`.trellis/.current-task` 已清空。

归档工具会机械地把 task JSON 写成 `completed`；这里的真实含义是“旧任务组织已终止并归档”，不是其 acceptance criteria 已完成。下一位 agent 不应恢复这些 task，也不需要创建替代 task。

Trellis 仍可用于查找：

- 历史 spec；
- 已知 issues；
- 实验和实现调查；
- 旧决策为何产生。

它不是开工许可、上下文注入器、队列或 agent 定位系统。

`experiments/active/formula-layer-migration/` 目前也只剩历史调查价值。下一位 agent 在吸收其中仍有效的 caller/owner 事实后，应把它移出 `active`；不要继续追加“迁移完成度”或用它恢复旧公式层路线。

## 八、有价值的既有材料在哪里

以下材料可作为事实和思考来源，但都不能覆盖本文的最新裁决。

### 当前代码与项目入口

- `README.md`：项目定位、构建、测试和测量入口；其中公式层目标待按本文修正。
- `.trellis/spec/index.md`：现有 spec 地图；作为参考库入口，不是自动工作流。
- `include/Weft/Plugin/RVV/RVVFormulaDecision.h`：本轮已抽出的部分公式，适合辨认可保留的真实算法和需要删除的协议包装。
- `lib/Conversion/RVV/RVVCodebookGatherPlanMaterialization.cpp` 与 `lib/Conversion/RVV/RVVRepackScheduleMaterialization.cpp`：本次偏航的代表性实现，适合作为删除和反例审查对象。

### 公式层历史调查

- `experiments/active/formula-layer-migration/AUTHORITY-MATRIX.md`：旧 caller/owner 盘点；只用作查找线索。
- `experiments/active/formula-layer-migration/LEDGER.md`：迁移历史；不作为目标架构。
- `.trellis/tasks/archive/2026-07/07-20-ab-formula-performance-mainlines/research/a-line-audit.md`：A 线启动时的代码事实；其中 recommended modules 已被本次裁决否定。
- `.trellis/tasks/archive/2026-07/07-20-ab-formula-performance-mainlines/research/b-line-audit.md`：B 线已有资产与剩余性能问题的摘要。
- `docs/method/C2_marginal_cost_ledger.md`：旧的接入成本材料，可用于了解历史扩展与复用，但不再定义当前 C2 或重构方法。

### 旧但有价值的外部于当前 spec 的资料

- `_attic/docs/method/REPOSITORY-MAP-五大件.md`：旧仓库结构地图，名称和路径可能过期。
- `_attic/docs/method/FALSIFIER-INDEX.md`：历史反假思路，可保留“用真实反例检验主张”的方法，不要复活 gate-first 工程。
- `_attic/docs/method/P4-family-integration-doc-TEMPLATE.md`：旧 family 接入资料，可用于理解既有 plugin surface，不作为当前目标模板。
- `_attic/docs/reports/2026-07-12-P4-Tier-2-cleanroom-接入演练-T1c.md` 与相邻 clean-room 报告：历史外部接入观察。
- `_attic/docs/reports/2026-07-09-SEL-1-design.md`：旧选择设计背景；其中 legality/stamping 结论需重新判断。
- `_attic/docs/reports/2026-07-12-full-refactor-战略-recon-下一战役.md`、`_attic/docs/reports/2026-07-13-full-refactor-母program-pillar-对账-真欠地图.md`：旧大重构与缺口分析，可用于了解项目如何走到当前状态。

### 性能与实验证据

- `tools/bench/bench`、`tools/bench/cells/`：正式测量入口与 cell。
- `experiments/master/`、`experiments/runs/`、`experiments/runs.log`：正式数字和 lineage。
- `docs/reports/SEALED-WIN-REGISTRY.md` 与 `docs/reports/2026-07-10-Win-K1-VLEN-加固报告.md`：已有代表性性能结论及其边界。
- `.trellis/spec/measurement/`：现有测量约定；B 线使用它，不把它扩张为 A 线架构门禁。

仓库中没有发现一份独立、现行的外部论文/网页参考清单。上述“外部资料”指位于当前 spec 之外、但仍值得参考的历史方法与报告。若后续引入新的外部论文，应建立简洁的来源索引，不把论文摘录复制成新的规则系统。

## 九、下一位 agent 如何开始

不需要启动 Trellis，不需要创建 task，不需要恢复 `.current-task`，也不需要先证明自己遵循某个流程。

直接执行：

1. 阅读本文并检查当前工作树、代码和近期提交。
2. 先修正会误导后续工作的 README/spec 核心方向。
3. 从当前生产树删除 stamp/materializer/reader/compat 多状态，保留真实 `g`、`c` 与公式价值。
4. 重新组织公式层，使人能在少数地方读懂代码生成决策。
5. 把所有调用者切到唯一最终 Plan/body，删除旧路径，不留回退。
6. 代码走不通时根据真实失败修正设计，不预建 legality/verifier 框架。
7. A 线稳定一段真实路径后，让 B 线做 paired correctness/performance；B 线的独立测量工作可以同时继续。

不要把本文再拆成十个任务。它表达的是项目方向，不是任务生成器。

## 十、完成后的项目应是什么样

完成不是某个 gate 数字，而是一种直观代码事实：

- 开发者知道到哪里查看和修改公式；
- `g` 和 `c` 的来源及消费清楚；
- 一个决策只有一个实现；
- 最终 Plan/body 是唯一生产状态；
- emitter 短、直接、机械；
- 没有 stamp 生命周期、兼容回退和重复 reader；
- 新增格式、能力或候选不需要在多个层复制协议；
- 代码总量和概念数量明显下降；
- B 线仍能用真实硬件证据判断这些公式是否产生有竞争力的代码。

这才是本轮“第一性原理重构”的含义。
