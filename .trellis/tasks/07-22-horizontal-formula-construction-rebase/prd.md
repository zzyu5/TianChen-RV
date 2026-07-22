# 横向公式/构造层重基

## 目标

按当前稳定 spec 一次性建立 project-wide formula/construction stage：所有当前可达的 operator construction entry 都由唯一的 family-local typed construction authority 消费真实 `g/c/ω`，产生会被 legality、selection 与 realization 直接消费的 candidate/plan/body 参数；同时删除旧 provider、planner replay、emitter-side redecision、隐藏默认和兼容旁路。

这不是按 dequant mechanism、format、backend 或“先做几个公式”的纵向迁移。任务只有在完整 current production domain 原子切换、旧 authority 为零、四类覆盖门都闭合后才能完成。

## 当前横向范围

范围由合入时的注册表、默认 pipeline、公开 direct route 与真实 production call graph
共同给出，不冻结手填 census，也不把目录、formula id 或旧 ledger 当分母：

- 全部 builtin plugin 的 proposal、analytic prior/cost、legality、selected
  construction 与 emission 路径；
- 全部已注册 source front door 与公开 direct construction route；
- selected-body realization registry 及其 composite case；
- dequant、quantize、repack、contraction、schedule、resource、tiling、loop-order、
  numerics 与 body-realization 中的生产解析决定；
- builder、materializer、verifier、planner、selector 或 emitter 中重放同一决定的
  production caller；
- test-only/unregistered demo 只需结构上不能成为 production fallback。

Catalog/registry 测试可以打印当前枚举结果帮助定位，但数量不是贡献、进度或完成门。

## 横向设计

### 1. 薄公共 catalog

公共层只提供 descriptor、axis-use/result-kind/construction-strength 小枚举、plugin registry 的只读枚举和静态一致性诊断。Catalog 不保存 evaluator function pointer，不按字符串动态求值，不引入 Formula IR、provider service、通用 mechanism AST 或 untyped result。

Descriptor 与 family-local typed evaluator/constructor declaration 同源，至少表达：owner、operator domain、真实 `g/c/ω` 依赖、结果种类、语义 cases、production entry 与当前 construction strength。

### 2. 入口与结果的真实关联

- 每个 source front door 与 direct route 关联唯一 construction descriptor；
- proposal/cost/analytic prior 明确关联其公式 owner；
- deterministic single-candidate 路径也进入 typed construction，不以“没有选择”为由绕过；
- formula result 必须进入实际 legality/selection/final-body construction，后层不得覆盖或重算；
- `g` 不携带 capability、measurement winner 或完整 body id；`c` 来自 canonical capability projection；`ω` 不夹带 winner memory。

### 3. RVV 决策收口

- dequant 的 nibble、codebook、K-quant、grid、ternary/binary-sign 等 mechanism/leaf 选择全部在 emission 前构造并落入 typed plan；
- emitter 只机械消费 selected typed plan，删除 `decode_model` 字符串决策、局部 provider 调用和 VLEN 常量重推；
- repack LMUL 与 contraction algorithm 拆开“候选/解析 prior”和“受限制 measurement winner”，selector 不得创造候选；
- schedule/resource/tiling/loop-order/numerics/body realization 的现有 family-local typed 公式保留语义，但统一注册和关联真实 caller；
- verifier 只检查 IR/ABI/ISA/typed plan 形状与一致性，不重放公式来决定 compute。

### 4. 诚实的构造强度

Catalog 覆盖不等于强义 reconstruction。每条路径按当前事实标为 strong construction 或 deterministic constructed-weak/dispatch-wired。后者必须移除隐藏决策和双 authority，但不会被虚称为已经通过“删除逐点 leaf 后仍可重建”的 `C_construct` 判据。

## 退役项

- `experiments/active/formula-layer-migration/` 下把 8 个决定误当完整公式域的 active matrix/ledger authority；
- 只验证旧 matrix 行数、旧 task 引用和历史 provider 名的脚本门；
- `RVVFormulaDecision` 中“三个 vertical slices”的旧定位与把 measurement 混入 formula context 的接口；
- dequant emitter 内依据 `decode_model` 重选 mechanism/plan 的路径；
- verifier/planner 对同一 formula 的 compute-authoritative replay；
- 同一 declared decision 的兼容 adapter、dual caller、missing-stamp fallback 与被忽略/覆盖的 result 字段。

历史实验与提交仍在 Git/归档中，不继续作为 active build 或架构 authority。

## 完成事实（2026-07-22）

- `ExtensionPlugin::constructFormulaPlans(ModuleOp)` 已成为 project-level construction
  lifecycle；emission planning 先调用该 hook，direct RVV backend 复用同一个 RVV owner。
- RVV 公式按 quantize、dequantize、generic schedule、repack schedule、low-precision/
  dot-reduce resource 与 catalog 分模块；typed `g/c/ω` 只在 family-local formula 汇合。
- quant/dequant typed body、standalone dequant `unroll_factor`、repack/generic final knobs 与
  low-precision selected-body plan 均在 emitter 前构造；route derivation、verifier 与 emitter
  不再缺失时补值或重新选择。
- 旧 dequant/repack materializer、Gearbox pass、low-precision performance/resource-selection
  policy、candidate/selected/audit attrs、handoff/marker 与 composite resource replay 已删除；
  没有兼容 alias 或 fallback 世界。
- 旧 active 8-row matrix/ledger 已归档，当前 catalog/behavior/authority tests 从代码验证
  owner、entry、semantic case、依赖边与旧 symbol/attr 零回流。
- 本任务闭合的是 authority convergence。逐点 leaf 的 strong delete-leaf reconstruction
  仍是后续横向增强目标，不作为本任务已完成事实。

## 完成标准

- [x] 所有 current registered/direct construction entry 恰好关联一个 catalogued family-local typed owner；
- [x] 所有 proposal、analytic prior、legality 与 realization caller 都能追到该 owner，且无 result ignore/override/recompute；
- [x] catalog integrity、production-entry linkage、dependency-edge behavior 与 semantic-case behavior 由当前代码和测试直接检查，不依赖手填百分比；
- [x] unclassified analytic authority、legacy caller、compat adapter、emitter-side redecision、verifier formula replay、selector-created candidate、hidden target/global read 均为 0；
- [x] dequant 全 mechanism typed plan 在 emission 前完成，emitter 不再按 format/decode-model 选算法；
- [x] measurement 只在 legality 后的薄 selector 使用，不进入 formula candidate construction；
- [x] strong 与 constructed-weak 分类由测试和 descriptor 如实暴露，catalog coverage 不冒充 `C_construct`；
- [x] 删除旧 8-row matrix gate，并以 registry/catalog + behavior/mutation tests 直接保护新 authority；
- [x] focused tests、`check-weft`、script self-test 与 `git diff --check` 通过；
- [x] 无后续“其余公式迁移”任务、无生产 dual path；提交后工作树干净。

## 不在本任务中改变的内容

- 不修改或重新解释两柱、六律、C1/C2/C3 与论文科研主线；
- 不改变已确定的主构造公式和薄 selector 数学语义；
- 不以本轮 catalog 化宣称所有逐点 leaf 已被强义重建；
- 不开展性能 winner 翻转、硬件 campaign 或论文 novelty 重写；
- 不建立通用 tensor/tile IR 或 runtime observer 主系统。

## 验证

- catalog/registry unit tests：唯一性、入口覆盖、caller linkage、semantic cases 与 construction strength；
- 每个 decisive `g/c/ω` edge 的独立扰动，honest-null 的不变性；
- missing/unknown/illegal/empty-legal-set 的 fail-closed 负例；
- dequant mechanism 全表的 typed-plan construction 与 emitter denylist；
- selector 不创造 candidate、measurement 不进入 formula 的类型与行为测试；
- 旧 symbol/include/caller 与 active matrix gate 的零残留检查；
- clean incremental build、focused lit/unit、完整 `check-weft`。
