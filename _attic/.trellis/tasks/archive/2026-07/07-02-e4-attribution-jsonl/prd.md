# E4 — 编译期归因 JSONL 出口 + [D-2a] 装载期解析记录

> **父 program:** `07-02-full-refactor`。证据轴 M1 硬门 **C_attr^CT=100%**([D-4①])。**"重构现有"**:富 in-IR 选择属性已在(`VariantSelection.cpp:326-358`),E4 加**附加式** JSONL 出口 + declared_instance_hash + D-2a 编译期 stamp。E1 已落 canonical-JSON normalize→SHA256 idiom,E4 复用其精神。
> **权威:** 科研总纲 [D-4①](JSONL `{kernel,candidates[],keys_evaluated{},chosen,reason∈{only_feasible,prior,measured},declared_instance_hash,ts}`)、[D-2a](装载期最小解析:消费显式 schema 事实→declared-instance-hash→落一条解析记录;热路径零逐次;[NG-3] per-dispatch 永禁)。执行总纲 §5。canon 7d781994:declared-instance-hash 对**展开归一事实集**取(profile ≡ 显式列表 ⇒ 同哈希)。
> **research(已完成):** `research/attribution-current-state.md`、`jsonl-export-design.md`、`d2a-resolution-record-design.md`、`e4-scope-summary.md`。

---

## Decisions(ADR-lite;research 5 escalation 定案)

1. **⭐ `prior` reason 诚实性(脊梁决策)= 选项2(用户 2026-07-03 裁决):独立过渡值 `static_order`,不复用 `prior`。** 枚举扩成 **{only_feasible, static_order, prior, measured}**;`prior` 从此**严格保留**给能力派生的先验(SEL-1 落地后才出现),今天 stage ① 绝不发。exec selector 今天 capability-blind(常量分 RVV1.0/IME20.0/Scalar1000.0 + explicit-preference)→ ≥2 候选选择记 `static_order`。**用户三条理由:** ① 选项3违反 M1 硬门(C_attr^CT=100% 要全部编译期选择落日志),且 ≥2 候选恰是最需记录的("矩阵静默落败"潜伏 bug 住这、是 T4b 消融 before 侧数据,缓了就没);② reason 是所有分析过滤主键,守卫字段迟早被某查询漏掉→"能力键选中数"静默虚高,**区分做在主键不做在脚注**(T4a"是否能力键选中"列直接从 reason 推:static_order→否/prior/measured→是/only_feasible→N/A);③ 燃减附赠:SEL-1 落地后 static_order 出现数应归零,不归零=先验层覆盖缺口,过渡值自动成诊断信号。**弃选项1(prior+守卫字段)——信息等价但把区分埋在脚注。**
2. **reason 派生逻辑:** `only_feasible`=恰 1 合法候选(N/A 能力键);`static_order`=≥2 合法候选经能力盲 cold-start 排序选中(今天唯一的多候选值);`prior`=严格保留给能力派生先验([SEL-1]/G3 后),**① 层今天绝不发**;`measured`=[SEL-3] 后,① 层今天绝不发。**故今天 ① 层只出现 `only_feasible` 与 `static_order`。** **常量分记进记录**(candidates[].score 对 feasible 候选恒发 / keys_evaluated),使 static_order 决策可完整重建(用户连带动作)。**in-IR `reason` 属性(`variant-selected`)不改**——JSONL.reason 是新派生 side field。**canon 修订(附加式,不动 schema.def):** 科研总纲 [D-4①]/spec generation-selection-tuning.md/实验总纲 T4a 的 reason 枚举已同步扩成四值,并取代旧 v1.1 [J-6]「prior 现值=插件常量分」句。
3. **JSONL sink 机制 = pass option**(`--attribution-jsonl=<path>`,lit-drivable,拥有 plan);**默认 off** → 现有 `.mlir`/`.cpp` 期望零变。sink 点 = `VariantSelection.cpp:576` 后;覆盖全部四种 `VariantSelectionKind`(含 `NoViableVariant` → `chosen=null`)。
4. **lit 确定性:** `--attribution-jsonl-no-timestamp`(ts 用固定 sentinel 或省略)+ 稳定哈希 + canonical JSON(sorted keys,candidates 按 rank 序)。
5. **declared_instance_hash = net-new C++ 序列化器**:`support::computeDeclaredInstanceHash(const TargetCapabilitySet&)` —— 展开事实集 `getCapabilities()` **按 id 排序** → canonical 串 → `llvm/Support/SHA256.h` hex。**排序是正确性 must-do**(否则 canon "profile ≡ 显式列表 ⇒ 同哈希" 破)。一 helper 同服 JSONL.declared_instance_hash + D-2a。
6. **D-2a cut = 编译期 stamp + 记录形态文档,不落 live per-process 运行期记录**(live 运行期链 = D-2b/M2)。E4:在 `DispatchRuntimeGuard` 路 stamp `declared_instance_hash`(`dispatch_available` RuntimeParamOp/KernelOp)+ 文档化"每进程一条"记录 shape;热路径零逐次不变([NG-3] 保)。

---

## Scope(IN;附加式为主 + 小 net-new)

- **`support::computeDeclaredInstanceHash`**(net-new helper,`lib/Support/` + header):sorted 展开事实集 → canonical JSON → SHA256 hex;带确定性 + profile-等价(profile ≡ 显式列表 ⇒ 同哈希)单测/lit 锚。
- **JSONL 出口**(`VariantSelection.cpp` sink,option-gated):每 kernel 一 JSON 对象 `{kernel, candidates[](从 plan.rankedVariants + VariantCostEstimate), keys_evaluated{}(sink 处从 variant.requires × TargetCapabilitySet.lookupBySymbolName 重派生), chosen, reason(+守卫字段), declared_instance_hash, ts}`;四 SelectionKind 全覆盖。两 pass option:`--attribution-jsonl=<path>` + `--attribution-jsonl-no-timestamp`。
- **D-2a**:`DispatchRuntimeGuard` 路 stamp `declared_instance_hash` + 记录形态文档(每进程一条 `{declared_instance_hash, ts, resolved_variant_set}`)。
- **测试**:新确定性 lit(`test/Transforms/VariantSelection/` 下,`--attribution-jsonl=%t.jsonl --attribution-jsonl-no-timestamp` + `FileCheck --input-file=%t.jsonl`)+ declared_instance_hash 确定性/等价 lit-or-unit。

## 验收(可机检)

1. **C_attr^CT=100%**:每个编译期选择(四 SelectionKind 全含 NoViableVariant)在 JSONL 出一条记录,字段齐(candidates/keys_evaluated/chosen/reason/declared_instance_hash)。
2. **reason 诚实(选项2)**:① 层今天只出现 `only_feasible`(单合法候选)与 `static_order`(≥2 候选能力盲排序);`prior`/`measured` **绝不**在 ① 层出现(保留给 SEL-1/SEL-3);无守卫脚注字段(区分在主键);feasible 候选恒发 score 使 static_order 可完整重建。
3. **附加式不回归**:现有 VariantSelection lit(`ime-mma-*-materialization.mlir`/`plugin-variant-materialization-builtin.mlir`/`variant-selection-pass-invalid.mlir`)+ C++ smoke(`tianchenrv-variant-selection-test`)**全绿不改期望**;in-IR 属性一字未动;E4 diff 改零个现有 `.mlir`/`.cpp` 期望(若改 = 非附加,查)。
4. **declared_instance_hash 确定性**:同事实集两次 → 同 hex;显式列表 vs profile 展开(事实集相同)→ 同哈希(排序生效)。
5. **新 JSONL lit 确定性**:`--no-timestamp` 下 FileCheck 稳定通过(无 ts/hash 漂移)。
6. **build 门(project memory `build-incremental-unreliable`)**:任何属性稳定性/byte 主张前 forced/clean rebuild(`ExecOps.cpp.inc` 每次重生、`tcrv-opt` 有时不重链);`cmake --build build` + `check-tianchenrv`。

## Out of Scope

- **schedule 阶段归因**(已富:`RVVScheduleMaterialization.h:195-216` 的 candidate_count/measured_ns/selection_reason)= **M2 增量**,不拉进 E4。
- **运行期归因 ③**(hwprobe/instance-hash 调度表)= D-2b/D-3(M2/M3)。
- **live per-process 运行期解析记录** = D-2b;E4 只编译期 stamp + 文档形态。
- **SEL-1 能力先验层** = G3;E4 不建(守卫字段为其留钩子)。
- `measured` reason 在 ① 层的可达性 = SEL-3。

## Technical Notes

- 数据轴顺利:`candidates[]`=`plan.rankedVariants`(live plan 成员)、`keys_evaluated{}` 输入(`TargetCapabilitySet`+变体 `requires`)sink 处都在,**无需重接线**。
- `only_feasible/prior/measured` token 代码 grep=0(net-new);`symbolName` 是否入 hash 是开放正确性点(见 d2a design)——默认**不入**(profile≡显式列表 同哈希优先)。
- Python 不涉(纯 C++ pass + helper);门/覆盖脚本是 E6。
