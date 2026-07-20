# PRD — GridLookupPlan + TernaryDecodePlan: 公式层铺满 3/5→5/5 + 合并双头家(刀①收官 + 刀②)

## 权威 = 公式层审计三刀①②(扩覆盖 + 合并双头家) + 用户主线令(5 阶段·柱一)
公式层现 3/5(Nibble/Codebook/KQuant 已进 plan)。本役 = 最后 2 家(grid IQ + ternary)进 plan → **5/5**·**同时解审计钉的双头家**(GridDecodePlan 独立注册表不经 FormulaProvider·且 grid+ternary 揉一起违 [K-10])。byte-exact reproduce-current。

## 缘起(审计钉的两病)
1. **grid 家格式名硬分派**: `RVVToEmitCForwardElementwise.cpp:5130-5133` `decodeModel == "iq2_xxs"||"iq2_xs"||"iq2_s"||"iq3_xxs"||"iq3_s"` → `emitDequantizeRowIQGridBodyShared(:4250)`·body 读 `codebook_entry_lanes` 描述符 + **查 GridDecodePlan 注册表**(`lib/Support/GridDecodePlan.cpp` `lookupGridDecodePlan`·`decodeModel`-keyed 静态表·**不经 FormulaProvider/Gearbox 家·无 DequantMechanism tag**=双头家)。
2. **ternary 家格式名硬分派**: `:5142-5143` `decodeModel == "iq1_s"||"iq1_m"||"tq1_0"||"tq2_0"`·且 ternary(iq1_s/iq1_m)现骑在 GridDecodePlan 内(`GridFoldArith::TernaryDelta/DeltaGrid`·`GridDecodePlan.h:211-289`)=grid+ternary lumped 违 [K-10] 5 分立。

## 做(律1·byte-exact·仿前三家·[K-10] 5 分立·刀②合并)
### A. GridLookupPlan(grid 家·= 5 mechanism 第 4 个 GridLookup)
- 定义 `GridLookupPlan` struct(`include/Weft/Support/GridLookupPlan.h`·复用 DequantMechanism taxonomy·mechanism=GridLookup)·**收编 GridDecodePlan** = 让 grid 的 FormulaProvider `gridLookupPlanFromFacts` 内部**调用/包 `lookupGridDecodePlan`**(GridDecodePlan 作 grid 几何数据源·但经 FormulaProvider→plan→emitter 链·不再是 emitter 直查注册表)·或把 GridDecodePlan 的 grid 部分并进 plan。字段: mechanism/gridDecodePlan-ref 或 grid 几何(entryLanes/groupLanes/codebook idx 结构)/scaleModel/loadLMUL(reproduce-current)/stripLanes/legality/reason/provenanceFormat。
- emitter grid body(:4250)读 `plan.*`·dispatch(:5130)→`plan.mechanism==GridLookup`。

### B. TernaryDecodePlan(ternary 家·= 5 mechanism 第 5 个·**拆出 lumped**)
- 定义 `TernaryDecodePlan` struct(ternary 家 iq1_s/iq1_m/tq1_0/tq2_0)·mechanism=TernaryDecode·**从 GridDecodePlan 拆出 ternary 部分**(GridFoldArith::TernaryDelta/DeltaGrid → TernaryDecodePlan·恢复 [K-10] grid/ternary 分立)。字段: ternary delta/grid/sign 几何/loadLMUL/stripLanes/legality/reason/provenanceFormat。
- emitter ternary body 读 `plan.*`·dispatch(:5142)→`plan.mechanism==TernaryDecode`。

### ★双头家合并的诚实边界(自验·可能须分步)
- GridDecodePlan 是既有 fail-closed 注册表(`unknown→nullptr→REJECT`)——**收编须保其 fail-closed 语义**·别弱化。若一步收编风险大·**诚实分步**: 先 grid 经 FormulaProvider 包 GridDecodePlan(单头化·byte-exact)·ternary 拆出作独立步·别硬一把梭破 byte-exact。
- ⚠grid/ternary 是 W5(dequant 收割)刚碰过的家(iq2_xxs/iq2_s/iq1_s owned body)——**自验现状**·plan 化须保 W5 的 owned body byte-exact。

## ★验收 = byte-exact + plan 承重判决(仿前三家)
- byte-exact: grid 5 格 + ternary 4 格各 emit C == 现值 golden(md5·CORE==PROD)。
- plan 承重判决 lit(committed): 改 GridLookupPlan/TernaryDecodePlan 一字段→emit 随之变。独立复核。
- **诚实覆盖报**: 若双头家收编不能一步 byte-exact·报实际达成(如 grid 单头化 done·ternary 拆分 deferred)·别假 5/5。

## 触碰集 & 禁区
- **碰**: `include/Weft/Support/GridLookupPlan.h`+`TernaryDecodePlan.h`(新) · `GridDecodePlan.{h,cpp}`(收编/拆·保 fail-closed) · `RVVGearboxSchedule.h`(provider 声明) · `RVVToEmitCSupport.cpp`(provider 定义) · `RVVToEmitCForwardElementwise.cpp`(grid/ternary body + dispatch) · `RVVToEmitCInternal.h` · 新判决 lit。
- 🔴 禁碰: nibble/codebook/KQuant plan(已建) · vec_dot/block-dot 路 · A线 · verifier 行为(ISSUE-122 defer)。
- **自验锚点**(naive-grep·前几家都订正过): :5130/:5142/:4250 + GridDecodePlan 真结构 + W5 owned body 现状·与 PRD 不符标实际。

## 门
- 覆到的 grid/ternary 格 byte-exact · plan 承重判决 PASS · [K-10] 5 分立(grid≠ternary·各独立 plan·GridDecodePlan 收编经 FormulaProvider) · GridDecodePlan fail-closed 保 · verifier 不变 · 全套无 NEW 失败(基线 3) · 无 inline-asm · 未 git commit · 诚实覆盖报。
⚠ Worktree 可能 stale·reset --hard 到 tip。

## 汇报(首节 byte-exact + plan 承重 + 双头家状态)
GridLookupPlan+TernaryDecodePlan 字段/住址/provider + grid/ternary byte-exact md5 + plan 承重判决 + 双头家收编状态(单头化 done? ternary 拆 done?) + [K-10] 5 分立达成? + 全套失败数 + 覆盖诚实报(5/5? or 分步) + 锚点。禁新建分析文档·结论进 final message。
