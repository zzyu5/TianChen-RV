# PRD — phase-2: NibbleDecodePlan（DequantMechanismPlan 抽象首个·reproduce-current·byte-exact·增量迁移下一刀）

## 权威
**读 `experiments/active/formula-layer-migration/LEDGER.md` §六（尤其 六.1 目标架构 / 六.5 纪律 / 六.6 诚实边界）为设计正本。** 本 PRD = 执行提示。这是 DequantMechanismPlan 5 阶段路线 **phase-②**（承 phase-1 已完成的 nibble 描述符）。

## 缘起 & scope（增量迁移·一步一验·用户选的路）
phase-1（`c6a9d6026`）已把 nibble 家（q8_0/q4_0/q4_1/q5_0/q5_1）decode-mechanism 8-tuple 迁进结构化描述符·emitter 读描述符 8-tuple + dispatch on `carrier_kind`。phase-② = **引入 MechanismPlan 抽象**：`FormulaProvider(g,c)` 从描述符产出一个 **`NibbleDecodePlan`**·emitter 改**只读 plan**（非散读 8-tuple）。
- **本刀 = reproduce-current·byte-exact**（严格复现现固定选择·**零行为变更**·无 perf 变化·phase-③ 才让 c 驱动）。
- **[K-10]**：`NibbleDecodePlan` 是 5 mechanism 之一（NibbleDecode）·**禁**把 5 mechanism 收进单一 plan 用判别式 switch。本刀**只做 nibble**（grid/ternary/KQuant = phase-④·别碰）。

## 做（律1·byte-exact·每步独立验）
1. **定义 `NibbleDecodePlan` struct**（我裁住 **Support 家** `include/Weft/Support/`·同 `GridDecodePlan.h` 先例·Conversion→Dialect 唯一合法方向·verifier+emitter 双消费）。字段（§六.1·复用既存闭式·别新造几何）：`mechanism(=NibbleDecode)` / `load_lmul`（复用现 nibble LMUL 来源·**本刀照现值**·非新算）/ `widen_chain`（复用 `deriveWideningChain`）/ `strip_lanes`（复用 `getRVVStripVLMAXElements`）/ `legality` / `reason`（provenance trace）/ `provenanceFormat`（仅诊断）。
2. **`FormulaProvider` 函数**（住 `RVVGearboxSchedule.h`·§〇 公式层家）：`nibbleDecodePlanFromFacts(DequantizeRowStreamFacts, minVLEN)` → `NibbleDecodePlan`·**复现现固定选择**（本刀 load_lmul/strip_lanes 从现有 geometry 派生·**与 phase-1 现值逐字节同**·不引 c 驱动）。
3. **emitter 改读 plan**：nibble 发射体从散读描述符 8-tuple → 读 `plan.*`。
4. **verifier**：**只加 new-attr 白名单**（若有）·**🔴 不改 verifier 行为**（bounds-check→重算升级 = **ISSUE-122 待裁·本刀不做**·保守默认维持现 verifier）。
5. **顺手**：构造表头注释 stale（`.h:61-69` 说 21/nullopt·实 24 格全构造）订正（doc-only·非 canon 变更）。

## ★验收 = byte-exact + plan 承重判决
- **byte-exact by-construction**：5 格各跑 constructed 前门路 + monolith fallback 路过 `--weft-rvv-lower-to-emitc`·**emit C 逐字节 == phase-1 后现值 golden**（md5 对齐·CORE==PROD）。plan 只是把 8-tuple 换个封装·产物不变。
- **plan 承重判决 lit**（committed·防"plan 是死数据 emitter 没真读"）：改 `NibbleDecodePlan` 一个字段值（如 test 里注入不同 strip_lanes/load_lmul）→ emit 必须随之变（证 emitter 真消费 plan·非仍散读描述符）。**独立复核跑判决不看 diff。**
- **[K-10] 合规自证**：`NibbleDecodePlan` 只覆 nibble·grid/ternary/KQuant 未收进·`git grep` 确认没造单 plan 判别式 switch。

## 触碰集 & 禁区
- **碰**：`include/Weft/Support/`（新 NibbleDecodePlan.h·或并入既有）· `RVVGearboxSchedule.h`（FormulaProvider 函数·**MIG-5 已完成·文件现 free**）· `RVVToEmitCForwardElementwise.cpp`（nibble 发射体读 plan）· `RVVDequantizeRowConstruction.{h,cpp}`（若 FormulaProvider 从 facts 构造）· 新判决 lit。
- 🔴 **禁碰**：`RVVToEmitCGridCodebook.cpp`（grid=phase-④）· `RVVLowerQuantContraction.cpp` · verifier 行为（ISSUE-122 待裁·只加白名单不改门措辞）。禁 fold-in K-quant/IQ（phase-④）。禁 c 驱动 load_lmul/strip_lanes（phase-③·本刀 reproduce-current）。
- **自验锚点**（naive-grep 教训·phase-1/MIG-5/W2 都撞过）：开工先机核 nibble 发射体/描述符读点真实位置·与 §六/PRD 不符按实际重路由·交付标「PRD 锚点误判·实际在 X」。

## 门
- 5 格 byte-exact（emit == phase-1 现值·md5 对齐·CORE==PROD）· plan 承重判决 lit PASS · [K-10] 合规（只 nibble·无单 plan switch）· verifier 行为不变（ISSUE-122 defer）· **全套 lit 无 NEW 失败**（基线 = 3 pre-existing Scripts·只许持平）· 无 inline-asm · 未 git commit。

## 汇报（自然·首节 byte-exact 5/5 + plan 承重判决）
NibbleDecodePlan 字段形态 + 住址 + FormulaProvider 签名 + 5 格 byte-exact md5 + plan 承重判决 lit 结果 + [K-10] 合规自证 + 全套 lit 失败数(应仍 3) + 锚点是否与 §六 一致。final message 放关键结论。
