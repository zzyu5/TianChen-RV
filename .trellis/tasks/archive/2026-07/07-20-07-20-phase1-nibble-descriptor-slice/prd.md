# PRD — phase-1: nibble 家 vertical slice（DequantMechanismPlan 5 阶段路线第一刀·柱一 C1/C2 falsifier）

## 权威
**读 `experiments/active/formula-layer-migration/LEDGER.md` §六（尤其 §六.3 scope / §六.5 纪律 / §六.6 诚实边界）为权威**——本 PRD 是执行提示，§六 是设计正本。这是"增加公式 per census2"的结构主线：把 dequant 执行分派权从格式名字符串搬到结构化 g 描述符。

## 缘起（用户令：增加公式 + 性能都要搞清楚·架构 Workflow 已核实）
dequant "来源部分解耦·推导与消费仍耦合在 `format`/`decode_model` 字符串"。nibble 家（q8_0/q4_0/q4_1/q5_0/q5_1·QK=32·flat·非 grid）= 最低风险第一刀（共享 nibble emitter）。**crux（Workflow 已核实）**：构造表（`RVVDequantizeRowConstruction.cpp:23-111`）只 stamp 5 facts；**decode-mechanism facts（mOff/qhOff/sub/hasMin/hasQh）未入表·在 emitter re-bake**（构造 `.cpp:28-32` 注释自证）——**一个常量两处源 = 编辑分家即 silent byte-exact break**。本刀 = 把这 8-tuple 中心化进描述符。

## ★第一步硬规: 自验锚点（naive-grep 教训·MIG-1/W2 都撞过）
§六.6 标 ⚠ 的 emitter loci 是 research line-number·**未逐一复核**。开工先机核自验（`grep -n`）·**与本 PRD/§六 不符时按实际重路由**·在交付标「PRD 锚点误判·实际在 X」。待验锚点：α `:2658-2674` / β `:3209-3244` / γ `:3500-3543` + header decls `:4925-4963` + 共享体 `:2966`/`:2702`（`lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`）。per-format tuple 中 `stride/dOff/qsOff` Workflow 已复核一致（q8_0 34/0/2·q4_0 18/0/2·q4_1 20/0/4·q5_0 22/0/6·q5_1 24/0/8）；`mOff/qhOff/sub/hasMin/hasQh` **须从 loci α 复核后再中心化**（别信 research 表·自读）。

## 做（律1 分三步·每步独立 byte-exact + 独立复核·禁一把梭）
结构化 nibble descriptor 字段（§六.3·全 ggml ABI-shape 事实·律2 合规）：`name`(仅 provenance) / `qk` / `weight_block_stride` / `scale_byte_offset(dOff)` / `quant_byte_offset(qsOff)` / `min_byte_offset+min_present` / `qh_byte_offset+qh_present` / `nibble_bias(sub)` / `carrier_kind∈{bare_int8,nibble4}` / `fold_kind∈{single_mul,fused_mac_min}`（**DERIVED from hasMin·不裸存**）。
- **(i)** 扩构造表 5-format 臂 + `DequantizeRowStreamFacts` struct 加字段 + 前门 stamp；emitter β/γ 改**读 stamped attr** 替 re-bake 常量·**证 0-byte golden diff**（5 格各跑 constructed 前门路 + monolith fallback 路过 `--weft-rvv-lower-to-emitc`·diff vs 现役 lit golden = 0）。
- **(ii)** 塌 decode dispatch（δ）到 `carrier_kind`（一 nibble4 支 + 一 bare_int8 支·q8_0 独立叶·**禁 fold 进 nibble 体**）。
- **(iii)** verifier 白名单从表 key-set 派生 + 落 **F1 falsifier**。

## ★5 byte-exact 风险点（§六.3·逐点 assert·shape-diff 看不出）
① fold 形状 `y=val*d`(q4_0/q5_0/q8_0) vs `y=val*d+m`(q4_1/q5_1)·不同 C+不同 fp 舍入 ② bias `-=sub` 精确 −8/−16 ③ 5th-bit qh merge gated on hasQh ④ **byte offset 喂地址算术**（错字面量读错字节·shape-diff 看不出·**须 assert stamped-attr == descriptor value**）⑤ q8_0 signed `vsext` vs nibble `vzext`+bias（q8_0 留独立叶）。

## ★验收 = F1 falsifier（= 本刀验收门·C1/C2 度量）
合成 `q4_synth`（4-bit nibble·distinct-but-legal tuple·如 stride=19/dOff=0/qsOff=3/sub=8/hasMin=F/hasQh=F）。**双跑对照**：
- **before**：加 q4_synth 需编辑 α+β+γ+δ+header = **5 处 emitter 改** → F1 FAIL（证今日耦合·记 `git diff --stat` 5 文件）。
- **after**：descriptor 行 + test·dispatch 命中共享 `nibble4` 支无新 if → F1 **PASS**（记 `git diff --stat` = Construction.cpp + test 共 2 文件·0 emitter/机制行）= **论文 C1/C2 数字**。
- **name-relative oracle**（无真 ggml block）：assert「q4_synth tuple == q4_0 → emit C 与 q4_0 逐字节相同(除 provenance 注释)」+「格式名在可执行 C 中只作注释」= 直测"格式名不再承担分派权"。
- **负控**：非法组合（`bareInt8 && hasMin`·或 stride<layout 最小）**须 fail verify closed**·证 legality 门是真的。

## 触碰集 & 禁区
- **碰**：`RVVDequantizeRowConstruction.cpp`/`.h`（构造表+struct+stamp）· `RVVToEmitCForwardElementwise.cpp`（β/γ/δ 读 attr·**自验锚点**）· `RVVDialectWideningOps.cpp`（verifier 白名单表-派生·**只加 new attr 白名单·[D-1] fail-closed 不弱化**）· 新 lit（`test/Conversion/RVV/rvv-dequantize-row-q4-synth-descriptor-only.mlir` 等）。
- 🔴 **禁碰 `include/Weft/Plugin/RVV/RVVGearboxSchedule.h`**（MIG-5 在飞·文件不相交才能并行）· 禁碰 `RVVToEmitCGridCodebook.cpp` · 禁碰 `RVVLowerQuantContraction.cpp`。
- 🔴 **禁 fold-in**：extended `enum Fmt` StringSwitch（`:5023-5045`/stride switch `:5366-5386`）= K-quant/IQ/codebook 家 = phase-④·**别碰**；`GgmlQuantContractionOp::verify` per-format switch = vec_dot/contraction 家（非 dequant-row·dequant≠vec_dot）·**严禁**。
- **[K-10] 合规**：`carrier_kind` 选的是两个**已分立的叶**（q8_0 独立叶 vs nibble 共享体）·非 plan 内 switch·合规。**禁**把 5 mechanism 收进单 plan 用判别式 switch。
- **可顺手**：构造表头注释 stale（`.h:61-69` 说 21/nullopt·实 24 格全构造）= doc 漂移·可随本刀顺手订正（非 canon 变更）。

## 门
- 5 格各 **0-byte golden diff**（by-construction）+ 5 风险点各 assert · F1 falsifier before-FAIL/after-PASS 双跑记 `git diff --stat`+LOC · 负控 fail-closed · 律1 分三步每步独立 byte-exact · 三闸 §四.5 Δ≤0 · 无 inline-asm · **未碰 RVVGearboxSchedule.h**（自查 `git diff --name-only` 确认）· 未 git commit。
- **canon-adjacent 不自决**：verifier 从「bounds-check+白名单」升「从公式重算几何拒 stamped≠recomputed」= **phase-② 能力·本刀不做**（本刀 verifier 只加 new-attr 白名单·不改重算门措辞）；plan struct 住址 = phase-② 决策·本刀不创建 struct。

## 汇报（自然·首节 0-byte golden diff x/5 + F1 falsifier before/after）
5 格 golden diff 结果 + F1 before(5文件FAIL)/after(2文件PASS) 的 `git diff --stat`+LOC + 5 风险点 assert 结果 + 负控 fail-closed + 锚点是否与 §六 一致(不一致标实际) + 确认未碰 RVVGearboxSchedule.h。final message 放关键结论 + C1/C2 数字。
