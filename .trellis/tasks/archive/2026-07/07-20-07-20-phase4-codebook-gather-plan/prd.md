# PRD — CodebookGatherPlan: 公式层扩覆盖第二家(1/5→2/5·reproduce-current·byte-exact)

## 权威 = 公式层审计@HEAD(d23a9e32c) + 用户主线令(g/c→plan·柱一·5 阶段路线)
公式层审计判 1/5 覆盖(仅 NibbleDecode 闭环)·三刀之一 = 扩覆盖去格式名。用户 phase-4 顺序: nibble→**small codebook**→IQ grid→KQuant→ternary。本役 = codebook-gather 家(small codebook)建 plan·= phase-2 之于 nibble 的复制·byte-exact reproduce-current(c 驱动作 follow-up)。

## 缘起(审计钉的病)
codebook-gather 家(iq4_nl/iq4_xs/mxfp4/nvfp4·16-entry LUT)现由**格式名硬分派**: `RVVToEmitCForwardElementwise.cpp:4978-4981` `decodeModel == "iq4_nl"||"iq4_xs"||"mxfp4"||"nvfp4"` → `emitDequantizeRowCodebookGridBodyShared(:4305)`·**零 plan**·body 读描述符/格式散读。getRVVCodebookGatherAnchorLMUL(闭式已备·`RVVToEmitCCodebookFp4.cpp:120`)但 call-site VLEN 钉死 128(:118 `kCodebookByteExactMinVLEN`·"VLEN256 flip INFRA-ready but deferred behind measured gate")。

## 做(律1·byte-exact·仿 phase-2 NibbleDecodePlan·[K-10])
1. **定义 `CodebookGatherPlan` struct**(住 `include/Weft/Support/CodebookGatherPlan.h`·同 NibbleDecodePlan/GridDecodePlan 先例·dual-consumer)。字段(复用既存几何·别新造): `mechanism(=CodebookGather)` / `codebookEntries`(16) / `codebookByteOffset` / `scaleModel` / `load_lmul`(**reproduce-current·照现值·非 c-驱动**) / `widen_chain` / `strip_lanes` / `legality` / `reason` / `provenanceFormat`(仅诊断)。
2. **FormulaProvider** `codebookGatherPlanFromFacts(facts, minVLEN)`(声明住 `RVVGearboxSchedule.h` 公式家·定义 `RVVToEmitCSupport.cpp` 旁 nibble provider)·**复现现固定选择 byte-exact**(load_lmul 照现值·minVLEN 作 phase-3 seam·**本刀 (void) 或钉死·不 c-驱动**)。
3. **emitter 改读 plan**: codebook body(:4305)从散读 → 读 `plan.*`·**dispatch(:4978-4981)从 `decodeModel==` 4 格 → `plan.mechanism==CodebookGather`**·格式名降 `plan.provenanceFormat`(仅诊断)。
4. **[K-10]**: CodebookGatherPlan = 5 mechanism 第 2 个·**禁**收进单 plan 判别式·codebook 独立体(与 nibble/grid 分立)。

## ★验收 = byte-exact + plan 承重判决(仿 phase-2)
- **byte-exact by-construction**: iq4_nl/iq4_xs/mxfp4/nvfp4 各跑 constructed 前门路 + fallback 路过 `--weft-rvv-lower-to-emitc`·emit C 逐字节 == 现值 golden(md5 对齐·CORE==PROD)。plan 只换封装·产物不变。
- **plan 承重判决 lit**(committed·防 plan 死数据): 改 CodebookGatherPlan 一字段(如 codebookByteOffset/strip_lanes)→ emit 随之变(证 emitter 真读 plan 非散读)。独立复核跑判决不看 diff。
- **c-驱动 follow-up(本刀不做·登记)**: :118 gate(minVLEN 钉死 128)= phase-3 codebook c-驱动点·须板验(measured gate·VLEN256 anchor byte-exact)·本刀 reproduce-current·seam 留好(plan.load_lmul 通道·minVLEN 参数在)·登记为下一刀。

## 触碰集 & 禁区
- **碰**: `include/Weft/Support/CodebookGatherPlan.h`(新) · `RVVGearboxSchedule.h`(provider 声明) · `RVVToEmitCSupport.cpp`(provider 定义) · `RVVToEmitCForwardElementwise.cpp`(codebook body :4305 + dispatch :4978 读 plan) · 可能 `RVVToEmitCCodebookFp4.cpp`(FP4 codebook body 读 plan·自验) · 新判决 lit。
- 🔴 禁碰: nibble plan(phase-2 已建·别动) · grid dequant/vec_dot(别碰) · KQuant/ternary 家(phase-4 后续家·本刀只 codebook) · A线(cap-complete 在飞·别碰 RVVCapabilityProfile/version/vreg) · verifier 行为(只加白名单·ISSUE-122 defer)。
- **自验锚点**(naive-grep 教训·phase-1/2/W1/W2/W4 都撞过): 开工机核 :4305/:4978/:118 真实位置·codebook 家真实格式集(iq4_nl/iq4_xs/mxfp4/nvfp4·自验 iq1_s/tq 是否属此家还是 ternary)·与 PRD 不符标实际重路由。

## 门
- 4 格 byte-exact(emit == 现值 golden·md5·CORE==PROD) · plan 承重判决 lit PASS · [K-10] 合规(codebook 独立·无单 plan switch·`decodeModel==` codebook 4 格降为 plan.mechanism) · verifier 行为不变(ISSUE-122 defer) · 全套 lit 无 NEW 失败(基线 3 pre-existing) · 无 inline-asm · 未 git commit。
⚠ Worktree 可能 stale pre-rename base·reset --hard 到 refactor/full-refactor-m1 tip。

## 汇报(首节 byte-exact 4/4 + plan 承重判决)
CodebookGatherPlan 字段 + 住址 + provider 签名 + 4 格 byte-exact md5 + plan 承重判决结果 + [K-10] 合规(dispatch decodeModel== codebook 降 plan.mechanism) + 全套失败数(应仍 3) + :118 c-驱动 seam 登记 + 锚点是否一致。禁新建分析文档·结论进 final message。
