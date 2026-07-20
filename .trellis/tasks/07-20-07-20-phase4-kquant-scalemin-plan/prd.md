# PRD — KQuantScaleMinPlan: 公式层扩覆盖第三家(2/5→3/5·reproduce-current·byte-exact)

## 权威 = 公式层审计三刀①(扩覆盖) + 用户主线令(5 阶段路线·柱一)
公式层现 2/5(NibbleDecode + CodebookGather 已进 plan)。本役 = KQuant 家(super-block scale/min·q2_K/q3_K/q4_K/q5_K/q6_K)建 plan·= **CodebookGatherPlan(0beb9cb71)/NibbleDecodePlan(phase-2) 的忠实复制**·byte-exact reproduce-current(c 驱动作 phase-3 follow-up)。

## 缘起(审计钉的病)
KQuant 家现由**格式名硬分派**: `RVVToEmitCForwardElementwise.cpp:5027+` `decodeModel == "q4_K"||"q5_K"` → `emitDequantizeRowQ45KVectorBody(:3554)`·q2_K→`emitDequantizeRowQ2KVectorBody(:3733)`·q3_K→`:3860`·q6_K→`:4038`(自验精确 dispatch 行)。这些 body 散读描述符/格式·零 plan。

## 做(律1·byte-exact·仿 CodebookGatherPlan·[K-10])
1. **定义 `KQuantScaleMinPlan` struct**(住 `include/Weft/Support/KQuantScaleMinPlan.h`·同 NibbleDecodePlan/CodebookGatherPlan 先例·复用 `DequantMechanism` taxonomy·mechanism=KQuantScaleMin)。字段(复用既存 super-block 几何·别新造·自验 KQuant decode_core 现携哪些 stamped facts): `mechanism` / super-block scale/min 几何(scale offset/min offset/high-bit/sub-block 结构·hasMin/hasQh 类) / `superBlockElements`(qk=256) / `weightBlockStride` / `loadLMUL`(reproduce-current·照现值) / `widen_chain`(发射侧 deriveWideningChain) / `strip_lanes` / `legality` / `reason` / `provenanceFormat`。
   - ⚠KQuant 各格几何差异大(q2_K 2-bit·q3_K 3-bit·q4_K/q5_K 4/5-bit+min·q6_K 6-bit)——**自验**是否一个 plan 参数化 5 格·还是须 per-K-format 子结构。若几何差异使单 plan 不能 byte-exact 覆 5 格·**诚实报**(可能只覆 q4_K/q5_K 一组·或需 sub-mechanism 字段)·别硬塞。
2. **FormulaProvider** `kquantScaleMinPlanFromFacts(facts, ..., minVLEN)`(声明 `RVVGearboxSchedule.h`·定义 `RVVToEmitCSupport.cpp` 旁 codebook provider)·复现现固定选择 byte-exact·minVLEN (void)=phase-3 seam。
3. **emitter 改读 plan**: KQuant body(:3554/:3733/:3860/:4038)读 `plan.*`·**dispatch(:5027+)从 `decodeModel==` K-格→`plan.mechanism==KQuantScaleMin`**·格式名降 provenance。
4. **[K-10]**: KQuantScaleMin = 5 mechanism 第 3 个·独立 plan·禁单 plan 判别式 switch。

## ★验收 = byte-exact + plan 承重判决(仿前两家)
- byte-exact: 覆到的 K-格各跑 constructed + fallback 路·emit C == 现值 golden(md5·CORE==PROD)。
- plan 承重判决 lit(committed): 改 KQuantScaleMinPlan 一字段(如 scale/min offset)→ emit 随之变。独立复核跑判决。
- **诚实覆盖报**: 若单 plan 不能 byte-exact 覆全 5 K-格·报实际覆几格 + 为什么(几何差异)·别假装 5/5。

## 触碰集 & 禁区
- **碰**: `include/Weft/Support/KQuantScaleMinPlan.h`(新) · `RVVGearboxSchedule.h`(provider 声明) · `RVVToEmitCSupport.cpp`(provider 定义) · `RVVToEmitCForwardElementwise.cpp`(KQuant body + dispatch 读 plan) · `RVVToEmitCInternal.h`(签名) · 新判决 lit。
- 🔴 禁碰: nibble/codebook plan(已建·别动) · grid/ternary 家(本刀只 KQuant) · vec_dot/block-dot KQuant 路(RVVToEmitCKQuant·别碰·本刀只 dequant-row) · A线(RVVCapabilityProfile/version/vreg) · verifier 行为(ISSUE-122 defer)。
- **自验锚点**(naive-grep 教训): :5027 精确 dispatch 行 + 各 KQuant body 真实位置 + KQuant 家真实格式集·与 PRD 不符标实际重路由。

## 门
- 覆到的 K-格 byte-exact(== 现值 golden·md5·CORE==PROD) · plan 承重判决 lit PASS · [K-10] 合规(K-格 dispatch 降 plan.mechanism) · verifier 不变 · 全套 lit 无 NEW 失败(基线 3 pre-existing·env-gated e2e 不计) · 无 inline-asm · 未 git commit · 诚实覆盖报(覆几格)。
⚠ Worktree 可能 stale pre-rename base·reset --hard 到 refactor/full-refactor-m1 tip。

## 汇报(首节 byte-exact x/5 + plan 承重判决)
KQuantScaleMinPlan 字段 + 住址 + provider 签名 + 覆到几格 byte-exact md5 + plan 承重判决 + [K-10] 合规 + 全套失败数 + 若未覆全 5 格诚实报为什么 + 锚点是否一致。禁新建分析文档·结论进 final message。
