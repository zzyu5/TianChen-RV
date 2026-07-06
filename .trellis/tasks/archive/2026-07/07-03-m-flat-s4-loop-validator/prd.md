# M-FLAT step4/6 — loop-aware allowlist validator

**parent:** 07-03-g1-q8-0-strong-construct · **base:** refactor/full-refactor-m1
**权威蓝图:** `../07-03-m-flat-loop-scaffold-design/research/loop-scaffold-design.md` **§4**。step1-3 已落（641c0e80 / 45c23f4d / 23a6ccf1）。

## 授权语境（Option 1）
≥5-7会话/step-not-slope/正-LOC先。**这是 step 4 = 强义门**：让 step1 循环 op 的 body 被**递归认证**为全 typed 原语（无 opaque helper）——这是 q8_0 未来翻转能过 [L-8] 机检（E5 增量①已建）的强义依据。**注意：这是 validator（检查逻辑），非 emitter，byte-exact 不适用；关键是 allowlist 正确 + 真调用点 + 零回归。**

## 一句话
建一个 **loop-aware allowlist 递归 validator**：进 step1 `TypedFlatBlockDotLoopBodyOp` 的 region，逐 op 断言 ∈ **allowlist**，fail-closed 拒任何非 allowlist op（尤其 opaque `call_opaque` helper）。挂一个**真实调用点**使其真的 fire。

## 关键设计（蓝图 §4）
- **反写陷阱**：现 `rejectMixedPreRealizedContractionBody`（`lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyInternal.h:527-541`）是**模板 blocklist**、走 `variant.getBody().front()`、**不递归 region**。**别把砖塞进 blocklist 模板**（会反向拒砖体）。**正解 = 新 allowlist 递归 walk**。
- **allowlist（前向兼容 step5 全 body）** = { 砖① `BlockFp16ScaleProductOp`（per-block-source）, `WideningProductOp`, `StandaloneReduceOp`, step2 `TypedVectorLaneToScalarExtractOp`, 砖② `BlockComputedScaleDequantOp`, 砖③ `CrossBlockF32AccumulateOp`, `SetVLOp`, `WithVLOp`, `LoadOp`, `StoreOp`, `TypedFlatBlockDotLoopBodyOp`+`TypedFlatBlockDotLoopYieldOp` }。**region 内出现任何其它 op（尤 opaque call_opaque / *_block_dot helper）⇒ fail-closed 拒。**
- **真调用点**（蓝图 §4：无既有 call site 做 region 递归 ⇒ 这是**新 walk / 新 post-realization 校验**，诚实说清不是"扩既有 call site"）：挂在 contraction realization-owner 边界（`lib/Plugin/RVV/Construction/RVVContractionSelectedBodyRealizationOwner.cpp`，`createRealizedWithVL:332` 邻的 validate 序），仿 `...PreRealizedValidators.cpp:937` 但**新增递归**，且只对新循环 op_kind 触发。
- **零回归**：既有 3 单块强路仍走**原** `rejectMixed`（**未改动**）；新 validator 只吃循环 op_kind ⇒ 结构性零回归。

## 赢的条件（全 laptop）
1. loop-aware allowlist validator 存在，递归进循环 region 逐 op allowlist 校验，**fail-closed 拒非 allowlist op（含 opaque helper）**。
2. **有真实调用点**（validator 真的 fire，不是死代码——cost-map 曾指 rejectMixed 无独立调用点）。说清挂哪。
3. **既有单块 rejectMixed 路零回归**（3 强路仍过原 blocklist；未改 `rejectMixedPreRealizedContractionBody`）。
4. lit：positive（合法循环 body——step1-3 的砖①+砖③+yield——过 validator）+ **negative（region 内注入一个非 allowlist op / opaque call ⇒ 被拒）**。
5. `cmake --build build --target tcrv-opt` 干净（forced ODS rebuild+relink 若涉 .td；[[build-incremental-unreliable]]）；RVV lit（`cd build/test && lit Conversion/RVV Target/RVV`）全绿零回归（step1-3 / 3 强路 / rejectMixed / DequantizeOp）。
6. **ΔC_construct=0 · Δ手写LOC 正** = 授权内。

## 红线
- **别改 `rejectMixedPreRealizedContractionBody`**（既有单块契约）——新增并列 allowlist walk。别把砖塞 blocklist 模板（反写陷阱）。
- 不 wire q8_0、不删 emitFlatBlockDot/承重体、不动砖①②③/step1-2 op/既有强路契约。不做 step5-6。
- 数值 pending-hardware。不造 n/N 计数。
- 唯一 STOP = 递归 allowlist walk / 真调用点接入**结构性不可行**（语义/架构做不到）——极不可能（标准 MLIR walk）。"难/厚" 不是 STOP。

## 交付物
loop-aware allowlist 递归 validator + 真调用点 + lit（positive + fail-closed negative）。**回报：ΔC_construct(=0) · Δ手写LOC(正,+多少) · 当前步(step4) · validator 挂在哪个真调用点+怎么验它真 fire · 既有 rejectMixed 零回归确认 · negative（opaque/非allowlist 被拒）是否验过 · build/lit/回归 · 是否撞结构死墙。** 别长篇。

## 权威 spec
core-invariants(I5/I7,强义=零 opaque + [L-8])· 蓝图 §4 · 执行总纲 [K-4]/[L-8] · burn-down 纪律。
