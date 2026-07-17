# P3 Fork-B: parameterize the flat-plain block-dot emitter bodies into ONE descriptor-driven `emitFlatBlockDot`

> 父 [[07-01-arch-refactor-noperand-core]] P2/P3。承接 front-door collapse(24→1 table-driven,commit ed616558)的 **EMITTER-side 镜像**。设计依据 = `research/track-b-autoconstruct-scope.md`(committed f05cffd1)。

## 目的(engineering-maturity + Track-B mechanism demonstrated,byte-exact)

scope 已证:flat-plain bucket(q4_0/q8_0/q4_1/q5_0/q5_1)的 `emitQxxx` block-dot emitter 是 **~85% mechanical-mirror** —— 共享 skeleton(sumf 累加 / `nb=n/qk` / block 循环 factor-1 或 multi-block / blockBase 地址 / fp16 scale 读 / integer core elided-vs-robust / strip-reduce seed+vwredsum+extract / emitFold expression-grouping / store),**唯一 per-op 分歧 = 3 点**:(a) `emitStripReduce` 里的 decode+product(已 factored 成 helper:q8_0 内联 vwmul、q4_0 `emitOffsetBinaryDecodeProductValue`、q4_1 `emitUnsignedNibbleDecodeProductValue`、q5_0 `emitFiveBitOffsetBinaryDecodeProductValue`),(b) `block_len` = qk vs qk/2,(c) `fold_model`(dx_dy_first / left_assoc / +MIN)。

**Fork B = 把 decode-primitive + fold-model 从 op-identity 选择 升成 descriptor-field dispatch**:一个 `emitFlatBlockDot(rewriter, scope, facts, descriptor)` 发共享 skeleton,decode 步 `switch(descriptor.decode_primitive)` 走**现有** helper,fold 步 `switch(descriptor.fold_model)`。这是 **body-CONSOLIDATION(byte-exact refactor)非 body-GENERATION**(后者=Fork A/composable micro-ops,被 P1 卡,不在此)。价值 = 成熟工程性(DRY/架构明确)+ 把 "capability-facts + descriptor 驱动 body" 的 Track-B 机制在 flat-plain 上 byte-exact demonstrated。

## 步骤(scope §6c,逐 op byte-exact-gated)

1. 定 `FlatBlockDotDescriptor`(scope §3 Group-B:`decode_primitive` enum / `block_len` / `num_activation_loads` / `fold_model` enum)+ `deriveFlatBlockDotDescriptor(op)`(读 op attr:`kind`/`scale_model` + `activation_high_byte_offset`/`activation_quant_byte_offset` 存在性),镜像已有 `deriveBlockDotFacts`。
2. 抽 `emitQ8_0Q8_0BlockDot` 的共享 skeleton 成 `emitFlatBlockDot(…, descriptor)`;decode/fold 步走 switch。
3. **先把 q8_0(plain_i8 实例)指向它**,证 byte-exact,再逐个迁 q4_0 → q4_1 → q5_0 → q5_1,每个独立 byte-exact-gate。每迁一个删一个 `emitQxxx`,`kBlockDotKernels`(RVVToEmitC.cpp:333)entry 指向 `emitFlatBlockDot` + descriptor(或 thin shim)。

## 纪律(硬)

- **byte-exact for existing**:每步 forced CLEAN relink tcrv-opt + 全 `ninja check-tianchenrv` **783 passed / 780 / 3**(3 failures = 既有 Scripts `computed-masked-strided...dry-run` ×2 + `self-test`)+ dequant md5 不变。发的 `emitc` op 必须逐字节同(harness 靠 22/24 block-dot e2e lit 的 `diff core.mlir prod.mlir` 双 VLEN + q4_0/q8_0/q4_1/q5_0/q5_1 各自 CORE/e2e lit 强制)。build incremental 不可靠 → forced clean relink + BEFORE==AFTER 相等(非绝对指纹)。
- **LAND-WHAT-WORKS**:先 q8_0 证 byte-exact;若某 op 抗 byte-exact 合并 → 保留其 `emitQxxx` 作 documented exception,报告真分歧,别硬合。**secondary fork 用 typed-MLIR(emitc builder)非 C-string template**(scope §6b:byte-exact-during-migration 靠发同样的 emitc op)。
- 只碰 flat-plain(q4_0/q8_0/q4_1/q5_0/q5_1)。**不碰** super-block/codebook/IQ/ternary emitter(scope §5:那些非 flat skeleton,bespoke-dominated,是 separate 更大机制)。不碰 route/export/front-door 层(emitter-local)。
- 走 trellis-implement;**不 git commit**(主会话独立复验 byte-exact 后 commit)。

## 报告

`emitFlatBlockDot` + `FlatBlockDotDescriptor` 落地;几个 op clean 合并 vs exception(为何);LOC delta;byte-exact 证据(suite count line + 哪些 lit);最终架构(一 emit 方法 + descriptor switch,decode/fold helper 复用)。诚实报 mechanical 估计是否成立。
