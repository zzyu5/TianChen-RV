# step5a 具体实现 spec（主 agent 已做完硬推理，照做即可，别再研究/consult）

## 根因（3 agent 卡这，已解）
brick① 的 per-block emitter（`emitBlockFp16ScaleProduct` :6426，present 分支 ~:6508）把 `scale = dX*dY` 建成**独立** `emitc::MulOp`（materialized value 存 valueMap）。monolith 的 `emitFold`（:5750）把 `d_x*d_y` 建在 **fold `ExpressionOp` region 内**（inline）。独立 MulOp 渲染成单独 C 值、不 inline → **byte-exact text 失败**。这是"hard structural constraint"，但**可解**（非 STOP）。

## 解法 = 干净 factor + region-driven 复用（advisor 批准）
**核心**：loop-body-fold 路**别**复用 brick① 的独立 emitter（那会 materialize scale）。改成：把 monolith 的 `emitBlockCore`（:5820 lambda：读 dX/dY + 整数核 → sumi）+ `emitFold`（:5750 lambda：fused ExpressionOp）**抽成共享方法**，`emitTypedFlatBlockDotLoopBody` 与 `emitFlatBlockDot` **都调它们**。brick① 的 dX*dY 由共享 `emitFold` 在其 ExpressionOp region 内构造（deferred，byte-exact inline）。

### 步骤
1. **抽共享 helper（= advisor 的"干净 factor"，也是 step6 LOC 诚实的前提）**：
   - 把 `emitFlatBlockDot` 内的 `emitBlockCore` / `emitIntegerCore` / `emitFold` lambda 提成 `VariantToEmitCFunc` 的**私有方法**（参数化：接 weight/activation base + strides + descriptor foldModel + sumfVar + ib 等；**不捕获 emitFlatBlockDot 局部/不与 GgmlBlockDotQ80Op 入口纠缠**）。`emitFlatBlockDot` 改成调这些方法（**其行为/emit 必须 byte-exact 不变**——跑既有 q8_0/q4_0/... block-dot lit 零回归验证）。
2. **loop-body 全链 emit（region-driven）**：`emitTypedFlatBlockDotLoopBody`（:5966）的 region dispatch（:6048-6075）扩成**识别完整 brick 链**（顺序：brick① → [setvl/load/widening_product/standalone_reduce/extract 或直接由 sumi 来源] → brick② → brick③ → yield）。识别到 → 调共享 `emitBlockCore`（用 brick① 的 base/stride 操作数 + 整数核 op 的操作数）+ 共享 `emitFold`（用 brick②③ 的操作数）→ byte-exact。**region-driven gate**：链缺项/错序 → `notifyMatchFailure`（emit 变/报错）。valueMap[brick①.result] 映射到 fold 内 inline 的 dX*dY（不单独 materialize）。
   - 注：整数核（load/widening/reduce/extract）可复用共享 `emitBlockCore`/`emitIntegerCore` 的 sumi 计算；确保 sumi 来源与 monolith `vwmul→vwredsum→vmv_x_s` 一致。
3. **sumf load 位置**：共享 `emitFold` 已在 block core 后 load sumf（:5757），复用即得对（收尾 step1 推迟项 2）。
4. **fold ExpressionOp**：共享 `emitFold` 已是 fused ExpressionOp（:5758），复用即得对（收尾 step1 推迟项 1）。

## 两条 advisor 硬要求
- **region-driven mutation 负例（lit）**：加一个负例——loop body region 里删 brick②（或 brick③ 折在非-brick① scale/错序）→ emit **必须变或 `not tcrv-opt` 报错**。证明 emit 由实际算子驱动、非 attrs 重导。
- **干净 factor**：共享方法不与 `GgmlBlockDotQ80Op` 入口纠缠（step6 删 monolith op+dispatch 时算术 guts 作 typed lowering 存活）。

## 验证（主 agent 会亲自复核）
- full-body byte-exact：`emitTypedFlatBlockDotLoopBody` 的完整 q8_0 body emit vs 真 `emitFlatBlockDot`（fixture `rvv-to-emitc-q8-0-q8-0-block-dot.mlir`，elided/mbf=1/m2）text-diff 骨架+核+fold **一致**（provenance verbatim 携本 op identity=wire 前故意不同）。
- 零回归：既有所有 block-dot lit（q8_0/q4_0/q4_1/q5_0/q5_1 经 emitFlatBlockDot）+ step1-4 lit + 3 强路 + rejectMixed + DequantizeOp 全绿（**helper 抽取后 emitFlatBlockDot emit 不变是硬门**）。
- step4 allowlist validator 放行完整 body。
- C_construct 停 3、Δ手写LOC 正、不 wire、不翻六态、不删 monolith（=5b/step6）。

## STOP
仅当抽 helper 后**无法让 emitFlatBlockDot emit 保持 byte-exact 不变**、或 loop 路的 fold ExpressionOp **语义上无法** byte-exact——那才是结构 STOP。工作量/要多次迭代 emit-text 不是 STOP。
