# M-FLAT step5a — 全 body 装配 + full-body byte-exact

**parent:** 07-03-g1-q8-0-strong-construct · **base:** refactor/full-refactor-m1
**权威蓝图:** `../07-03-m-flat-loop-scaffold-design/research/loop-scaffold-design.md` **§3.2 / §3.3**。step1-4 已落（641c0e80 / 45c23f4d / 23a6ccf1 / 61cea426）。

## 授权语境（Option 1）
≥5-7会话/step-not-slope/正-LOC先/byte-exact不可 hacky 绕。**这是 step5 的第一半（5a：装配+byte-exact，仍未 wire）**。C_construct 停 3、Δ手写LOC 正 = 授权内。**wire q8_0 dispatch + 翻六态 = 5b（下一步），本轮不做。**

## 一句话
扩 `emitTypedFlatBlockDotLoopBody`（`RVVToEmitCBlockQuantLinear.cpp`），把循环 body 从 step1 的 stub（只 brick③）填成**完整 q8_0 typed 原语链**，并证明**整个 body** byte-exact 到真 q8_0 `emitFlatBlockDot`（elided/m2/SumiTimesScales 实例）。这收尾 step1 显式推迟的 2 项 full-body byte-exact。

## 完整 body 原语链（蓝图 §3.3，对照 `emitBlockCore:5843`/`emitIntegerCore:5663`/`emitFold:5750`）
每块 region body（q8_0 = PlainI8 / SumiTimesScales / 无 min / 无 qh）:
1. **per-block-source ①'**：`block_fp16_scale_product(vx, vy, block_index)` → f32 scale（step3 扩后砖①，复刻 blockBaseValue+fp16ReadAt+dx*dy）。
2. **load ×2**：setvl + with_vl + load i8 weight/activation（复刻 `loadI8:5521` @ elided vsetvl `:5685`）。
3. **widening_product**：`WideningProductOp` i8×i8→i16（复刻 PlainI8 `vwmul:5538-5542`）。
4. **standalone_reduce**：`StandaloneReduceOp` signed_widening_reduce_add i16→i32m1 lane0（复刻 `vwredsum:5648-5652`）。
5. **extract 桥**：step2 `TypedVectorLane0ToScalarExtractOp` i32m1 lane0 → 标量 i32 sumi（复刻 `vmv_x_s:5653-5655`）。
6. **brick②**：`block_computed_scale_dequant(sumi, scale)` → f32 term（复刻 `(float)sumi*scaleProduct:5768-5774`）。
7. **brick③**：`cross_block_f32_accumulate(acc, term)` → acc_next（复刻 `sumf+term:5775-5776`），yield。

## 收尾 step1 推迟的 2 项（蓝图 §3.2，byte-exact 核心）
1. **fold 语句 wrapper**：`emitFlatBlockDot` 把 fold 包进 `emitc::ExpressionOp`（`:5758`）渲染成**一条 C 语句** `sumf += (float)sumi*(d_x*d_y)`；step1 的 stub 用 brick③ 裸 `AddOp`。5a 要让 brick②③ 的 fold 发射产出**同一 ExpressionOp wrapper**，byte-exact。
2. **sumf load 位置**：monolith 在 block core **之后**load sumf（`:5757`）；step1 在 loop top load。5a 要把 acc block-arg **惰性**映射，使 load 落在 block core 之后（body 填满后二者不再重合）。

## ⚠ 核心裁决（advisor：genuine construction 是可证的经验性质，非哲学）
**"真构造 vs cosmetic relabel" 的判别 = emit 是否 REGION-DRIVEN（由 region 里实际的 brick 算子 lower 出来），还是 attribute-derived（从 qk/stride/fold_model 属性重导、无视 region 内容）。** 复用 monolith 的 `emitFold`/`emitBlockCore` 逻辑**不算** cosmetic（共享 lowering helper 正常）；cosmetic 的标志 = **不管 brick② 在不在 region、发出同样的 bytes**。若 region-driven，则 q8_0→constructed 是**真的**：CORE IR 变成 typed pattern-library 组合，byte-exact 是**安全性质**（证明表征转换没回归输出）而非价值本身；价值 = 表征改变（[L-8]/六态/provenance 测的正是它；E5 机检 walk realized body op-identity 读到 brick①②③+widening/reduce/extract、无 *_block_dot）。

## 赢的条件（全 laptop）
1. `emitTypedFlatBlockDotLoopBody` 发完整 typed 原语链（①'+load+load+widening+reduce+extract+②+③），**零 opaque helper**（RVV intrinsic call 不算，见 step2 说明）。
2. **full-body byte-exact**：完整 body lower 与真 q8_0 `emitFlatBlockDot`（elided/mbf=1/m2/SumiTimesScales）**字节一致**（含 fold ExpressionOp + sumf load 位置）。emit-consistency lit 锁**整个 body**（不只骨架）。**验证 = 对真 q8_0 fixture text-diff 骨架+核+fold 一致**（provenance verbatim 仍携本 op identity，wire 前故意不同，非 miss）。
3. **★ REGION-DRIVEN 证明（防 cosmetic，必做）**：fold emission 必须是 brick②/brick③ **实际算子的 lowering**，不是从 attrs 重导。**加一个 mutation 负例**：region chain 错了（如缺 brick②、或 brick③ 折在非-brick① 的 scale 上）时，emit **必须改变或报错**。若构造不出"改 region 却不改 emit"的反例 = 走了 cosmetic 的信号（红旗）。（step1 BADFOLD / step4 allowlist 已部分覆盖；本轮加一个专咬 attribute-derived 的。）
4. **★ 干净 factor（为 step6 LOC 诚实，必做）**：共享 emit 逻辑（emitFold/emitBlockCore 算术 guts）要**干净 factor 成 typed body 的 lowering**，**不与 monolith 的 `GgmlBlockDot*Q80Op` 入口纠缠**。因为 step6 删 monolith = 删 `GgmlBlockDot*Q80Op` op+dispatch+orchestration 入口,而**算术-emit guts 作为 typed lowering 存活（repurposed）**——Δ手写LOC 赢是真的但**限于 orchestration/dispatch 层、小于 emitFlatBlockDot 原始体量**。别把 guts 和 monolith 入口缠死。
5. **step4 allowlist validator 放行**该完整 body（全 op 已在 allowlist）。
4. `cmake --build build --target tcrv-opt` 干净（forced ODS rebuild+relink 若涉 .td；[[build-incremental-unreliable]]）；RVV lit（`cd build/test && lit Conversion/RVV Target/RVV`）全绿零回归（step1-4 lit / 3 强路 / rejectMixed / DequantizeOp / 既有 q8_0 block-dot fixture 未回归）。
5. **ΔC_construct=0 · Δ手写LOC 正** = 授权内（未 wire、未翻六态）。

## 红线
- **byte-exact 不可 hacky 绕**（fold ExpressionOp + sumf load 位置要真复刻，不是近似）。
- **不 wire q8_0 dispatch、不翻六态、不删 emitFlatBlockDot**（=5b/step6）。不动砖契约/既有强路/step1-4 op 签名。
- 数值 pending-hardware。不造 n/N 计数。
- 唯一 STOP = **full-body byte-exact 结构性不可行**（fold ExpressionOp 或 sumf load 位置语义上做不到 byte-exact）。"难/要重写多次/更厚"不是 STOP——byte-exact 要重写就重写。

## 交付物
扩后 `emitTypedFlatBlockDotLoopBody`（完整原语链 + fold ExpressionOp + sumf 惰性 load）+ full-body emit-consistency lit（整 body byte-exact 到真 q8_0）。**回报：ΔC_construct(=0) · Δ手写LOC(正,+多少) · 当前步(step5a) · full-body byte-exact 是否达成+怎么验的（对真 q8_0 哪些行）· fold ExpressionOp+sumf load 2 项是否收尾 · step4 validator 放行? · build/lit/回归 · 是否撞结构死墙。** 别长篇。

## 权威 spec
core-invariants(I5/I7,强义=零 opaque)· 蓝图 §3.2/§3.3 · [K-4]/[L-8] · burn-down 纪律.
