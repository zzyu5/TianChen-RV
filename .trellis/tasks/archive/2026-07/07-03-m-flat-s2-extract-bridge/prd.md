# M-FLAT step2/6 — scalar-lane-extract 桥(vmv_x_s typed 化)

**parent:** 07-03-g1-q8-0-strong-construct(M-FLAT 里程碑)· **base:** refactor/full-refactor-m1
**权威蓝图:** `../07-03-m-flat-loop-scaffold-design/research/loop-scaffold-design.md` **§8**(阻断 B 确认 + 设计)。step1 已落(commit 641c0e80,循环 op + byte-exact acc)。

## 授权语境(Option 1,不变)
≥5-7 会话/step-not-slope/正-LOC 先/byte-exact 不可 hacky 绕。**这是 step 2**。C_construct 停 3、Δ手写LOC 正 = **预期授权内**。

## 一句话
建**一个 typed 原语** `TypedVectorLane0ToScalarExtractOp`(暂名):i32m1 向量(vwredsum 结果)lane0 → **标量 i32**。拆阻断 B(蓝图 §8 已 grep 确认:方言**无** typed 标量-extract op;`vmv_x_s` 今天只以 opaque 出现在 `emitFlatBlockDot:5653-5655`)。这是"整数核 → 标量 fold" handoff 从裸 opaque 变 typed 的枢纽——循环强体必须它(brick② 硬要标量 i32 sumi;byte-exact fold `(float)sumi` 是标量 cast)。

## 范围(只此一 op,同砖①②③模式 copy-then-adapt)
- **op 语义:** ins = i32 LMUL m1 向量 token(`standalone_reduce` 输出形,lane0=标量边界)+ vl;attr = kind(如 `lane0-i32-extract`);out = **标量 i32**。fail-closed(I7)verifier:输入 vector-i32、结果 scalar-i32、vl present;emission 由 op identity 决定(I5)。
- **纵向:** ODS op(RVVOps.td,接 loop op 后)+ verifier(RVVDialectWideningOps.cpp,**append**)+ emitc lowering(**byte-identical** 到 `emitFlatBlockDot` 的 `__riscv_vmv_x_s_i32m1_i32(...)`,`RVVToEmitCBlockQuantLinear.cpp:5653-5655`)+ emit-consistency lit(positive + fail-closed negative)。
- **强义说明(重要,别误判):** 此 op lower 到 `vmv_x_s` **intrinsic**(如既有强路的 widening_product→vwmul / standalone_reduce→vwredsum)。**发 RVV intrinsic ≠ opaque 手写 helper**——[L-8] 机检(E5 增量①已建)只拒 opaque `*_block_dot` monolithic helper,不拒 typed op 的 intrinsic lowering。故此 op 是强义兼容的 typed 原语。

## 赢的条件(全 laptop)
1. typed op 存在(ODS+verifier),i32m1→标量 i32,fail-closed。
2. emit-consistency:lowering 与 `emitFlatBlockDot` 的 `vmv_x_s` 调用**字节一致**(lit 锁那行 emitc.call_opaque)。
3. `cmake --build build --target tcrv-opt` 干净(forced ODS rebuild+relink,[[build-incremental-unreliable]]);RVV lit(`test/Conversion/RVV`+`test/Target/RVV`)全绿零回归(既有 3 单块强路/rejectMixed/DequantizeOp/step1 loop op 未回归)。
4. **ΔC_construct=0 · Δ手写LOC 正** = 授权内,照报。

## 红线
- byte-exact 不可 hacky 绕。**不 wire q8_0、不删 emitFlatBlockDot 或承重体、不动既有契约**(新 op 独立 append,同砖①②③)。不做 step 3-6。
- 数值 pending-hardware。不造 n/N 计数。
- 唯一 STOP = byte-exact `vmv_x_s` lowering **结构性不可行**(语义做不到)——极不可能(直接 intrinsic 复刻)。"难/厚" 不是 STOP。

## 交付物
新 typed op(ODS+verifier append+emitc lowering)+ emit-consistency lit。**回报:ΔC_construct(=0)· Δ手写LOC(正,+多少)· 当前步(step2)· byte-exact 是否达成+对哪行 · build/lit/回归 · 是否撞结构死墙。** 别长篇。

## 权威 spec
core-invariants(I5/I7,强义定义)· 蓝图 §8 · 执行总纲 [K-4] · burn-down 纪律。
