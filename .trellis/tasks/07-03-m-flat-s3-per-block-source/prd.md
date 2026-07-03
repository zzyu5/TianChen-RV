# M-FLAT step3/6 — 扩砖① loop-capable per-block-source(base+ib*stride)

**parent:** 07-03-g1-q8-0-strong-construct · **base:** refactor/full-refactor-m1
**权威蓝图:** `../07-03-m-flat-loop-scaffold-design/research/loop-scaffold-design.md` **§2**（根因缺口）。step1/step2 已落（641c0e80 / 45c23f4d）。

## 授权语境（Option 1，不变）
≥5-7会话/step-not-slope/正-LOC先/byte-exact不可绕。**这是 step 3 = 拆两次 STOP 反复点名的根因**：砖① `BlockFp16ScaleProductOp::verify`（`RVVDialectWideningOps.cpp:9750-9765`）硬要 scale base 是导入 ABI 基址，**读不了循环里每块的 `base+ib*stride`**。

## 一句话
把砖① `BlockFp16ScaleProductOp` **扩成 loop-capable**：加一个 optional `block_index` 操作数（step1 循环 op region 的归纳变量）+ `lhs/rhs_block_stride`(I64) attrs，使它能 typed 表达每块 d_x/d_y 的 `base + block_index*stride (+ byte_offset)` 地址。**扩而非新 op**（蓝图 §2：单块砖① = 这条循环变体的 block_index-缺省退化特例；scale_model/`dx*dy`/f32-cover 全复用；操作数已 AnyType → 改动限 verifier+attr+operand）。

## 范围（只 step 3，向后兼容扩展）
- **操作数/attr:** `lhs_scale_base`/`rhs_scale_base` 仍导入 `RuntimeABIValueType`（LHS/RHSInputBuffer role，**不放宽基址导入性**）+ **新增 optional `block_index`**（index 类型，来自循环 op 归纳变量）+ **新增 `lhs/rhs_block_stride`(I64)** attr（AoS 块步长，如 q8_0=34）。复用既有 `lhs/rhs_scale_byte_offset`。
- **verifier 放宽（append/改，fail-closed I7）:** `block_index` **present** 时——结构校验它来自 enclosing 循环 op（step1 `TypedFlatBlockDotLoopBodyOp`）的 region 归纳变量 block-arg，允许计算地址；`block_index` **absent** 时——**保留现单块常量偏移行为**（向后兼容）。
- **lowering 扩:** `block_index` present → 发 `base + block_index*stride + byte_offset` 的 emitc.add/mul + `(float)*(const _Float16*)` 读，**byte-exact 复刻** `emitFlatBlockDot` 的 `blockBaseValue`（`RVVToEmitCBlockQuantLinear.cpp:5451-5462`）+ `fp16ReadAt`（`:5467-5476`），产 f32=`f32(dx)*f32(dy)`；absent → 现单块 lowering **不变**。
- **emit-consistency lit:** 新 positive（per-block/loop 案例：把扩后砖① 放进 step1 循环 op region、用其 block_index，锁 `base+ib*stride` 地址 + fp16 读 byte-exact 到 :5451-5476）+ fail-closed negative。

## 赢的条件（全 laptop）
1. 扩后砖① 能 typed 表达 per-block scale（block_index+stride）；`block_index` absent 时行为不变。
2. emit-consistency：per-block lowering 与 `emitFlatBlockDot` blockBaseValue+fp16ReadAt **字节一致**（lit 锁）。
3. **既有单块砖① lit 零回归**（`rvv-to-emitc-block-fp16-scale-product*.mlir` 全绿）——这是向后兼容硬门。
4. `cmake --build build --target tcrv-opt` 干净（forced ODS rebuild+relink，[[build-incremental-unreliable]]）；RVV lit（`test/Conversion/RVV`+`test/Target/RVV`，`cd build/test && lit Conversion/RVV Target/RVV`）全绿零回归（step1 loop op/step2 extract/3 单块强路/rejectMixed/DequantizeOp 未回归）。
4. **ΔC_construct=0 · Δ手写LOC 正** = 授权内。

## 红线
- **byte-exact 不可 hacky 绕**。**关键：既有单块砖① 契约向后兼容、其 lit 零回归**（扩展是 additive：block_index optional）。
- 不 wire q8_0、不删 emitFlatBlockDot/承重体、不动砖②③/既有强路/rejectMixed/DequantizeOp 契约。不做 step 4-6。
- 数值 pending-hardware。不造 n/N 计数。
- 唯一 STOP = per-block `base+ib*stride` byte-exact 地址 lowering **结构性不可行**（语义做不到）。"难/要重写/更厚" 不是 STOP。

## 交付物
扩后砖①（ODS operand/attr + verifier 放宽 + lowering 扩）+ emit-consistency lit（per-block 新 positive + 单块 backward-compat）。**回报：ΔC_construct(=0) · Δ手写LOC(正,+多少) · 当前步(step3) · per-block byte-exact 是否达成+对哪行 · 既有单块砖① lit 是否零回归 · build/lit/回归 · 是否撞结构死墙。** 别长篇。

## 权威 spec
core-invariants(I5/I7)· 蓝图 §2 · 执行总纲 [K-4] · burn-down 纪律。
