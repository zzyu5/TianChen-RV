# iq2-grid (iq2_xxs / iq2_xs / iq2_s) vluxei16 解码式 repack-GEMM 前门构造设计

- 状态: **needs-design，未立项**（FLAT 族收口后立项即可开工；本文档只做设计，零实现）
- 目标覆盖格: `gemm_tile` × {iq2_xxs, iq2_xs, iq2_s} × engine=rvv
- 现状（`schema/coverage-sixstate.v1.json`）: 三格 `gemm_tile` = **dispatch-wired**，`provenance="direct-emitter (transitional scaffolding)"`，`retirement_batch=3`。三格 `dequantize_row` = **constructed**（前门已建）。
- 立项动作类型 = **retirement（退役直发射器 → 前门构造）**，与 K-quant / iq4_nl / iq4_xs / ternary 已走过的完全同构。不是从零写 kernel：byte-exact 参考体已存在。
- 队列关系: 本批（batch 3）= iq2-grid；**mxfp4 殿后（batch 4）**，走 codebook-core 路（iq4 族），不复用本文的 grid-core。

---

## 0. TL;DR + 关键设计决策

1. **这不是新 kernel，是 retirement。** iq2_xxs/iq2_xs/iq2_s 的 16×1-repack GEVM/GEMM **已经有完整 byte-exact 直发射器**（`emitRepackGemvIq2XxsQ8K` / `emitRepackGemvIq2DualScaleQ8K` + GEMM 兄弟），已 dispatch-wired、已跑通、已带真 grid+sign 解码。构造 = 把这条直发射器的 body **改由前门 `lowerToRepackGem{v,m}Grid` 构造 `typed_repack_gem{v,m}_loop_body` region + 一个新 `repack_gem{v,m}_grid_core` anti-bypass brick**，emitter 侧新增 grid 分支 **复用同一 body 逻辑**再发射。byte-exact 是 by-construction（与 q4_K→emitRepackKQuantGemvBodyQ4K、iq4_nl→emitRepackCodebookGemvBodyIq4Nl 同纪律）。

2. **signs64 op-attr 障碍已解除。** 三格 ODS 明确写 “The fixed grid + signs plane are **NOT op attrs**：they are FIXED canonical tables emitted ONCE as static const decls”。grid/sign 平面是 emit 期 **DERIVED**（`emitIQ2XXSCanonicalSigns64TableDecl` / `emitIQ2SCanonicalSigns256TableDecl`），**不作为 op-attr 携带**。core brick 只带 `decode_model` + 字节 offsets，不带 signs 数组。此路已由 dequant 前门与直发射器双双验证 —— 这条历史 blocker 不再是障碍（见 §5.1）。

3. **register cliff 预判 = AlreadyLean → prior Plain**（与 iq4_xs 同判，非 q4_K 的 MinFoldRegisterCliff）。理由：iq2-grid 解码是**内存 gather**（grid+sign 从表里逐 lane 取，不在寄存器里重建权重平面），且 **NO min-fold**（无 bsums×dmin 的 min 累加器平面）。S6 输出 tiling 对它是结构性 no-op。**但需在构造时实测**：iq2 比 iq4_xs 多一层累加器（sub→sumi 两级）+ dual-ls，4 列 GEMM 下的峰值 vreg 要量一次确认不越入需要 tiling 的新形（见 §2）。

4. **body 拆分沿用今天的两分**：xxs（single-ls，u8 grid-idx，signs64）用一个 body；xs/s（dual-ls，u16 grid-idx，signs64 vs signs256）共享 `Iq2DualGridVariant` body。`decode_model ∈ {iq2_xxs, iq2_xs, iq2_s}` 键控三条结构轴（§附录 facts 表）。

5. **触碰集与 FLAT 族/K-quant/iq4 线不相交**（§4）：新增集中在 `RVVLowerQuantContraction.cpp`（新 facts + 新 lowering）、`RVVOps.td`（新 grid-core op ×2）、`RVVDialectWideningOps.cpp`（新 verifier）、`RVVToEmitC*.cpp`（新 emit 分支 + 退 6 条直发射 dispatch）。grid table / signs plane / dual-scale body 全部**复用**已存在的直发射器件。

---

## 1. vluxei16 解码式 repack 布局图（数据流）

### 1.1 三格公共骨架（QK_K=256 super-block，16-way block-as-lane）

权重侧 = `block_iq2_*x16` block-as-lane 布局；每 16 个原始 super-block 交织成一个 “group”，按 VLEN 切成 `numHalves = weight_interleave/half_lanes` 条 strip（VLEN128→half=8→2 halves；VLEN256→half=16→1 strip）。激活侧：GEVM = 一条 plain `block_q8_K`（stride 292，fp32 d @0，int8 quant @4）；GEMM = 交织 `block_q8_Kx4`（stride 1168，4×fp32 d @0，int8 quant @16 排布 pos*4+c）。

每 super-block 分 8 sub-block（每 32 元），每 sub-block 4 个 group（每 group 8 元 = 一个 grid entry）。核心 per-lane 数据流：

```
                      per (sub-block ib, group grp, column-strip h)
  weight repack plane                              activation (q8_K)
  ┌─────────────────────┐
  │ grid-index strip     │   idx  ── (widen→u16) ──► base16 = idx*8   (vsll ,3)
  │  (u8 xxs / u16 xs,s) │                              │
  └─────────────────────┘                              │  for j = 0..7:
  ┌─────────────────────┐            grid[idx*8 + j] ──┤   vluxei16_v_i8(grid8, base16+j)  → gridV (i8)
  │ sign-selector strip  │   sel  ── (widen→u16) ──►   │   vluxei16_v_i8(signs, sbase+j)   → signV (i8, ±1)
  │  (7-bit xxs/xs;      │            sign[sel*8 + j] ─┤        │
  │   8-bit explicit s)  │                              │   w = vmul_vv_i8(gridV, signV)   ← 符号折进 GRID（不折进 q8：q8 可 -128）
  └─────────────────────┘                              │        │
                                                        │   aq = *(int8*)(al + quant + k)  (标量广播)
                                                        │   prod = vwmul_vx_i16(w, aq)
                                                        │   sub += vwadd_wv_i32(sub, prod)  ← i32 两级累加（grid*sign*q8 达 |43*127|，溢 i16）
                      per sub-block 结束:                         │
  ┌─────────────────────┐                                        │
  │ ls scale strip (i8)  │  ls = vsext_vf4_i32(ls_i8)  sumi += vmacc_vv_i32(sumi, ls, sub)
  │  single xxs / dual   │                                       (dual: 前2 group×ls1，后2 group×ls2)
  │  xs,s（groups 0-1/2-3）│
  └─────────────────────┘
                      per super-block 结束（NO min）:
  ┌─────────────────────┐
  │ fp16 d strip (16)    │  d0 = vfwcvt(fp16 d) * y.d(fp32)      sumf += vfmacc_vv(sumf, cvt(sumi), d0)
  └─────────────────────┘
                      per column-strip 输出:
                            out[x*16 + h*half] = 0.125f * sumf     (vse32；iq2 家族 1/8 因子)
```

**三条结构轴（`decode_model` 键控）**：

| 轴 | iq2_xxs | iq2_xs | iq2_s |
|---|---|---|---|
| grid-index 宽度 & 取法 | **u8**：`vle8`→`vzext_vf2` u16→`vsll ,3` | **u16**：`vle16` 直接（9-bit 装不进 byte）→`vsll ,3` | **u16**：`vle16` 直接（10-bit assembled）→`vsll ,3` |
| grid 表 | 256 entry × 8B = 2048B `iq2xxs_grid` | 512 × 8 = 4096B `iq2xs_grid` | 1024 × 8 = 8192B `iq2s_grid` |
| sign 平面 | **signs64** DERIVED（7-bit selector→128×8 ±1，ksigns 派生） | **signs64** DERIVED（同 xxs 平面，selector = q2>>9） | **signs256** DIRECT（explicit 8-bit sign byte→256×8 ±1，NO ksigns 选择器） |
| ls scale arity | **single** ls（sub-block 顶 nibble） | **dual** ls（ls1 group0-1 / ls2 group2-3） | **dual** ls |

其余（0.125 折、no-min、i32 两级累加、fp16 d 折、weight_interleave=16、half∈{8,16}）三格一致。

### 1.2 GEVM（decode）vs GEMM（prefill）的区别 = 仅激活轴

- **GEVM**：单激活行 plain `block_q8_K`；`numHalves` 条权重 strip，每 strip 一个 `sumi`/`sumf`。列组循环 `x` 内层块循环 `l`。
- **GEMM**：交织 `block_q8_Kx4`，`columnsPerPass = kActivationInterleave = 4`（mf2）/ `1`（m1）。**权重 grid+sign 解码 per-16-weight-group 摊销一次，跨 4 列（及 M 行）复用**（`w` 只算一次，vwmul 展开 4 个激活标量）。每权重字节读一次。这是 repack 的 memory-locality 赢点所在（decode 摊销），与 iq4_xs GEMM 同构。

构造后两者共用同一 `typed_repack_gem{v,m}_loop_body` region 机制，只是 GEMM 的 loop-body op 多带 `row_count / output_row_stride / activation_interleave`，region 多一个 `strip_offset` block-arg（照抄 iq4_xs 的 `lowerToRepackGemmCodebook`）。

---

## 2. 寄存器预算静态账 + [PAT-S6] 判类

RVV 架构 32 个向量寄存器 = register cliff。判类键 = **`fold_model` 的瓶颈 SHAPE，不是 format 名**（`include/TianChenRV/Plugin/RVV/RVVRepackTilingSelection.h` `classifyTilingBottleneckShape`）。三类锚点：

- `MinFoldRegisterCliff`（q4_K/q2_K/q5_K，`kquant_dmin_bsums_min`）：min-fold 的 bsums×dmin 平面是峰值，S6 stack-panel 把 spill 打到 0（q4_K GEMM 实测 peak live vreg **94→~29**，`schema/pattern-registry.v1.json` PAT-S6）。→ prior **S6Tiled**。
- `DualPlaneWeightBound`（q6_K/q3_K，`kquant_single_scale_no_min`）：峰值是双平面权重重建，输出 tiling 是 NULL 杠杆。→ prior **Plain**。
- `AlreadyLean`（iq4_nl/iq4_xs codebook + flat q4_0 `lane_wise_vector_scale`）：body 已 ≤32-vreg（memory-gather 解码，无可 stage 的解码 strip 可缓解），S6 是结构 no-op。→ prior **Plain**。

### 2.1 iq2-grid GEVM 峰值 vreg 静态估（VLEN128，core=mf2：l8=mf2, l16=m1, l32=m2；numHalves=2）

累加器 LMUL = m2（2 vregs/个）。同时活跃集（内层 j 累加时）：

| 角色 | 类型 | 条数 | vreg |
|---|---|---|---|
| `sumfVar[h]`（跨块持久） | f32m2 | 2 | 4 |
| `sumiVar[h]`（跨 sub-block 持久） | i32m2 | 2 | 4 |
| `ls32[h]`（per sub-block） | i32m2 | 2 | 4 |
| `subVar[h]`（per group） | i32m2 | 2 | 4 |
| `gridBase[h]` + `signBase[h]`（跨 j=0..7 held） | u16m1 | 4 | 4 |
| 瞬态 `gridV/signV/w`(i8mf2)+`prod`(i16m1) | — | ~4 | ~4 |
| **峰值** | | | **~24 < 32** |

VLEN256（core=m1，numHalves=1，累加器 m4=4 vregs）：单 strip 但更宽，累加器 4+4+4+4=16 + gather base(u16m2=2×2=4) + 瞬态 ~4 ≈ **~24 < 32**。两板均在 cliff 下。

### 2.2 iq2-grid GEMM 峰值（S6 相关路径）

GEMM 4 列（mf2 columnsPerPass=4）× numHalves=2 → 每列每 half 一个累加器。sumf/sumi 各 4×2×m2=16 vreg，但二者**不同时全活**（sumi 在块尾折进 sumf）。sub-block 累加相内活跃 = sumi(16) + sub(per-col-half) + ls(共享) + gridBase/sign(共享 4) + 瞬态。**与 iq4_xs GEMM 同构**（同 q8_Kx4 4 列交织 + super-block 累加，已判 AlreadyLean→Plain）。

### 2.3 判类结论（设计期预判，需构造时实测确认）

> **iq2-grid 三格预判 = `AlreadyLean` → prior `Plain`。**
> 依据：memory-gather 解码（grid+sign 逐 lane 从表 gather，不在 vreg 重建权重平面）+ **NO min-fold**（无 bsums×dmin 累加器平面，正是 q4_K 落 MinFoldRegisterCliff 的那一项 iq2 没有）⇒ S6 输出 tiling 无可缓解的 decode strip ⇒ 结构 no-op。与 iq4_xs 同映射（键=SHAPE 非 format）。
>
> **诚实 caveat（必须构造时量一次）**：iq2-grid 比 iq4_xs 多一层累加器（`sub → sumi` 两级）+ **dual-ls**（xs/s）。4 列 GEMM 下峰值 vreg 要在真板 objdump 量一次：若仍 ≤32 → 确认 AlreadyLean（预期）；若越过 cliff 且 S6 能缓解 → 那是一条**新 bottleneck SHAPE**，应在 `RVVRepackTilingSelection.h` 注册新 case（**不得硬塞进 AlreadyLean**，键是 shape）。predicted 不等于 sealed —— 这条预判绑 “相×板×格式” 待实测。
>
> **selector reason 预判**：无 offline seed → cold-start `[XFER-1]` prior，`reason=prior`（键控新 `fold_model` 字符串 → shape → variant）。构造时给三个新 `fold_model`（建议 `grid_sign_single_scale_eighth`(xxs) / `grid_sign_dualscale_eighth`(xs,s)）加 `classifyTilingBottleneckShape` 分支。

---

## 3. byte-exact 验证方案（cert 三要件）

构造的 byte-exact 参考体 = **现存的直发射器**（`emitRepackGemvIq2XxsQ8K` / `emitRepackGemvIq2DualScaleQ8K` + GEMM 兄弟）。构造纪律：前门 region 里唯一 brick = `repack_gem{v,m}_grid_core`，emitter 从 brick identity **RE-EMIT 整个 body**，与退役前 dispatch 的直发射器逐字节同（modulo 仅 source-op provenance token）。这是 by-construction，非事后对拍——与 q4_K/iq4_nl 退役时用的同一门。

三要件（ZERO-MODEL 证书 canon，`memory/zero-model-adjudication-cert-hardening.md`）：

1. **语料完备**：lit 必须喂**非退化** grid-index / sign-selector / ls-scale：
   - grid-index 覆盖到表尾（xxs idx∈[0,255]，xs∈[0,511]，s∈[0,1023]）——不能全 0（0 号 entry 常是退化的对称 grid，掩盖 idx*8 offset 错）。
   - sign-selector：xxs/xs 覆盖 7-bit 全非零 selector（signs64 派生正确性），s 覆盖 explicit sign byte 含高位翻转位（signs256 direct 索引）。
   - ls：xs/s 的 ls1≠ls2（否则 dual-scale 退化成 single，掩盖 group0-1/2-3 分派错）。
   - 至少 2 个 super-block（nb≥2）确保 `block_index*stride` anti-bypass 真被行使（否则 block-0 常数掩盖 stride）。
2. **输入路径同源**：oracle 与被测走**同一** `block_q8_K` 激活缓冲、同一 repack 权重缓冲；q8_K quantizer 用同一条路量化（`memory/zero-model...` 的 q8 quantizer 失配是 min-term 误诊根因 —— iq2 激活是 q8_K，务必两侧同一 quantize 调用，不各量各的）。
3. **oracle 独立**：oracle 从 **raw iq2_* block bytes 零复用重算**全算术项——不捕获被测的 intermediates。oracle 应是 ggml 参考 `ggml_vec_dot_iq2_*_q8_K`（或等价独立标量重算：从 raw `qs`/`qh`/`scales`/`d` 逐位解 grid-index → 查 canonical grid → 应 sign → ls → 0.125），**独立于 repack 布局**（repack-GEMM 的 oracle 恒等于 mat-quant 参考，见 cert canon）。0-mismatch = 正确（比 output-vs-oracle 强）。

**门**：lit `test/Conversion/RVV/rvv-repack-gemv-iq2-*-construct.mlir` 两 RUN（① 前门 REALIZE region + brick 结构 stamp；② region→C byte-exact vs 退役直发射器 golden + vs 独立 oracle）。构造前先跑 forced/clean rebuild（`memory/build-incremental-unreliable`：`RVVOps.cpp.inc` 每次重生、`tcrv-opt` 有时不重链，byte-exact 对拍必用 BEFORE/AFTER-EQUALITY，绝对指纹已 STALE）。

---

## 4. 与现有件的共享清单 + 触碰集（disjoint-file 判断）

### 4.1 直接复用（不改）

| 件 | 位置 | 复用方式 |
|---|---|---|
| grid 表 decl 发射器 | `RVVToEmitCTernaryBinary.cpp`: `emitIQ2XXS/XS/SCanonicalGridTableDecl` | 原样 —— body emit 里已调 |
| signs 平面 decl | 同上: `emitIQ2XXSCanonicalSigns64TableDecl` / `emitIQ2SCanonicalSigns256TableDecl` | 原样（DERIVED，非 op-attr） |
| dual-scale body 逻辑 | `RVVToEmitCBlockQuantLinear.cpp`: `emitRepackGemvIq2DualScaleQ8K` / `emitRepackGemmIq2DualScaleQ8K`（`Iq2DualGridVariant`） | body 计算逻辑整体复用，仅改**入口**（从直发射 op → 从 grid-core brick 取 ABI + facts） |
| xxs single-scale body | 同上: `emitRepackGemvIq2XxsQ8K` / GEMM 兄弟 | 同上 |
| vluxei16 gather / vwmul / vmacc helper | body 内 lambda（`gatherByte`/`idxBaseU16`/`signFold`/`lsScale32`） | 原样内联 |
| region 构造模板 | `RVVLowerQuantContraction.cpp`: `lowerToRepackGem{v,m}Codebook`（iq4_nl/iq4_xs） | copy-then-adapt（最近模板） |
| SP4 tiling stamp | `stampTilingSelection` + `RVVRepackTilingSelection.h` | 复用，加新 fold_model→shape 分支 |
| decode-facts 模板 | `CodebookDecodeFacts` / `kIq4XsDecodeFacts` | copy-then-adapt 成 `Iq2GridDecodeFacts` |

### 4.2 新增（本批 touch-set，未来立项文件集判交用）

| 件 | 位置 | 动作 |
|---|---|---|
| `Iq2GridDecodeFacts` 结构 + `kIq2Xxs/Xs/SDecodeFacts` constexpr | `lib/Plugin/RVV/RVVLowerQuantContraction.cpp` | 新增（grid 表 id、sign 平面 id/kind、gridIdx/ls/sign offset、grid-idx 宽度 flag、scale arity flag、strides） |
| `lowerToRepackGem{v,m}Grid` | 同上 | 新增（照 codebook 版；建 `typed_repack_gem{v,m}_loop_body` region + `repack_gem{v,m}_grid_core` brick） |
| scale_model→facts 路由 | 同上（~L795 三元链 + ~L822 fail-closed 表） | 新增三个 `kIq2*ScaleModel` 分支 |
| `RepackGem{v,m}GridCoreOp` ×2 | `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` | 新增（照 `RepackGem{v,m}CodebookCoreOp`：operands weight/act/vl/block_index[+strip_offset]，attrs `decode_model` + grid/ls/sign byte offsets + n_subblocks；**无 grid/signs 数组 attr**） |
| grid-core verifier | `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` | 新增（fail-closed pin decode_model∈{iq2_xxs,xs,s}、offsets、integer_core_lmul） |
| emit 分支 grid case | `lib/Conversion/RVV/RVVToEmitC.cpp`: `emitTypedRepackGem{v,m}LoopBody` | 新增 grid 分支（按 brick `decode_model` 调 xxs / dual body） |
| body 入口改签名 | `RVVToEmitCBlockQuantLinear.cpp` + `RVVToEmitCInternal.h` | body 函数增 “from grid-core brick” 入口（照 `emitRepackKQuantGemvBodyQ4K` “Called ONLY from ...LoopBody's branch”） |
| 退役 6 条直发射 dispatch | `RVVToEmitC.cpp`:437-442, 500-505 | 删（`isRepackGem{v,m}Iq2{Xxs,Xs,S}Q8KBody`→`emit...` 六项）；旧 `GgmlRepackGem{v,m}Iq2*Q8KOp` monolith op + recognizer 退役（照 iq4_xs NOTE 注释纪律留 retirement note） |
| decode_model surface gate | 若走 dequant 同款 bounded gate，需在 grid-core verifier 加 iq2 三 model | 新增 |
| lit | `test/Conversion/RVV/rvv-repack-gem{v,m}-iq2-*-construct.mlir` | 新增（§3 两 RUN） |
| 覆盖翻格 | `schema/coverage-sixstate.v1.json` 三 `gemm_tile` iq2 行 dispatch-wired→constructed | **立项时翻，本设计不翻**（纪律：纯文档不 flip 覆盖格/不改 schema） |
| pattern-registry | `schema/pattern-registry.v1.json` PAT-S6 三子类若确认 AlreadyLean | 立项实测后登 |

**disjoint-file 判交（`memory/parallel-lines-need-disjoint-files`）**：
- 与 **FLAT 族**（q4_0/q4_1/q5_0/q5_1 nibble）线：`RVVToEmitCForwardElementwise.cpp` 的 nibble body **不碰**，无交。
- 与 **K-quant / iq4 codebook** 线：都碰 `RVVLowerQuantContraction.cpp` / `RVVOps.td` / `RVVToEmitC.cpp` / verifier —— **天然共享文件（ODS/verifier/emitter 骨架跨线共享），必须串行**：若 iq4_xs codebook 收口或 SP4/SEL-1 有未 commit 改动，先 verify+commit 那条再起本批。本批自身新增与它们逻辑不重叠（新 op、新 lowering、新 facts），但同文件编辑需 rebase 序化。
- 结论：**本批不适合与任何动 `RVVLowerQuantContraction.cpp`/`RVVOps.td`/`RVVToEmitC.cpp` 的线并行**；与纯 FLAT-nibble 线或纯 spec/doc 线可并行。

---

## 5. 风险 / 未决

### 5.1 signs64 op-attr —— 已解除，非障碍（历史 blocker 更新）

memory `emitter-maturity-vluxei16-widelmul` 记 “iq2 被 signs64 op-attr 卡”。**现状已解决**：三格 ODS 与直发射器都把 grid/signs 平面作为 emit 期 DERIVED canonical static const 表（`emitIQ2*Signs*TableDecl`），**不经 op-attr**。core brick 只带 `decode_model` + 字节 offset；signs 平面由 body emitter 自己 spell。构造直接继承此性质（与 dequant IQ-grid 前门 `emitDequantizeRowIQGridBodyShared` 注释 “signs64 / grid-table 平面是 DERIVED at emit, not carried as op-attrs, so the leaf is self-contained and CLEANLY constructible” 完全一致）。**风险级别：低**。唯一要守：新 grid-core op **不得**为图省事把 signs/grid 塞成 `DenseI8ArrayAttr`（那会重蹈 op-attr 覆辙且撑大 IR）——保持 DERIVED。

### 5.2 grid entry 宽度 & 索引装载

三格 grid entry 恒 = 8 packed int8（byte-view 后 `idx*8+j` gather）。索引宽度不同（u8 vs u16）是**唯一**装载分叉：xxs 需 `vzext_vf2`（u8→u16）再 `vsll ,3`；xs/s 直接 `vle16`（9/10-bit 装不进 byte）。grid 表大小 2048/4096/8192B，均超 VLMAX，**必须 memory gather（vluxei16），register vrgather 在 mf2 anchor VLMAX<16 非法**——这是 iq1_s 已验证的同一约束（gap 7.4→2.3× byte-exact）。**风险：低**（直发射器已跑）。注意：iq2_s 的 10-bit 索引是 repack 期从 `qs[g]` + `qh` 高 2 位 **assembled**，构造只搬 body、不重做 assembly（assembly 在上游 repack 布局，已定）。

### 5.3 mxfp4 殿后（batch 4）—— 独立，不共 grid-core

mxfp4 = FP4 e2m1 codebook（E8M0 shared exp），结构上是 **iq4_nl 族**（16-entry codebook memory gather），退役走 **codebook-core**（`lowerToRepackGem{v,m}Codebook` 扩 `kMxfp4DecodeFacts`），**不复用本文 grid-core**。故 batch 3（iq2 grid）与 batch 4（mxfp4）触碰集在**新 op / 新 facts 上不相交**，但都编辑 `RVVLowerQuantContraction.cpp` / `RVVToEmitC.cpp`（序化，不并行）。ordering：先 iq2（batch 3）再 mxfp4（batch 4），照 `retirement_batch` 号。iq2 grid-core 建成后，nvfp4（absent）若将来上，也是 codebook 族不是 grid 族。

### 5.4 dual-ls group 分派正确性（xs/s 专属）

xs/s 的 `sub` 两级累加必须把 group0-1 折进 ls1-加权、group2-3 折进 ls2-加权。这是直发射器 `emitRepackGemvIq2DualScaleQ8K` 里 `groupsPerHalf=2` 的逻辑，构造复用即继承；但 §3 语料必须 ls1≠ls2 才能 catch 分派错。**风险：中低**（复用逻辑，但 cert 语料要显式覆盖）。

### 5.5 register 预判 predicted≠sealed（§2.3 重申）

AlreadyLean 是**设计期结构预判**，非硅上 sealed。构造完成第一件事：真板（`ssh rvv` VLEN128 + `ssh k1` VLEN256）objdump 量 GEMM 峰值 vreg + Plain/S6Tiled 两 variant 对拍。落 AlreadyLean 则确认（预期，与 iq4_xs 同）；越 cliff 则**升级为新 bottleneck shape 注册**，不硬塞 AlreadyLean。这条绑 “相×板×格式×八门”，不得以直觉投影收口（`CLAUDE.md` 性能常驻判断规则 2）。

---

## 附录 A：三格精确 facts 表（构造 `Iq2GridDecodeFacts` 填值源）

| fact | iq2_xxs | iq2_xs | iq2_s | 来源 |
|---|---|---|---|---|
| `decode_model` | `iq2_xxs` | `iq2_xs` | `iq2_s` | — |
| weight_block_stride | 1184 | 1824 | 1824 | ODS + emitter |
| gevm activation stride (q8_K) | 292 | 292 | 292 | ODS |
| gemm activation stride (q8_Kx4) | 1168 | 1168 | 1168 | emitter L20697 |
| gemm activation_interleave | 4 | 4 | 4 | emitter |
| gevm activation quant offset | 4 | 4 | 4 | ODS |
| gemm activation quant offset | 16 | 16 | 16 | emitter |
| weight grid-idx offset | +160 | +288 | +288 | ODS |
| grid-idx 宽度 | u8（512 bytes） | u16（512 lanes） | u16（512 lanes, assembled） | ODS |
| weight ls-scale offset | +32 | +32 | +32 | ODS |
| ls arity | single | dual | dual | ODS |
| weight sign offset | +672 | +1312 | +1312 | ODS |
| sign 平面 | signs64 DERIVED | signs64 DERIVED | signs256 DIRECT | ODS |
| grid 表 | 256×8=2048B `iq2xxs_grid` | 512×8=4096B `iq2xs_grid` | 1024×8=8192B `iq2s_grid` | ODS |
| n_subblocks | 8 | 8 | 8 | ODS |
| weight_interleave | 16 | 16 | 16 | ODS |
| half_lanes | {8,16} | {8,16} | {8,16} | 能力键（VLEN） |
| min 项 | NO | NO | NO | ODS |
| 尾因子 | 0.125 | 0.125 | 0.125 | ODS |
| scale_model（GEVM） | `superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth` | `superblock-d.fp16-grid-sign-dualscale-nomin-eighth` | （iq2_s 兄弟串） | ODS |
| 建议 fold_model | `grid_sign_single_scale_eighth` | `grid_sign_dualscale_eighth` | `grid_sign_dualscale_eighth` | 本设计 |
| 预判 bottleneck shape | AlreadyLean | AlreadyLean | AlreadyLean | §2.3（needs-measure） |

## 附录 B：构造后覆盖翻格判据（立项时执行，本文档不翻）

三 `gemm_tile` iq2 行 `dispatch-wired → constructed` 的翻格判据（照 iq4_xs/q4_K 前例）：
1. 前门 `lowerToRepackGem{v,m}Grid` CONSTRUCT `typed_repack_gem{v,m}_loop_body` region 携 `repack_gem{v,m}_grid_core` brick（非 direct-emitter）；
2. brick anti-bypass 生效（block_index = region arg0）；
3. region→C byte-exact vs 退役直发射器 golden（BEFORE/AFTER-EQUALITY，forced rebuild）；
4. 独立 oracle 0-mismatch（§3 cert 三要件）；
5. 6 条直发射 dispatch 退役、旧 monolith op + recognizer 退役（留 retirement note）。

`realized-body manifest` 预期形（对齐 auto_readout）：`typed_repack_gem{v,m}_loop_body + repack_gem{v,m}_grid_core(iq2_*) + typed_repack_gem{v,m}_loop_yield; opaque_helper=false`。
