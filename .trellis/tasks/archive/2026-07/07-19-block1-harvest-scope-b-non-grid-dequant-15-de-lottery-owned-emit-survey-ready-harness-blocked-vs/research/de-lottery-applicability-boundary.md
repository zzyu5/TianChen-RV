# Research: de-lottery 模式适用边界（直接复制 / 需适配 / grid 排除）

- **Query**: 每格能否复制已证 owned emit 模式（block-quant / K-quant 超块 / tiny-codebook / ternary）·哪些直接复制·哪些需适配·哪些是 grid（排除）+ emitter/harness 就绪度
- **Scope**: internal（survey · 零代码改）
- **Date**: 2026-07-19

## 已证 owned emit 模式（收割范本）

**范本 = `experiments/active/r-dequant/kernels/q8_0_dequant.c`**（发射器 `emitDequantizeRowQ8_0VectorBody` `RVVToEmitCForwardElementwise.cpp:3387`）：

```
vsetvl_e32m1 → for nb: vle8(quant) → vsext_vf4 → vfcvt_f_x → vfmul_vf(d) → vse32
```

5 owned intrinsic · **gather=0** · 宽度读自块几何非焊字面量([K-5]) · byte-exact by construction（q8_0 无 add/min → 无 fp-contract 歧义）。

**flat nibble 范本**（`emitDequantizeRowNibbleVectorBody:2966`·扇 q4_0/q4_1/q5_0/q5_1）：
```
vle8 → vand/vsrl(nibble split) → vzext → [q5 5th-bit spread] → vsub → vfcvt →
  vfmul_vf(d) / [q4_1/q5_1 fused vfmacc_vf(d,m)] → vse32
```
- **SAFE 集**（q4_0/q5_0）：单 mul·无 fp-contract 歧义·byte-exact by construction。
- **FMA 集**（q4_1/q5_1）：`q*d+m` fused vfmacc·须匹配对手 contraction（vfmadd）·byte-exact 须证一致（blockquant PRD §二）。

**模式三要素（可复制的方法学）**：① owned 真向量 body（非 autovec·OWNED 探针 ≫2）；② byte-exact ZERO-MODEL 对拍（harness verify·3 臂反空心）；③ 便宜档成色钉死（opp = autovec scalar-C）。

---

## 一、K-quant super-block（q2_K/q3_K/q4_K/q5_K/q6_K）— **需适配 · 无 gather 墙**

- **结构**：QK_K=256 超块 = fp16 super-scale `d`(+`dmin`) × per-sub-block 6-bit scale × 2/3/4/5/6-bit 位解包 quant。**无码本 gather**（纯位运算 + scale）。
- **可复制性**：**需适配**——block 几何（256 vs 32）、sub-scale 解包（6-bit packed scales）、per-bit-width quant 解包全异于 flat nibble。author 一个超块 owned body（vle8 + 位解包 + vsext/vzext + vfcvt + 多级 vfmul）·family 内按 bit-width 参数化复制。
- **★墙判定（关键·勿误植）**：K-quant **dequant = streaming（无 q8_K 归约·无累加器）**。census §四 **F1「weight-reconstruction floor / vwredsum / register-fusion EXHAUSTED」墙是 vec_dot 的**（`named-X-leverage-census.md` L170·L64–75）·**不适用 dequant**。dequant 只解包+scale+store → 结构上像 q8_0 干净 de-lottery·**几乎必 PASS**（memory-bound streaming·成色便宜档）。
- **byte-exact 风险**：多级乘（d × sub-scale × quant + dmin）= FMA-风险集（同 q4_1/q5_1）·须证 fp-contract 一致·未证不硬凑。
- **发射器就绪**：dispatch 已 wire（`emitTypedDequantizeRowLoopBody` L3732–3736 · decode_model 门 L3653–3654 已认 q2_K..q6_K）·verifier 已 gate decode_model·**只差把 `emitDequantizeRowKQuantBodyShared`(L3554) 的 scalar-forwarder 换成 owned super-block body**。

## 二、tiny-codebook 16-entry（iq4_nl/iq4_xs/mxfp4/nvfp4）— **需适配 · vrgather · 非 grid 墙**

- **结构**：4-bit index → 16-entry 码本值（iq4_nl/iq4_xs 非线性 kvalues · mxfp4/nvfp4 FP4 e2m1）× scale（iq4_nl flat fp16 / iq4_xs 超块 signed-6 / mxfp4 E8M0 共享指数 / nvfp4 4×UE4M3 sub-scale）。
- **可复制性**：**需适配**——用 `vrgather.vv`（16 项码本常驻单向量寄存器的寄存器置换）替 flat nibble 的算术。**≠ 大 grid 内存 gather（vluxei）** → **不撞 ISSUE-107 gather 墙**（16 项平凡装入寄存器·vrgather 是正确原语·见 nongrid-full-table §三）。
- **正锚**：**mxfp4 3.47× WIN**（ledger 机制⑤ L17·「vrgather 16-entry LUT·mxfp4 路已存」）——vrgather 码本路已证可达·本族适配置信最高。
- **墙风险**：低-中。nvfp4 **vec_dot@rvv** 侧诊出「码本 TABLE spill」(F5 L174·clang reg-alloc spill)·但那是 reduction 核；dequant streaming 应更干净。scale 粒度逐格异 = 逐格适配点。
- **发射器就绪**：dispatch 已 wire（L3751–3757 · decode_model 门 L3659–3661 已认）·只差 `emitDequantizeRowCodebookGridBodyShared`(L3611) 换 owned vrgather body。

## 三、ternary super-block（tq1_0/tq2_0）— **需适配 · 无 gather**

- **结构**：tq2_0 = 2-bit ternary + fp16 scale；tq1_0 = base-3 packed ternary + scale。无码本 gather·无 reduction。
- **可复制性**：**需适配**·**且 tq1_0（base-3 解包）与 tq2_0（2-bit 解包）不同构**——family 内复用弱（近两次独立适配）。tq2_0（2-bit）最接近 flat nibble（适配最轻）·tq1_0（base-3 除法/取模解包）最重。
- **墙风险**：低（unpack + scale·streaming）。byte-exact fp-contract 须证。
- **发射器就绪**：dispatch 已 wire（L3751–3757 · decode_model 门 L3661 已认 tq1_0/tq2_0）·同 codebook forwarder 换 owned。

## 四、GRID（排除 · gather 墙 ISSUE-107）— **不可干净 owned**

- **格**：iq3_xxs · iq3_s · iq1_m〔ISSUE-107 具名〕+ iq2_xxs · iq2_xs · iq2_s · iq1_s〔同族结构墙〕。
- **结构**：大 grid-table（2048/256-entry）内存 indexed gather（`vluxei`）。
- **墙**：board-tested（iq3_xxs@rvv·性能与测量.md L365–366）·**3 owned 变体全 <0.8**（naive 0.181 / 标量-load 0.33 / HW-gather 0.36）·**honest-null·杠杆清单空(construction)**·同族结构推断覆盖 iq3_s/iq2*/iq1_m/iq1_s（census §2.2 F6）。
- **处置**：**排除本次收割**（owned 会撞 gather 天花板·非干净收割）。[L-8]-vs-PASS tradeoff（接受 owned 具名-X or 保 lottery-PASS 非 owned）= 战略裁待用户（ISSUE-107/106·非第一块）。

---

## 五、适用边界总表

| 族 | formats | 可复制性 | 墙 | 发射器 dispatch | harness | 判定 |
|---|---|---|---|---|---|---|
| flat block（已收割） | q4_0/q4_1/q5_0/q5_1/q8_0 | 直接（已完成） | 无 | owned ✓ | ✓ | **DONE** |
| K-quant 超块 | q2_K/q3_K/q4_K/q5_K/q6_K | **需适配**（超块位解包·family 内 bit-width 复用） | 无（F1 vec_dot 墙不适用 dequant） | wired·body=scalar | blocked | **收割批1** |
| tiny-codebook | iq4_nl/iq4_xs/mxfp4/nvfp4 | **需适配**（vrgather 码本·mxfp4 正锚） | 无（16 项寄存器·非 vluxei） | wired·body=scalar | blocked | **收割批2** |
| ternary 超块 | tq1_0/tq2_0 | **需适配**（tq1_0/tq2_0 不同构） | 无 | wired·body=scalar | blocked | **收割批3** |
| **grid（排除）** | iq3_xxs/iq3_s/iq1_m/iq2_xxs/iq2_xs/iq2_s/iq1_s | **不可干净 owned** | **gather 墙 ISSUE-107** | wired·body=scalar | 部分（iq3_xxs 已建） | **排除**（战略裁待用户） |
| 域外 | q1_0 | — | — | — | — | 排除（非分母） |

## Caveats / Not Found

- **「收割」语义澄清**：11 格无一是 flat-body 的**代码直接复制**——收割 = 复制 q8_0 的**方法学**（owned + byte-exact + 便宜档），每族仍需 author 新的 super-block/vrgather/ternary body。「直接复制」只在 **family 内**（首格适配后·同族余格随参数复制）成立。
- **发射器已 wire·body 未 owned**：3 收割族的 dispatch/verifier/decode_model 门全已建（L3653–3661 · L3729–3757）——**收割不需碰派发骨架·只需把对应 `...BodyShared` scalar-forwarder 换 owned body**（+ 各自 kernel/driver/harness/oracle）。这是「便宜」的真义：脚手架已在。
- **byte-exact FMA-contract 是每族的真前置**：K-quant（d×sub-scale×quant）、codebook（码本值×scale）、ternary（value×scale）皆多重乘·`-ffp-contract=on` 下 vfmadd-vs-vfmul+vfadd 差 1 rounding（同 q4_1/q5_1 已解范式）·**须逐格证一致·未达 byte-exact 该格具名(FMA-contract·非架构)·不硬凑**（blockquant PRD §六）。
