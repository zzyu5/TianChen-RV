# [LODE-8] repack-GEMM 前门构造设计侦察 — 8 旁路格类枚举 + decode-cost 并入

- **状态**: ★ 侦察态(SCOUT-only)—— **未立项 / 待用户裁 / 未实现 / 零代码改动 / 不 commit(由用户提交)**。
  本文只读了 `schema/emit-bypass-whitelist.v1.json`、`lib/` 的 direct-emitter 体 + 前门 typed-region 体 + codebook
  core brick 验证器 + iq2 grid/sign 表 decl helper,以及并入的 `docs/reports/2026-07-09-decode-cost-scout.md`,
  只写这一份 `.md`。所有"设计空间/成本草图/feasibility"仅为**枚举 + 粗判**,**不是方案、不是承诺、不是 speedup 数字**。
- **裁决锚**: G3 矿脉-8 前门构造设计侦察(枚举每旁路格类的 repack-GEMM 布局设计空间,不实现);并入 [DECODE-COST]
  侦察包 → 合并"矿脉 onboarding-design 包"。
- **日期**: 2026-07-09
- **代码 HEAD**: `2363fd3c`(只读;未 commit;树在会话中被并行线推进过,本文读的是当前树)
- **上游事实来源(均已存在,只读)**:
  - `schema/emit-bypass-whitelist.v1.json`(baseline_count 8;8 direct-emitter gemm_tile 旁路 + 已退役 ledger)
  - `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`:flat 直发体 `emitRepackGemvQ5_0Q8_0`(:3082)/
    `…Q8_0Q8_0`(:4122)/`…Q4_1Q8_1`(:4541);codebook `emitRepackGemvMxfp4Q8`(:18325)/`…GemmMxfp4Q8`(:18768);
    iq2 grid `emitRepackGemvIq2XxsQ8K`(:20173)/`…GemmIq2XxsQ8K`(:20634)/dual-scale `emitRepackGemvIq2DualScaleQ8K`(:21155);
    **前门** `emitTypedRepackGemmLoopBody`(:2279,ternary/codebook/K-quant dispatch 三分支已构造)
  - `lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp`:iq2 grid/sign 表 decl(:694/:720/:743/:768/:791/:817)——
    **grid 256/512/1024 × int64 + sign 1024/2048 字节,keyed off op 身份 emit(NON-attr)**
  - `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` `verifyRepackCodebookCoreCommon`(:3452,codebook 只收 16-entry
    DenseI8ArrayAttr + decode_model iq4_nl/iq4_xs)
  - `lib/Conversion/RVV/RVVToEmitCInternal.h`:4453–4456(**signs64/grid-table op-attr blocker 的确切边界**:
    "grid/sign 平面 DERIVED at emit(NOT op-attrs)…blocker 只针对 block-dot repack GEVM 路,不针对 streaming dequant 路")
  - `docs/reports/2026-07-09-decode-cost-scout.md`(81a8050b,[DECODE-COST],被并入)

---

## 0. 一句话结论(先给,细节在下)

★ 侦察态,未立项:8 个 gemm_tile 旁路(baseline_count 8,`provenance="direct-emitter (transitional scaffolding)"`,
喂 C_dispatch **永不** C_construct)的前门 repack-GEMM 构造设计空间**三档分明**:

- **FLAT 类(q4_1/q5_0/q5_1/q8_0,batch 0)= near-term**:decode-leaf 是 q4_0 的**同族最小 delta**(q8_0 无 nibble、
  q4_1 加 min、q5_0 加 qh、q5_1 加 qh+min),而**所需机件全部已建**(qh attr `weight_qh_byte_offset` 来自
  q5_K/tq1_0;single-scale fold 来自 q4_0/ternary;dual d/dmin min fold 有 K-quant 先例)。whitelist 自称 batch 0
  "the readiest"。**q8_0 最近(GEVM-only、无 nibble)**。
- **mxfp4(batch 4)= near-term**:iq4_nl 的**直系 sibling**——**同一 16-entry codebook gather**(正好卡进现有
  16-entry DenseI8ArrayAttr 约束)+ E8M0 shared-exp scale fold;净新 = 一个 fold_model,与 iq4_xs 的 signed-6 fold
  同量级;GEMM ships PLAIN(tiny-codebook 已在 ≤32 悬崖,S6 no-op)。
- **iq2 GRID 类(iq2_xxs/iq2_xs/iq2_s,batch 3)= needs-design-work(非 blocked)**:设计分叉在**"codebook"是
  256/512/1024-entry grid(2–8 KB)+ 1–2 KB sign 平面,装不进 iq4 那个 16-entry DenseI8ArrayAttr** —— 这**就是**
  "signs64/grid-table op-attr blocker"。但它是**已解模式、非开放 blocker**:repack/streaming 路一律**把 grid+sign
  keyed off decode_model 身份 emit 成 FIXED canonical static-const 表(不当 attr 背)**,现役 iq2 monolith 与 dequant
  streaming 路都这么做。前门构造照搬即可,须新建一个 **GRID/SIGN core brick**(不是碰 16-entry codebook brick)。

**decode-cost 并入**:本包(lode-8)= **构造前**的 onboarding 设计空间(dispatch→construct);decode-cost 包 =
**构造后**已构造格(q6_K/q3_K)的 weight-bound perf 残留画像。★关键交叉事实:**这 8 个 lode 格没有一个落在 decode-cost
的两桶里**(bucket A grid-index-storm = iq3;bucket B weight-recon = q6/q3);iq2 的 index 是 in-vector `w&511`(scout
明列在"参照 WIN 桶"),gather 已近零 —— 所以**8 格前门构造是纯 C_construct/成熟度任务,不是 perf 杠杆**,B1–B4 权重杠杆
对它们**不适用**。

---

## 1. FLAT 类(q4_1 / q5_0 / q5_1 / q8_0)—— retirement_batch 0

**格类**:线性 scale(± min)+ nibble(± qh 5th-bit)的平面 32-元 block-as-lane repack,结构近 **q4_0**(q4_0 decode
+ prefill 是唯一已 front-door-CONSTRUCTED 的 flat,前门体 `emitTypedRepackGem{v,m}LoopBody` = 精确模板)。

**到已构造 sibling(q4_0)的结构距离**——逐格 decode-leaf delta:

| fmt | 直发体 | 已 wired | decode-leaf = q4_0 + … | fold(vs q4_0 single-fp16) | 所需机件是否已建 |
|-----|--------|---------|------------------------|---------------------------|-----------------|
| **q8_0** | `emitRepackGemvQ8_0Q8_0`(:4122) | **GEVM only** | **减** nibble unpack(int8 直载 → 直接 widening 积) | single fp16(**同 q4_0**) | 是(严格更简单,是 flat 核最小者) |
| **q4_1** | `emitRepackGem{v,m}Q4_1Q8_1`(:4541) | **GEVM+GEMM** | nibble **同** q4_0(`vand 0x0F`/`vsrl 4`) | **dual d/dmin**(Family-B scale+MIN;min fold 已在 q4_K "kquant_dmin_bsums_min" 有先例、flat 版更简) | 是 |
| **q5_0** | `emitRepackGemvQ5_0Q8_0`(:3082) | **GEVM only** | q4_0 nibble **+ qh 5th-bit lane-inject**(`u8[0,31]`→reinterp i8→`vsub 16`,PROVEN block-dot `fifthBitLane` 路) | single fp16(**同 q4_0**) | 是(qh inject = q5_K 已用机制;`weight_qh_byte_offset` attr 已存) |
| **q5_1** | `emitRepackGemvQ5_1Q8_1`(:3590) | **GEVM only** | q5_0 的 qh **∪** q4_1 的 dual-fold | dual d/dmin | 是(两 delta 的并集,都已建) |

**repack-GEMM 布局设计空间**:唯一自然路 = **retire 进 `emitTypedRepackGem{v,m}LoopBody` 的 flat 分支**,照 q4_0
`typed_repack_gem{v,m}_loop_body` + core/fold brick 模板,按 `decode_model ∈ {q8_0,q4_1,q5_0,q5_1}` 选 decode-leaf、
按 `fold_model ∈ {flat_single_fp16, flat_dual_dmin_min}` 选 fold,qh 平面走已有的 `weight_qh_byte_offset` optional
attr。**不需要新 brick 家族**(flat 核就是 q4_0 核的参数化);预重组/LUT 布局对 flat **无意义**(位宽已线性、无 codebook)。

**register/instruction 成本草图**:与 q4_0 同量级——GEVM 每 strip = `vle8`(+q5 再一条 qh `vle8`)+ nibble
`vand/vsrl`(q8_0 省)+ (q5) `vand/vsll/vor` qh inject + `vwmul_vx`/`vwadd_wv` i32 dot + fp16 scale fold(q4_1/q5_1
再加 dmin·min 项)。**全在 ≤32 vreg 内**(flat 核无多平面权重重建、无 grid gather),GEMM 与 q4_0 prefill 同构、
S6 tiling 沿用 q4_0 的既定态。

**feasibility 判定**:**near-term(全 4 格)**。readiest 批;机件零新建。GEVM-only 的 q8_0/q5_0/q5_1 是单体退役,
q4_1 需 GEVM+GEMM 两体。**排序:q8_0(最简 · GEVM-only)> q5_0(+qh · GEVM-only)> q4_1(+min · GEVM+GEMM)>
q5_1(+qh+min · GEVM-only)**。

---

## 2. iq2 GRID 类(iq2_xxs / iq2_xs / iq2_s)—— retirement_batch 3

**格类**:QK_K=256 super-block,4-bit/2-bit **grid index → grid-of-8 codebook**(每 index 展 8 个 int8 权重)+ 一个
**±1 sign 平面** + per-sub-block ls scale,dot 后 `0.125` 缩放存。三格的 grid/sign 规模逐级放大:

| fmt | 直发体 | 已 wired | grid | sign 平面 | index 载 | ls scale | 到已构造 sibling 距离 |
|-----|--------|---------|------|-----------|---------|----------|----------------------|
| **iq2_xxs** | `emitRepackGem{v,m}Iq2XxsQ8K`(:20173/:20634) | GEVM+GEMM | **256**-entry int64(2 KB) | **derived signs64**(ksigns→1024 B ±1) | 8-bit → `vzext`+`vsll 3` | vsext(no min) | 无直系已构造 sibling;**codebook 家族**(iq4)是最近亲、但 grid≫16-entry |
| **iq2_xs** | `emitRepackGem{v,m}Iq2XsQ8K`(via `…Iq2DualScaleQ8K`:21155) | GEVM+GEMM | **512**-entry int64(4 KB) | ksigns signs64(1024 B) | 9-bit → `vle16`直载+`vsll`(**不能 vzext,byte 装不下**) | **dual-half**(4-bit×2:ls1/ls2) | iq2_xxs + 512-grid + dual-half ls |
| **iq2_s** | `emitRepackGem{v,m}Iq2SQ8K`(via `…Iq2DualScaleQ8K`) | GEVM+GEMM | **1024**-entry int64(8 KB) | **explicit signs256**(raw sign 字节直索引 → 2048 B) | 10-bit(qs|qh 平面) | dual-half | iq2_xs + 1024-grid + explicit-sign(无 ksigns 选择子) |

**repack-GEMM 布局设计空间(★三选项枚举,每个成本草图 + signs64 op-attr 障碍状态)**:

- **(A) 解码式-repack(kernel 内 vluxei16 gather 解码 grid)—— 当前直发法 + 自然前门路**。
  成本草图:每(sub-block,group)= `vluxei16` grid gather + `vluxei16` sign gather + `vmul_vv_i8` sign-fold-onto-grid
  + `vsext_vf4` ls-scale + i32 `vwmul/vwadd_wv/vmacc` dot;index 组装 = `w&511` **in-vector**(iq2 **非** bucket-A 标量
  index-storm,那是 iq3;见 §4)。**register 全在 ≤32 内**(grid/sign 是 MEMORY 表,不占 vreg)。
  **signs64/grid-table op-attr 障碍状态 = 已解、非开放**:前门 codebook brick 现在把 16-entry codebook 当
  `DenseI8ArrayAttr` 背(`verifyRepackCodebookCoreCommon` 硬门 `codebook.size()==16`);grid 是 256–1024 entry×int64
  + sign 1–2 KB,**塞不进这个 attr —— 这正是 blocker**。**解法(已被 dequant streaming 路 + 现役 iq2 monolith 采用)**:
  **新建一个 GRID/SIGN core brick**,只在 brick 上背一个 `decode_model ∈ {iq2_xxs,iq2_xs,iq2_s}` 字符串,前门 emit 时
  按身份调 `emitIQ2*CanonicalGrid/SignsTableDecl` 把表 emit 成 **FIXED canonical static-const**(不当 op-attr) →
  **绕过 blocker**。`RVVToEmitCInternal.h:4453` 明文:该 blocker "specific to the block-dot repack GEVM path, not this
  streaming dequant path" —— 即把它 keyed-off-identity 就消失。**这是 iq2 前门构造的推荐路**。
- **(B) 预重组-repack(离线把 grid+sign 预展开成显式 int8 权重)**。成本:kernel 侧删掉全部 grid/sign gather,leaf 塌成
  `vle8`+widening dot(退成 q8_0-类)。**代价:↑ 流字节 ~×3–4**(iq2 ~2.06–2.6 bit/权重 → 8-bit),**伤 memory-bound
  e2e decode**(同 decode-cost 包 B1/GRID-L4 caveat);且是 **[repack]-布局改动 = 另一战役地界**,非 emitter 前门构造。
  **不推荐作前门首选**(放弃压缩、只在 compute-bound prefill 站得住)。
- **(C) 查表化 / in-register LUT(`vrgather`)**。**不可行**:grid 256–1024 entry,mf2 anchor VLMAX<16 装不下,寄存器
  vrgather 非法(与 iq4 同因 —— iq4 也被迫走 MEMORY vluxei16)。iq2 grid 本就**必须**是 memory gather。排除。

**feasibility 判定**:**needs-design-work(全 3 格,非 blocked)**。唯一设计工作 = 定义 **GRID/SIGN core brick**
(decode_model→canonical-table-decl 映射,不背大 attr)+ **dual-half ls-scale fold**(iq2_xs/iq2_s)+ **sign-plane 变体
选择**(iq2_xxs/xs = derived signs64;iq2_s = explicit signs256)。比 flat/codebook 家族多"新 brick 类型 + 3 grid×2
sign 变体",但**无真 blocker**;GEMM 与 iq4 同走 PLAIN(gather 已在 ≤32 悬崖,S6 no-op)。
**排序:iq2_xxs(最简 sign 方案 derived-signs64 + 单 ls)< iq2_xs(512-grid + dual-half ls)< iq2_s(1024-grid +
explicit signs256 + dual-half ls,decode 变体最多)**。

---

## 3. mxfp4 —— retirement_batch 4

**格类**:FP4(doubled-E2M1)16-entry int8 **codebook** + **E8M0 shared-exponent** per-block scale(`2^(e-128)`)。
直发体 `emitRepackGem{v,m}Mxfp4Q8`(:18325/:18768,GEVM+GEMM 都 wired)。

**到已构造 sibling(iq4_nl / iq4_xs)的结构距离 = 极近(直系)**:
- codebook gather **逐字节共享 iq4_nl 的 helper**(:18442 注释自述 "SHARED with iq4_nl";`vzext_vf2`→u16 offset→
  `vluxei16_v_i8`,NOT fake-linear)——且 mxfp4 codebook **正好 16 entry**,**卡进现有
  `verifyRepackCodebookCoreCommon` 的 `codebook.size()==16` 约束**(无须动 codebook attr 形状,只需 decodeModel 白名单
  加 "mxfp4",与 iq4_xs 的 widen 同一手法)。
- **唯一净新 = fold_model**:`codebook_e8m0_shared_exp`——E8M0 位构造 `e32=vzext_vf4`;`bits=(e<2)?(0x00200000u<<(e&0x1F)):
  ((e-1)<<23)`;`scale=vreinterpret→f32`(:18513 起,向量化 compare-mask 走 `vbool8/16`)。与 iq4_xs 的
  `codebook_superblock_signed6_no_min` fold **同量级工作量**(都是"复用 codebook brick + 一个新 scale fold")。

**repack-GEMM 布局设计空间**:与 iq4 同——**retire 进 `emitTypedRepackGem{v,m}LoopBody` 的 codebook 分支**,复用
`repack_gem{v,m}_codebook_core` brick(decode_model "mxfp4" + 16-entry doubled-E2M1 codebook 当 DenseI8ArrayAttr 背,
`lowerToRepackGem{v,m}Codebook` 从抽象请求 RECONSTRUCT codebook)+ 新 fold_model E8M0。GEMM ships **PLAIN/untiled**
(tiny-codebook 单 gather、已在 ≤32 悬崖,S6 结构 no-op,同 iq4_nl/iq4_xs)。预重组/LUT 布局对 tiny-codebook 无收益。

**feasibility 判定**:**near-term**。framework re-pay 近零(iq4_xs 已证同款路径)。
**caveat(非阻塞)**:mxfp4 是 **vec_dot 轴的 NEGATIVE CONTROL**(vec_dot 保持 undecomposed);**repack-GEMM 轴是不同
cell,正常退役、与该 caveat 不冲突**(whitelist entry 明载)。

---

## 4. 并入 decode-cost 侦察包 → 合并"矿脉 onboarding-design 包"

两份侦察是**同一格生命周期的两端**,合并成一个 onboarding 视图:

- **lode-8(本文)= 构造前 · 设计空间**:8 个 dispatch-wired 旁路 **如何** front-door 构造(dispatch→construct,
  C_construct++,byte-exact by construction via region-vs-oracle lit)。产出 = 格类 × 结构距离 × 布局选项 × feasibility。
- **decode-cost(81a8050b)= 构造后 · perf 残留**:已构造 K-quant 格(q6_K/q3_K)构造完之后 ship 出的 weight-bound
  ~900-spill 残留画像 + B1–B4 杠杆枚举(**仍侦察态、未立项**)。产出 = 瓶颈占比 × 杠杆 × e2e 风险。

**★关键交叉事实(把两份粘起来的那颗钉)**:decode-cost 的**两桶都不含这 8 个 lode 格**——
- bucket A(grid-index/sign 标量 storm)= **iq3_s/iq3_xxs**(qh 第 9 位逐 lane 变量注入 / ksigns 二级抽取);iq2 的 index
  是 **in-vector `w&511`**,decode-cost scout 明把 iq2 列在"参照 WIN 桶(非 decode-bound)",gather 已被 [GAP-SB] 打到
  近零。**iq2 GRID ≠ iq3 storm**。
- bucket B(多平面权重重建)= **q6_K/q3_K**(双面逐 lane 变量移位装配);flat/codebook/iq2 全无此病。

**推论**:**8 个 lode 格没有一个带 weight-bound perf 残留** → 它们的前门构造是**纯 C_construct / 编译器成熟度**推进
(coverage novelty:旁路存量 ratchet-down 8→0),**不是** perf 杠杆;decode-cost 的 B1–B4 权重杠杆对它们**不适用**。
反过来,decode-cost 的 B1(离线预重组)与本文 iq2 的选项 (B) 是**同一个"↑流字节伤 memory-bound"的布局杠杆的两张脸**——
都属"另一战役地界",都非前门 emitter 构造的首选。

**onboarding-design 包 — 一览**:

| 阶段 | 覆盖格 | 产物 | 关键判据 | 是否 perf 立项 |
|------|--------|------|----------|----------------|
| 构造前(lode-8) | q4_1/q5_0/q5_1/q8_0 · iq2_xxs/xs/s · mxfp4 | 布局设计空间 + feasibility | 到已构造 sibling 的 decode-leaf delta | **否**(纯 C_construct) |
| 构造后(decode-cost) | q6_K/q3_K(+q5_K 尾) | weight-bound 残留 + B1–B4 | 权重是否散布 ≥2 平面 + 逐 lane 变量移位 | 侦察态未立项 |

---

## 5. 8 格 feasibility 排名(★枚举 + 交回)

| 排名 | fmt | batch | 类 | wired | 到 sibling 距离 | 判定 | 净新工作 |
|-----|-----|-------|-----|-------|----------------|------|----------|
| 1 | **q8_0** | 0 | flat | GEVM | q4_0 **减** nibble(最简) | **near-term** | flat 核参数化(decode_model q8_0) |
| 2 | **q5_0** | 0 | flat | GEVM | q4_0 + qh inject | **near-term** | decode_model q5_0(qh attr 已存) |
| 3 | **q4_1** | 0 | flat | GEVM+GEMM | q4_0 nibble + dual-min | **near-term** | flat dual-dmin fold_model |
| 4 | **q5_1** | 0 | flat | GEVM | q5_0 qh ∪ q4_1 min | **near-term** | 两 delta 并集 |
| 5 | **mxfp4** | 4 | codebook | GEVM+GEMM | iq4_nl sibling(同 16-entry codebook) | **near-term** | 一个 E8M0 fold_model |
| 6 | **iq2_xxs** | 3 | grid | GEVM+GEMM | codebook 家族亲、但 grid≫16 | **needs-design-work** | 新 GRID/SIGN brick(derived-signs64,单 ls) |
| 7 | **iq2_xs** | 3 | grid | GEVM+GEMM | iq2_xxs + 512-grid | **needs-design-work** | + dual-half ls fold |
| 8 | **iq2_s** | 3 | grid | GEVM+GEMM | iq2_xs + 1024-grid + explicit-sign | **needs-design-work** | + explicit signs256 变体 |

**无一格 blocked**。"signs64/grid-table op-attr blocker" = iq2 的**设计分叉**(把 grid 当 attr 背 = blocker;keyed-off
decode_model 身份 emit canonical 表 = 已解模式,dequant streaming 路先例),**非开放障碍**。

**若立项(交回裁决,不自决)**:两个自然子批次 ——(a)**flat batch 0**(readiest,q8_0→q5_0→q4_1→q5_1,机件零新建,
沿 q4_0 typed_repack 模板);(b)**mxfp4**(iq4 codebook 框架 near-zero re-pay,一个 E8M0 fold)。**iq2 grid batch 3** 需
先定 GRID/SIGN brick 设计(新 brick + dual-half ls + sign 变体),是三档里唯一需设计工作的。**三档全 = C_construct 推进
(旁路 8→0 ratchet),非 perf**;byte-exact by construction(region-vs-oracle lit),无 e2e/perf 主张。

★★ **红线复述**:以上全部为**设计侦察画像**,**未立项、未实现、零 lib/schema 改动**(只读 `lib/` 直发体 + 前门体 +
brick 验证器 + iq2 表 decl + whitelist + decode-cost scout,只写本 `.md`)。是否立项、采纳哪档/哪路(或都不做)由**用户裁**。
本文不 commit(由用户提交)。

### 关键 caveat / 不确定
- feasibility "near-term / needs-design-work" 是**结构判**(decode-leaf delta 大小 + 机件是否已建),**非工时估**;真退役
  仍须 region-vs-oracle byte-exact lit + rebuild 通过,属实现期。
- flat "机件全已建"依据 = q5_K(qh attr)、q4_K(min fold)、q4_0(single-scale + typed_repack 模板)已在树内;但 flat
  用的是 **ggml-mirror op 家族**(`GgmlRepackGemvQ50Q80Op` 等),retire 时须把它们接进 typed_repack core-brick 构造路
  —— 这一步 flat 尚无先例(ternary/K-quant/codebook 有),是 flat 的**唯一未验证接缝**(判为 near-term 而非 trivial 的原因)。
- iq2 "GRID/SIGN brick keyed-off-identity 绕 blocker"依据 = dequant streaming 路已这么做(`RVVToEmitCInternal.h:4453`);
  但 **repack-GEMM 前门尚未有 grid brick 先例**(现有 core brick 只有 flat/ternary/kquant/codebook 四类),故判
  needs-design-work。canonical 表规模(iq2_s 8 KB grid + 2 KB sign)是 rodata 常量、非 op-attr,不撞 attr 大小问题。
- mxfp4 的 vec_dot NEGATIVE-CONTROL 身份**只锁 vec_dot 轴**;若 repack-GEMM 轴立项须在 provenance 上把两轴分记
  (whitelist 已如此),避免误读成"负控被拆"。
- **无任何 speedup / e2e 数**:本文只判"能否/多近上前门",不判上门之后快慢;8 格既非 weight-bound(§4)、gather 已近零,
  预期上门后 kernel 侧与对手 parity(采用 ggml 自身 gather/decode)、非 beat —— 但那是构造后 board 事,不在本侦察背书。
