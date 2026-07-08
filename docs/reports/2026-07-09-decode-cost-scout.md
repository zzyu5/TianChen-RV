# [DECODE-COST] 权重重建瓶颈画像 + GRID 并入 — 解码成本侦察包

- **状态**: ★ 侦察态(SCOUT-only)—— **未立项 / 待用户裁 / 未实现 / 零代码改动**。本文只读了 `lib/`、
  board objdump seal、cell findings 与 emit 函数体,只写这一份 `.md`。所有"杠杆"仅为**枚举 + 粗判**,
  **不是方案、不是承诺、不是数字**。
- **裁决锚**: 裁决四 weight-bound 杠杆侦察(画像,不立项);并入归档的 [GAP-GRID] 侦察 → 合并"解码成本侦察包"。
- **日期**: 2026-07-09
- **代码 HEAD**: `0e21c1e5`(只读;未 commit;由用户提交)
- **上游事实来源(均已存在,只读)**:
  - `experiments/active/l1-t3-q6k-repack-gemm/{objdump_tile_q6k_t3_seal.objdump, tile_q6k_t3_findings.md}`(q6_K S6 NULL)
  - `experiments/active/l1-t3-q3k-repack-gemm/{objdump_tile_q3k_t3_seal.objdump, tile_q3k_t3_findings.md}`(q3_K S6 NULL,含逐助记词分解)
  - `experiments/active/l1-t3-q5k-repack-gemm/MANIFEST.md`(q5_K S6 WIN + 残留 weight-floor,分界锚)
  - `experiments/active/kquant-l1-q6q2q3-repack/MANIFEST.md`(prefill 0.18/0.176 LOSS 基线)
  - `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` 的 `emitRepackKQuantGemmBodyQ6K`(:8467)/ `…Q3K`(:15228)(只读)
  - `docs/reports/2026-07-07-gap-grid-decode-scout.md`([GAP-GRID] 侦察,被并入)

---

## 0. 一句话结论(先给,细节在下)

★ 侦察态,未立项:q6_K/q3_K 的 S6 tiling NULL(残留 spill ~949/894、maxVreg 卡死 v31、无 ≤32 悬崖)= **多平面
权重重建 bound**,不是 decode-scale strip bound。逐指令分解:**权重装配的向量 ALU(vand/vsrl/vsll/vor/vsub)
= q6_K ~2944 条(≈1.28× 于 dot 的 2304 vwmacc)、q3_K ~7168 条(≈3.3× 于 dot 的 2176 vwmacc)**——**权重重建
比 dot 本身还贵**,这就是 ~900 残留 spill 的来源(S6 只砍了 ~84 = scale strip,是 ~900 桶里一滴)。
并且它与 [GAP-GRID] 的 iq3 "index/sign 逐 lane 变量移位组装"是**同一种病(散布字段的重组装)的两张脸**:
一张长在 index(GRID / bucket A),一张长在 weight value(本文 / bucket B)。合并成一个"解码成本"分类学(§4)。

---

## 1. 权重重建指令占比(逐助记词,q3_K 板测 + q6_K 结构推导-vwmacc 校验)

### 1.1 板测锚:两格都是 ~900-spill weight-bound(非 register-cliff-bound)

| fmt | 平面布局 | fold | S6 前 spill | S6 后 spill | maxVreg | vwmacc | S6 判 |
|-----|----------|------|-----------:|-----------:|--------:|-------:|-------|
| q6_K | 双面 6-bit `ql`@1312 + `qh`@288 | no-min | 913(-O2) | **949(↑)** | v31→v31 | 2304 不变 | NULL(spill 升) |
| q3_K | 双面 3-bit `qs`@800 + `hmask`@288(subtractive) | no-min | 978(-O2) | **894(↓84)** | v31→v31 | 2176 不变 | NULL(只砍 scale strip) |

- 两格 untiled spill 都 ~900(**比 q4_K golden 的 84 / S1 的 23 高一个数量级**),maxVreg 全程 v31(32 寄存器
  文件饱和,无 spare)。S6 int16 scale panel 只碰到 q3_K 的 **~84 条 scale-strip spill**(= 32 条 staged
  strip + reload),留下 **~894 残留**;q6_K 连符号都反了(panel store/load 流量让 spill **升** 913→949)。
- VLEN128 参照:1 向量寄存器 = 16 B;**整个 32 寄存器文件 = 512 B**。~900 全寄存器 spill = weight-bound 签名。

### 1.2 逐助记词分解(权重装配 vs dot vs decode-scale strip)

**q3_K(板测,`objdump_tile_q3k_t3_seal.objdump` §MECHANISM 直读):**
| 类别 | 助记词分解 | 计数 | 对 dot 比 |
|------|-----------|-----:|----------:|
| **多平面权重装配** | **2048 `vand` + 2048 `vsrl` + 1024 `vsll` + 1024 `vor` + 1024 `vsub`** | **~7168** | **≈3.29×** |
| dot(真算术,byte-exact 保号) | `vwmacc_vv` (256) + `vwmacc_vx`(weight×act 主体) | **2176** | 1.0× |
| decode-scale strips(S6 panel 打的) | 32 signed-6bit scale `vsext_vf2` + vse16/vle16 | ~84 spill | 边角 |

**q6_K(结构推导自 `emitRepackKQuantGemmBodyQ6K`,以 vwmacc=2304 板测数校验闭合):**
- `assembleWeight`(:8626)每条权重 strip = `vand|vsrl`(取 nib)+ `vsrl`(qh sel 移位)+ `vand`(0x03)+
  `vsll`(4)+ `vor` + `vsub`(bias 32)。静态展开:4-quad 组 = 6 vand + 5 vsrl + 4 vsll + 4 vor + 4 vsub。
- 静态 (p,h) 组数 = 2(j)×2(sh)×2(k)×8(p)×2(numHalves) = 128;`assembleWeight` 共 512 条。

| 类别 | 助记词分解 | 计数 | 对 dot 比 |
|------|-----------|-----:|----------:|
| **双面权重装配** | **768 `vand` + 640 `vsrl` + 512 `vsll` + 512 `vor` + 512 `vsub`** | **~2944** | **≈1.28×** |
| 权重面装载 | `vle8`(qlA + qlB + qh)= 3×128 | 384 | — |
| dot(byte-exact 保号) | `vwmacc_vx_i16`(2048)+ `vwmacc_vv_i32` scale fold(256) | **2304** | 1.0× |

> **校验闭合**:2048 + 256 = **2304 = 板测 objdump vwmacc**(§1.1)。结构推导的展开计数与硬件反汇编一致 →
> q6_K 逐助记词占比可信。

### 1.3 判据小结
- **权重重建(向量 ALU 装配)≥ dot 本身**:q6_K 1.28×、q3_K 3.29×。这些是 1:1 intrinsic(每条 `__riscv_vand_vx_u8`
  等 lower 成一条机器指令,操作数各异 → 几乎无 CSE)→ 静态指令数 ≈ 发射数。**~900 残留 spill = 这批 wq 装配中间量
  (nib/raw/hbit)+ 装配出的 int8 weight strip + f32 accumulator fan 全部同时活**,而 S6 scale panel 触不到。
- **q5_K 是分界锚(部分态)**:4-bit + 单条 `qh` 5th-bit 面、走 **min-fold**;S6 命中 ≤32 悬崖(v31→v30,+41.7% WIN),
  但 spill 155→**105 未塌到 0**——单条额外平面留了一个**小 weight-materialization floor**。即:weight-recon 成本随
  **平面数 + 位散布程度**上升:q5_K 单面 = ~105 floor;q6_K/q3_K 双面 + 逐 lane 变量移位 = ~900 floor。

---

## 2. 候选杠杆枚举(★枚举、别实现;每个给粗略 register/instruction 草图 + 为何【可能】达 ≤32 悬崖)

两条 finding 都点名了两个杠杆("block-loop re-roll to shrink the unroll" / "a dedicated ql·qh(qs·hmask)
materialization panel")。下面把它们 + 两个补充候选一起枚举:

| # | 杠杆 | 打的是哪段 | register/instruction 成本草图 | 为何【可能】达 ≤32 悬崖 | 空间/边界 | 粗评级 |
|---|------|-----------|------------------------------|------------------------|-----------|--------|
| **B1** | **离线预重组 repack 布局**(把 packed 低位权重**离线**预展开成连续一字节 sign-biased int8,使热循环免装配) | **整段权重重建 + wq 活跃集** | 热循环塌成 `vle8`+`vwmacc`;删掉 q6_K ~2944 / q3_K ~7168 条装配 ALU **全部** | 消除主导活跃集(wq strip)→ live set 塌到 loaded strip + accumulator。**结构上最硬的悬崖候选** | 流字节膨胀:q6_K 6-bit→8-bit ≈ ×1.23–1.33、q3_K 3-bit→8-bit ≈ ×2.3–2.67。**↑ 流字节 = 伤 memory-bound e2e decode**(同 GRID L4);是 **[repack]-布局改动(离线权重合并),非 kernel-emitter 修** = 另一战役地界;等价把 K-quant 退成 q8_0-类、放弃压缩,只在 compute-bound prefill/GEMM 站得住 | **高(结构悬崖,但跨 repack 战役 + 伤 memory 性)** |
| **B2** | **专用 ql·qh(qs·hmask)权重物化 panel**(S6 scale-panel 的同构,但 stage 装配出的 int8 weight strip:`vse8` 装配后即存、`vle8` reload 推迟到 vwmacc) | **~900 残留主导集本身** | panel 容 4 quad × numHalves × ≤8 chunk 的 int8 strip;切断 wq 活跃跨度,allocator 不必让 wq 常驻全 unroll 体 | **直打 ~900 残留**(不像 S6 只碰 ~84 scale strip)→ 有望像 q4_K/q2_K 那样塌 spill | **kernel-emitter only,无布局改动**;byte-exact by construction(reload==store)。风险:panel store/load 流量(q6_K S6 打错集时曾让 spill 升——但这次打的是主导集,流量对冲被删的 spill) | **高(kernel-only,在战役内)** |
| **B3** | **block-loop re-roll / 部分展开**(现 16-subblock 体全 compile-time 展开 → 全 wq/acc 同时活;引入 runtime 子块循环或 unroll 因子 U<16) | **峰值活跃集** | 峰值 live wq 从"全体"降到"每迭代";per-iter live < 32 vreg 即停 spill | 直接**给峰值活跃集设上界** | kernel-only。风险:重引入循环开销 + **可能扰动 vwmacc fold 序(byte-exact 风险,需重导累加树)** | **中(结构,但 perf 开销 + byte-exact 重导)** |
| **B4** | **多平面装配查表化**(用 `vluxei8`/`vrgather` 按 packed 字节查 sign-biased int8,替 vand/vsrl/vsll/vor/vsub 链) | **装配 ALU 链 + 中间量** | 1 gather 替 ~5–6 ALU/strip + 塌掉 nib/raw/hbit 中间量;q6_K ≤256-entry byte LUT、q3_K ≤8-entry(可 in-reg `vrgather`) | 更少 op **且** 更少中间 live 量 | kernel-only,无布局改动。**关键对照 GRID**:此处 LUT index **就是 packed 字节(直接),无逐 lane 变量移位风暴** → 避开 GRID 侦察点名的标量 index-storm 陷阱,近 iq2 in-vector WIN。风险:qh 2-bit 仍须 shift+mask+or **拼进 index** 才能查 → 可能只折叠尾段 `vor+vsub`、净收益不定;q3_K 8-entry codebook 则 `vrgather` 极便宜 | **中(最干净的 kernel-only,但 plane-merge-进-index 可能封顶收益)** |

---

## 3. 预计受益格清单(★枚举 + 按解码结构判)

**多平面权重重建 bound(逐 lane 变量移位装配,本文 bucket B):**
| fmt | 平面布局 | 装配 ALU(§1) | fold | 是否 weight-recon-bound? |
|-----|----------|-------------|------|--------------------------|
| **q6_K** | 双面 6-bit `ql`+`qh` | ~2944(1.28× dot) | no-min | **是(确证)** — S6 NULL,spill 949 |
| **q3_K** | 双面 3-bit `qs`+`hmask` subtractive | ~7168(3.29× dot) | no-min | **是(确证)** — S6 NULL,spill 894 |
| q5_K | 4-bit + 单 `qh` 5th-bit 面 | 小(单额外面) | min-fold | **部分** — S6 WIN(v30)但留 ~105 weight-floor(§1.3);主体 min-fold,B2/B4 可清那 ~105 尾 |
| q4_K / q2_K | 单面 4/2-bit | 平凡 | min-fold | 否 — S6 WIN(spill 3/7),非 weight-bound |
| iq3_s / iq3_xxs | grid + index/sign 位面注入 | 见 GRID | — | **邻族(bucket A,不同 flavor)** — index/sign 逐 lane 变量移位 storm(标量化),同病不同长处 |
| iq1_s / iq1_m | iq3_xxs sibling | 见 GRID | — | 邻族 bucket A 疑似(未测) |
| iq4_xs/nl · mxfp4/nvfp4/fp4 | 16-entry 微码本 | 1 gather | — | 否 — tiny-codebook(单次 lookup 已便宜) |

**小结**:多平面权重重建 bound = **q6_K、q3_K 确证(2 格)**;q5_K 部分(~105 尾,B2/B4 可清);
其余要么 min-fold-lean(q4_K/q2_K/q5_K 主体,S6 已 WIN)、要么 grid-index-bound(iq2/iq3 = bucket A 邻族)、
要么 tiny-codebook(iq4/fp4,不入本桶)。**判据不是"位宽",是"权重值是否散布在 ≥2 位面且需逐 lane 变量移位重组"。**

---

## 4. 并入 GRID → 合并"解码成本侦察包"(统一分类学)

把两份侦察合并成一个"**解码成本**"(packed 字节 → vwmacc dot 之间的一切)分类学:

- **共同事实**:(a)dot(`vwmacc`)在所有实测 LOSS 格里 **byte-exact 保号、非瓶颈**;(b)内存 gather(`vluxei`)
  已被 [GAP-SB] 打到近零(iq3 4 条/kernel)。**残留成本 = DECODE = 把散布的 packed 字段重组成向量-ready 值。**
- **Bucket A — grid index/sign 重组 bound**([GAP-GRID] scout):**iq3_s**(qh 第 9 位逐 lane 变量注入)、
  **iq3_xxs**(ksigns 二级表逐 lane 抽取)= 确证 2 格;iq1_s/iq1_m 疑似。病灶=喂 gather 的 index/sign 被
  emitter 标量化 + 栈回程。杠杆 = **in-vector index 组装(L5)**;gather 本身已收口。
- **Bucket B — 多平面权重重建 bound**(本文 scout):**q6_K**(双面 6-bit)、**q3_K**(双面 3-bit subtractive)
  = 确证 2 格;q5_K 部分(~105 尾)。病灶=权重值散布在 ql·qh / qs·hmask 双面,逐 lane 变量移位装配 → ~900 vector spill。
  杠杆 = **B1 离线预合并 repack / B2 权重物化 panel / B3 block re-roll / B4 装配-LUT**。
- **★统一结构签名**:两桶都是 **"散布字段重组装"** —— 片上格式为压缩把位散布到多平面 / 变量移位,我方 emitter 把重组
  串行化成 **标量 storm(bucket A)** 或 **~900-向量 spill(bucket B)**。两者都用**压缩换解码成本**,都是 emitter-
  成熟度 / 布局问题,都套 ≤32-vreg 悬崖 framing。
- **参照 WIN 桶(非 decode-bound,别混)**:min-fold K-quants(q4_K/q5_K/q2_K,S6 register-cliff WIN);
  in-vector index(iq2,`w&511` 就地);tiny-codebook(iq4/fp4,单 gather)。
- **跨桶 caveat**:两桶的 kernel 侧 decode 胜都是 **latency/compute-bound**,**未必传导 memory-bound e2e decode**
  (memory `kernel-wins-dont-transplant-to-e2e`);且**离线 repack 类杠杆(GRID L4 / 本文 B1)↑ 流字节 = 反噬那个
  memory-bound regime 本身**——kernel 侧结构悬崖 ≠ e2e 胜,须分开报。

### 解码成本侦察包 — 一览
| bucket | 病灶 | 确证格 | 疑似格 | 主杠杆(枚举) | 是否改布局 | e2e 风险 |
|--------|------|--------|--------|--------------|-----------|----------|
| A grid-index/sign | index/sign 标量 storm + 栈回程 | iq3_s, iq3_xxs | iq1_s, iq1_m | L5 in-vector index 组装 | 否 | kernel-only,e2e 未测 |
| B multi-plane-weight | ~900 向量 spill(vand/vsrl/vsll/vor/vsub) | q6_K, q3_K | q5_K(~105 尾) | B2 panel / B4 LUT(kernel);B1 repack / B3 re-roll | B1 改;B2/B3/B4 不改 | B1 ↑流字节伤 memory-bound |

---

## 5. 交回结论(★侦察态,待用户裁立项,未实现)

> **权重重建瓶颈画像(交付)**:q6_K/q3_K S6 NULL 的 ~949/894 残留 spill = **多平面权重装配**(q6_K ~2944 /
> q3_K ~7168 条 vand/vsrl/vsll/vor/vsub,**分别 1.28×/3.29× 于 dot 的 vwmacc**),**非** S6 打的 decode-scale strip
> (仅 ~84)。判据 = "权重值是否散布多平面 + 逐 lane 变量移位重组"。
>
> **若立项**:kernel-only 优先候选 = **B2 权重物化 panel**(打 ~900 主导集、不改布局、byte-exact by construction)
> 或 **B4 装配-LUT**(q3_K 8-entry vrgather 尤其便宜);**B1 离线预合并 repack** 是最硬的结构悬崖但跨 [repack] 战役
> 且 ↑ 流字节伤 memory-bound e2e;**B3 re-roll** 有 byte-exact fold-序重导风险。**预计受益 = q6_K、q3_K 确证 2 格 +
> q5_K ~105 尾**;bucket A(iq3/iq1)是同病邻族、走 L5(GRID scout)。

★★ **红线复述**:以上全部为 **侦察画像**,**未立项、未实现、零代码改动**(只读 `lib/` emit 体 + board objdump seal +
cell findings + ODS,只写本 `.md`)。是否立项、采纳哪个杠杆(或都不做)由**用户裁**。本文不 commit(由用户提交)。

### 关键 caveat / 不确定
- q6_K 逐助记词是**结构推导**(以板测 vwmacc=2304 校验闭合),非板测逐助记词直方图;q3_K 逐助记词是**板测直读**
  (`objdump_tile_q3k_t3_seal.objdump`)。q6_K 若要板测直方图应取其 .o 反汇编复核(未做)。
- 所有杠杆**无任何预测 speedup**:只给"砍掉 X 条装配指令 / 塌 wq 活跃集"的**结构判**;真收益须 board A/B 实测,
  且 micro 胜未必传导 e2e(见跨桶 caveat)。
- B1/B4 的 index/codebook 组装可能**只折叠尾段**(qh 位仍须拼进 index/查表键)→ 净收益不定,属实现期风险。
- B2/B3 是否真达 ≤32 悬崖 + 保 byte-exact(fold 序)属**实现期风险**,本侦察不背书,应由 implement + byte-exact gate 验。
- q5_K "~105 尾"是单条 5th-bit 面的**小 weight-floor**,非双面 ~900 级;把它算进 bucket B 是"部分态",别当满格受益。
