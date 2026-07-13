# G7 L2 — q5_K KNEST GEVM 结构设计 + G1 静态账（零板·纯本地·[VERIFY-LADDER] G1 层·禁跳级）

> 前序：qh-plane REDESIGN-B（q5_0/q5_1·`../qh-plane-G1` + `../qh-plane-G2`·**board-validated** kernel 1.44–1.67× ≥parity·e2e 1.11× 部分传导 [CASE-MICRO-E2E]）。本任务 = **q5_K（K-quant super-block）GEVM decode M=1** 的结构分析 + [K-10] 判定 + KNEST plan 设计 + G1 静态账。
> **★诚实前提（定 scope）**：本线求 **结构 novelty（C1 plan 库 extensibility）· 非 perf-covered 移动**。[CASE-MICRO-E2E] 已证 GEVM M=1 削重建 e2e 受 memory-wall 限；q5_K super-block **只会更受限**（instruction-count 已远高于 stock）。**不承诺 e2e green**。
> 依据 canon：[VERIFY-LADDER] G1 双子门 · [K-10] 结构级独立 plan · [X-ZVBB] 纯 base-RVV · [PAT-2] P9 GEVM regime-plan · 性能宪章规则1-2/4（Amdahl）· [CASE-MICRO-E2E] · [CASE-COMPILER-ASYMMETRY]（部署 gcc-15 域）。
> **本任务止于 G1（静态账）+ 设计 · 禁 git · 禁上板 · 禁改 tracked 源码**（`RVVToEmitCBlockQuantLinear.cpp` 只读分析·hand-construct 证据在 `raw/`）。
>
> **★裁决 TL;DR**：q5_K GEVM = **[K-10] 结构级独立 KNEST plan（三问 YES/YES/YES）**。G1 = **分裂裁决**：
> - **byte-exact（硬门）= PASS**（`raw/byteexact_model.c` 0/32）·**register-budget-fit（子门）= PASS**（部署 gcc-15 现行 emitc **0 向量 spill**·TG=1·KNEST 削中间量 → 严格更省）·**削重建 = 兑现**（qh-recon −22.6%·total −10.3% vs 现行 OLD）。
> - **instruction-count ≤1.10× vs stock（子门）= NOT MET（~3.76× stock·结构性不可达）**——q5_K M=1 GEVM 系 element-wise broadcast·per-output 指令密度天然高于 stock 的 vl=32 wide block-dot；赢点在 **memory-locality 非 instruction-count**（[CASE-MICRO-E2E]）。
> - ⇒ **KNEST plan = 合法结构级 C1 改良（修 [GAP-EMIT-KNEST] 一半）+ byte-exact + 预算合规**·但 **G2 perf 资格 = 打回**（instruction-count 子门结构性不达）·**具名 blocker = [GAP-EMIT-KNEST-QH-SUBBLOCK-PACK]**（full REDESIGN-B vlm 杠杆被 q5_K sub-block-bit-packed qh 布局阻断）。

---

## 0. 方法与工具链（本地静态账·零板·可复现）

- **★部署编译器域**：rvv 板 e2e 部署 = **SpacemiT `riscv64-unknown-linux-gnu-gcc-15.2.0`**（`/home/kingdom/spacemit-ime/spacemit-toolchain-.../bin`·`-march=rv64gcv_zvfh -mabi=lp64d -O3`）——与 qh-plane-G1 精化① 同族·**本轮全部 spill/vreg/v-insn 以部署编译器族测**（非 clang 代理）。
- **现行 q5_K GEVM 发射真源（只读）**：`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` `emitRepackKQuantGemvBodyQ5K`（**8517-9080**）·前门 dispatch `kquant_dmin_bsums_min` + `decode_model "q5_K"`（**2444-2547**）。
- **现行 emitc 抽取**（真路径·非 hand-guess）：`weft-opt <fixture> --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`·fixture = `test/Conversion/RVV/rvv-emit-identity-quant-contraction-q5-K-repack-vlen128.mlir` → `raw/q5k_gevm_CURRENT.emitc.c`（12581 行·**block loop 真 for·super-block body 全展开**·per super-block-column-group=16 outputs）。
- **对手（stock 同路径）**：`ggml_vec_dot_q5_K_q8_K`（`llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:2081-2197` __riscv_v 臂·block-dot·`raw/stock_q5_K_blockdot.c` VERBATIM 抽出·同 gcc-15 编）。
- **rolled apples-to-apples**（避 unrolled-vs-rolled trap·casefile §4.1 精神）：`raw/q5k_gevm_variants.c` = `cur_q5k` / `knest_q5k` 两函数·**BYTE-IDENTICAL 除 qh-recon 一处**·同 loop trip counts·gcc-15 -O3 编 → 隔离 recon delta。
- **归一化**：per-output（GEVM 一 super-block-column-group 产 weightInterleave=16 outputs → ÷16 vs stock 1 output/super-block）。

---

## 1. ★结构分析（q5_K super-block 解码热点 + 现状缺 plan 证据）

### 1.1 q5_K super-block 结构（比 q5_0 难两层）
`block_q5_K`（QK_K=256·`ggml-common.h:334-346`）：`d`(fp16 super-scale) · `dmin`(fp16 super-min-scale) · `scales[12]`(**8 sub-block × 6-bit scale + 8 × 6-bit min·kmask 位打包**) · `qh[32]`(**5th-bit plane·256 bit·byte 内 8 bit = 8 sub-block**) · `qs[128]`(4-bit nibble)。stock 额外 activation-side `block_q8_K` 带 `bsums[16]`（block sums·q8_0 无）。

### 1.2 stock 解码热点归因（`raw/stock_q5_K_blockdot.gcc15.O3.s`·130 v-insn/super-block/output）
| 热点 | 手段 | 占比 |
|---|---|---|
| **super-block 6-bit scale/min 重建** | **纯 scalar kmask 舞（utmp[]·memcpy+移位）·非向量** | 0 v-insn（标量）|
| **qh 5th-bit 重建** | **native mask**：`vand(vqh,m)`+`vmsne`+`vadd_mu(+16)` = **3 op × 8 sub-block = ~24 v-insn** | ~18% |
| min-fold（bsums×mins） | vlse16×2+vadd+vzext+vwmul+vredsum | ~9 v-insn |
| 主 dot（productive） | vwmul.vv×9 + vwmul.vx×8 + vredsum×9（**vl=32 宽·256 elem/8 chunk**）| ~40 v-insn |
| vsetvl 税 | 28（vl=8↔32 toggle）| — |

**关键**：stock 用 **vl=32 wide** 一次吃 32 元素 → 主 dot 仅 ~8 组指令覆盖 256 元素·qh-recon **native mask 已最优**（3 op/sub-block）。stock **6 向量 spill**（vwmul→i16m4/vwmul→i32m8 大 group·VLEN-agnostic 编）·18 vreg。

### 1.3 现行 q5_K GEVM 热点归因（`raw/q5k_gevm_CURRENT.emitc.c`·8714 intrinsics/super-block-col-group·16 outputs·0 向量 spill·32/32 vreg）
| 热点 | 现行手段（RVVToEmitC*.cpp:8878-9007）| per-super-block v-insn |
|---|---|---|
| **super-block 6-bit scale/min 重建** | **向量 lane-wise**（vand 0x0F/0x03/0x30 + vsrl + vsll + vor + vzext·per strip×sub）| ~小（32 unit × ~9） |
| **★qh 5th-bit 重建（主热点·[GAP-EMIT-KNEST]）** | **OLD per-lane 展开**：per elem-step `vsrl(qh,bit)+vand(0x01)+vsll(4)+vor` ×2 臂 = **7–8 v-insn/elem-step-strip** | **~3968（qh-recon total）** |
| nibble 抽取 | vand(0x0F)+vsrl(4) | 2/elem-step（shared）|
| min-fold | vwmacc_vx_i32m2 ×(2×4×numHalves) | 32 |
| 主 dot（productive）| **vwmacc_vx_i16m1 ×1024**（**per-element broadcast·非 vl=32 wide**）| 1024 |

**★现状证据（[GAP-EMIT-KNEST] 坐实）**：现行 q5_K GEVM decode **无独立 plan·qh 重建走 OLD per-lane `vsrl+vand+vsll+vor` 展开**（与 q5_0 REDESIGN-B **未共享** native-mask 杠杆）·**重建税全在 qh 5th-bit**（~3968 v-insn = total 的 45%·远超 super-block scale 重建 ~300）。**"super-block scale 重建 vs qh 5th-bit" 占比裁决 = qh 5th-bit 压倒性主热点**（与直觉相反·super-block scale unpack 是次要）。

---

## 2. ★[K-10] 三问判定（结构级 KNEST vs 参数级铺开）

对比基准 = q5_0 qh-plane GEVM（REDESIGN-B·`../qh-plane-G1` 已判 **参数级**·发射编排改·非新 plan）。

| # | 问 | q5_K GEVM vs q5_0 GEVM | 答 |
|---|---|---|---|
| 1 | **迭代拓扑变？** | q5_0 = 单层 flat block-loop（32-elem·单 scale·单累加）。q5_K = **super-half(j) × sub-block-pair(pair) × overflow-chunk(k) × elem(ii) 四层嵌套** + **并行 min-fold 归约路**（bsums×mins·q5_0 完全无）+ **双累加层级**（i16 partial → per-sub-block scale vwmacc → i32 → ×d/dmin f32）| **YES 变** |
| 2 | **布局契约变？** | q5_0 = 单 fp16 scale + transposed-qh **u16-per-step（16 bit = 16 row·可 vlm 直装 mask）**。q5_K = **dual fp16 d/dmin + 6-bit packed scales/mins region（kmask 舞）+ qh byte = 8 sub-block bit（非 16 row·不可 vlm 直装）+ activation bsums（q8_K≠q8_0）** | **YES 变** |
| 3 | **优化目标变？** | q5_0 = 单一 qh 削重建。q5_K = **双热点**（super-block 6-bit scale unpack + qh 5th-bit）**+ register-budget 为 binding 约束**（resident: sumi×numHalves + bsums×numHalves + sumf×numHalves + scale/min strips·P2 教训）| **YES 变** |

**⇒ 判定：q5_K GEVM = [K-10] 结构级独立 plan（"KNEST plan"）**·**非** q5_0 qh-leaf 参数化铺开。qh-recon 子技术（native-mask）可从 REDESIGN-B **参数级复用**·但 **envelope（super-block 流式嵌套 + min-fold + 6-bit scale unpack）= 独立结构级 plan**。与 agent a548692a 判定 + [K-10] 一致（GEMM↔GEVM 结构级先例·806a7cc1）。

---

## 3. ★KNEST plan 设计（结构级·TG=1·register-budget-fit）

### 3.1 超块流式嵌套 envelope（现行已实现·本设计不改拓扑）
super-half 外层流式 K-归约 × sub-block scale/min resident（per super-half 解 4 sub × numHalves strip）× 双累加（sumi 主 + bsums min）× end-block d/dmin f32 fold。**TG=1 不加宽 bank**（P2 加宽-bank M=1 spill 反噬已证·见 §3.3）。

### 3.2 qh 5th-bit 削重建（native-mask·移植 stock 技巧进 transposed envelope）
现行 OLD per-lane（per elem-step·per 臂）：`vsrl(qh,bit) + vand(0x01) + vsll(4) + vor(nibble)` = **7–8 op**。
**KNEST（stock native-mask 复刻）**：
```
loNib_i8 = reinterpret(vand(packed,0x0F))          // 复用 nibble
loMask   = vmsne(vand(qh, 1<<sLoBit), 0)           // 从 qh byte 抽 sub-block bit → per-lane bool
nLo      = vadd_vx_i8mf2_mu(loMask, loNib, loNib, 16)   // masked +16 融进重建（无 vor/vsll）
```
per elem-step 两臂 = **6 op**（`vand+vmsne+vadd` ×2·reinterpret free）。**撤销 OLD 的 vsll/vor 全链·mask 直接驱动 vadd_mu**。
- **★与 q5_0 REDESIGN-B 的关键差**：q5_0 transposed-qh u16 = **16 row bit·`vlm_v_b16` 直装 native mask**（mask 白嫖·省 vand+vmsne）。q5_K qh byte = **8 sub-block bit**（同 lane 内 8 sub-block）→ **单 sub-block mask 不可 vlm 直装·必 `vand(1<<bit)+vmsne` 抽**。**∴ full REDESIGN-B vlm 杠杆【被阻断】**·KNEST 只能落到 stock 自身的 `vand+vmsne+vadd_mu`（3 op/臂·非 q5_0 的 vlm+vmnot+vsub 1.x op/臂）。具名 = **[GAP-EMIT-KNEST-QH-SUBBLOCK-PACK]**。

### 3.3 register-budget-fit 前置检查（★P2 教训）
| 配置 | resident 向量态 | 部署 gcc-15 spill |
|---|---|---|
| **现行 emitc（TG=1·numHalves 全宽）** | sumi + bsums + sumf（各 ×numHalves）+ scale/min strips + 临时 | **0 向量 spill**（`raw/q5k_gevm_CURRENT.gcc15.O3.s`·1259 sp-ref **全标量** addi/ld/sd·`\(sp\)` 无 v-reg）·**32/32 vreg 用满·TIGHT 但不 spill** |
| **KNEST（TG=1）** | 同上·**但削 loSel/loBit/hiSel/hiBit 4 中间 u8 → 2 bool mask** | **0 向量 spill·严格 ≤ 现行**（rolled 实测 25 vreg·§4）|
| **TG=2 假想（加宽 accumulator bank）** | 12 resident accumulator → 翻倍 | **必 spill**（P2 [GAP-P1] 反噬·**设计明文拒**）|

**⇒ register-budget-fit = PASS·TG=1 硬约束保持**（加宽 bank 是 falsified 反面·不做）。

---

## 4. ★G1 静态账（部署 gcc-15·`raw/*.s`）

### 4.1 rolled apples-to-apples（隔离 recon delta·同结构·仅 qh-recon 异）
| 变体 | v-insn（single-strip body）| 向量 spill | distinct vreg |
|---|---|---|---|
| `cur_q5k`（OLD per-lane）| **183** | **0** | 25 |
| **`knest_q5k`（native-mask）** | **165** | **0** | 25 |
| Δ | **−9.8%** | — | — |

recon 指令迁移（rolled asm 直读）：OLD `{2 vsll + 2 vor + 1–2 vsrl-sel}` → KNEST `{2 vmsne + 2 vadd}`·per elem-step **6→4 recon-arith**（sLoBit≠0）/ **5→4**（sLoBit==0）。

### 4.2 per super-block-column-group（现行 emitc 实测 + KNEST 派生）
| 度量 | STOCK（1 out/super-block）| CURRENT GEVM（16 out）| **KNEST GEVM（16 out）** |
|---|---|---|---|
| total v-insn | **130** | 8714 | **~7818（−10.3%）** |
| qh-recon v-insn | ~24（native·已最优）| **~3968** | **~3072（−22.6%）** |
| 向量 spill | **6**（vs2r/vs4r）| **0** | **0** |
| distinct vreg | 18 | 32 | ≤32 |
| **per-output（÷outputs）** | **130** | 545（4.19× stock）| **489（3.76× stock）** |

### 4.3 判读
- **byte-exact 硬门 = ✅ PASS**：`raw/byteexact_model.c` 穷举 nibble∈[0,15]×qh_bit∈{0,1} → **stock=current=KNEST 0/32 mismatch**。代数：q5_K RAW 5-bit `nibble|(bit<<4)` ≡ `bit? nibble+16 : nibble` = `vadd_mu(mask=bit, nib, nib, 16)`（**无 q5_0 的 −16 offset-binary·bias 在 6-bit MIN**）。
- **register-budget-fit 子门 = ✅ PASS**：现行 emitc **0 向量 spill**·KNEST 削中间量 → 严格更省·TG=1 保持（§3.3）。
- **削重建 = ✅ 兑现**：qh-recon **−22.6%**·total **−10.3%**（vs 现行 OLD·区别 q4_K 的 0-yield·`../G1-static`）。
- **★instruction-count ≤1.10× vs stock 子门 = ❌ NOT MET（~3.76× stock）**：**结构性不可达**——q5_K M=1 GEVM 系 **element-wise broadcast**（`vwmacc_vx` per element × 16 column-lane·256 elem-step/super-block）·stock 系 **vl=32 wide block-dot**（8 chunk 覆盖 256 elem）。GEVM 主 dot productive-per-output 与 stock parity·但 **per-element decode 展开税 × super-block 双累加 overhead** 令 per-output 指令密度 ~4× stock。**赢点 = repack memory-locality（weight 流式 · 16-column 摊 activation）· 非 instruction-count**（[CASE-MICRO-E2E] 已证此 locality 在 decode 受 memory-wall 稀释）。

---

## 5. ★G1 裁决（获 G2 资格 / 打回·具名 blocker）

**分裂裁决**：
- **结构 C1（[K-10] 结构级 KNEST plan）= 成立**：三问 YES/YES/YES·byte-exact 由构造·修 [GAP-EMIT-KNEST] 一半（qh-recon OLD→native-mask·−22.6%）。
- **G2 perf 资格 = 打回**：G1 双子门中 **instruction-count ≤1.10× vs stock 结构性不达（~3.76×）**·**不获 G2**（[VERIFY-LADDER] 禁跳级·双子门须皆过·参 qh-plane-G1 q5_0 双门皆过才升 G2）。
- **具名 blocker**：
  1. **[GAP-EMIT-KNEST-QH-SUBBLOCK-PACK]**（主）：full REDESIGN-B `vlm` direct-mask 杠杆被 q5_K **sub-block-bit-packed qh 布局**（byte=8 sub-block bit·非 16 row）阻断·KNEST 只落到 stock 的 `vand+vmsne+vadd_mu`（3 op/臂·非 q5_0 vlm 的 ~1 op/臂）。**恢复需 re-transpose qh repack 布局**（per-sub-block 16-row-bit 连续 2 byte → vlm 可装）= **更深结构级 repack-format 改**·**G1 scope 外**（改前门 repack tool + 重 validate）·留主会话裁 [远期]。
  2. **M=1 element-wise 结构密度**：super-block GEVM per-output 指令 ~4× stock·**instruction-count 非本格赛场**（memory-locality 才是·[CASE-MICRO-E2E]）。

**⇒ 负结果如实报**：KNEST plan = **合法结构级 C1 改良 + byte-exact + 预算合规·但 perf-gate（instruction-count vs stock）结构性打回**。这是 **[GAP-EMIT-KNEST] 具名的结构边界·C3′ 素材**（能力键控削重建在 K-quant super-block M=1 GEVM 的适用边界·正如 [PAT-1] format-keyed 在 q4_K@ime epilogue 稀释的负结果）。**诚实前提坐实**：本线求结构 C1 验证·非 perf-covered·e2e 大概率 [CASE-MICRO-E2E] 受限·不承诺 green。

---

## 6. [X-ZVBB] / [PAT-2] P9 / 下一步建议（留主会话·仅建议）

- **[X-ZVBB]**：KNEST 全 base-RVV V1.0 mask op（`vand`/`vmsne`/`vadd_vx_i8mf2_mu`·stock q5_K 自身同族）·**无 zvbb**（舰队缺席不阻·`../G1-static` 已证）。
- **[PAT-2] P9 GEVM regime-plan**：q5_K 新增 **"super-block native-mask KNEST"结构条目**——嵌套 super-half 流式 + qh sub-block-bit vand+vmsne+vadd_mu。status：**结构级 plan 成立·byte-exact 由构造·qh-recon −22.6% 兑现·但 instruction-count perf-gate 打回（结构性）**。
- **发射器实装（若主会话取·G1 scope 外·deferred）**：`RVVToEmitCBlockQuantLinear.cpp:8976-8987`（`emitRepackKQuantGemvBodyQ5K` qh-recon 段·**与 sealed GEMM leaf `emitRepackKQuantGemmBodyQ5K` 9104+ 隔离·勿碰 q5_0 REDESIGN-B leaf 1651+**）——`vsrl+vand+vsll+vor` 换 `vand(1<<bit)+vmsne+vadd_vx_i8mf2_mu`。**但先决**：instruction-count 结构打回 ⇒ 实装仅修 [GAP-EMIT-KNEST] 结构缺口·**不改 perf 结论**·价值 = C1 plan 库 extensibility（发射编排成熟）·非 perf-covered 移动。
- **[远期] qh re-transpose**（恢复 vlm 杠杆）= 更深 repack-format 结构 plan·须主会话立项裁决（改前门 repack tool·不可逆度高）。

**禁上板 · 禁 git · 无 schema label 改动 · tracked 源码只读未改（hand-construct 全在 `raw/`）。**

---
### 复现
```
GCC=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin/riscv64-unknown-linux-gnu-gcc
WEFT=build-weft/bin/weft-opt; F=test/Conversion/RVV/rvv-emit-identity-quant-contraction-q5-K-repack-vlen128.mlir
# 现行 emitc（真路径）
$WEFT $F --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > raw/q5k_gevm_CURRENT.emitc.c
$GCC -march=rv64gcv_zvfh -mabi=lp64d -O3 -S raw/{stock_q5_K_blockdot,q5k_gevm_variants}.c   # 计数 v-insn/spill/vreg
gcc -O2 raw/byteexact_model.c -o raw/be_q5k && raw/be_q5k   # 0/32 byte-exact 硬门
```
