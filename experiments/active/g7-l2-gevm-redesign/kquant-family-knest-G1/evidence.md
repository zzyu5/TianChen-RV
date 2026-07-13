# G7 L2 — q2_K/q3_K/q6_K KNEST GEVM 家族完成线：结构 survey + G1 静态账（零板·纯本地·[VERIFY-LADDER] G1 层）

> 前序：q5_K KNEST G1（`../q5k-knest-G1/evidence.md`·`7a45a2fe`）= 分裂裁决（[K-10] 结构级 KNEST·byte-exact PASS·register-budget-fit PASS·qh-recon −22.6%·instruction-count 3.76× stock NOT MET → G2 打回）。q4_K P1 mechanized（`806a7cc1`）。
> **本线目标（诚实前提·定 scope）**：完成 **5 个 K-quant GEVM plan 库的 C1 extensibility 陈述** + 确认/精化 **家族级 perf 边界**。**非 perf-covered 移动**（[CASE-MICRO-E2E] 已证 K-quant M=1 GEVM instruction-count 结构性不可达·本线求结构 C1 完成 + 边界确认·不承诺 perf green）。
> 依据 canon：[VERIFY-LADDER] G1 双子门 · [K-10] 结构级 vs 参数级 · [CASE-MICRO-E2E] · 性能宪章规则1-2/4 · [CASE-COMPILER-ASYMMETRY]（部署 gcc-15.2 域·rvv 板 VLEN128）。
> **止于 G1（静态账）+ 结构分析 · 禁 git · 禁上板 · 禁改 tracked 源码**（emitter 只读·hand-construct 全在 `raw/`）。

---

## ★裁决 TL;DR

1. **家族 a-priori 框架被证据反转（重要正结果）**：任务预设"q6_K 最接近 q5_K·优先深做"。**证据反转**——真正的 q5_K 削重建靶（单-bit plane native-mask KNEST 技巧）落在 **q3_K**（hmask=3rd-bit **单-bit** plane），**不在 q6_K**（qh=2-bit field·shift+or 布放·native-mask 不适用）。故**深做改投 q3_K**（有靶）·q6_K/q2_K 确认负结果。
2. **C1 完成 = 达成**：5 个 K-quant（q2/q3/q4/q5/q6_K）**全有 GEVM leaf 且 byte-exact-by-construction**。plan 库结构 = **2 个 envelope 结构级 plan（MIN-bearing {q2,q4,q5} / NO-MIN {q3,q6}·codebase dispatch 坐实）× format-keyed decode leaf + 1 个可复用 native-mask 单-bit-plane 削重建 KNEST 元件（q5_K ∧ q3_K）**。这是**组合式 extensibility**（比"5 个正交结构 plan"更诚实、更强的 C1 陈述）。
3. **C3′ 边界 = instruction-count 结构限【家族一致】确认**：q2 1.62× / q3 4.77× / q6 4.92× vs stock（per-output·全 NOT MET ≤1.10× 子门）·anchor q5_K 3.76×。**无反例**。M=1 element-wise broadcast + **vsetvli 税（25–30%·家族最大单项）** + per-element decode 展开 结构性超 stock 宽 block-dot。
4. **q3_K native-mask KNEST 削重建 = byte-exact + 静态兑现**：byte-exact 硬门 0/8 PASS·rolled apples-to-apples **103→58 v-insn（−43.7%）·recon-arith 24→8（−66.7%）·双 spill-free**。= q5_K [GAP-EMIT-KNEST] 的 q3_K 同类·结构级 C1 改良可构造（emitter 实装 deferred）。
5. **register-budget-fit 家族纠偏**：q2/q3/q6 **部署 emitc 全有向量栈流量**（vs*r.v whole-reg store 143/174/137·**vl*r.v reload=0**·= per-element-broadcast 结构把 16-col 解码 tile round-trip 过栈 scratch·非经典 RA spill/reload 对·**≠ q5_K 零栈流量**）·**rolled strip 重构消除**（q3_K rolled 0 vs*r.v / 13-14 vreg 实证）→ 部署=FAIL·rolled-redesign=PASS 可达。
6. **G2 perf 资格 = 家族全打回**（instruction-count 子门结构性不达·[VERIFY-LADDER] 禁跳级）。诚实前提坐实：结构 C1 完成 + 边界确认·非 perf green。

---

## 0. 方法与工具链（本地静态账·零板·可复现）

- **部署编译器域**：rvv 板 e2e 部署 = SpacemiT `riscv64-unknown-linux-gnu-gcc-15.2.0`（`-march=rv64gcv_zvfh -mabi=lp64d -O3`·VLEN128）。全 spill/vreg/v-insn 以部署编译器族测（非 clang 代理）。emitc 是 C++（`extern "C"`）→ 用 toolchain `g++`。
- **现行 GEVM 发射真源（只读）**：`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`·`emitRepackKQuantGemvBodyQ2K`（11014）/`Q3K`（16735）/`Q6K`（9864）。前门 dispatch：q2_K → `kquant_dmin_bsums_min`（MIN-bearing·2444）；q3_K/q6_K → no-min fold（2548·"q3_K = q6_K's no-min sibling"）。
- **现行 emitc（真路径·非 hand-guess）**：`weft-opt <fixture> --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`·fixture = `test/Conversion/RVV/rvv-emit-identity-quant-contraction-q{2,3,6}-K-repack-vlen128.mlir` → `raw/q{2,3,6}k_gevm_CURRENT.emitc.c`（block loop 真 for·super-block body 全展开·16 outputs/col-group = 2×vfloat32m2×8-lane）。
- **对手（stock 同路径·同 gcc-15.2）**：`ggml_vec_dot_q{2,3,6}_K_q8_K_vl128`（`/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c`·**手写 inline asm hand-brick**·VERBATIM 抽出 `raw/stock_q{2,3,6}_K_vl128.c`）。stock j-loop（QK_K/128=2·常数 trip）被 gcc 展开 → 静态 asm count = 运行时 per-super-block（1 output）。
- **归一化**：ours = per super-block-column-group（16 outputs）÷16 → per-output；stock = per super-block（1 output）。ratio 与 nb 无关。

---

## 1. ★结构分析（每格·重建热点占比 + 现状 plan 状态）

### 1.1 stock decode 热点（`raw/stock_q{2,3,6}_K_vl128.*`·手写 inline asm·0 spill）
| 格 | scale 重建 | plane 重建 | plane 手段 | 主 dot | v-insn/super-block(1 out) |
|---|---|---|---|---|---|
| **q2_K** | 4-bit scale + 4-bit min·**vand 0xF + vsrl 4**（trivial·无 kmask 舞）| **无 plane** | — | vwmul.vv×8 + vwredsum.vs×16（vl=16）| **134** |
| **q3_K** | 6-bit scale·kmask 舞（vsll/vor/vsub 32·scalar-ish·once/super-block）| **hmask 3rd-bit（单-bit plane！）**| **native-mask**：`vand(hm,m)+vmseq+vadd.vi(-4,v0.t)`（3 op/sub·bias 折入）| vwmul.vv + vwredsum.vs（vl=32）| **116** |
| **q6_K** | int8 scale·**ld+slli/srai**（scalar sign-ext·零向量）| **qh 2-bit field**（4 sub/byte）| **shift-or**：`vsll4/vsll2/vsrl2/self + vand 0x30 + vor`（2-bit 布放·**已最优**）| vwmul.vv(m4) + vwredsum.vs | **94** |

**★关键结构订正**：任务预设"q2/q3/q6 无 qh plane"——**部分错**。**q3_K 有 hmask=3rd-bit 单-bit plane**·stock 用与 q5_K qh **同一 native-mask 技巧**（vand+vmseq+vadd_mu bias 折入）→ **q3_K 才是 q5_K 的真结构同类**。q6_K 有 qh 但是 **2-bit field**（非单-bit）·必 shift+or 布放（4-way 偏移·单 mask 不可表达）。q2_K 无任何 high-bit plane。

### 1.2 现行 GEVM emitc decode 热点（`raw/q{2,3,6}k_gevm_CURRENT.gcc15.O3.s`·per super-block-col-group·16 out）
| 格 | total v | **vsetvl 税** | vwmacc(productive) | decode-leaf 手段（source histogram）| plane recon 状态 |
|---|---|---|---|---|---|
| **q2_K** | 3466 | **861（24.8%）** | 576 | vand 544 + vsrl 416（纯 2-bit unpack·无 vsll/vor）| **无 plane·无靶** |
| **q3_K** | 8846 | **2675（30.2%·家族最重）** | 544 | vand 1024 + vsrl 832 + **vsll 512 + vor 512 + vsub 512（OLD-style hmask 单-bit recon）** | **★OLD-style·有 native-mask 靶** |
| **q6_K** | 7403 | **1931（26.1%）** | 576 | vand 768 + vsrl 640 + **vsll 512 + vor 512 + vsub 512（2-bit field 布放·REQUIRED·匹配 stock）** | **2-bit field·无 native-mask 靶** |

**★占比裁决**：三格 emitc 的 **#1 成本一致 = vsetvli 税（25–30%）**（per-element 展开体无 vsetvli hoisting/CSE·SEW/LMUL 频繁 toggle）·**压倒 recon 与 productive**。productive dot（vwmacc ~544-576）per-output 与 stock parity。**削重建靶只在 q3_K**（OLD-style hmask·[GAP-EMIT-KNEST] q3_K 同类）。

---

## 2. ★[K-10] 三问判定（结构级独立 plan vs 参数级复用）

对比基准 = 既有 K-quant plan（q4_K P1·q5_K KNEST）+ codebase envelope 分族（dispatch 坐实：MIN-bearing {q2,q4,q5} / NO-MIN {q3,q6}）。

| 格 | vs 最近 plan | Q1 迭代拓扑变？| Q2 布局契约变？| Q3 优化目标变？| 判定 |
|---|---|---|---|---|---|
| **q2_K** | vs q4_K（MIN-bearing）| **NO**（同 min-fold envelope）| 部分（4-bit scale trivial·**无 kmask 舞**·无 plane）| 参数级（2-bit leaf·**无 recon 靶**）| **参数级 decode-leaf 复用 q4_K MIN-bearing plan**·非独立结构级 |
| **q3_K** | vs q6_K（no-min）+ q5_K（native-mask）| 共享 q6_K no-min envelope（vs q5_K 去 min-path=变）| hmask **单-bit** plane + 6-bit scale | **★单-bit native-mask recon = q5_K KNEST 技巧**（结构缝）| **混合：no-min envelope（参数级 vs q6_K）+ native-mask 单-bit-plane 削重建 KNEST 元件（从 q5_K 复用）** |
| **q6_K** | vs q3_K（no-min）+ q5_K | 共享 q3_K no-min envelope（vs q5_K 去 min-path=变）| qh **2-bit** field + int8 scale | 参数级（2-bit 布放 shift-or·**无 native-mask 靶**）| **参数级：no-min envelope + 2-bit-field leaf**·非独立结构级（vs q3_K 仅 plane 宽度异）|

**⇒ 家族 [K-10] 判定（订正 a-priori）**：三格**皆非** q5_K-式全新独立结构级 plan。plan 库真实结构 = **2 个 envelope 结构级 plan（MIN-bearing / NO-MIN·dispatch 坐实）× format-keyed decode leaf + 1 个可复用 native-mask 单-bit-plane 削重建 KNEST 元件（覆盖 q5_K ∧ q3_K）**。5 格由 **(envelope × decode-leaf) 组合**覆盖 = **组合式 C1 extensibility**（比"5 个正交结构 plan"诚实且更强）。

---

## 3. ★G1 静态账（部署 gcc-15.2·双子门判）

### 3.1 instruction-count 子门（per-output vs stock·硬对手 = 手写 inline asm）
| 格 | ours per super-block-col-group(16 out) | ÷16 = per-output | stock(1 out) | **ratio** | 子门 ≤1.10× |
|---|---|---|---|---|---|
| **q2_K** | 3466 | 216.6 | 134 | **1.62×** | ❌ NOT MET（最"近"·仍不达）|
| **q3_K** | 8846 | 552.9 | 116 | **4.77×** | ❌ NOT MET |
| **q6_K** | 7403 | 462.7 | 94 | **4.92×** | ❌ NOT MET |
| q5_K(anchor) | — | 545→489 | 130 | 3.76× | ❌ NOT MET |

**⇒ instruction-count 结构限【家族一致】·无反例**。根因三重：① M=1 element-wise broadcast（vwmacc_vx per-element × 16-col-lane）vs stock vl=16/32 宽 block-dot；② **vsetvli 税 25–30%**（emitter 无 vsetvli 调度成熟）；③ per-element decode 展开税。即便清零 vsetvli 税，element-wise 密度仍不过 ≤1.10×。

### 3.2 register-budget-fit 子门（★家族纠偏 vs q5_K）
向量栈流量度量 = whole-reg vector store `vs[0-9]r.v` / reload `vl[0-9]r.v`（stock=0·q5_K emitc=0）。
| 格 | vs*r.v store | vl*r.v reload | 部署判 | rolled-redesign |
|---|---|---|---|---|
| q2_K | **143** | 0 | ❌ FAIL（有栈流量）| 消除可达（rolled）|
| q3_K | **174** | 0 | ❌ FAIL | **✅ PASS 实证**（`raw/q3k_gevm_variants` rolled = **0 vs*r.v·13-14 vreg**）|
| q6_K | **137** | 0 | ❌ FAIL | 消除可达（rolled）|

**⇒ 部署 emitc 全有向量栈流量（vs*r.v store 143/174/137·reload=0·≠ q5_K 零栈流量）**。**性质** = per-element-broadcast GEVM 结构把 full-unroll 的 16-col 解码 tile round-trip 过栈 scratch（store 后经 vle8 element-wise 重读·非经典 RA spill/reload 对·故 reload=0）。**rolled strip 重构消除栈流量**（q3_K rolled 实证 0 vs*r.v）。**register-budget-fit：部署 FAIL·rolled-redesign PASS 可达**（per-element-broadcast tile round-trip·[GAP-P1]-邻）。**注**：即便从 total v-insn 扣除全部 vs*r.v（视作 memory-traffic 非 compute）·ratio 仍 q2 1.55× / q3 4.67× / q6 4.83× 全 NOT MET（结论对此稳健）。

### 3.3 q3_K native-mask KNEST 削重建（byte-exact 硬门 + rolled apples-to-apples）
- **byte-exact 硬门 = ✅ PASS**（`raw/be_q3k`·穷举 base2∈[0,3]×hbit∈{0,1}=8 例）：**native-mask（vmseq+vadd_mu −4）=== OLD（vand+vsll+vor+vsub）=== stock scalar·0/8 mismatch**。代数：q3_K 3-bit 值 = base2 with hmask 为高位·bias −4；stock `if(hbit==0) q-=4`；native `vadd_vx_i8mf2_mu(mask=hbit0, base, base, -4)` bias 折入。
- **rolled apples-to-apples**（`raw/q3k_gevm_variants.*`·single-strip·BYTE-IDENTICAL 除 RECON·同 loop trip·gcc-15.2 -O3）：
  | 变体 | v-insn | vs*r.v(栈流量) | vreg | recon-arith(vsll/vor/vsub/vmseq/vadd) |
  |---|---|---|---|---|
  | `cur_q3k`（OLD）| **103** | 0（真零栈·非 tile round-trip）| 13 | **24** |
  | `knest_q3k`（native）| **58** | 0 | 14 | **8** |
  | Δ | **−43.7%** | — | — | **−66.7%** |
  recon 迁移：per elem-step OLD `{vand(hb 0x01), vsll<<2, vor, vsub −4}`（4 op）→ native `{vmseq(hplane,0), vadd_mu(−4)}`（2 op·bias 折入·drop vand/vsll/vor/vsub）。

### 3.4 q6_K/q2_K 负结果（无 native-mask 靶·具名）
- **q6_K**：qh = **2-bit field**（贡献 {0,16,32,48}·4-way 偏移）。2-mask 分解假想 = `vand+vmsne+vadd_mu ×2 + vsub` = **7 op** > stock shift-or（vsll+vand+vor 摊 4 sub）= **3 op** → **native-mask WORSE**·负结果量化。stock/ours 的 shift-or 已最优·无靶。
- **q2_K**：**无 high-bit plane**·decode = vsrl+vand 纯 2-bit unpack（已算法最优·匹配 stock）·无 recon 可削·无靶。

---

## 4. ★家族裁决（C1 完成 + C3′ 边界）

### 4.1 C1 完成陈述："5 K-quant GEVM plan 覆盖" = ✅ 达成
| 格 | GEVM leaf | envelope 结构族 | decode leaf | byte-exact | 削重建靶 | 状态 |
|---|---|---|---|---|---|---|
| q2_K | `emitRepackKQuantGemvBodyQ2K` | MIN-bearing | 2-bit·4-bit scale | by-construction（oracle）| 无 | ✅ 覆盖 |
| q3_K | `emitRepackKQuantGemvBodyQ3K` | NO-MIN | 3-bit·hmask 单-bit·6-bit scale | **0/8 PASS**（本线）| **native-mask（byte-exact + rolled −43.7%）** | ✅ 覆盖 + 削重建可构造 |
| q4_K | `emitRepackKQuantGemvBodyQ4K` | MIN-bearing（P1 mechanized `806a7cc1`）| 4-bit·6-bit scale | by-construction | — | ✅ 覆盖 |
| q5_K | `emitRepackKQuantGemvBodyQ5K` | MIN-bearing | 5-bit·qh 单-bit·6-bit scale | 0/32 PASS（`7a45a2fe`）| native-mask（−22.6%）| ✅ 覆盖 + KNEST |
| q6_K | `emitRepackKQuantGemvBodyQ6K` | NO-MIN | 6-bit·qh 2-bit·int8 scale | by-construction | 无（2-bit field）| ✅ 覆盖 |

**⇒ 5/5 K-quant GEVM plan 全覆盖**。plan 库 extensibility = **组合式**（2 envelope × format-keyed leaf + native-mask 单-bit-plane 削重建元件[q5_K,q3_K]）。C1 完成。

### 4.2 C3′ 边界确认："instruction-count 结构限家族一致？" = ✅ 是·无反例
- 测得 q2 1.62× / q3 4.77× / q6 4.92× / q5 3.76×（anchor）· 全 NOT MET ≤1.10× 子门。q4_K 同族 MIN-bearing envelope（同 element-wise M=1 结构·per 806a7cc1）。**4/5 直接测 NOT MET·无一格 MET·无反例**。
- **家族一致确认**：K-quant M=1 GEVM instruction-count 结构性不可达 ≤1.10× stock·系 M=1 element-wise broadcast + vsetvli 税 + decode 展开 的结构限·**不因格式（2/3/4/5/6-bit·min/no-min·有/无 plane）而变**。赢点若有 = memory-locality（repack 流式·16-col 摊 activation）·非 instruction-count（[CASE-MICRO-E2E]·decode memory-wall 稀释）。
- **q2_K 1.62× = 家族"最近"格**（decode 最轻·无 plane）·但仍不达 → 更坐实"结构限"而非"个别格实现差"。

---

## 5. ★新 gap 具名

1. **[GAP-EMIT-KNEST-Q3K-HMASK]**（= q5_K [GAP-EMIT-KNEST] 的 q3_K 同类）：q3_K GEVM hmask 单-bit recon 现走 OLD-style（vand+vsll+vor+vsub 4 op）·未用 q5_K 已有的 native-mask（vmseq+vadd_mu 2 op·bias 折入）。byte-exact 0/8 + rolled −43.7% v-insn / −66.7% recon-arith 已证。**结构级 C1 改良可构造**（emitter 实装 = `emitRepackKQuantGemvBodyQ3K` 的 hmask 段·**与 sealed q3_K GEMM leaf 17208+ 隔离**·deferred·价值 = plan 库成熟·非 perf-covered 移动·instruction-count 仍打回）。
2. **[GAP-EMIT-VSETVL-TAX]**（NEW·家族级·instruction-count 真杠杆）：K-quant GEVM emitc 的 **#1 成本 = vsetvli/vsetivli（q2 861 / q3 2675 / q6 1931 = 25–30% total）**·per-element 展开体无 vsetvli hoisting/CSE·SEW/LMUL config 未调度。**是最大单项税**·但即便清零仍不过 ≤1.10×（element-wise broadcast 密度限）。emitter-maturity 缺口（非削重建·非 perf-gate 通道·但方法学正确 framing）。
3. **[GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP]**（NEW·[GAP-P1]-邻·[CASE-MICRO-E2E] 印证）：q2/q3/q6_K GEVM **部署 emitc 有向量栈流量**（vs*r.v store 143/174/137·reload=0·**≠ q5_K 零栈流量**）= per-element-broadcast 结构把 full-unroll 的 16-col 解码 tile round-trip 过栈 scratch（后经 vle8 element-wise 重读）。rolled strip 重构消除（q3_K 实证 0 vs*r.v）→ register-budget-fit 部署 FAIL·rolled PASS 可达（实装 deferred）。**这正是 memory-locality-vs-instruction-count 张力的 kernel 侧证据**（解码 tile 无法驻寄存器跨 per-element 广播·被迫过内存）。
4. **native-mask 削重建【不推广至 multi-bit plane】负结果**（具名）：q6_K qh 2-bit field 的 native-mask 2-mask 分解（7 op）> stock shift-or（3 op）→ **KNEST native-mask 元件仅适用【单-bit】plane 格（q5_K·q3_K）·不适用 q6_K(2-bit)/q2_K(无 plane)**。= [PAT-1] format-keyed 削重建的适用边界（C3′ 素材·如 q4_K@ime epilogue 稀释先例）。

---

## 6. G1 裁决（获 G2 资格 / 打回）+ 诚实前提坐实

**家族分裂裁决（与 q5_K 一致）**：
- **结构 C1（5 K-quant GEVM plan 覆盖 + 组合式 extensibility）= ✅ 成立**：5/5 leaf present·byte-exact-by-construction·q3_K native-mask KNEST byte-exact + rolled 兑现。plan 库 = 2 envelope × format-keyed leaf + native-mask 单-bit-plane 削重建元件。
- **G2 perf 资格 = 家族全打回**：instruction-count ≤1.10× vs stock 子门结构性不达（q2 1.62× / q3 4.77× / q6 4.92×·家族一致·无反例）·[VERIFY-LADDER] 禁跳级（双子门须皆过）。register-budget-fit 部署 FAIL（rolled 可达 PASS）叠加。
- **C3′ 边界确认 = ✅**：K-quant M=1 GEVM instruction-count 结构限【家族一致】·不因格式变·赢点若有 = memory-locality 非 instruction-count（[CASE-MICRO-E2E]）。

**⇒ 诚实前提坐实**：本线求**结构 C1 完成（5 格 plan 覆盖）+ 家族级 perf 边界确认**·**非 perf-covered 移动**·e2e 大概率 [CASE-MICRO-E2E] 受限·不承诺 green。q3_K 削重建 = 合法结构级 C1 改良 + byte-exact + rolled 兑现·但 perf-gate 打回（同 q5_K 结构边界·[PAT-1] 适用边界 C3′ 素材）。

**禁上板 · 禁 git · 无 schema label 改动 · tracked 源码只读未改（hand-construct 全在 `raw/`）。**

---
### 复现
```
GCC=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin/riscv64-unknown-linux-gnu-gcc
GPP=${GCC%gcc}g++ ; WEFT=build-weft/bin/weft-opt
for f in q2 q3 q6; do
  F=test/Conversion/RVV/rvv-emit-identity-quant-contraction-${f}-K-repack-vlen128.mlir
  $WEFT $F --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > raw/${f}k_gevm_CURRENT.emitc.c
  $GPP -march=rv64gcv_zvfh -mabi=lp64d -O3 -S raw/${f}k_gevm_CURRENT.emitc.c -o raw/${f}k_gevm_CURRENT.gcc15.O3.s
  $GCC -march=rv64gcv_zvfh -mabi=lp64d -O3 -S raw/stock_${f}_K_vl128.c -o raw/stock_${f}_K_vl128.gcc15.O3.s   # 手写 inline asm 对手
done
gcc -O2 raw/byteexact_q3k.c -o raw/be_q3k && raw/be_q3k                                    # q3_K byte-exact 0/8 硬门
$GCC -march=rv64gcv_zvfh -mabi=lp64d -O3 -S raw/q3k_gevm_variants.c -o raw/q3k_gevm_variants.gcc15.O3.s  # rolled cur vs knest
# 计数：grep -cE '^\s+v[a-z]' (v-insn) · grep -cE 'vs[0-9]r\.v|vl[0-9]r\.v' (vspill) · grep -cE '^\s+vset' (vsetvl 税)
```
