# G7 perf-ceiling — whole-K-nest S6 重构【可行性评估】· G1 静态账（零板·纯本地·informs 必问）

> **线定位（诚实前提）**：评估最深结构 gap **[GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP]** 的重构可行性 —— whole-K-nest S6 restructuring 值不值得建。**【非】perf-covered mover**（[CASE-MICRO-E2E] 已证 K-quant M=1 GEVM decode 不传导 e2e；G6-B 判 whole-nest 重构 "non-warranted perf 战·mechanism 名义"）。本线 = **结构可行性评估（C1 emitter maturity 深度）+ informs 用户 radical-lever 必问决策**。**不承诺 perf green**。
> 依据 canon：[VERIFY-LADDER] G1 双子门 · [K-10] 结构级 · [CASE-MICRO-E2E] · [GAP-P1] register-pressure trap（P2 教训）· 性能宪章规则1-2/4 · [CASE-COMPILER-ASYMMETRY]（部署 gcc-15.2 域·rvv 板 VLEN128）。
> **止于 G1（静态账）+ 设计 · 禁 git · 禁上板 · 禁改 tracked 源码**（RVVToEmitC* 只读·hand-construct 全在 `raw/`）。
> 代表格 = **q5_K**（super-block 结构最典型·qh plane + dual d/dmin·有既有 byte-exact oracle + rolled 模板）。

---

## ★裁决 TL;DR

1. **whole-K-nest roll = 结构可行 ∧ byte-exact ∧ 预算合规（三门全过）**。控制 A/B（同源·byte-identical 算法·仅 K-nest 循环指令异·部署 gcc-15.2 -O3）实测：
   - **(a) tile round-trip ELIMINATED**：`vs*r.v` 栈流量 **104 → 0**（whole-nest roll 消除）。
   - **(b) register-budget-fit PASS**：rolled **28 distinct vreg · 0 向量 spill**（单一 wide resident accumulator sumi/bsum/sumf 驻寄存器·TG=1）。**P2 的 TG=2 bank 加宽反噬【被规避】**（本设计 TG=1 前置守恒）。
   - **(c) byte-exact**：两形态 DYNAMIC op-stream 恒等（280 vwmacc/super-block）· 削重建 host oracle 0/32 · G6-B 板 A/B 先例（同 transform 43008/43008 byte 0-mismatch）。
2. **附带杠杆兑现**：whole-nest roll 同时 **vsetvli 税 1419 → 46（−96.8%·hot ii-loop 内 0·gcc 外提）** + static v-insn **4277 → 311（−92.7%·13.8×）** + textB **−94.6%**。= [GAP-EMIT-VSETVL-TAX] 一并塌缩。
3. **tile-roundtrip 是【家族级】pathology**：5 个 K-quant GEVM CURRENT full-unroll **全带** vs*r.v（q2 143/q3 174/q4 201/q5 209/q6 137·全 vl*r.v=0·vle8 element-wise 重读）。whole-nest roll 是家族级 closure。
4. **★但 perf 资格【仍打回】**：rolled DYNAMIC 指令数 per-output **~232 vs stock 130 = ~1.78×** > 1.10× 子门。虽比 unrolled static 4.19× **好 ~2.3×**（去 tile-RT + vsetvli 税），但 **M=1 element-wise broadcast 密度（256 element-vwmacc/8-output vs stock vl=32 wide block-dot）结构性超 stock**。[CASE-MICRO-E2E] decode memory-wall → **无 perf-covered green**。
5. **★★GEVM roll 规避了 GEMM roll 的杀手**：G6-B P3 板证伪 narrow-tile GEMM roll（0.41–0.63×）的元凶 = **redundant weight re-decode**（narrow output-tile → 每权重多次解码）。**GEVM（M=1）无 output-tile 维度可窄 → 无 redundant-decode 罚**（每权重恰解码一次）。∴ GEVM roll **不会**像 GEMM roll 那样回归吞吐；但仍不过 instruction-count 门。
6. **informs 必问结论**：whole-K-nest S6 重构 = **结构可行 + byte-exact + 低风险（roll 机制 G6-B 已建）+ 家族级最深 emitter-maturity 靶**·**但纯 kernel-axis·零 perf green**。**值不值得建 = 取决于是否为 C1 emitter-maturity 深度付多-leaf 共享发射器工**（明知无 perf 数字）。若求 perf-covered movement → **确证【非】**；若求 C1 plan 库成熟度 → **是唯一最深可行靶**。**只能以 mechanism/method 名义立项，不得以 perf 名义**（性能宪章规则4 · G6-B 自身判词）。

---

## 0. 方法与工具链（零板·纯本地·可复现）

- **部署编译器域**：`riscv64-unknown-linux-gnu-g++ (gf3b8c022145) 15.2.0`（`/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin`·`-march=rv64gcv_zvfh -mabi=lp64d -O3`）= rvv 板 e2e 部署编译器族。emitc 是 C++（`extern "C"`）→ 用 g++。**全 spill/vreg/v-insn 以部署编译器测**（非 clang 代理·[CASE-COMPILER-ASYMMETRY] 戒）。
- **CURRENT emitc 真源（只读）**：`weft-opt <fixture> --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`·fixture = `test/Conversion/RVV/rvv-emit-identity-quant-contraction-q{2,3,4,5,6}-K-repack-vlen128.mlir`。发射真源 = `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` `emitRepackKQuantGemvBodyQ5K`（8517-9080·只读）。
- **控制 hand-construct**（`raw/q5k_knest_roll.c`）：ONE source·两函数 `q5k_rolled`/`q5k_unrolled`·**共享 `ELEM_STEP` 宏**（同 vwmacc/decode op 序·asc j,pair,k,ii）·唯一差 = K-nest 循环 `#pragma GCC unroll 1`（rolled·runtime `emitc.for`-等价）vs `64`（unrolled·全展开）。累加器 = 单一 wide resident bank（sumi/bsum/sumf·跨 super-block K-loop SSA 携带）。结构忠实 emitRepackKQuantGemvBodyQ5K·numHalves=1 single-strip（v-insn count VLEN-invariant·q5k-knest-G1 校准）。
- **stock 对手**：`ggml_vec_dot_q5_K`（`llama.cpp/.../riscv/quants.c` block-dot·130 v-insn/super-block/output·同 gcc-15.2 编·复用 q5k-knest-G1 `raw/stock_q5_K_blockdot`）。
- **byte-exact**：① DYNAMIC op-stream 恒等（280 vwmacc/super-block 两形态）② host decode oracle（`../q5k-knest-G1/raw/byteexact_model.c`·0/32）③ 数值 A/B 硬件跑 = **DEFERRED**（GNU sim `riscv64-unknown-linux-gnu-run` 无 RVV·signal 4 illegal insn·板本线禁）→ 依 construction-identity + **G6-B P3 板 A/B 先例**（同 emitc.for-vs-unroll transform·q2_K GEMM·rolled==unrolled 43008/43008 byte 0-mismatch·真硅）。

---

## 1. ★承接已证 prior（步骤1·不重做）

### 1.1 为何 full-unroll 有栈税（[GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP]·kquant-family casefile 05f96739）
K-quant GEVM full-unroll 把 16-col 解码 tile **round-trip 过栈 scratch**：`vs*r.v` whole-reg store（q2 143/q3 174/q4 201/q5 209/q6 137）→ 后经 **vle8 element-wise 重读**（非经典 RA reload·故 `vl*r.v`=0）。本线控制复现坐实此签名：q5k_unrolled 104 `vs*r.v` + 0 `vl*r.v` + **94 `addi *,sp,*` scratch 指针** + 597 vle8。= per-element-broadcast 结构令解码 tile 无法驻寄存器跨 per-element 广播·被迫过内存。

### 1.2 为何 narrow-tile-roll 更差（G6-B P3 板证伪·real-lever-tiling-sweep）
G6-B P3 **板实测**（rvv clang-17 对称）：narrow-tile GEMM roll 全 LOSS（rolled-lvalue_t4 0.63× / t1 0.41× / rolled-panel 0.41× vs shipped unrolled）。**nr 旋钮在吞吐上【证伪】**（窄 tile → vector-residence↑ 但吞吐↓）·元凶 = **redundant weight re-decode**（窄 output-tile → 每权重多 pass 解码·t1=4 cLo passes）。具名边界（P3 verdict 逐字）：*"compact(16 vwmacc)∧resident(spill 4) 形态 NOT reachable by rolling inner mm loop + narrowing tile — requires rolling the **entire K-super/sub-block nest** into runtime loops with a **single wide resident accumulator** (a different S6 loop structure)"* = **本线评估靶**。

### 1.3 whole-nest roll 的假设机制（步骤1 提炼）
把整个 super-block × sub-block × element nest roll 成 runtime loop·单一 wide resident accumulator 跨 K 归约驻寄存器。假设：① 解码内联进 loop·不再 materialize tile 过栈（消 vs*r.v）② uniform loop body → vsetvli 可 hoist/CSE（消 vsetvli 税）③ 累加器 resident（G6-B P3 Evidence-2 已证 `sumfVar` 可跨 iter-arg-less `emitc.for` 驻寄存器）。**★与 GEMM narrow-tile roll 的判别键**：GEVM M=1 无 output-tile 维度 → 无 redundant-decode 罚（G6-B roll 杀手不适用）。

---

## 2. ★whole-nest rolled 单-wide-accumulator 结构设计（步骤2）

### 2.1 结构
- **外层 K 归约 = runtime `emitc.for` over super-blocks（nb）**（现行已是·不改）。
- **super-block BODY nest（j × pair × k × ii element）= runtime loop**（现行 = 全展开·本设计 roll 成 loop）。dominant = **ii-loop（16-trip·per-element decode + vwmacc）** —— roll 它是消 tile-RT + vsetvli 税的关键杠杆。
- **单一 wide resident accumulator bank**：`sumi`（i32·主整数点积）+ `bsum`（i32·min-fold）+ `sumf`（f32·终 fold）· **跨 super-block K-loop 作 SSA VariableOp 携带**（G6-B P3 Evidence-2 机制：seed above blockLoop·load/assign inside·地址不取 → mem2reg 驻 vreg）。**per-element 解码 tile 不驻**（即消即用 vwmacc 进 accumulator·不 materialize）。

### 2.2 ★register-budget-fit 前置分析（P2 教训·决定性）
| 配置 | resident 向量态 | 部署 gcc-15.2 spill | 判 |
|---|---|---|---|
| CURRENT full-unroll | tile materialize 过栈（vs*r.v 209）| tile-RT（非经典 spill·vl*r.v=0）| tile-roundtrip |
| **whole-nest roll（TG=1·本设计）** | sumi + bsum + sumf resident + 工作集 | **0 向量 spill·28 vreg**（实测）| **✅ FIT** |
| TG=2 假想（加宽 accumulator bank）| accumulator ×2 → 溢 32-vreg | **必 spill**（P2 板证 534 spill·[GAP-P1] 反噬）| **设计明文拒** |

**⇒ register-budget-fit = PASS**。单一 wide accumulator（sumi/bsum/sumf·TG=1）**28 distinct vreg·maxVreg v31·0 向量 spill**。**P2 的加宽-bank 反噬【前置规避】**：roll 消 tile-RT 靠**驻单一 wide accumulator**·非加宽 bank —— 不引入新 spill（与 P2 falsified 路径正交）。

---

## 3. ★G1 静态账（部署 gcc-15.2 -O3·硬门 = byte-exact 代数模型）（步骤3）

### 3.1 控制 A/B（同源·byte-IDENTICAL 算法·仅 K-nest 循环指令异）
| 度量 | **q5k_rolled（whole-nest）** | q5k_unrolled（full）| Δ |
|---|---|---|---|
| static v-insn | **311** | 4277 | **−92.7%（13.8×）** |
| vsetvli | **46** | 1419 | **−96.8%（30.8×）** |
| **vs*r.v（tile-RT store）** | **0 ← ELIMINATED** | **104** | **tile-roundtrip GONE** |
| vl*r.v（reload）| 0 | 0 | （皆非经典 spill）|
| vle8 | 28 | 597 | −95.3% |
| vwmacc（static）| 40（loop→280 dyn）| 280 | 同 DYNAMIC |
| distinct vreg | **28** | 32 | **fit ≤32** |
| maxVreg | v31 | v31 | 同 |
| sp_sd / sp_ld（标量）| 8 / 10 | 12 / 12 | rolled 更少（仅 frame）|
| asm lines | 550 | 10229 | −94.6% |
| **hot ii-loop 内 vsetvli** | **0（HOISTED）** | n/a（展开）| **vsetvli 税塌缩** |

**★unrolled 忠实复现部署 pathology**：q5k_unrolled（4277 v-insn / 1419 vsetvli / 104 vs*r.v·single-strip）≈ CURRENT q4_K 部署 emitc（4227 / 1420 / 201）·q5_K CURRENT 8751/2943/209 = numHalves=2 ≈ 2× single-strip。控制构造可信。

### 3.2 byte-exact 硬门（代数模型·mismatch=0）
- **DYNAMIC op-stream 恒等**：两形态每 super-block 执行 **280 vwmacc**（256 ELEM_STEP + 16 sumi_vv + 8 bsum_vx）·roll = 纯 loop-structure 变·**不改算术序** → byte-exact by construction（同 G6-B P1 "identical integer vwmacc accumulation order; only residence changes"）。
- **host decode oracle**（`byteexact_model.c`·穷举 nibble∈[0,15]×qh_bit∈{0,1}）：**stock == current == knest 0/32 mismatch**。
- **数值 A/B（硬件）= DEFERRED**：GNU sim 无 RVV（signal 4）·板本线禁。依 construction-identity + **G6-B P3 板 A/B 先例**（同 emitc.for-vs-unroll transform·q2_K GEMM·rolled==unrolled 43008/43008 byte 0-mismatch·真硅证）。**诚实标注**：本线 byte-exact = by-construction + oracle + 板先例·非本线新板跑。

### 3.3 instruction-count 子门（rolled DYNAMIC·per-output）
rolled DYNAMIC/super-block = ELEM_STEP 8×16×13.5=1728 + unpack 88 + bsum 8 + sumi 16 + folds 12 = **~1852 v-insn（8 outputs）** → per-output **~232**·stock **130** → **~1.78×**（> 1.10× 子门·**NOT MET**）。
- cf. UNROLLED static per-output **545 = 4.19×**。**rolled ~2.3× 更接近**（去 tile-RT + vsetvli 税）·**但仍不达**。
- 根因 = **M=1 element-wise broadcast 密度**（256 element-vwmacc 产 8 outputs·vs stock vl=32 wide block-dot 8-chunk 覆盖 256 elem）·**结构性 > stock·不因 roll 变**（roll 是 icache/code-volume/vsetvli 杠杆·非 instruction-density 杠杆）。

---

## 4. ★可行性裁决（步骤4·informs 必问）

### 4.1 结构可行性（三门）
| 门 | 判据 | 结果 |
|---|---|---|
| **(a) 真消 tile round-trip（栈→0）**| vs*r.v 104 → 0（控制 A/B）| **✅ YES** |
| **(b) fit register-budget（单 wide accumulator ≤32 vreg 无新 spill）**| 28 vreg · 0 向量 spill · TG=1（P2 trap 规避）| **✅ YES** |
| **(c) byte-exact** | by-construction（280 vwmacc 恒等）+ oracle 0/32 + G6-B 板先例 | **✅ YES** |
| **⇒ 结构可行？** | | **✅ 结构可行**（非 (d) 不可行/预算爆）|

### 4.2 收益/风险静态估
- **收益（emitter-maturity / C1·非 perf）**：closes **[GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP]**（家族级·5 leaf 全带 vs*r.v 137-209 → 0）+ 塌缩 **[GAP-EMIT-VSETVL-TAX]**（vsetvli 30× ↓·家族最大单项 25-30% 税）+ instruction-count ratio 4.19×→~1.78×（腰斩）。= **家族级最深可行 emitter-maturity 靶**·纯结构 C1 plan 库成熟。
- **风险/工作量**：
  1. 触 **共享发射器** `RVVToEmitCBlockQuantLinear.cpp`（5 K-quant GEVM leaf 共 super-block-nest envelope·multi-leaf 重写）·须串行（memory `parallel-lines-need-disjoint-files`）。
  2. **roll 机制【已建·低风险】**：G6-B P1 已建 rolled-loop 发射 MODE（`emit_loop_schedule`·capability-keyed·byte-exact-neutral default）+ P3 Evidence-2 已证 resident-across-`emitc.for`。GEVM 实装 = 把该 MODE 延到 GEVM decode nest（机制存在·非框架级不可能）。
  3. **比 [远期] repack-format 改【低风险】**：qh re-transpose（q5k-knest blocker [GAP-EMIT-KNEST-QH-SUBBLOCK-PACK]）改前门 repack tool·不可逆度高；whole-nest roll 只改发射编排·byte-exact-neutral·可逆。
- **★perf 收益 = 零 green**：instruction-count ~1.78× > 1.10×（kernel-axis·结构 M=1 限）+ [CASE-MICRO-E2E] decode memory-wall（GEVM M=1 削重建 e2e 受限已证·q5_K super-block 只更受限）。**GEVM roll 规避 GEMM roll 的 redundant-decode 罚（无 output-tile 维度）→ 不回归吞吐·但也不 beat stock**。

### 4.3 ★informs 必问结论（whole-K-nest S6 重构值不值得建·给用户静态数据）
| 维度 | 静态数据 |
|---|---|
| **结构可行性** | ✅ 可行（三门全过·tile-RT 消·预算 fit·byte-exact）|
| **预期收益** | emitter-maturity / C1（家族级 tile-RT closure + vsetvli 税塌缩）·**非 perf green**·kernel-axis-only |
| **风险/工作量** | multi-leaf 共享发射器重写（中）· roll 机制已建（低）· 比 repack-format 改低风险 · byte-exact-neutral 可逆 |
| **立项名义约束** | **只能 mechanism/method 名义**（性能宪章规则4·G6-B 自身判词）·**不得 perf 名义** |

**⇒ 值不值得建的判别键 = 是否为 C1 emitter-maturity【深度/完整度】付多-leaf 工·明知无 perf 数字**：
- **若用户求 perf-covered movement** → whole-nest S6 重构 **确证【非 mover】**（[CASE-MICRO-E2E] + instruction-count 1.78×）·**不建**（放空 perf 名义弹）。
- **若用户求 C1 plan 库成熟度深度**（论文 C1 "mature GEVM emission" 证词）→ 这是**唯一最深可行靶**·结构可行 + byte-exact + 低风险·**可建·但须 mechanism 名义 + 明记零 perf**。

**诚实至上**：结构可行但纯 kernel-axis 无 perf = **有效的必问输入**（= 排除了 "结构不可行" 与 "有 perf 藏在结构里" 两种误判·把决策收窄为纯 C1-maturity 价值判断）。

---

## 5. 复现
```
TC=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin
GPP=$TC/riscv64-unknown-linux-gnu-g++ ; WEFT=build-weft/bin/weft-opt
# 控制 A/B（rolled vs unrolled·同源）
$GPP -march=rv64gcv_zvfh -mabi=lp64d -O3 -S raw/q5k_knest_roll.c -o raw/q5k_knest_roll.gcc15.O3.s
#   计数：per-func split on _Z10q5k_rolled.. / _Z12q5k_unrolled..（C++ mangled）
#   v-insn '^\s+v[a-z]' · vsetvli '^\s+vset' · tile-RT store 'vs[0-9]+r\.v' · reload 'vl[0-9]+r\.v'
# 家族 CURRENT full-unroll（真路径）
$WEFT test/Conversion/RVV/rvv-emit-identity-quant-contraction-q4-K-repack-vlen128.mlir \
  --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > raw/q4k_gevm_CURRENT.emitc.c
$GPP -march=rv64gcv_zvfh -mabi=lp64d -O3 -S raw/q4k_gevm_CURRENT.emitc.c -o raw/q4k_gevm_CURRENT.gcc15.O3.s
# byte-exact host decode oracle
gcc -O2 ../g7-l2-gevm-redesign/q5k-knest-G1/raw/byteexact_model.c -o raw/be_q5k_host && raw/be_q5k_host
# 数值 A/B（DEFERRED：GNU sim 无 RVV → signal 4；板本线禁）
$GPP -march=rv64gcv_zvfh -mabi=lp64d -O3 -DDRIVER raw/q5k_knest_roll.c -o raw/q5k_ab   # 板可跑·本线未跑
```

**禁上板 · 禁 git · 无 schema label 改动 · tracked 源码只读未改（hand-construct 全在 `raw/`）。**
诚实前提坐实：结构可行【等价】不可行【同价值·都 informs 必问】——本线证 = 结构可行但无 perf green·把 radical-lever 决策收窄为纯 C1-maturity 价值判断。
