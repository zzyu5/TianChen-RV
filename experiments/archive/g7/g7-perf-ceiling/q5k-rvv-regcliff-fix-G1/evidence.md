# G7-PERF-CEILING / q5k-rvv-regcliff-fix-G1 — [GAP-Q5K-VLEN128-QH-REGCLIFF] 可修性评估（纯本地 G1·零板·零源码改动）

> **线**: research + design + G1（read-only 分析 + hand-construct standalone .c·**无 emitter 源码改动·无板·无 e2e**）· 分支 refactor/full-refactor-m1 · HEAD 6b8b7d39
> **问**: q5_K@rvv prefill GEMM 的 [GAP-Q5K-VLEN128-QH-REGCLIFF] 能否由 **register-pressure 减压**（native-mask / rolled-strip / lifetime-reorder）修到 prefill ≥parity = genuine perf 改良？还是 VLEN128 结构内禀不可修？
> **域**: rvv/VLEN128（部署 clang-18 [L-10] / 对手 stock gcc-15 block-dot）· 本地工具 gcc-15.2.0（= stock 出货编译器）+ clang-20（= 我方 clang 家族·[L-10] 出货 clang-18）· `-march=rv64gcv_zvfh`（zvl128b）
> **裁决（一句）**: **NOT FIXABLE by register-pressure reduction — 结构坐实（VLEN128-intrinsic）**。所谓 "register-cliff" **是误名**：VECTOR 寄存器 spill 在 VLEN128 与 VLEN256 **完全相同（6·q4_K=4）且不可由任何 byte-exact qh-recon 减压降低**；WIN(k1/VLEN256)/LOSS(rvv/VLEN128) 的判别 = **numHalves=2 的 tile 翻倍（同 16-col 输出跑 2× 指令）+ qh 5th-bit 解码不可约指令量**，摊到 VLEN128 的 8-lane 上。register-pressure 减压是【错的杠杆】。诚实：这是干净的**负判读**（不可修），与可修同价值（任务明示）。

---

## 0. 前置 — 承接的已证 prior（不重测）
- **L2 系统账地面真值**（`../../l2-kquant-rvv-systemacct/q5_K/`·`53b2d3b0`）: q5_K@rvv **prefill 0.8688× LOSS**（clang-18 emitted repack vs stock gcc-15 block-dot·n=4·relIQR 0.28%）· q5_K@k1 **1.641× WIN**（VLEN256·同 VLA 核）· 部署 clang-18 spill 报 **GEMM=155 / GEVM=97**（gcc-15 GEMM=4124）。correctness GREEN@bounded-ULP（ZERO-MODEL INTEGER bit-exact 独立 cert）。
- **kquant-prefill-quality-G1**（`../kquant-prefill-quality-G1/`·db44829f）: 天花板 CONFIRMED·q5_K@rvv 标 **"唯一有真 headroom"（register-cliff·弱 generic 对手）但"非 count mover"**（cell 已绿 via k1）。**本线 = 兑现该 headroom 评估 — 结论：headroom 不真实**。
- **qh-subblock-pack-retranspose-G1**（`../qh-subblock-pack-retranspose-G1/`）: re-transpose qh 布局恢复 vlm native-mask·**结构可行∧byte-exact BUT HIGHER-RISK（repack-layout + GEMM∧GEVM 两路 + re-seal）= informed-decline**。retrans isolated qh-recon 16→12 v-insn。
- **G6-B 板测**（`../../g6-b-emit-unroll/`）: rolled/register-fit/nr 三杠杆板测穷尽·**全 <parity（最好 0.63×·rolled 0.41×）** = register-pressure-reduction-via-rolling **PROVEN BACKFIRES**（redundant weight re-decode 主导）。

## 1. 步骤1 — register-cliff 根因精确刻画（read-only + hand-construct·VLEN128 gcc-15.2/clang-20 objdump）

### 1.1 emitter 现状：register-pressure 减压【已最大化部署】
`emitRepackKQuantGemmBodyQ5K`（`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:9314-10059`·read-only）已实施的减压（`:9653-9672` 注释自述 "peak ~94 live vregs > 32 → tile-local set, under the spill cliff"）：
1. **S1 h-strip tiling**（`:9673`）: numHalves = weight_interleave/half = **2@VLEN128 / 1@VLEN256**（capability-keyed·VLEN128 拆 2 个 8-lane strip 各自 block-loop·分离 live-set）。
2. **scale/min/bsums panels 落栈**（`:9731-9808`）: `vse16/vle16/vse32` 把冷解码带 + idle min 累加器搬**离热 live-set**（DELIBERATE tile-materialization·非 spill）。
3. **d/dmin widen 延迟到 fold**（`:9985`）: 2 个 f32m2 strip 不跨热 dot 循环存活（−4 vreg）。
⇒ **减压已到结构底**（tiling + panel + defer 三件齐）。本线问：在此之上，qh-recon 减压能否再降 spill？

### 1.2 q5_K vs q4_K 唯一 delta = qh 5th-bit inject（`:9928-9951`）
热 ii-循环（`for ii<16`）内 q5_K 比 q4_K 多的**唯一**结构 = qh 5th-bit 解码：
```
qhStrip=vle8(...); loSel=vsrl(qhStrip,s); loBit=vsll(vand(loSel,1),4); nLo=vor(loNib,loBit)  // OLD·4 op/臂
```
「the ONLY q5_K delta vs the S6-tiled q4_K GEMM body」（`:9932-9935` 源码自述）。

### 1.3 ★precision-lesson 校正：VECTOR spill vs SCALAR spill vs tile-mat（"155 spill" 被误读）
hand-construct `raw/strip_body_regpressure.c`（**忠实复现单 strip 热 nest + 4-col 累加器 + 8-sub-block 展开 + qh recon**·panels 落栈）·clang-20 -O3 VLEN128：

| clang-20 变体 | **VEC-spill**（vs*r.v）| SCA-spill（sd）| panel-vse（tile-mat·非 spill）| v-insn | vsetvl |
|---|---|---|---|---|---|
| q4k（无 qh·基线）| **4** | 39 | 37 | 359 | 29 |
| q5k_old（部署）| **6** | 42 | 37 | 402 | 29 |

- **★harness 保真度确证**: 本地 q4k **VECTOR spill = 4 = 板 clang-18 部署 q4_K spill=4 完全吻合** → hand-construct 忠实复现 VECTOR 寄存器分配。
- **★"155 spill" 被误读**: L2 报的 "spill=155" 是 **TOTAL Folded Spill（scalar+vector·2-strip·full-nest）**·实测拆分 = **VECTOR 仅 6/strip·SCALAR(地址计算 sd) 42/strip·tile-mat(deliberate vse) 37/strip**。真正的 VECTOR 寄存器压力 delta = **q5_K 6 vs q4_K 4 = 仅 +2/strip**。所谓 "register-cliff" 在 VECTOR 层面**只有 +2·不是崖**。

## 2. 步骤2 — register-pressure 减法设计（三候选杠杆·全 byte-exact）
| 杠杆 | qh-recon | 来源/风险 | register-budget-fit 前置 |
|---|---|---|---|
| (a) **native-mask KNEST** | `m=vmsne(vand(qh,1<<s));nLo=vadd_mu(m,loNib,16)`·mask 进 v0/mask-reg | q5k-knest-G1·byte-exact·decode-only 低风险 | 削 vsll/vor VREG 中间量 → 预期降热 live-set |
| (b) **retranspose** | `m=vlm(mask);nLo=vadd_mu(m,loNib,16)` | qh-subblock-pack-retranspose-G1·byte-exact·**HIGHER-RISK informed-decline**（repack-layout + 两路 + re-seal）| 最强削减（连 vand 都省）|
| (c) **rolled K-nest** | roll 展开的 ii/sub-block | **G6-B 板测 PROVEN BACKFIRES 0.41×** | 不合格（perf 反向）|

## 3. 步骤3 — G1 静态账（hand-construct·部署 gcc-15.2/clang-20·VLEN128）

### 3.1 byte-exact 硬门（`raw/byte_exact_qhrecon.c`·穷尽 256 qh × 8 subbit × 16 nibble = 32768）
```
OLD == KNEST == RETRANS == ref (w = nibble | (qh_bit<<4))    mismatch = 0 / 32768    RESULT: PASS
```
三 qh-recon 变体 bit-identical（KNEST `nib+16 == nib|16` 因 nibble<16·retrans 是 permutation 读同 bit）→ 减压不改算法·硬门绿。

### 3.2 ★register-pressure 静态账（clang-20·VLEN128·VECTOR-spill = register-cliff 真度量）
| 变体 | **VEC-spl** | VEC-rld | v-insn | vsetvl | qh Δv-insn |
|---|---|---|---|---|---|
| q4k（无 qh）| 4 | 4 | 359 | 29 | +0 |
| **q5k_old**（部署）| **6** | 6 | 402 | 29 | **+43** |
| **q5k_knest**（减压 a）| **6** | 6 | 399 | 32 | +40 |
| **q5k_retrans**（减压 b·最强）| **6** | 6 | 384 | 29 | +25 |

- **★★VECTOR spill = 6 · OLD/KNEST/retrans 三者完全相同**：任何 byte-exact qh-recon 减压 **均不降 VECTOR 寄存器 spill**。register-pressure 减压对 spill = **零效果**。
- **KNEST 全身近无用**: v-insn +40 vs +43（仅 −3）· **vsetvl 反升 29→32**（mask-op SEW toggle 税·[CASE-KQUANT-GCC-CODEGEN] 同型）· spill 不动 → **净 wash**。
- **retrans 削 v-insn 最多（+25）但仍不降 spill(6)** 且 = HIGHER-RISK informed-decline（qh-subblock casefile 已裁不值）。
- **真正的 q5_K 成本 = qh 5th-bit 解码【指令量】**：+43 v-insn/strip（OLD）·不可约地板 +25（连最强 retrans 都要 +25：loNib/hiNib OR + mask/vlm + vadd_mu）。**5th bit 必须被抽出 → 指令不可约**。这才是 q5_K@rvv 输 block-dot 而 q4_K（qh Δ=+0）赢的原因，**非 spill**。

### 3.3 ★★决定性：VLEN256 同 spill 却赢 → spill 非判别键（`raw/clang20_vlen256_*.s`）
同 hand-construct 编 VLEN256（half=16·numHalves=1 单 strip）：
| 变体 | VEC-spl（VLEN256 单strip）| v-insn（单strip）| = **16-col 输出总 v-insn** |
|---|---|---|---|
| q4k | 4 | 359 | VLEN256 **359** / VLEN128 ~**718**（2 strip）|
| q5k_old | **6** | 402 | VLEN256 **402** / VLEN128 ~**804**（2 strip）|

- **★★VECTOR spill 在 VLEN128 与 VLEN256 完全相同（q5_K=6·q4_K=4）** → **不存在 VLEN128-specific vector-spill 崖**。register 压力两板一致。
- **WIN/LOSS 判别 = numHalves=2 tile 翻倍**：同 16-col 输出，VLEN128 需 **2 strip（~804 v-insn·8-lane/insn）**，VLEN256 **1 strip（402 v-insn·16-lane/insn）**。对手 block-dot 速率与我方 VLEN 无关 → VLEN128 的 2× 指令量 + qh 成本输·VLEN256 的 16-lane 摊销赢。**VLEN128-INTRINSIC·非 register-pressure**。

## 4. 步骤4 — 可行性裁决

| 判据 | 结果 |
|---|---|
| (a) register-cliff 可由减压修？| **❌ 否**：VECTOR spill = 6·OLD/KNEST/retrans 三者相同·**任何 byte-exact 减压零效果** |
| (b) register-budget-fit（VLEN128 spill-free ≤32 vreg）？| **N/A / 已在预算内**：q5_K VECTOR spill 仅 6（q4_K 4）·热 live-set 已近 32 但 **崖不在 vector·在指令量** |
| (c) 达 prefill ≥parity？| **❌ 否**：即便 spill=0·q5_K@rvv 的 +43 v-insn/strip × 2 strip（VLEN128 tile 翻倍）仍输 block-dot·**减压不触此项** |
| (d) byte-exact？| ✅ PASS（0/32768·减压不改算法）|
| **可修性** | **❌ NOT FIXABLE by register-pressure reduction — 结构坐实（VLEN128-intrinsic）** |

**★具名结论**：**[GAP-Q5K-VLEN128-QH-REGCLIFF] = 结构坐实·VLEN128-intrinsic**·且**名称应精化**：
- **不是 vector register-cliff**（VECTOR spill 两板同 = 6·+2 来自 qh·不可约·非崖）。
- **是 qh-decode 指令量在 8-lane VLEN128 的摊销缺口**：numHalves=2 tile 翻倍（16-col 需 2 strip）+ qh 5th-bit 不可约指令量（+43/strip·地板 +25）· 摊到 8 lane。
- L2 锚定 register-cliff 的 "155 spill" ≈ **90%+ SCALAR GPR 地址 spill + tile-mat**（实测 6 vector : 42 scalar : 37 tile-mat / strip），**非 vector 寄存器压力**。同一结构裁决·更精更诚实的机制名。

## 5. ★诚实定性（可修与不可修同价值）
- **不增 perf-covered 计数**：q5_K cell 已绿（via k1 any-board）·**9/83 不变**（任务诚实前提遵守）。
- **本线兑现了 kquant-prefill-quality-G1 标记的 "唯一 perf-improvable headroom" 评估 → headroom 不真实**：register-pressure 减压不是有效杠杆（spill 不降·指令量不降·retrans HIGHER-RISK 已 informed-decline）。**"唯一有真 headroom" caveat 应【撤回/精化】** —— q5_K@rvv 与 q2_K/q6_K 同属 VLEN128-throughput-bound（gcc-death 必要非充分律第三例·L2 已证）·非 register-fixable 特例。
- **天花板 CONFIRMED 强化**：perf-covered 9/83 天花板经此格 register-pressure 轴 = 坐实·**四格皆结构/算法级**（q2_K weight 摊销不足 / q6_K 6-bit plane 内禀 / q3_K decode-only / q5_K VLEN128 tile-doubling + qh 指令量）·无 instruction-level register 减压可 cross。
- **负判读 = 有效正结果**（任务明示"诚实至上·可修与不可修同价值"）。设计 + byte-exact 模型 + 静态账就绪·若主会话未来以别名义（uniform-emitter-maturity）取 KNEST/retrans 可直用本 casefile（但**明记：perf=零·不 flip VLEN128**）。

## 6. Raw artifacts（`raw/`）
- `strip_body_regpressure.c` — 忠实单 strip 热 nest hand-construct（QHMODE 0=q4k/1=q5k_old/2=knest/3=retrans）
- `byte_exact_qhrecon.c` + `byte_exact_qhrecon` — 硬门（0/32768·OLD≡KNEST≡RETRANS）
- `clang20_{q4k,q5k_old,q5k_knest,q5k_retrans}.s` / `gcc15_*.s` — VLEN128 objdump 双编译器
- `clang20_vlen256_{q4k,q5k_old,q5k_knest}.s` — VLEN256 单-strip 对照（spill 同=6 决定性证）
- **CAVEAT**: 本地 clang-20 TOTAL-spill 绝对数 ≠ 板 clang-18（计数法/版本差·prior casefile 已记）·但 **VECTOR spill 本地 q4_K=4 = 板 clang-18 q4_K=4 吻合** → VECTOR 层 relative + 绝对均可信。gcc-15.2 不支持 `-mrvv-vector-bits`（非阻塞）。

**禁上板 · 禁 git · 无 schema label 改动 · tracked 源码只读未改（hand-construct 全在 `raw/`）。**

---
### 复现
```
GCC=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin/riscv64-unknown-linux-gnu-gcc-15.2.0
SYSROOT=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/sysroot
CLANG="clang-20 --target=riscv64-unknown-linux-gnu --sysroot=$SYSROOT"
cd raw
gcc -O2 byte_exact_qhrecon.c -o byte_exact_qhrecon && ./byte_exact_qhrecon         # 硬门 0/32768
for m in 0 1 2 3; do $CLANG -march=rv64gcv_zvfh -mabi=lp64d -O3 -S -DQHMODE=$m strip_body_regpressure.c -o /tmp/c$m.s; done
# VECTOR spill = grep -cE 'vs[0-9]+r\.v.*Folded Spill' ; v-insn = grep -cE '^\s+v[a-z]'
```
