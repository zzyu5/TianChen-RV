# [G7-PERF-CEILING / kquant-prefill-quality-G1] — K-quant prefill GEMM emitter-quality 破天花板调查

> **线**: research + G1 (纯 read-only 分析 + hand-construct·**无 emitter 源码改动·无板**) · 分支 refactor/full-refactor-m1
> **问**: q2_K/q3_K/q6_K@rvv 或 q5_K@rvv 的 prefill GEMM emitter-quality 缺口能否闭合到 cross parity = 新 perf-covered green?
> **域**: rvv/VLEN128·部署编译器 clang-18 (我方出货 [L-10]) / 对手 stock gcc-15 as-shipped block-dot
> **本地工具**: gcc-15.2.0 (spacemit toolchain·= stock 出货编译器) + clang-20 (= 我方 clang 家族) · `-march=rv64gcv_zvfh` (zvl128b)
> **裁决 (一句)**: **天花板坐实 (CONFIRMED)**·四候选无一是 perf-covered mover·具名 blocker 全为结构/算法级·非 emitter-instruction 级·无 instruction-level hand-construct 可 cross。

---

## 0. 前置 — 承接的已证 prior (不重测)
- **L2 系统账 e2e 地面真值** (`experiments/active/l2-kquant-rvv-systemacct/`): q4_K@rvv **1.344× WIN** · q2_K@rvv **0.857× LOSS** · q5_K@rvv **0.8688× LOSS** (均 ours-clang-repack-GEMM vs stock-gcc-block-dot·系统账)。
- **G6-B 板测 exhaustive 证伪** (`experiments/active/g6-b-emit-unroll/`): q2_K prefill GEMM·kernel-axis symmetric clang-17·**unrolled = 最快 (1.000×)**·rolled-panel 0.41×·rolled-lvalue(register-resident)最好 0.63×·nr-knob 在 throughput 上**反证** (窄 tile→register-resident 但更慢·redundant weight re-decode 主导)。**命名 blocker**: hand-brick 的 compact(16 vwmacc)∧resident(spill 4) 形态**不可**由 roll 内 mm-loop + 窄 tile 达成·需把**整个 K-super/sub-block nest** roll 成 runtime loop + 单一 wide resident accumulator (不同 S6 loop structure) = mechanism 目标·非 perf 战。
- 本线 = 用**部署域 gcc-15.2 objdump** (G6-B 因板 gcc-12.3.1 缺 riscv_vector.h 而 **DEFERRED**) + **跨格 relative 静态账** + **weight-size amortization 论点** 收口天花板。

## 1. 步骤1 — emitter-quality gap 量化 (gcc-15.2 部署域 objdump·本地新证)
prefill GEMM 64-output-tile (8× f32m2 accumulator·4 行 × 16 列·2240–2304 vwmacc/super-block·全展开):

| 格 | gcc-15.2 -O3 vinsn/vsetvl/textB | clang-20 -O3 vinsn/vsetvl | stock hand-brick /output | weight B/sblk | e2e |
|---|---|---|---|---|---|
| **q4_K** | 4193 / **398** / 42064 | 5708 / **320** | 105 vinsn·7 vsetvl | **144** | **WIN 1.344×** |
| q2_K | 5769 / **950** / 62574 | 6247 / **394** | 125 vinsn·9 vsetvl | 84 | LOSS 0.857× |
| q6_K | 14733 / **6195** / **269934** | 9591 / **668** | 84 vinsn·10 vsetvl | 210 | LOSS |
| q3_K | (非本地·GEMM 板导出) | — | 102 vinsn·14 vsetvl | 110 | 未测 e2e |
| q5_K | (非本地·GEMM 板导出) | — | **无 hand-brick=generic** | 176 | k1 WIN / rvv LOSS 0.87× |

**gap 根因分层 (量化)**:
1. **vsetvli 税 (SEW-toggle 调度)** = 大半 **gcc-death**: gcc q6_K **6195** → clang **668** (9.3× 收缩)·gcc q2_K 950→clang 394·gcc q4_K 398→clang 320。gcc 在 mixed-SEW 全展开核上 vsetvli 爆炸 ([CASE-KQUANT-GCC-CODEGEN])·clang 部署域已大幅治愈。
2. **clang 残留跨格 overhead 排序**: q4_K(320) < q2_K(394) < q6_K(668)。**q2_K 在部署编译器下 overhead 已近 q4_K(WIN 者) parity**·q6_K = 2× q4_K。
3. **q6_K plane-recon = 6-bit 内禀·非 vsetvli/spill**: 6-bit(ql 4-bit + qh 2-bit-plane placement) 需 clang vsll=**512**+vor=**512** vs q4_K nibble vsll=16+vor=32 = **32×** plane-insertion 算术·**结构内禀于 6-bit**·削不掉。

**tile-materialization vs true-spill (精度教训遵守)**: 本地 clang-20 objdump 的 `# Folded Spill` 显示这些是**真 RA spill** (非纯 tile-mat)·但**本地 clang-20 ≠ 板 clang-18**·L2 文档记部署 q4_K spill=4·**本地未复现绝对 spill 数** (计数法/编译器版本差)。故本账 **只用跨格 relative 结构** (同代 emit + 同本地编译)·绝对 spill 不外推板。

## 2. 步骤2 — q4_K@rvv cross-parity 关键 (模板提取)
q4_K WIN **不是** 单纯 emitter-quality-beat·是**三因合取**:
1. **gcc-death 治愈** (clang 消 gcc 的 vsetvli/spill 爆炸·[CASE-COMPILER-ASYMMETRY]·必要非充分)。
2. **emit-form 精简** (跨格最低 vsetvli 税·nibble unpack 最简·plane-insertion trivial vsll=16)。
3. **★ weight-size sweet spot (决定性·新识)**: repack GEMM 的赢机制 = **weight memory-traffic 摊销** (block-dot 每 output-row 重读 weight block·repack 只 stream 一次)·收益 ∝ weight 大小。q4_K **144 B/sblk** 足够大·摊销收益 > 我方全展开核 overhead → 净赢。**这是 q4_K 独占的键·非通用 emitter 成熟度**。

## 3. 步骤3 — hand-construct G1 (最有 headroom 候选)·byte-exact + cross-parity 静态证据
**byte-exact**: 本线 **零算法改动** (只编译现存 shipped/emitted 核 + stock)·无新算术引入 → byte-exact 平凡保持。候选 emitter 杠杆 (rolled/register-fit/nr) 的 byte-exactness 已由 **G6-B 独立 oracle 证** (INT_mismatch=0 全 8 shapes + A/B memcmp 43008/43008 byte-identical)。

**候选逐格 cross-parity 静态判读**:
- **q2_K**: 部署编译器下 overhead 已近 q4_K parity (vsetvl 394 vs 320)·**且** total-vinsn/output 已 ≤ stock hand-brick 量级 → **e2e LOSS 非 emitter-instruction-count gap**。G6-B 已把 rolled/register-fit/nr 三杠杆板测穷尽·全不 cross (最好 0.63×)。**打回·结构 blocker (whole-K-nest roll·不同 S6 结构)**。
- **q6_K**: clang vsetvli 668 (可削向 320·但) per-output compute 由 6-bit qh-plane recon (vsll/vor 32× q4_K) 主导·**削 vsetvli 不改 compute-volume 主项** → 即便 vsetvli 达 q4_K 级·per-output 仍 > stock (compute-bound plane recon)。knest 已证 q6_K native-mask **更差** (7 ops vs stock shift-or 3 ops·负结果量化)。**打回·6-bit plane recon 内禀**。
- **q3_K**: weight 110 B (近 q2_K insufficient-amortization 区)·唯一杠杆 native-mask 仅适单-bit hmask plane·**且属 DECODE(GEVM) leaf** (q3_K native-mask 线在改·本线不碰) + decode 不传导 perf-covered ([CASE-MICRO-E2E])。prefill GEMM 无独立杠杆。**打回 (结构·无 prefill-distinct lever)**。
- **q5_K@rvv**: **唯一有真 headroom** = [GAP-Q5K-VLEN128-QH-REGCLIFF]·同 VLA 核 k1(VLEN256) WIN 1.641× / rvv(VLEN128) LOSS 0.87×·qh 5th-bit-plane register-cliff·且 stock = **generic** (弱对手·非 hand-brick)。**但 cell 已 green (via k1 any-board)** → 即便 rvv-half 修正 cross·**不增 perf-covered 计数** (仅 single→dual-board green 升级·消 caveat)·**非 count mover**。

## 4. 步骤4 — 破天花板裁决
**perf-covered 9/83 天花板 = 坐实 (CONFIRMED·prefill 已尽)**·四候选无一是 mover:

| 候选 | 现状 | headroom | 裁决 | 具名 blocker |
|---|---|---|---|---|
| q2_K@rvv | 黄 LOSS | 否 | **结构打回** | whole-K-nest roll (不同 S6 结构·G6-B 板测证伪 roll/lvalue/nr 全 <parity)·gap 非 emitter-instr-count (clang overhead 已近 q4_K parity 仍 LOSS) |
| q6_K@rvv | 黄 LOSS | 否 | **结构打回** | 6-bit qh-plane recon compute-volume 内禀 (vsll/vor 32× q4_K·native-mask 更差)·削 vsetvli 不触主项 |
| q3_K@rvv | 黄/未测 | 否 | **结构打回** | weight 110B 近 insufficient-amortization·唯一 lever(native-mask)属 decode leaf(禁碰)+不传导 |
| q5_K@rvv | **已绿(k1)** | 是(register-cliff) | **非 count mover** | 修 cliff 只升 single→dual-board green·不增分子·且 cell 已计绿 |

**共性天花板机制 (新识·合取)**: prefill green cross-parity 的键 **不是 emitter-instruction 成熟度**·而是 **repack-GEMM-vs-block-dot 的 weight-size 摊销 sweet spot** (q4_K@144B 独占) ∧ **unpack 简度** (nibble·plane-insertion trivial)。q2_K weight 太小 (84B·摊销不足)·q6_K unpack 太复杂 (6-bit plane recon)·q3_K 居中偏 q2_K·q5_K 已绿。**四格皆结构/算法级偏离 sweet spot·非可由 instruction-level emit 改动闭合。**

## 5. 战略结论
- **perf-covered 9/83 天花板经 prefill emitter-quality 轴 = 坐实。** decode 死于 [CASE-MICRO-E2E]·prefill 唯一 potential mover 三格 (q2/q3/q6_K@rvv) 全结构打回·q5_K 已绿非 mover。
- **下一步 potential mover 不在 emitter-instruction 轴**·需 用户裁 必问 radical 新 lever·候选方向 (仅列·不自决立项):
  1. **whole-K-nest rolled 单-wide-accumulator S6 重构** (G6-B 命名·= 不同 loop structure·mechanism 目标·非 knob)·若达 hand-brick compact∧resident 形态·或触 q2_K@rvv。**风险高·G6-B 判 non-warranted perf 战·仅 method 名义**。
  2. **IME 发射器路** (G6-A 已证 q8_0@ime 2.233× beat·q4_0@ime tie)·= 换 compute substrate·非 VLEN128 block-dot·**已是 perf-covered 现役 mover 轴** (非本线 prefill-emitter scope)。
  3. **q5_K@rvv register-cliff 修** = dual-board green 质量升级 (消 [GAP-Q5K-VLEN128-QH-REGCLIFF] caveat)·**不增计数**·低优先。
- **诚实注**: 天花板坐实 = 有效战略结论 (与找到 headroom 同等价值·任务明示)。本线新证 = 部署域 gcc-15.2 objdump (G6-B DEFERRED 的补完) + 跨格 relative 结构 + weight-size 摊销论点·**强化而非推翻** L2/G6-B 已证结论。

## 6. Raw artifacts (`raw/`)
- `{q4_K_WIN,q2_K_unrolled,q2_K_rolled,q6_K}.{gcc15.O3,clang20.O3}.s` — emitted prefill GEMM objdump (双编译器)
- `stock_{q2_K,q3_K,q4_K,q6_K}.gcc15.s` — stock hand-brick (inline asm·compiler-fixed·per-superblock-per-output baseline)
- `stock_q4_K.clean.c` — 从 llama.cpp `ggml-cpu/arch/riscv/quants.c` 提取的 q4_K_vl128 hand-brick harness
- **CAVEAT 复述**: 本地 clang-20 绝对 spill ≠ 板 clang-18 部署 (L2 记 spill=4·本地未复现)·只用跨格 relative·绝对 spill 不外推。gcc-15.2 不支持 `-mrvv-vector-bits` (compile-fail·非阻塞·relative 结构不受影响)。
