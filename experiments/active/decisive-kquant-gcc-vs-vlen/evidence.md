# [DECISIVE-KQUANT-GCC-VS-VLEN] — rvv q4_K e2e LOSS 归因隔离：gcc codegen vs VLEN128 重建摊销

> board `ssh rvv` openEuler VLEN128 64c · cores 8-15 perf-gov 2.6GHz (DVFS locked) · load ~2.2 (<10 gate)
> 裁二.2（gcc-death 冠名权前置）· A-tree `/home/ubuntu/tcrv-llamacpp` HEAD f3e1828 · **NO git · board reversible**
> **internal-A/B 机制探针记账（[NG-4]·不计 perf-covered）** · 对手 = stock generic `ggml_vec_dot_q4_K_q8_K` block-dot

## 决定性问题
[WORK-ITEM] 已证 rvv q4_K e2e repack-vs-vecdot **LOSS**（`g5-wiring/M2-q4_K` + `T-PERF1b`: DeepSeek-8B prefill
**0.334×**（q4kON.gcc / q4kOFF.gcc·both gcc-15·block-dot 分母）·decode 0.18×）。候选因素未隔离 =
{gcc codegen（742-spill）| VLEN128 重建摊销 | uarch}。**本实验隔离 gcc 因素**：同板·同格·同分母（gcc block-dot）·
**仅换 emitted repack kernel 的编译器**（gcc-15 → clang-18），重跑 e2e A/B。

## 隔离设计（surgical compiler-isolation·同分母）
三 `libggml-cpu.so` 变体，全部出自 gcc-15 build tree（block-dot/ops/其余 TU **恒 gcc-15** = T-PERF1b 账），
**唯一变量 = 2 个 repack TU（GEN `ggml-cpu/repack.cpp` + ARCH `arch/riscv/repack.cpp`，后者含 emitted vl=8 kernel）
的编译器**：
| 变体 | emitted q4_K repack kernel 编译器 | 角色 |
|---|---|---|
| `q4kOFF.gcc` (md5 05a62e6a) | — (gate OFF → block-dot) | **两账共同分母** |
| `q4kON.gcc` (md5 f20d61c5) | gcc-15.2.0 -O3 (742/217 spill) | gcc 账分子（复现 0.334×） |
| `q4kON.clangrepack` (md5 915e6733) | clang-18.1.8 -O3 (4/8 spill) | **clang 账分子（决定性）** |

- emitted kernel = M2 current-HEAD `tcrv_emitted_gemm_q4_K.inc` md5 **6cbd9c19** + `..._gevm_...inc` md5 **e909a9bd**
  （VLEN128-safe vl=8·intercept 破损上游 AVL=16）。deploy = M2 `deploy_patch_q4k_emitted.py`（case128 flip + intercept）。
- clang-18 = TCRV toolchain LLVM（env.sh·k1 clang-18 同族）；只重编 2 个 intrinsic-only repack TU（clang-18 装
  `__riscv_vcreate_v_*`）+ g++ -shared 重链（gcc 其余 .o 不变·C++ Itanium ABI 兼容）。march clang-compat
  `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause`（drop zfa/zvfhmin/zicond/zawrs experimental·不涉 kernel 语义）。

## [CASE-COMPILER-ASYMMETRY] spill 反汇编（EXACT M2 kernel·闭 provenance gap）
既有卷宗的 742-spill 是 sibling md5 90d454da；**本实验首次 objdump EXACT 部署 kernel（6cbd9c19/e909a9bd）**，
三编译器同板同 march（-c only）：

| kernel | gcc-15.2.0 -O3 | clang-17.0.6 -O3 | clang-18.1.8 -O3 |
|---|---|---|---|
| GEMM (prefill) | **742 spill / 820 vsetvli / 14903 insns** | 3 spill / 71 vsetvli / 6423 insns | **4 spill / 70 vsetvli / 6410 insns** |
| GEVM (decode) | **217 spill / 1387 vsetvli / 10824 insns** | 9 spill / 57 vsetvli / 2345 insns | **8 spill / 57 vsetvli / 2328 insns** |

⇒ gcc-15 对 EXACT 部署 kernel = ~185×(GEMM)/27×(GEVM) 整-向量-寄存器 spill 于 clang。部署 `.so` 内 seal 一致
（q4kON.gcc GEMM=742/GEVM=217 · q4kON.clangrepack GEMM=4/GEVM=8）。这是 codegen-asymmetry 根因铁证（同源·contention-immune）。

## correctness gate（clang-recompiled kernel · 载 timing-independent·contention 无关）
greedy A(q4kON.clangrepack repack)==B(q4kOFF.gcc block-dot) byte-identical + emitted-kernel engage banner + no NaN。
- prompt[1] "The capital of France is" → **A==B byte-identical**（coherent·无 garbage/NaN）
- prompt[2] "Q: What is 2 + 2? A:" → **A==B byte-identical**
- engage_banner=12(>0) · nan=0 · pass=2 fail=0 → **CORRECTNESS_GATE: GREEN**
⇒ clang-recompiled emitted kernel 数值正确（区别裸翻 gate PPL 822057 garbage）·perf 数有意义。

## e2e A/B 分相（DeepSeek-8B-Q4_K_M · taskset -c 8-15 -t 8 · pp128/tg16 · 交织·DVFS 2.6GHz 锁）
clang 账 = 决定性测量（load 2.2 clean · 3 internal-rep median · relIQR<0.2%）·同分母 q4kOFF.gcc block-dot：

| 账（q4_K repack kernel 编译器）| 相 | ours(repack) t/s | stock(block-dot) t/s | 比 ours/stock | 判读 |
|---|---|---|---|---|---|
| gcc-15 (742/217 spill)†  | prefill pp128 | 1.230 (T-PERF1b) | 5.186 / (3.68 Bq4kOFF) | **0.237× / 0.334×** | LOSS |
| gcc-15 (742/217 spill)†  | decode  tg   | 0.381 (M2 tg8) | 2.10 | **0.18×** (M2 0.1815×) | LOSS |
| **clang-18 (4/8 spill)** | prefill pp128 | **6.877** | **5.186** | **1.326×** | **WIN ≥parity** |
| **clang-18 (4/8 spill)** | decode  tg16  | **1.050** | **2.097** | **0.500×** | LOSS (residual) |

- clang 账 3-rep: repack prefill [6.881,6.879,6.870] · decode [1.049,1.051,1.049] · block-dot prefill
  [5.186,5.187,5.184] · decode [2.097,2.098,2.096] → cv<0.2% 全项（确定性 kernel-speed·非噪声·load 2.2 clean）。
- † gcc 账 = T-PERF1b(prefill 1.2299·同板/同 model/同协议·2026-07-10)+ M2(decode 0.381@tg8·0.1815×) 权威锚。
  **本会话 in-session gcc 锚重测被弃**：测中一并行 agent 的 formal perf-bench（`kernspan-r3-perf-formal`）占用
  core 8·load→10（越 board-load gate）→ 污染，如实弃用；决定性 clang 账在此之前 load 2.2 clean 已取。
  注：同 block-dot 分母(5.186)下 gcc repack = 0.237×（T-PERF1b Bstock 口径一致）→ **flip 0.237×→1.326×**。

## 出口判（预注册·裁二.2）
**MIXED — 主 = 出口 A（prefill·headline）· decode 残留 VLEN128-recon**：
- **Prefill（compute-bound·即 headline 0.334×/0.24× LOSS 所在相）**：gcc→clang **翻正 ≥parity（0.24×→1.326× WIN）**
  = **出口 A · gcc-death 坐实**。承重 headline LOSS 主因 = gcc-742-spill codegen 病理（clang 4-spill 消除即翻 WIN）。
  → **gcc-death 冠名权解锁（T）**·[CASE-COMPILER-ASYMMETRY] K-quant 侧兑现（742 spill 直证）。
- **Decode（memory-bound GEVM）**：gcc→clang 大幅改善（0.18×→0.500×·~2.8×）但**仍 LOSS**（block-dot 快 2×）
  = **残留 = VLEN128 权重重建摊销**（repack GEVM 每 token 重建全权重·VLEN128 半宽·带宽受限相未摊销）·非 gcc。
- 净：rvv K-quant e2e headline LOSS **主因 = gcc codegen**（prefill 翻 WIN 坐实）；decode 有真实但次要的 VLEN128-recon 残留。

## ★ 关键结构发现（premise 修正）
任务 premise "stock-repack vs stock-vecdot" 在 VLEN128 **不可行**：上游/stock q4_K repack kernel
(`ggml_gem[mv]_q4_K_16x1_q8_K`) load/acc **硬编码 AVL=16**（f32m2 VLMAX=8@VLEN128 → 钳半 → 输出垃圾 PPL 822057）
= VLEN256-only。故 0.42×/0.334× e2e LOSS 的实际 repack = **我方 emitted vl=8 kernel**（M2·VLEN128-correct·intercept
破损上游）。本实验隔离的即该**实部署 kernel**（gcc vs clang），忠于真 baseline（stock repack 在 VLEN128 从不存在）。

## board restore = TRUE
- source byte-exact: GEN=deb61a29 · ARCH=99131cf7 · CML=4426f543（全 baseline·测后校验通过）
- live libggml-cpu.so = 05a62e6a（pristine block-dot·pristine rebuild·tcrv_sym=0）
- 无 stray .inc in ARCHDIR · scratch build-clang17-dkgv removed · 0 dkgv 遗留 proc
- 唯一板改动 = source patch 期间（已 restore）+ /tmp/dkgv 临时 evidence scratch（/tmp·md5-recorded·非 shipped 改动）
- 并行邻居 `kernspan-r3-perf-formal`（他 agent·非本实验）占 core8 → 弃 in-session gcc 锚（用 T-PERF1b 权威锚）

## VERDICT（裁二.2·预注册双出口）
**MIXED · 主 = 出口 A（prefill headline 翻 WIN·gcc-death 坐实·冠名解锁 T）· decode 残留 VLEN128-recon**
- clang-重编 repack vs block-dot（同板·同格·同分母·仅换 repack kernel 编译器 gcc→clang）:
  **prefill 1.326× (WIN·翻正) · decode 0.500× (LOSS·残留)** · correctness GREEN · both 3-rep cv<0.2%
- prefill headline LOSS (gcc 0.237×/0.334×) → clang 1.326× = **gcc-742-spill codegen 主因坐实**（clang 4-spill 消 spill 即翻 WIN）
- decode gcc 0.18× → clang 0.50× (~2.8× 改善) 仍 LOSS = 残留 **VLEN128 权重重建摊销**（memory-bound GEVM·非 gcc）
- [CASE-COMPILER-ASYMMETRY] K-quant 侧兑现：EXACT 部署 kernel spill gcc **742/217** (GEMM/GEVM) vs clang **4/8**
- **gcc-death 冠名权解锁 = TRUE**（prefill/headline 翻正 ≥parity 满足出口 A）· 板 restored = TRUE
