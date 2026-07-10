# T-VALIDITY Stage-1 — rvv/VLEN128 对称重测 (3 承重-不对称格)

> 生成 2026-07-10 · 分支 `refactor/full-refactor-m1` · 板 `ssh rvv`（openEuler / VLEN128 / 64c / core 8 /
> gov=performance 2.6 GHz）· [CASE-COMPILER-ASYMMETRY] **Stage-1 rvv 重测段**。
> **口径冻结**：纯 remeasure 证据。**禁**改 T8/8-gate/memory canon（canon 重写 = Stage-2）。未 git add/commit/stash/rm。
> **方法学 = Stage-0b 复刻**（`tools/e2e-harness/board/case-compiler-asymmetry/stage0a-dualcompile/` +
> board `/tmp/case_0b0d/board_0b0d.sh`；q4_K 对称 gcc/gcc = 0.272×、clang-域 = 1.874× 之复刻）。

## 一句话 VERDICT
**3 格全部蒸发（EVAPORATED）。** 同一 kernel 源、部署编译器 gcc-15.2.0 双侧对称重测下，3 个承重"kernel-轴 vs-opponent WIN"
全部翻成 LOSS；historical 数是 clang-ours(-O2) vs gcc-shipped-block-dot 的**编译器不对称测量**，clang-域参照逐格
精确复现 historical（1.41 / 2.21 / 1.55），坐实为 clang-vs-gcc codegen artifact。与 q4_K（1.884→0.272）、
iq4_xs（1.4556→0.7178）同先例。

## 方法（两侧同工具链 = 出货成色）
- **我方 kernel**（cached front-door export，md5 冻结）编两次：**gcc-15.2.0**（部署编译器 → SYMMETRIC）与
  **clang-17.0.6**（→ CLANG-域参照，复现 historical）。`-O2 -march=rv64gcv_zvfh -mabi=lp64d -ffp-contract=on -x c++`。
  （Stage-0a：gcc march-invariant；clang 拒全部署 march；`rv64gcv_zvfh` = 两编译器公共最富非实验 march，两侧同。）
- **对手 block-dot** = gcc-15 上游 `libggml-cpu.so`（read-only 动态链接；**未修改 A-tree**）。q2_K = 真派发 tuned
  `ggml_vec_dot_q2_K_q8_K`；q5_K = generic `ggml_vec_dot_q5_K_q8_K`（无 vl128，UNTUNED）。
- Paired 同 binary 内 ours-vs-opp 背靠背（drift 共模抵消），N=14 cold-round（224 MiB flush）interleaved 全 6 binary。
- **对手在 gcc/clang 两 binary 间恒定**（q2K-opp 均值 4.525 n=28；q5K-opp 均值 1.324 n=56）⇒ 唯一变量 = 我方 kernel 编译器。

## 逐格结果

| # | cell (T8) | kernel md5 | **对称 gcc/gcc** | clang-域参照 | historical | objdump gcc / clang (vsetvli·spill) | 判读 |
|---|---|---|---:|---:|---:|---|---|
| F1 | `q2_K-tile-S6` (r69/32b) | `9c5bac28` (tiled S6) | **0.386×** (best 0.390) | 1.376× (best 1.407) | 1.413× | **1433 / 1446** vs 87 / 7 | **蒸发** (1.413→0.39, 3.6× 反转→LOSS) |
| F2 | `q5_K-tile-T3-S6` (r71/34b) | `c209226b` (tiled T3-S6) | **0.775×** (best 0.775) | 2.208× (best 2.221) | 2.193× | **2354 / 525** vs 71 / 105 | **蒸发** (2.193→0.775, 2.85× 反转→LOSS) |
| F3 | `q5_K-L1-candidate` (r61/25) | `ba30ba54` (untiled front-door) | **0.120×** (best 0.120) | 1.545× (best 1.554) | 1.50–1.62× | **6627 / 2066** vs 53 / 155 | **蒸发** (1.545→0.12, ~12.9× 反转→灾难 LOSS) |

- 全部 `ratio_med` 中位数（14 round）；ours/opp（>1 = ours 更快）。clang-域参照逐格命中 historical ⇒ harness 正确、historical 数确 = clang-vs-gcc。
- objdump vwmacc multiset 逐格 **gcc==clang**（F1 2304 / F2·F3 2240），且 **gcc-build 输出 == clang-build 输出 byte-exact**
  （显式 RVV intrinsics = op multiset 固定，编译器无 FP 重结合自由）⇒ gcc kernel **数值正确**，速度对拼公平（非"快因错/慢因错"）。
- clang seal 复现 historical tile seal（q2_K clang 87/7 == tile TILED-O2 87/7；q5_K-T3 clang spill 105 == tile TILED 105）。
- gcc 病理 = 宪章 rule 1「全展开 regfile spill」的 GCC-specific 实证：全展开直线码 + `csrr vlenb` 反复重算 VLA 帧偏移
  → vsetvli 风暴（q5L1 6627）+ 整寄存器 spill（q5L1 2066）；S6「stack panel」在 gcc 下退化成整寄存器访存流。clang 从未看见。

## q5_K-L1 性质判定（task 明问：routing 共模幸存 vs kernel-质量对拼蒸发）
**判定 = kernel-质量对拼（蒸发），NOT byte-identical routing。** 依据：
1. 我方 kernel（front-door 发射 repack GEMM，md5 `ba30ba54`）与对手（ggml generic `ggml_vec_dot_q5_K_q8_K` block-dot）
   是**两个不同算法/二进制**，**非** byte-identical——不同于 q4_0 5.9×（两树同 git f3e1828 **同一 repack kernel**，赢=纯 routing 闸翻、零 kernel diff）。
2. `ba30ba54` == r71 UNTILED（`untiled_q5K_gemm.c`）逐字节同一 ⇒ q5_K-L1 就是 untiled front-door repack GEMM（r61 1.5× ≡ r71-untiled 1.547×）。
3. 对称 gcc 下我方 kernel = 6627 vsetvli / 2066 spill = 0.158 GMAC/s，比 UNTUNED generic block-dot（1.32 GMAC/s）**还慢 8×** → 0.12×。
   ⇒「structural path-win candidate」（对手 VLEN128 无 repack path）是**真结构事实**，但 **1.5× 吞吐数 100% 是 clang-vs-gcc codegen**，
   非 memory-locality/routing 赢。三格中蒸发最猛。

## 自检 SOP（测量闸门）
- **测前 3× 重测-重测**（bin_q2K_gcc, N=12）：ours-side best_gmacs 1.7669/1.7700/1.7729 → spread **0.34% < 1.2% gate** → 开测 PASS。
- **测中 paired-ratio IQR%**（14 round）：F1-gcc 1.55 / F2-gcc 1.03 / F3-gcc 1.67 / clang 侧 0.39–1.53 —— 与 3-样自检 ratio spread(~2.8%)
  同量级，**无 ≥3× 劣化** ⇒ 会话 VALID。ours-side 逐格 spread 0.43–1.64%（loadavg 漂移 2.4→3.4 driven；paired ratio 共模抵消）。
- **效应量 2.85–12.9× 远压噪声**（sub-2%）⇒ 蒸发判读对噪声完全稳健。
- loadavg（start 2.42 / end 3.33，1-min）= 指纹记录项**非闸门**；无 llama-bench/重邻居在跑。

## A-tree 复原证明
| | md5 (`libggml-cpu.so.0.15.1`) | nm ours-symbols | nm opp q2/q5 |
|---|---|---|---|
| BEFORE | `d1adc634c2ca04ffc389536b30d9d9c4` | 0 | 6 |
| AFTER  | `d1adc634c2ca04ffc389536b30d9d9c4` | 0 | 6 |
**UNCHANGED**（micro A/B 仅 read-only 动态链接 .so 作对手，从不 swap/写入）⇒ 复原自动满足（active .so = stock, nm 0/0）。

## 一手证据指针
- 板 harness：`tools/e2e-harness/board/case-compiler-asymmetry/stage1-rvv-remeasure/{board_stage1_compile_seal,board_stage1_selfcheck,board_stage1_measure}.sh`。
- 板原始 log：`/tmp/case_stage1_rvv/{phaseAB.log,phaseD.log}`（本地耐久副本 `/tmp/stage1_evidence/`）。
- kernel 源（cached export，md5 冻结）：board `/tmp/kquant_tile_q2k_t3_ab/tiled_q2K_gemm.c` (9c5bac28) ·
  `/tmp/kquant_tile_q5k_t3_ab/tiled_q5K_gemm.c` (c209226b) · `/tmp/case_0b0d/gemm_q5_K_q8_K.kernel.c` = `untiled_q5K_gemm.c` (ba30ba54)。
- 先例：Stage-0b q4_K 1.884→0.272（`/tmp/case_0b0d/0b0d.log`）；iq4_xs batch2b>1 蒸发（T8 r13）。
- historical 对照：`experiments/active/l1-t3-q2k-repack-gemm/`（r69 1.413×）· `l1-t3-q5k-repack-gemm/`（r71 2.193×）·
  `kquant-l1-q4k-q5k-repack-prefill/`（r61 1.50-1.62×）。

*纯 remeasure 底账（Stage-1 产物），未 commit。所有「撤回/收窄/clang-域保留」终审 pending Stage-0/Stage-2。不改既有 verdict。*
