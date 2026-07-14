# MANIFEST — g7-perf-ceiling / kquant-prefill-quality-G1

- campaign: G7 perf-ceiling — K-quant **prefill emitter-quality 破天花板调查线** (research + G1·纯 read-only 分析 + hand-construct·**无 emitter 源码改动·无板**).
- question: q2_K/q3_K/q6_K@rvv 或 q5_K@rvv 的 prefill GEMM emitter-quality 缺口能否闭合到 cross parity = 新 perf-covered green? (= perf-covered 9/83 唯一 potential mover·decode 死于 [CASE-MICRO-E2E]).
- verdict: **天花板坐实 (CONFIRMED)**·四候选无一是 mover·具名 blocker 全结构/算法级·非 emitter-instruction 级.
- account: **本地 G1 静态账** (无板·无 e2e) — 部署域 gcc-15.2 objdump (补 G6-B DEFERRED 的部署编译器 seal) + clang-20 对称 + stock hand-brick baseline. RELATIVE 跨格结构·绝对 spill 不外推板.
- read-only: RVVToEmitCBlockQuantLinear.cpp 及 prefill GEMM leaf **只读·未改** · 不碰 q3_K native-mask decode 线在改的 GEVM leaf (不同 leaf).
- byte-exact: 零算法改动 (只编译现存核)·候选 emit 杠杆(rolled/register-fit/nr) byte-exactness 已由 G6-B 独立 oracle 证 (INT_mismatch=0).
- NO git · 本地纯静态 · 无 board 动作.

## provenance (读死)
- emitted prefill GEMM 核源 (64-output-tile·8 f32m2 acc·全展开·2240-2304 vwmacc/sblk):
  - q4_K = experiments/active/g5-wiring/M2-q4_K/tcrv_emitted_gemm_q4_K.inc (19521 行·= /tmp/sealproven_gemm_q4_K.inc 同 md5·deployed 者)
  - q6_K = experiments/active/g5-wiring/M2-q6_K/tcrv_emitted_gemm_q6_K.inc
  - q2_K = /home/kingdom/.claude/jobs/6b55ec14/tmp/tcrv_emitted_gemm_q2_K.{UNROLLED,ROLLED}.inc (G6-B 导出)
- stock hand-brick 源 = /home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c
  - ggml_vec_dot_{q2,q3,q4,q6}_K_q8_K_vl128 (inline asm·compiler-fixed) · q5_K **无 vl128/vl256 = generic fallback**
- toolchain: gcc-15.2.0 (/home/kingdom/spacemit-ime/.../riscv64-unknown-linux-gnu-gcc-15.2.0·= stock 出货编译器) + clang-20 (--sysroot spacemit) · -march=rv64gcv_zvfh (zvl128b/VLEN128)
- 承接 prior (不重测): L2 系统账 e2e (q4_K 1.344× WIN/q2_K 0.857× LOSS/q5_K 0.87× LOSS) + G6-B 板测证伪 (unrolled 最快·roll/lvalue/nr 全 <parity·命名 whole-K-nest-roll blocker)
- CAVEAT: 本地 clang-20 绝对 spill ≠ 板 clang-18 部署 (L2 记 deployed q4_K spill=4·本地未复现·计数法/版本差)·只用跨格 relative. gcc-15.2 不支持 -mrvv-vector-bits (非阻塞).

## files
- evidence.md — 主证 (6 节·步骤1-4 + 战略结论)
- raw/*.gcc15.O3.s raw/*.clang20.O3.s — emitted GEMM 双编译器 objdump
- raw/stock_*.gcc15.s — stock hand-brick objdump (per-superblock-per-output baseline)
- raw/stock_q4_K.clean.c — q4_K_vl128 hand-brick 提取 harness
