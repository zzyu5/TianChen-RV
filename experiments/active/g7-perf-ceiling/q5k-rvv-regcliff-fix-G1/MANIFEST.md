# MANIFEST — q5k-rvv-regcliff-fix-G1

- line: **G7-PERF-CEILING / [GAP-Q5K-VLEN128-QH-REGCLIFF] 可修性评估**（research + design + G1·纯本地静态账·无 emitter 源码改动·无板·无 e2e）
- branch: refactor/full-refactor-m1 · HEAD 6b8b7d39 · **NO git · tracked 源码只读未改**
- account: 本地 G1 静态账 — hand-construct standalone .c（`raw/`）· gcc-15.2.0（= stock 出货编译器）+ clang-20（= 我方 clang 家族·[L-10] 出货 clang-18）· `-march=rv64gcv_zvfh`（zvl128b / VLEN128）+ zvl256b 对照
- read-only 源: `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:9314-10059`（emitRepackKQuantGemmBodyQ5K·prefill GEMM leaf）

## 裁决 TL;DR
- **NOT FIXABLE by register-pressure reduction — 结构坐实（VLEN128-intrinsic）**。
- "register-cliff" **是误名**：VECTOR 寄存器 spill 在 VLEN128 与 VLEN256 **完全相同（q5_K=6 / q4_K=4）**，且 **OLD/KNEST/retrans 三 byte-exact qh-recon 减压均不降 spill（恒 6）** → register-pressure 减压 = 错的杠杆。
- 真机制 = **numHalves=2 tile 翻倍**（同 16-col 输出 VLEN128 跑 2 strip ~804 v-insn / VLEN256 1 strip 402）**+ qh 5th-bit 不可约指令量**（+43/strip·地板 +25），摊到 8-lane。VLEN128-intrinsic。
- L2 "155 spill" 实测拆 = 6 vector : 42 scalar(地址) : 37 tile-mat / strip → **非 vector 寄存器压力**（precision-lesson 校正）。
- **不增 perf-covered 计数**（cell 已绿 via k1·9/83 不变）· 兑现 kquant-prefill-quality-G1 的 "唯一 headroom" 评估 = **headroom 不真实·caveat 应撤回/精化**。诚实负判读 = 有效正结果。

## 硬门
- byte-exact `raw/byte_exact_qhrecon.c` — OLD≡KNEST≡RETRANS≡ref **0 / 32768** PASS（减压不改算法）
- harness 保真 — 本地 clang-20 q4k **VECTOR spill=4 = 板 clang-18 q4_K=4** 吻合

## files
- `evidence.md` — 完整根因刻画 + 减法设计 + G1 静态账 + 可行性裁决 + 诚实定性
- `raw/strip_body_regpressure.c` — 忠实单 strip 热 nest hand-construct（QHMODE 0/1/2/3）
- `raw/byte_exact_qhrecon.c` (+ 编好的 `byte_exact_qhrecon`) — 硬门穷尽
- `raw/clang20_{q4k,q5k_old,q5k_knest,q5k_retrans}.s` · `raw/gcc15_*.s` — VLEN128 双编译器 objdump
- `raw/clang20_vlen256_{q4k,q5k_old,q5k_knest}.s` — VLEN256 单-strip 对照（spill 同=6 决定性证）

## CAVEAT
- 本地 clang-20 TOTAL-Folded-Spill 绝对数 ≠ 板 clang-18（计数法/版本差）·但 **VECTOR spill 本地 q4_K=4 = 板 q4_K=4** → vector 层 relative + 绝对可信
- 无 e2e / 无板：本线止于 G1 静态账·prefill perf 主张不落地（结论 = 不可修·无需板证）
- gcc-15.2 不支持 `-mrvv-vector-bits`（非阻塞·relative 结构不受影响）
