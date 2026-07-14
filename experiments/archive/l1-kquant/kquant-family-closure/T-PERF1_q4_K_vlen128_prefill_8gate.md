# [PERF-1] 8-gate status cell — q4_K @ VLEN128 e2e prefill (repack GEMM)

- **campaign**: repack / [KQUANT-L1] · **status**: ACTIVE (逐门账；**远未** sealed 8-gate Win)
- **claim under test (bound)**: on board `ssh rvv` (openEuler, VLEN128), the TianChen-RV compiler's
  front-door-**constructed** q4_K repack GEMM (`emitRepackKQuantGemmBodyQ4K`, S6-tiled working-tree
  emitter, cell `l1-tile-s6-q4k-repack-gemm`) beats the board's factory-dispatched
  `ggml_vec_dot_q4_K_q8_K_vl128` **block-dot** (git `f3e1828`) on the **kernel axis** by
  **1.884×** (nr=64, single-core, cold N=10-12). ggml has **no repack path** for K-quants at VLEN128
  (dispatch trait = `nullptr`), so block-dot **is** the factory baseline (this is a legitimate
  Win-B candidate axis, not an algorithm-matched diagnostic). **e2e is a PROJECTION only** (§二.2
  Amdahl ≈ 1.59× prefill); no measured e2e Δ — integration blocked (see gate ④).
- **authoritative gate definition**: 科研目标总纲 v2 §4.4 [PERF-1]
  (`docs/canon/TianChen-RV_科研目标总纲v2.md:168-170`)：①字节精确 ②VLEN翻转lit{128,256} ③双板objdump
  ④micro∧e2e(llama-bench分相,多长度) ⑤双板都验证 ⑥实验纪律 ⑦机制合成归因 ⑧措辞门。
- **primary evidence cells**: `experiments/active/l1-tile-s6-q4k-repack-gemm/`（1.884× + byte-exact
  identity gate + objdump seal）· `l1-tile-s1-q4k-repack-gemm`（S1 1.473×）·
  `kquant-l1-q4k-q5k-repack-prefill`（dispatch probe = nullptr repack）.

## Verdict: 3 / 8 PASS · 2 partial · 3 missing → **NOT** a sealed Win; beat 措辞 LOCKED ([NG-4])

| # | gate (§4.4) | status | evidence pointer |
|---|---|---|---|
| ① | 字节精确 (byte-exact) | **PASS via bounded-ULP branch — NOT ULP0** | q4_K repack GEMM 用 vector FMA（`vfmacc.vv f32m2` 末端 fold + `vfnmsac` min）→ 浮点路径 = **FMA-fold bounded-ULP**。构造 oracle（039133ea，no-FMA 左结合 scalar oracle vs f64）8/8 WORST_NORM **~7–8e-7**。S6 tile 的 **IDENTITY cmp = 0 mismatch**（tiling 相对 golden 字节等价，int+norm 双 shape）+ `vwmacc` 2240 multiset 不变 → **timed S6 = golden**。**诚实**：correctness 由 bounded-ULP 满足，措辞**禁**说 "byte-exact vs ggml"。 |
| ② | VLEN 翻转 lit (128/256 双配置) | **MISSING (256 缺)** | VLEN128 lit 存在：`rvv-emit-quant-contraction-q4-K-repack-gemm-prefill-vlen128.mlir` + `rvv-emit-identity-quant-contraction-q4-K-repack-vlen128.mlir`。**无 q4_K VLEN256 codegen-flip lit**（对比 q4_0/q4_1/q8_0 有 `…-gemv-*-vlen256.mlir`）。LMUL 旋钮 mf2→m1 由 `coreLmul` 键控存在，但不是 {128,256}×{m1,m2,m4} 矩阵。 |
| ③ | 双板各一次 objdump 验封 | **PARTIAL (1/2 板)** | 板 A (rvv/VLEN128): **SEALED** — `l1-tile-s6-q4k-repack-gemm/objdump_tile_s6_seal.objdump`（spill 84→3, reload 88→8, maxVreg v31→v30, `vwmacc` 2240 UNCHANGED，clang-17 -O2/-O3）。板 B (k1/VLEN256): **无 objdump seal** — q4_K repack GEMM 未在 k1 上封。→ pending 新板批。 |
| ④ | micro AND e2e (llama-bench 分相, 多 prompt 长度) | **PARTIAL — micro✓ / e2e BLOCKED(projection)** | **micro PASS**: S6 1.884× vs 出厂 block-dot, cold N=10-12, cv~0.8%, min A/B>1 T-N pass, shape-robust (nr16/64)。**e2e MISSING**: measured "ours vs stock" **BLOCKED** — 构造 kernel 未接入 llama.cpp 的 Q4_K mul_mat（`emitRepackKQuantGemmBodyQ4K` 在编译器 emitter，working-tree 未 commit，未热插进 ggml dispatch）；且 ggml 无 K-quant repack env-toggle（不同于 G2 fusion ON/OFF），**没有可翻的开关**。本 session 给 **Amdahl 传导账 ≈1.59× prefill 上限**（`transmission_account.md`）+ **phase-split 基线**（`phase_split.json`）作档案估计，**非** measured Δ。 |
| ⑤ | 双板都验证 | **MISSING** | 板 A rvv/VLEN128 kernel-axis 已验；板 B k1/VLEN256 q4_K repack GEMM **本 campaign 未测**（k1 cell 皆 q4_0）。→ 需 k1 run。 |
| ⑥ | 实验纪律 (freq/pin/median+IQR/same-commit/model+format+ctx) | **PASS** | micro: core8, 2.6GHz, gov=perf, cold, N≥10, cv, 对手同 `libggml-cpu.so`（f3e1828）。e2e-baseline（本 session）: pin 8-15, gov=perf 2.6GHz, model=Q4_K_M sha `f8eba201`, quant/ctx 报全，warmup-dropped, N=10 median+IQR。 |
| ⑦ | 机制合成归因 (selector 日志证能力键选中获胜变体) | **MISSING** | q4_K repack GEMM 由 op-identity + weight-layout **上游**定选（registry 单算法；option-2 尚未做成真 pass）；tiled 变体是**单条 emitter 路径**，无 capability-keyed `reason∈{measured,prior}` 归因日志（reason≈`static_order`/`only_feasible`，SEL-1 先验层未落）。→ 需把 tiled 变体的选择做在归因主键上。 |
| ⑧ | 措辞门 (主张精确到相, [L-1]) | **PASS** | 每个数字绑 phase(prefill) × board(rvv/VLEN128) × format(q4_K) × baseline(ggml f3e1828 factory block-dot) × axis(kernel/micro; e2e=projection)。**约束**: 禁 generalize 到 decode（§二.2 传导效率≈0）; 禁 claim byte-exact vs ggml（见①）; e2e 数字**必须**标 "projection/Amdahl 上限" 直到 gate④ e2e 侧闭合。 |

## Missing / pending items → 下板批
1. **门② — q4_K VLEN256 flip lit**（lib 侧 codegen 跨 VLEN128/256，出本 cell touch-set）。
2. **门③⑤ — 板 B (k1/VLEN256) 上部署 + objdump seal + kernel-axis 复测**。
3. **门④ e2e 侧 — 集成兑现或 measured e2e A/B**：需 (a) 离线把 q4_K 权重 repack 成 `block_q4_Kx16` 布局，
   (b) ggml mul_mat 新 dispatch 入口 + q8_K 激活量化胶水。历史 e2e-seal 难点（q4_K emitter scaffold timed out 2×）。
4. **门⑦ — tiled 变体 capability-keyed selector 归因日志**（reason=measured/prior，SEL-1 落地后）。

**在 ②③④e2e⑤⑦ 闭合前，q4_K = 3/8 characterized kernel-axis path-win（板 A / VLEN128 / prefill 条件）；
beat/outperform 措辞 LOCKED ([NG-4])。唯一可引句 = 上面的 fully-bound claim。**
