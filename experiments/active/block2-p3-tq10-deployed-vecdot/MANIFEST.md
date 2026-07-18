# block2-p3-tq10-deployed-vecdot — DEPLOYED-EMIT tq1_0 vec_dot direct M=1 board measure

**Task**: `.trellis/tasks/07-19-block2-tq10-deployed-measure` — clarify the tq1_0 vec_dot
territory (P2 iq2 paradigm). Export the PRE-EXISTING deployed emit
(`emitTypedSuperBlockScalarDeltaGridLoopBodyTQ10`, the retired-monolith byte-exact code-move
carrying the tq1_0 BASE-3 ternary integer core) and board-test it DIRECTLY as M=1 vec_dot vs
the deployed hand-tuned `ggml_vec_dot_tq1_0_q8_K` (compiler-symmetric clang-18). READ-ONLY:
no emitter change, no commit. The sealed core `RVVToEmitCTernaryBinary.cpp` md5 **338a31bb
untouched**; the export-e2e fixture asserts CORE==PROD diff=0.

## Files
- `kernels/tq1_0_deployed_emit.cpp` — the DEPLOYED EMIT, exported byte-identical to the sealed
  CORE emit (md5 `00142974e1f26ca67abdcb52b5ef54f8`, sym
  `weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot`). See `GEN_SEAL.txt` for
  the recipe + CORE==PROD diff=0 reproduction.
- `tq10_vecdot_driver.c` — verbatim copy of the P1 driver (same symbol, same INDEPENDENT
  ZERO-MODEL scalar oracle, 3-arm anti-hollow, INT/FLOAT gate, cold timing). The driver is
  agnostic to how the symbol is implemented, so it links the DEPLOYED-EMIT leaf unchanged.
- `remote_build_run.sh` / `board_run.sh` — P1 harness, one change: the deployed emit is
  `extern "C"` C++ so the leaf is compiled `-x c++` and the link pulls `-lstdc++`. Everything
  else (opponent, gate, cold) identical to P1 → apples-to-apples vs the P1 standalone.
- `raw/{rvv,k1}_tq1_0_deployed_emit_seal.txt` — board seal + run logs.

## Result (direct M=1 vec_dot, clang-18 compiler-symmetric, K=4096 M=1, oversized-pool 256MiB cold)
Byte-exact GREEN both boards: INT ours_vs_oracle_mism=0, worst_ulp(ours,oracle)≤1,
worst_ulp(ours,factory)≤2, anti-hollow RED-all 3/3, corpus 8192×2×2-seeds.

**THREE-NUMBER COMPARISON** (ratio = opp_ns / ours_ns; ≥0.8 = PASS地盘, <0.8 = 具名-X边界):

| board | opp (手调 _vlNNN) | master proxy (T3 matmul) | **deployed-emit DIRECT M=1 (this cell)** | P1 standalone universal |
|---|---|---|---|---|
| rvv (VLEN128) | ~180 ns | 0.207 | **0.22** (5-seed 0.217–0.228, iqr ~3%, ours ~808 ns) | 0.90 (0.90–0.93, ours ~198 ns) |
| k1  (VLEN256) | ~241 ns | 0.606 | **0.549** (5-seed ~0.549, iqr <0.2%, ours ~440 ns) | 0.82 (0.823, ours ~293 ns) |

## Verdict — **需部署 P1 leaf** (NOT deployed地盘; opposite of P2 iq2)
The deployed-emit DIRECT M=1 measurement **CONFIRMS** the master proxy (rvv 0.22 ≈ 0.207,
k1 0.549 ≈ 0.606 — same ballpark, both < 0.8). The proxy was **NOT** an artifact: the deployed
tq1_0 vec_dot emit really is ~0.2× (rvv) / ~0.55× (k1) of the hand-tuned opponent. This is the
OPPOSITE of P2's iq2 (where proxy 0.697 understated a near-parity deployed emit 0.84 = 地盘).

**Mechanism of the deployed-emit slowness** (self-probe, apples-to-apples same driver/opp/board):
the byte-exact retired-monolith code-move DECODES trits into an `aux8[256]` scratch (vse8) then
RELOADS them (vle8_v_i8m2) for the dot, with **8× serial per-super-block vwredsum** and an
e8m1/e8m2 LMUL mix — vsetvli=57, vmul=14, vwredsum=8, insn~269. The P1 standalone universal
FUSES into a single i16 accumulator + **single vwredsum**, no scratch round-trip — vsetvli=49,
vmul=2, vwredsum=1, insn~181, ~4× faster on rvv (~1.5× on k1). Same byte-exact result, same
opponent. ⟹ the P1 leaf is a GENUINE improvement, not a redundant exploration.

## Opponent caliber (成色)
Opp = ggml hand-written intrinsic `_vl128`/`_vl256` (tier=手调, NOT the cheap `_generic`).
M=1 kernel-axis, NOT e2e ([NG-4] — do not extrapolate). Because the deployed emit LOSES to the
hand-tuned opponent, the honest verdict for the DEPLOYED territory stays 具名-X on both boards
(now with a direct-M=1 provenance rather than the T3 matmul proxy). The flip to PASS is
contingent on DEPLOYING the P1 fused leaf into the emitter (see deployment scope in the task
report) — an emitter-body change, out of scope for this read-only cell.

## 出处
task `07-19-block2-tq10-deployed-measure` (2026-07-19, P2 iq2 paradigm; read-only export +
board measure, no emitter change, no commit).
