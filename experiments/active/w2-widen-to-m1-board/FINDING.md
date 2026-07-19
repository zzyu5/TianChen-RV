# W2 §③ — widen-to-m1 board re-measurement under the register-pressure inequality

**Purpose.** [GAP-P1] is not a permanent physical fact; it is the "the old emitter
would spill" measured boundary. This experiment (1) uses the register-pressure
inequality (`rvvRegisterPressureLegal`, commit 0e9faee0a) to prove the m1 chain is
**spill-free**, and (2) board-measures the m1 vs mf2 crossover on a spill-free
format. **Agent gives data only; the boundary ruling (keep [GAP-P1] hard vs loosen)
is the supervisor's.**

Not a selector change: these are STANDALONE kernels that include/link NO plugin
code and flip NO selector. The DEPLOYED default stays mf2 (`selectRepackAccumulatorLMUL`
unchanged).

## Register-pressure legality (STEP ②, closed-form, no board)

`peakCost = Σ footprint(LMUL)·unroll·liveVars ≤ vregBudget − fixedOccupancy`,
budget = 32 (`getRVVArchitecturalVectorRegisterCount`), footprints from
`getRVVLMULRegisterFootprint` (m1=1, m2=2, m4=4).

| chain | strip | product | accumulator | peak-live (unroll=1) | ≤ 32? |
|---|---|---|---|---|---|
| **mf2** | e8mf2 | i16m1 (1) | i32m2 (2) | 1+2 = **3** | spill-free |
| **m1**  | e8m1  | i16m2 (2) | i32m4 (4) | 2+4 = **6** | spill-free |

⟹ Both chains are budget-legal at unroll=1 for EVERY repack format (the "regfile
spill" half of [GAP-P1] does NOT apply to unroll=1 single-block bodies). m1
first spills at unroll ≥ 6 (6·6=36 > 32); mf2 at unroll ≥ 11.

## Board (pin)

- board `ssh rvv`: openEuler riscv64, **VLEN=128** (e8m1 VLMAX=16), 64 cores,
  ISA `rv64imafdcv_..._zve64d_zvfh_...`.
- compiler: system **clang 17.0.6** (both chains, same compiler ⇒ compiler-symmetric
  ratio). Pinned llvm-18.1.8 compiles objects but lacks a gcc sysroot to link.
- weft-rv commit **0e9faee0a**.
- sources: `widen_kernels.c` md5 `5d1b330f1ff67a7f24130f2d96831b45`,
  `harness.c` md5 `2ded15b0b917cfc9a49347d1bf55158f`.
- objdump confirms the two chains emit DISTINCT vtypes: `e8,mf2` and `e8,m1`
  (plus the i16m1/i16m2 and i32m2/i32m4 widening steps).

## Data — cold GEVM `out[M]=W[MxK]·act[K]`, 2 seeds, byte-exact

All rows **byte-exact PASS** vs the scalar oracle (ZERO-MODEL, both chains, both
seeds, both formats). Ratio = m1 / mf2 on median_ns (<1 ⇒ m1 faster).

### q8 format (pure int8 widening dot — clean chain measurement)

| footprint | seed | mf2 median_ns | m1 median_ns | m1/mf2 |
|---|---|---|---|---|
| M=256 K=1024 (cache-resident) | 0 | 83,020 | 57,360 | **0.69** |
| M=256 K=1024 | 1 | 82,941 | 56,740 | **0.68** |
| M=2048 K=4096 (~8 MB) | 0 | 2,489,551 | 1,761,488 | **0.71** |
| M=2048 K=4096 | 1 | 2,493,511 | 1,765,708 | **0.71** |
| M=8192 K=4096 (~32 MB, DRAM-bound) | 0 | 10,045,246 | 6,926,532 | **0.69** |
| M=8192 K=4096 | 1 | 10,024,006 | 6,963,732 | **0.69** |

**q8: m1 is ~1.41–1.46× FASTER than mf2, robust across cache-resident AND 32 MB
DRAM-bound footprints, both seeds.** The advantage does NOT wash out with footprint
⇒ within the isolated GEVM the bottleneck is instruction issue (m1 processes
16 lanes/iter vs mf2's 8 ⇒ half the vsetvl/loop overhead), not single-thread DRAM
bandwidth (32 MB / 6.9 ms ≈ 4.6 GB/s ≪ saturation).

### q4 format (nibble unpack + dot — CONFOUNDED, not a clean chain measurement)

| footprint | seed | mf2 median_ns | m1 median_ns | m1/mf2 |
|---|---|---|---|---|
| M=256 K=1024 | 0 | 794,664 | 756,904 | 0.95 |
| M=2048 K=4096 | 0 | 25,800,538 | 25,923,678 | 1.005 |
| M=8192 K=4096 | 0 | 102,556,526 | 97,448,702 | 0.95 |

**q4 ≈ parity (m1 0.95–1.005×).** CAVEAT: this kernel's SCALAR nibble-unpack (offset-8)
dominates (~26 ms vs ~2.5 ms for the dot at M=2048), diluting the chain difference.
The real repack path vectorizes the unpack, so this q4 number under-represents the
chain effect — treat it as a floor ("m1 never loses"), not a clean crossover.

## Verdict handed to supervisor (agent gives data, does NOT rule)

- **Spill-free**: confirmed by the inequality (m1 chain = 6 vregs at unroll=1 ≤ 32).
  The "regfile spill" half of [GAP-P1] is refuted for unroll=1 single-block bodies.
- **有格式赢 (a format where m1 wins)**: **q8-like** (pure int8 widening dot) —
  m1 wins ~1.41–1.46×, byte-exact, 2-seed, footprint-robust.
- **Open (NOT settled by this micro)**: [GAP-P1]'s "micro win washes at e2e" concern
  is about the FULL decode (KV-cache/activation traffic, Amdahl dilution). This
  ISOLATED-kernel micro cannot settle e2e transfer; only an e2e harness can. The
  micro win is robust *within the isolated GEVM* (even at 32 MB), which is
  necessary-not-sufficient for an e2e win.

Per PRD: "还输 = [GAP-P1] 真墙钉死; 有格式赢 = 边界该松 (结论 supervisor 裁)". Data:
q8 = 有格式赢; the e2e transfer question remains the boundary's live falsifier.
