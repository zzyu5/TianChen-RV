# q8_0 block-dot VLEN→LMUL flip — BOARD SEAL (commit a874a56b)

Win-A brick #1: the gearbox selects the q8_0 block-dot integer-core LMUL anchor
from the REAL VLEN capability fact. VLEN128(rvv)→**m2-elided**; VLEN256(k1)→
**m1-elided** (the new emit `vsetvl_e8m1(32)`, fold-back-risky). This is the
load-bearing two-board seal: byte-exact on each matching silicon + micro.

## Method (honest)
- Shipping artifact emitted via the FULL gearbox pass (NOT the driver's explicit-
  SHAPE path, which omits `minimum_vlen` and is rejected by the verifier):
  `tcrv-opt INPUT --tcrv-rvv-materialize-q8-0-schedule=march=<M> --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`.
  Gearbox stamps confirmed on host: zvl256b→ m1/factor2/elided/minimum_vlen=256;
  rv64gcv→ m2/factor2/elided/minimum_vlen=128.
- **Reference = ggml's REAL hand-written RVV q8_0 kernel** (separate TU, vl=32,
  i8m2→i16m4→vwredsum), contracted under `-ffp-contract=fast` exactly as it ships
  — the established gate (stronger than a scalar oracle; for q8_0 the integer
  reduction is exact, so a broken elided cover fails either reference). Same
  reference on both boards.
- GATE: byte-exact over 12 n-sizes {32,64,96,128,160,256,288,512,1024,2048,4096,
  8192} × 200 trials = 2400 checks. Odd nb (e.g. 96→nb3, 160→nb5, 288→nb9)
  exercises the ROBUST scalar tail; even nb the ELIDED main loop. clang-18 -O3.
- MICRO: best-of-200-reps min, n=4096, taskset-pinned (k1 -c 0-3 / rvv -c 3).

## k1 — VLEN256 (LOAD-BEARING) — PASS
Live VLEN confirmed this session: `VLEN_bytes=32 VLEN_bits=256` (Spacemit X60).
- Shipping shape = **m1-elided**. Instruction-level proof (llvm-objdump-18 of the
  k1-compiled .o): elided cores emit `vsetvli zero, s7, e8, m1, ta, ma` (offsets
  8a/dc/13a/174). NO `e8, m2` anywhere → **NO fold-back regression** on real
  VLEN256 silicon. The single-vsetvl(32) elided cover is correct ONLY at VLMAX≥32,
  which VLEN256 provides; the byte-exact gate proves it executes correctly.
- **GATE: 2400/2400 BYTE-EXACT vs ggml-real (0 mismatches).**
- MICRO (k1 VLEN256): m1-elided **6499.9 ns** vs m2-elided **7621.1 ns** →
  m1 (selected) beats m2 by **+17.2%**. vs ggml-real 8047.6 ns → m1-elided = **1.24x**.

## rvv — VLEN128 (no-regression) — PASS
- Shipping shape = **m2-elided**. Instruction-level proof (llvm-objdump-18, rvv-
  compiled .o): cores emit `vsetvli ..., e8, m2` (offsets 8a/dc/13a/174). NO m1.
- **GATE: 2400/2400 BYTE-EXACT vs ggml-real (0 mismatches).**
- MICRO (rvv VLEN128) — DRIFT-NORMALIZED: the two rvv invocations ran under
  different board state (same-kernel ggml-real baseline drifted +24%: 5937.2 ns
  in the m2 run vs 7349.5 ns in the m1-robust run). The harness times ours AND
  ggml interleaved best-of-200 WITHIN one process, so normalize each kernel to
  its OWN in-run ggml: m2-elided 5767.6/5937.2 = **0.971**; m1-robust 7705.7/
  7349.5 = **1.048** → m2 (selected) beats m1-robust by **+7.9%** (1.048/0.971).
  This matches 06-22's +7.0% and the ~7% prediction. (Raw cross-run 5767.6 vs
  7705.7 = +33.6% is mostly board drift — do NOT use it.) m1-robust is the LEGAL
  m1 family at VLEN128; m1-ELIDED is illegal here (counterfactual below), so this
  micro conflates family (m1↔m2) with elision (robust↔elided), inherent at VLEN128.

## Counterfactual: the legality prune is load-bearing (not arbitrary)
Forcing m1-**elided** at VLEN128 on rvv → **DRIFT (11/12 mismatches)**: e.g.
ours=-306656.969 vs ref=-495483.469. Because `vsetvl_e8m1(32)` returns 16 at
VLEN128, the elided cover sums only 16 of 32 quants. This is EXACTLY the bug the
gearbox's `VLMAX ≥ blockLen` legality prune prevents — it does NOT select
m1-elided at VLEN128 (it selects m2-elided). The VLEN→LMUL flip is a correctness
boundary, not a heuristic.

## Verdict
| board | VLEN | selected shape | binary vsetvli | byte-exact (2400) | micro: selected vs other-family | vs ggml |
|---|---|---|---|---|---|---|
| k1  | 256 | **m1-elided** | `e8,m1` | **PASS 0 mism** | m1-elided 6499.9 vs m2-elided 7621.1 (**+17.2%**, clean: ggml 8047.6/8047.1) | 1.24x |
| rvv | 128 | **m2-elided** | `e8,m2` | **PASS 0 mism** | m2 vs m1-robust **+7.9%** (drift-normalized; raw +33.6% is board drift) | 1.03x |

**Both boards PASS. No fold-back bug. The flip DIRECTION + byte-exactness are
sealed on each matching silicon; the micro magnitude (k1 +17.2% clean, rvv +7.9%
drift-normalized) confirms the selected family is the faster one on its VLEN.**
The k1 +17.2% exceeds 06-22's +6.88% because this compares m1-ELIDED (6499.9) vs
m2-elided, whereas 06-22 used m1-ROBUST (7138.4); elision buys m1 ~9% at VLEN256.
The k1 m1-elided artifact (the single shape NOT validated by
the 06-22 paired campaign, which enumerated at capability-march rv64gcv where
m1-elided is pruned) is now sealed on real VLEN256 silicon.

Note vs 06-22: that campaign measured m1/2/**robust** as the k1 winner (it never
had m1-elided in its ladder). This seal shows m1-**elided** (the actual shipping
shape) is byte-exact and 1.24x vs ggml on k1 — the elided strip runs the single
vsetvl(32) cover that robust would loop once; elided is the selected + correct +
fast shape on VLEN256.

Artifacts: `q8-vlen-flip-board/` (harness.cpp, ggml_ref.cpp, q8_v256.cpp
[m1-elided], q8_v128.cpp [m2-elided], q8_m1robust.cpp). Remote temp dirs
(/data/tcrv_q8flip_k1, /tmp/tcrv_q8flip_rvv) cleaned. No lib/ code changed; no
git commit.
