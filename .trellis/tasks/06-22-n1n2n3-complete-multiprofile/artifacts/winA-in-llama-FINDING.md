# Win-A IN llama's hot path — VLEN-aware repack strip, validated cross-profile (2026-06-22)

The reframed PRIZE delivered: the compiler-automatic, resource-aware SELECTION now lives in
llama's q4_0 decode hot path (the repack GEMV), and it measurably wins on real VLEN=256 silicon.

## The mechanism (committed 87c540ec)
New pass `tcrv-rvv-materialize-repack-strip-width=march=...` stamps `half_lanes = min(VLEN/16,
weight_interleave)` on the repack ops → the emitter selects the strip structure:
- `march=rv64gcv` (VLEN=128) → half_lanes=8 → **two 8-lane halves** (unchanged from before)
- `march=rv64gcv_zvl256b` (VLEN=256) → half_lanes=16 → **one 16-lane strip**
Divergence by construction: the two emitted .cpp differ only in strip count (1×16 has ONE
`vse32_f32m2(...,16)` + ONE accumulator vs 2×8's two). This is the compiler choosing the
resource-optimal layout from a capability (VLEN) fact — not a hand-tuned kernel.

## Validation on real K1 silicon (Spacemit X60, VLEN=256, clang18) — all four claims backed
| claim | result |
|---|---|
| structural | 1×16 = exactly one strip (one store/accumulator @ avl=16) |
| correctness | BOTH variants PASS vs ggml `_generic` @ VLEN=256: 1×16 norm 1.0e-5–1.6e-5, 2×8 1.3e-5–2.1e-5 (NC=16/32/64/336). 1×16's avl=16 only fills at VLEN≥256 → its PASS directly proves the 16-lane strip reads byte-identical data (the 16-way-interleave invariant). |
| **perf** | **1×16 is ~1.48× faster than 2×8 on K1** (n=4096, nc=336 decode shape; 404,930 vs 600,576 ns/call; reps=3000, best-of-11, 3 rounds 1.46–1.52×, identical sink). **Conservative lower bound** (1×16 built without `zvl256b` so the compiler couldn't assume VLMAX=16). |

## What this establishes (and what it is NOT)
- **N3 / N1**: a compiler-automatic, resource-aware (VLEN-capability-driven) SELECTION in llama's
  real q4_0 decode hot path, that measurably wins (1.48×) on the VLEN=256 profile vs the
  VLEN=128-default layout. The selection is the only thing that changed; it's emitted, not
  hand-written. This is the "Win-A in llama" the campaign targeted (distinct from the earlier
  cross-VLEN tune-ratio leg — this is a real SELECTION divergence with a measured win).
- **Honest scope**: this is the *decode GEMV repack* hot path. The 1.48× is the strip-width win
  on K1 (VLEN=256); on rvv (VLEN=128) half_lanes=8 is already optimal (the strip can't widen), so
  there is no single-chip strip ablation on rvv — the win is inherently CROSS-PROFILE (128↔256),
  exactly as the advisor framed. Not to be conflated with the repack-vs-naive Win-B kernel-swap
  (5–6×) or the contraction max-LMUL Win-A microbench (2–4× vs naive).
- **Portability note**: the earlier `v73 not declared` error was a FALSE alarm — a missing
  `-march=…zvfh` build flag (the fp16 scale-fold needs zvfh; K1 has it), NOT an emitter scoping
  bug. With `rv64gcv_zvfh` the emitted C compiles clean. No emitter fix needed.

## Artifacts
`k1-vlen256/strip-1x16-vs-2x8-k1.log`, `gemv_1x16.cpp`, `gemv_2x8.cpp`, `bench_gemv.cpp`.
