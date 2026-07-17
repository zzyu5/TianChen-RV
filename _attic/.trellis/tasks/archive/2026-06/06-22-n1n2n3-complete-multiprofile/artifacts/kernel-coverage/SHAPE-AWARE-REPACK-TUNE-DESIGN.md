# Shape-aware repack tune — design + honest verdict (2026-06-24); CORRECTS the §8/§8b "shape-mismatch" framing

> **⚠ SUPERSEDED in part by `path-selection-tune-DESIGN.md` (2026-06-24): the "Option B = in-compiler
> route-materialization gate in `RVVVectorSourceFrontDoor.cpp`, enumerated candidates via
> `RVVScheduleDescriptorRegistry`" plan below is NOT VIABLE.** Inspection proved: (1) the repack-vs-block-dot path
> is committed by **op identity + weight layout in the INPUT IR, upstream of the compiler** — the FrontDoor only
> materializes elementwise RVV-vs-scalar ops, never sees contraction ops, can't synthesize the layout-incompatible
> sibling; (2) the registry is one-key→one-shape (single-algorithm Win-A axis), **cannot enumerate two algorithms**.
> The real gate is the **producer / weight-packing build harness**; in-compiler the selector is **AUDIT-ONLY**.
> Declining = matching ggml (`case 128: break`) = loss-avoidance **hygiene, NOT a novelty**; the real e2e N3 win is
> the EXISTING q4_0@128 repack 2.6×. Whether to build it (option-1 characterization vs option-2 compiler-ownership)
> is an escalated **user fork**. The diagnosis below (competitor-strength × compute-density; the win/loss table) stands.

**Headline correction (the 8th over-claim caught this session — mine):** the §8/§8b explanation that the
block-as-lane repack "is VLEN256-shaped and DEGRADES to mf2/8-lane strips at VLEN128, strip-split overhead
outweighing the removed vredsum" is **MECHANISTICALLY WRONG**. Two proofs:
1. **Lane-math:** at VLEN128, `e8mf2` VLMAX = LMUL·VLEN/SEW = 0.5·128/8 = **8 lanes**. The emitter's VLEN128
   form (half_lanes=8, two i8mf2→i16m1→i32m2 strips at vl=8) is the CORRECT full-utilization VLEN128 tiling of
   16 columns — NOT a degraded VLEN256 kernel. (ggml's own 16x1 repack uses literal vl=16 on i8mf2, correct
   only at VLEN256, and is gated OFF at VLEN128 — `case 128: break` for q4_0/q8_0/q4_K/q2_K/iq4_nl.)
2. **Our own q8_0 ISO datum** (`q8_0-repack-winA-oracle-FINDING.md`): NARROW (2×8, ILP-2) BEATS WIDE (1×16
   serial m1) at VLEN128 — the 2-strip split is BENEFICIAL ILP, not overhead. The tune already defaults to it.

## The TRUE discriminator (the q4_0-wins-but-q4_K/q8_0-lose puzzle, RESOLVED)
All three repacks run the SAME correct 2×8 mf2 VLEN128 shape. The win/loss is set by **what ggml falls back to
at VLEN128** (its 16x1 repack is off there) — i.e. **competitor strength × our compute-density**, NOT lane-shape:

| our repack @VLEN128 | ggml VLEN128 baseline | baseline character | result |
|---|---|---|---|
| **q4_0** | `ggml_vec_dot_q4_0_q8_0` block-dot | **HEAVY** (nibble + per-block vredsum + scattered reads) | **WIN** 1.22×→~2.6× e2e |
| **q8_0** | `ggml_vec_dot_q8_0_q8_0` block-dot | **LEAN** (32 int8 fill one i8m2, 1 vwredsum/block) | LOSS 0.6–0.7× |
| **q4_K** | `ggml_vec_dot_q4_K_q8_K_vl128` | **HAND-TUNED VLEN128 asm** | LOSS 0.47–0.66× |

Two factors: (a) does ggml ship a hand-tuned VLEN128 kernel for the quant (q4_K yes → we lose to hand-tuning);
(b) if it's a block-dot, how lean (q4_0 heavy → repack out-streams it + e2e locality win; q8_0 lean → nothing
for the repack to remove + 2× inner iters at QK8_0=32). The 16-interleave is FINE at VLEN128.

## Option A (VLEN128-native variant): NOT viable
An 8-column `block_q4_0x8` single-strip variant needs a NEW repacked ABI + quantize-mat + verifier + oracle
(not a parametrization tweak of the fixed block_q4_0x16 ABI), AND would not win: the 2×8 mf2 form already fully
utilizes VLEN128 (fewer strips = LESS ILP per the q8_0 ISO datum), it just trades competitors, and it does
nothing about q4_K's loss-to-hand-asm. DECLINE.

## Option B (the legitimate N3 fix): measured-best PATH SELECTION {repack, block-dot}
Enumerate {repack, block-dot} as competing candidate routes per (quant-type, VLEN, M-regime); select
measured-best, branch-free, through the capability machinery. Keep repack where it wins (q4_0@128), DECLINE it
where it loses (q8_0/q4_K@128; q4_0@K1-VLEN256 — folds in the disclosed 0.74× regression, ledger §4.2). Prefill
(M≫1) is a selection axis (q4_0 GEMM 5.68×; q4_K narrows to 0.59–0.89×, doesn't cross).
- **It is a legitimate N3 result** — a capability/resource-aware SELECTION between two real compiler-emittable
  algorithms, the algorithm-path-level analogue of the proven Win-A VLEN-strip selection (1.31× e2e) and the
  N1 VLEN→LMUL-family selection. It answers the OPEN §5.3/§4.2 "per-microarch repack-vs-autovec selection" gap.
- **It is NOT a Win-B perf win:** declining selects block-dot = MATCHES ggml (no speedup). The contribution is
  *selection-correctness* (measured-best > the static "always-repack" argmin, which is wrong on 3 of the cells)
  — the same "measured > static" thesis as the N1 family-selection. Report as selection, NEVER as a new Win-B.

## Cost-model mechanism (where the code goes)
`deriveRepackHalfLanes` (RVVRepackStripWidthMaterialization.cpp:78) is the WRONG layer — width-only, runs AFTER
the repack op exists. The path choice is UPSTREAM, at route materialization (RVVVectorSourceFrontDoor.cpp /
the route layer where GgmlRepackGemv*Op vs the block-dot op is decided): a new `(quant,VLEN,M)→{repack|block-dot}`
gate, wired as enumerated candidates through `RVVScheduleDescriptorRegistry` (I1/I3-clean, measured-best). Static
prior (no-measurement profiles): prefer repack only when (no ggml hand-tuned VLEN-native kernel for the quant)
AND (block-dot fallback is "heavy") AND (VLEN==128 OR M≫1); measurement is the authority.

## Next-session plan
1. **(this session, doc) CORRECT the ledger §8/§8b + the spec shape-match discipline + the matrix** — replace
   "VLEN256-shape degradation/strip-split overhead" with the competitor-strength × compute-density discriminator
   (lane-math: 2×8 mf2 IS the correct VLEN128 shape; cite the q8_0 NARROW>WIDE ISO datum).
2. Build Option B as a route-materialization gate (NOT in deriveRepackHalfLanes), enumerated candidates via the
   descriptor registry, measured-best. Validate on q4_0@128→repack (win kept), q8_0/q4_K@128→block-dot (decline,
   tie ggml), q4_0@K1→block-dot (flips the 0.74× loss to tie). Report as selection-correctness, not a Win-B.
3. (adjacent, separate axis — do NOT conflate) K-quant emitter LMUL-parametrization (RVVToEmitCKQuant.cpp
   hardcodes m2) to test whether a Win-A LMUL tune *narrows* the q4_K Win-B loss.

Critical files: RVVRepackStripWidthMaterialization.cpp (width-only, the wrong layer); RVVScheduleDescriptorRegistry.cpp
(enumerate candidates); RVVVectorSourceFrontDoor.cpp (the correct gate layer); RVVToEmitCBlockQuantLinear.cpp
(emitters q4_0:2674/q8_0:3167/q4_K:4135); ggml repack.cpp:4593 dispatch + arch/riscv/quants.c (the VLEN128 baselines).
