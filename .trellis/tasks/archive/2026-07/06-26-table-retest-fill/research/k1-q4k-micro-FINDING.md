# k1 q4_K repack GEVM MICRO vs ggml's OWN VLEN256 16x1 repack — FINDING (注17 follow-up)

**Date:** 2026-06-29. **Board:** `ssh k1` (SpacemiT X60, RISC-V **VLEN256**, 8 cores,
`isa … zvfh ime`). **harts 0-3 only** (`taskset -c 0-3`). Serial, k1-only (rvv untouched —
the q4_K 8B judge runs on rvv; no conflict). **HEAD `afad2d62`.**
**Cell filled:** 自查表 表2 q4_K-repack **k1 MICRO** vs ggml 自己 VLEN256 repack (注17 留的
"更强对手" follow-up). **evidence-status = MEASURED.**

> **Headline:** OUR compiler-emitted q4_K repack GEVM (**manually-stamped h16** native strip —
> NOT the auto-selected default; see caveat) is **~1.26× FASTER** than ggml's own hand-written
> VLEN256 `ggml_gemv_q4_K_16x1_q8_K` at realistic decode shapes (stable 1.247–1.263 across 3 runs
> + 4 large shapes), **byte-exact**, and the win **survives recompiling ggml's own source with our
> exact flags** (build-flag artifact ruled out). This is a **Win-B2 WIN that EXCEEDS the PARITY
> success bar** — vs the *correct, stronger* opponent (ggml's own repack, NOT block-dot).
> **MICRO / compute-axis only — NOT an e2e claim** (decode e2e is memory-bound; expect washout).
> **Caveat:** the win is for the manually-stamped native-strip emit; VLEN→half_lanes
> auto-selection is a separate Win-A gap this task does NOT close.

---

## The opponent (why this is the methodology-correct, *stronger* baseline)

At VLEN≥256 ggml SHIPS a q4_K repack — it does NOT fall back to block-dot (that's the VLEN128
story on rvv). Confirmed in k1 source `ggml/src/ggml-cpu/repack.cpp:4617-4623`:

```c
#if defined __riscv_zvfh
switch (__riscv_vlenb() * 8) {
    case 128:  { break; }                                              // → nullptr → block-dot (rvv)
    case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }   // k1: REPACK
    case 512:  { break; }  case 1024: { break; }  default: { return nullptr; }
}
```

So on k1 the q4_K decode path is the **`q4_K_16x1_q8_K` trait → `ggml_gemv_q4_K_16x1_q8_K`**
(the `8x4`/`8x8` traits are x86-AVX). This is the SHIPPING decode kernel and the right opponent
named by 注17 — a much stronger competitor than the rvv q4_K winB (0.55×/0.74× was vs *block-dot*
at VLEN128, where ggml ships no repack).

It is a **real vector body**, not `_generic`: objdump of the linked symbol shows RVV ops
`vsetvli e16,m1 / e32,m2 / e8,m2`, `vfwcvt.f.f.v`, `vle8.v`, `vand.vx`, `vwmacc` — VLEN256-native
(16-lane `vfloat32m2`/`vint32m2` accumulators). `nm -D micro` → `U ggml_gemv_q4_K_16x1_q8_K`
(dynamic), `ldd` → `/data/k1build/bin/libggml-cpu.so.0` (the real shipped lib, built clang-18
`-O3 -DNDEBUG` Release, `-march=rv64gcv_zfh_zvfh_zicbop_zihintpause`).

---

## OURS (the kernel under test)

Our compiler-emitted `tcrv_emitc_…_ggml_repack_gemv_q4_K_q8_K` at the **VLEN256-native
`half_lanes=16` single mf2 strip** (`k_gemv_q4K_h16.cpp`, md5 `39b70399…`, the SAME emit the
banked k1 oracle validated). Both kernels consume **identical buffers** — `vx` = `block_q4_Kx16`
stream (group-major × nb), `vy` = plain `block_q8_K` w/ bsums, output nc contiguous floats
(ggml `bs` UNUSED) — so this is a clean apples-to-apples matched-repack comparison (NO per-column
handicap like the q5_0 block-dot baseline).

> **⚠ Manual-stamp caveat (load-bearing, not auto-selected):** the `h16` form is a **MANUAL
> `half_lanes=16` stamp** on the input op (sed-edited attribute, matching the archived 06-25 k1
> oracle), **NOT** the compiler's auto-selected default — the default q4_K emit is **h8**, and
> there is **NO VLEN→half_lanes auto-selection** (see memory `repack-winA-always-mf2`). So the
> honest claim is "**our manually-stamped native-strip h16 emit beats ggml's hand kernel 1.26×**",
> NOT "our auto-emit wins". Closing the **VLEN256 → native-strip auto-selection** is a separate
> **Win-A compiler-maturity gap** that this task does NOT resolve.

---

## GATE — byte-exact, BEFORE any perf (3 independent checks, all PASS)

1. **Banked oracle reproduced**: `oracle_gemv_q4K.cpp` + `k_gemv_q4K_h16.cpp` →
   **WORST_NORM 7.0651e-07 PASS** (identical to the banked value), negative controls fire hard
   (NOMIN 396124× / PERM 2080930× margins → min-term + per-sub-block-scale structure genuinely
   exercised). Gates OURS.
2. **Inline independent scalar ref** (in the micro, per shape): `norm(ours-ref)` and
   `norm(ggml-ref)` BOTH `< 1.4e-6` at every shape (bar 1e-4). Self-contained gate for the
   **linked ggml symbol** too (it computes the full correct q4_K math — not a stub).
3. **ours ≈ ggml** SANITY `norm < 1.6e-6` every shape (same computation, right kernel engaged).
4. **ggml-SRC ≈ ref** `srcRefNorm < 1.4e-6` (the recompiled-from-source ggml control is correct).

---

## MEASURED — ratio per shape (RATIO = ggml_ns / ours_ns; >1 ⇒ OUR repack faster)

k1 VLEN256, clang-18 -O3 `-ffp-contract=fast`, `taskset -c 0-3`, min-of-reps (reps=80,
adaptive iters). System ~77% idle at bench time (instantaneous top); load-avg ~2.5 was
desktop-daemon EWMA, no sustained compute on harts 0-3.

| shape (nc×n) | ours (ns) | ggml.so (ns) | ggml-SRC (ns) | **RATIO ggml.so/ours** | ggml-SRC/ours |
|---|---:|---:|---:|---:|---:|
| 4096×4096   |  3.66e6 |  4.61e6 |  4.61e6 | **1.258** | 1.257 |
| 11008×4096  |  9.80e6 | 12.34e6 | 12.34e6 | **1.259** | 1.259 |
| 4096×11008  |  9.83e6 | 12.39e6 | 12.37e6 | **1.260** | 1.259 |
| 14336×4096  | 12.79e6 | 16.10e6 | 16.10e6 | **1.258** | 1.257 |
| 16×256 (tiny, overhead-bound) | 0.80e3 | 1.11e3 | 1.10e3 | 1.381 | 1.373 |

(values = central across 3 runs; large-shape spread 1.247–1.263.)

**Headline: ~1.26× WIN at decode shapes.** The ratio is **shape-invariant** across the large
shapes → it is a **per-element inner-loop throughput** difference (data-parallel steady state),
NOT a fixed-overhead artifact. The 16×256 tiny shape (1.38×) is call-overhead-dominated.

### Build-flag artifact RULED OUT (decisive control)
`ggml-SRC` = ggml's `ggml_gemv_q4_K_16x1_q8_K` source extracted **verbatim** from k1
`arch/riscv/repack.cpp:331-461` and recompiled **inside our harness with our EXACT flags**
(`-O3 -march=rv64gcv_zvfh -ffp-contract=fast`, same clang-18). It lands within **0.1%** of the
shipped `.so` (both ~1.26× slower than ours). So the win is **genuine codegen/structure**, not a
build-setting difference. (The shipped `.so` even carries EXTRA `zicbop` prefetch + `zihintpause`
hints that could only help ggml.)

---

## VERDICT

**Win-B2 = WIN (~1.26×), exceeding the PARITY bar**, vs ggml's OWN VLEN256 16x1 repack GEVM —
the correct, stronger opponent (objdump-confirmed real vector body; ldd-confirmed shipped lib;
trait-selection-confirmed as THE k1 decode path). Byte-exact (3 gates), stable (3 runs × 4 large
shapes), build-flag-robust (source-recompiled control matches). **evidence-status = MEASURED.**

Plausible mechanism (HYPOTHESIS, not objdump-proven, not load-bearing): ggml's body uses wider
`e8,m2`/`e16,m4` scale-unpack (vand/vsrl/vsll/vor/vzext + per-slice `vget`) and frequent vtype
toggling (e16m1↔e32m2↔e8m2); our native single mf2 strip has a tighter, more uniform vtype
profile. Mechanism unconfirmed — the MEASURED ratio is the claim.

## Honesty / scope
- **MICRO / compute-axis ONLY — NOT e2e.** Decode GEVM e2e is memory-bound; per
  `kernel-wins-dont-transplant-to-e2e` (+ prior k1 q4_0 winB e2e 0.74×), this micro win is
  **expected to wash out** at decode e2e. The q4_K e2e (B2/B3) is a SEPARATE measurement
  (rvv board, build-half ready, board-pending) — do NOT fold this micro into an e2e claim.
  The **shape-invariant** ratio across 4 large shapes ⇒ a **compute-bound** regime (the kernel
  is cache-resident here); at e2e the weights stream from DRAM and the throughput edge washes —
  the same compute-bound/memory-bound split seen in the k1 q4_0 winB micro-TIE → e2e-0.74×.
- **Auto-selection gap (see §OURS caveat):** the win is for the **manually-stamped h16** native
  strip, not the compiler's default (h8) — `repack-winA-always-mf2`. Do NOT reclaim as an
  auto-tune/auto-select win; the VLEN→half_lanes selection remains a Win-A gap.
- Contrast with the rvv q4_K winB (0.55×/0.74× LOSS): that was vs **block-dot** at **VLEN128**
  (ggml ships no repack there). This is vs ggml's **OWN repack** at **VLEN256** — different
  board, different (stronger) opponent, opposite outcome. Both honest, separately scoped.
- Reported the MEASURED ratio only; no back-claim. A WIN here is the honest measured result, not
  an over-optimistic reach — it is triple-gated, 3-run-stable, and the build-flag confound was
  explicitly killed with a source-recompiled control.

## Artifacts (research/k1-q4k-micro/)
- `winB_micro_q4K_gevm.cpp` (md5 `d8ba2840…`) — the 3-way micro harness (ours / ggml.so / ggml-SRC).
- `ggml_src_gemv.cpp` (md5 `45100fe2…`) — ggml's gemv source + our-flag recompile control.
- `evidence.txt` — objdump RVV-ops snippet + `nm -D` (U) + `ldd` resolution + md5s.
- `run_full.log` — clean captured run (run 3).
- OURS kernel emit: `archive/2026-06/06-25-backend-maturity-winA/artifacts/k1-repack-fill/q4k/k_gemv_q4K_h16.cpp`
  (md5 `39b70399…`); banked oracle `q4k/oracle_gemv_q4K.cpp` (WORST_NORM 7.07e-7).
- Board scratch `/home/bianbu/q4k-micro-scratch` was REMOVED on completion (k1 restored; no
  shared process touched; rvv untouched).
