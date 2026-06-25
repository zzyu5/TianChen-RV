# IME e2e — the CLEAN one-toolchain SPACEMIT toggle (Control A + B), K1 X60

**Date:** 2026-06-25
**Hardware:** `ssh k1` = SpacemiT X60, RVV1.0 VLEN256, IME1, harts 0-3 (`taskset -c 0-3`).
**Goal:** Isolate whether the IME *unit* (not global codegen) moves prefill e2e, by flipping
ONLY `-DGGML_CPU_RISCV64_SPACEMIT` on ONE toolchain. This is the REBUILD the prior probe
(`IME-PREFILL-PROBE.md` §VERDICT scope-ceiling) explicitly deferred — it cancels the codegen
confound that made the prior cross-build "prefill win" (1.37-1.95x) dissolve when the decode
control read 1.25x.

**HONEST hypothesis (stated, then tested):** the prior probe NULLed the IME-unit e2e question
via the decode control (1.25x in a regime IME cannot touch ⇒ codegen, not IME). The clean toggle
that cancels codegen by construction is EXPECTED to show IME-unit e2e ~NULL on this
memory-bound tinyllama-1B-Q4_0 regime. The point of the clean toggle is to make that NULL
*defensible*, not to manufacture a win. **Report AS MEASURED.**

---

## SETUP — cached one-toolchain A/B (prior attempt built both arms; reused)

Both arms come off the SAME `cmake` config; **only `GGML_CPU_RISCV64_SPACEMIT` is flipped.**
Identical incidental flags confirmed from each `CMakeCache.txt`:

| flag | ON arm (`build-ime`) | OFF arm (`build-off`) |
|---|---|---|
| `GGML_CPU_RISCV64_SPACEMIT` | **ON** | **OFF** |
| `CMAKE_C_COMPILER` | `/usr/bin/clang-18` | `/usr/bin/clang-18` |
| `CMAKE_CXX_COMPILER` | `/usr/bin/clang++-18` | `/usr/bin/clang++-18` |
| `CMAKE_C_FLAGS` / `CXX_FLAGS` | `-fno-integrated-as` | `-fno-integrated-as` |
| `GGML_RV_ZBA` / `ZVFH` / `ZFH` | ON / ON / ON | ON / ON / ON |
| `IME_SPEC` | `RISCV64_SPACEMIT_IME1` | (none) |

This is the can't-confound A/B the methodology (Control A) mandates: NOT the old reuse of
`libggml-cpu-1x16.so` (which was a different build = the documented confound). Lib sizes differ
only by the IME kernels (ON 1222928 B vs OFF 920720 B).

**Lib resolution (clean — each build's own binary):** both `llama-bench` binaries carry a
RUNPATH to their OWN `bin/`, so the ON binary loads the ON lib and the OFF binary loads the OFF
lib with NO `LD_LIBRARY_PATH` juggling (`ldd` confirmed: build-ime→build-ime lib,
build-off→build-off lib). Same llama.cpp tree/commit, same compiler, same flags ⇒ the only
runtime difference between the two binaries' kernels is the IME unit.

## ENGAGEMENT (objdump the FINAL staged libs) — PASS

| proof | ON (`build-ime`) | OFF (`build-off`) |
|---|---|---|
| `objdump -d \| grep -c vmadot` | **32** | **0** |
| `strings \| grep -ci spacemit` | **624** | **0** |
| `objdump -T \| grep gemm_kernel_i8i4` | **1** (defined `T`) | **0** |

Sample ON encoding: `d2b50: e225382b  vmadot v16,v10,v2` (real IME1 4x4x8 int8 MAC, VLEN256).
⇒ The IME instruction is PHYSICALLY PRESENT in ON and ABSENT in OFF. Engagement proven by the
instruction, not assumed. Source dispatch (from prior probe, unchanged): Q4_0 + marchid→x60→
`use_ime1=true` ⇒ `gemm_kernel = ime1::gemm_kernel_i8i4`; repack needs `ne[1]%16==0 && use_ime1`.

Model: `~/tcrv-k1-llama/models/tinyllama-q4_0.gguf` (637 MB, tinyllama-1B Q4_0).

## CORRECTNESS ARBITER (guards a `-fno-integrated-as` IME-asm miscompile)

`build/bin/llama-cli` (only cli on box) + `LD_LIBRARY_PATH=<arm>/bin` (`ldd`-confirmed it
resolves the correct arm lib), prompt "The capital of France is", `--temp 0`, `taskset -c 0-3`:
- **ON arm: PASS** — emitted "The capital of France is Paris". The clang `-fno-integrated-as`
  frankenbuild did NOT miscompile the explicit-register IME inline asm ⇒ ON numbers admissible.
- **OFF arm: PASS** — emitted "The capital of France is Paris". (Both arms then idle at a `>`
  interactive prompt — the documented newer-llama-cli `-no-cnv`-rejected fallback, harmless; the
  answer prints first.)

## ATTRIBUTION RULE for THIS clean toggle (differs from the prior cross-build probe)

Codegen is **common-mode by construction** here (identical flags). So:
- **tg16 decode is now a VALIDATION, not a subtraction.** Both arms run identical code in M=1
  GEVM (no IME) ⇒ tg16 ON/OFF MUST collapse to ~1.0. If it does, the toggle is proven clean and
  the pp ratio attributes DIRECTLY to the IME unit. (The prior 1.25x was the cross-build artifact;
  it should NOT reappear.) If tg16 is materially off 1.0, the "identical flags" claim has a hole.
- **pp ON/OFF attributes to the IME unit ONLY to the extent SPACEMIT toggles only the matmul**
  (OFF prefill = ggml RVV Q4_0 mul_mat; ON = ime1 i8i4). pp>1 rising with M ⇒ defensible IME-unit
  prefill win; pp~1.0 ⇒ honest memory-bound null (NOT "codegen artifact" — codegen is cancelled).
  *(Caveat discovered in-flight: SPACEMIT toggles a whole 227-symbol kernel FAMILY, not just the
  IME matmul — see the decode-control result below; the clean rule above is qualified accordingly.)*

**Anchor caveat:** the methodology anchors (pp32≈23.87, tg16≈6.52) are from the *1x16 RVV* lib,
NOT this OFF arm. This OFF arm is clang `-fno-int-as` SPACEMIT=OFF and per the prior codegen
finding will read FASTER than those anchors. Idle cert here = interleaved 3-run tight stddev +
live `top %id`, not anchor-match.

## MINIMAL SWEEP (pp256 + tg16, r=3, interleaved ON-then-OFF, `taskset -c 0-3`)

Idle cert: `top` %id = 84.5 pre / 84.6 post (the ~2.0 loadavg is the known D-state virtio floor,
not contention; no llama/spin process). Each arm = its OWN `llama-bench` binary (RUNPATH→own lib).

| regime | OFF (SPACEMIT=OFF) | ON (SPACEMIT=ON) | ON/OFF ratio |
|---|---|---|---|
| **pp256** (prefill, CLAIM) | 24.43 ± 0.02 | 39.58 ± 1.49 | **1.62×** |
| **tg16** (decode CONTROL, M=1) | 5.52 ± 0.02 | 8.12 ± 0.04 | **1.47×** |

**Independent check-agent reproduction (2026-06-25, idle box, `taskset -c 0-3`, r=3, interleaved):**
the load-bearing points re-ran within noise — OFF tg16 **5.59 ± 0.01** / ON tg16 **8.23 ± 0.05**
⇒ decode **1.47×** (the falsifier, reproduced exactly); OFF pp256 **24.51 ± 0.01** / ON pp256
**40.11 ± 1.74** ⇒ **1.64×**. Structural artifacts re-verified live: CMakeCache flips ONLY
`GGML_CPU_RISCV64_SPACEMIT` (all other clang flags identical), objdump vmadot 32(ON)/0(OFF),
SpacemiT `T`-symbols 227(ON)/0(OFF), lib sizes 1222928(ON)/920720(OFF) B, each `llama-bench`
RUNPATH→its OWN lib (ldd-confirmed), ON-arm correctness arbiter re-emitted "…is Paris". The sweep
tables below were not re-run point-by-point but the decode falsifier — on which the entire
attribution pivots — and the pp256 headline both reproduce, so the numbers are MEASURED, not fabricated.

**The decode control DID NOT collapse to ~1.0 — it reads 1.47×.** This is the key result. In M=1
decode the IME matrix unit physically cannot help (memory-bound GEVM), yet ON is 1.47× faster
than OFF. So most of the prefill speedup is NOT the IME `vmadot` unit. The prefill ratio (1.62×)
EXCEEDS the decode control (1.47×) by ~10% — that ~10% excess is the only candidate IME-unit
increment, and **a single prefill point cannot tell a real IME increment that grows with M from
kernel-family M-variation / noise. The M-trend (pp512/pp1024) is the discriminator — see expanded
sweep below.** (ON pp256 stddev ±1.49 = ±3.8% means 1.62 ± ~0.06, so the ~10% excess is barely
above noise; the expanded sweep uses higher `-r` on the ON arm.)

**Why the decode control did not collapse (NOT a flags hole — SPACEMIT gates a kernel FAMILY):**
the `GGML_CPU_RISCV64_SPACEMIT` flag does NOT only add the IME i8i4 gemm. `objdump -T` shows the
ON lib has **227 SpacemiT `T` symbols** (OFF: 0), including a SpacemiT `forward_mul_mat` dispatcher
and `quantize_a_*row_i8` quant kernels (`ime.cpp` L253 main mul_mat, plus an `ne11==1` M=1 branch
L643). So SPACEMIT=ON swaps in an ENTIRE SpacemiT RVV+quant kernel family that serves the SAME
Q4_0×Q8_0 ops the decode path uses — the clean toggle cancels *compiler* codegen (identical clang
flags) but the SPACEMIT macro itself gates a kernel-family swap that reaches M=1 decode. The
decode speedup is the SpacemiT RVV GEVM/quant kernel, NOT the IME `vmadot` unit. (Engagement
objdump still holds: `vmadot` is ON-only and runs in prefill; it is just not the source of the
cross-regime speedup.) The flags ARE identical; SPACEMIT legitimately swaps more than the IME unit.

## EXPANDED PREFILL SWEEP (the M-trend discriminator) — pp512/pp1024, r=5, interleaved

Idle cert: `top` %id 88.2 pre / 85.6 post; `-r 5` on the ON arm to control its variance.

| regime | OFF (SPACEMIT=OFF) | ON (SPACEMIT=ON) | ON/OFF | excess over decode (1.47×) |
|---|---|---|---|---|
| tg16 (decode, M=1, CONTROL) | 5.52 ± 0.02 | 8.12 ± 0.04 | **1.47×** | — (baseline kernel-family swap) |
| pp256 | 24.43 ± 0.02 | 39.58 ± 1.49 | **1.62×** | +0.15 |
| pp512 | 23.70 ± 0.07 | 39.39 ± 1.95 | **1.66×** | +0.19 |
| pp1024 | 22.21 ± 0.05 | 36.70 ± 0.41 | **1.65×** | +0.18 |

**The prefill ratio is constant across pp256→1024: 1.62 → 1.66 → 1.65** — a fixed ~+0.15-0.19
above the 1.47× decode-control floor. **CAVEAT (do not over-read this flatness):** a 4-wide IME
tile saturates its weight-load amortization well before M=256, so a flat ratio across 256→1024 is
EXPECTED whether or not the IME unit contributes — this M-range cannot discriminate "IME win" from
"no IME win." The IME amortization transition, if any, lives in 1<M<256 (see small-M sweep below),
which the pp256/512/1024 points do not sample. So flatness here is NOT evidence against an IME
contribution. What the data DOES establish: the +0.15-0.19 prefill-specific increment is several
sigma above the decode floor (decode 1.471 ± 0.008 vs prefill 1.62-1.66 ± 0.02-0.08) ⇒ it is a
REAL, prefill-only effect, appearing only where the IME `vmadot` can run.

## SMALL-M SWEEP (the actual shape — the discriminating 1<M<256 range), r=5, interleaved

| M | OFF | ON | ON/OFF |
|---|---|---|---|
| tg16 (M=1, decode floor) | 5.52 ± 0.02 | 8.12 ± 0.04 | **1.47×** |
| pp8  | 20.09 ± 0.38 | 37.18 ± 1.08 | **1.85×** |
| pp16 | 23.10 ± 0.09 | 45.34 ± 0.17 | **1.96×** |
| pp32 | 23.56 ± 0.15 | 46.82 ± 0.51 | **1.99×** (peak) |
| pp64 | 24.31 ± 0.10 | 46.30 ± 2.18 | **1.90×** |
| pp128| 24.38 ± 0.03 | 38.72 ± 8.33 | 1.59× (ON ±8.33 — high-variance outlier sample) |
| pp256| 24.43 ± 0.02 | 39.58 ± 1.49 | 1.62× |
| pp512| 23.70 ± 0.07 | 39.39 ± 1.95 | 1.66× |
| pp1024| 22.21 ± 0.05 | 36.70 ± 0.41 | 1.65× |

**Read the within-PREFILL trend, NOT the jump off the decode floor.** The decode→pp8 jump
(1.47→1.85) is an OP-PATH change (GEVM→GEMM), not M-scaling. The within-prefill M-trend is:
pp8 1.85 → pp16 1.96 → **pp32 1.99 (PEAK)** → pp64 1.90 → pp256-1024 **~1.62-1.66 (lowest)**. This
is **peak-at-small-M, DECAY-at-large-M** — NOT a rise-with-intensity matmul signature. A genuine
IME FLOP-intensity win would be HIGHEST at large M (most arithmetic intensity); instead the ratio
is at its LOWEST (1.65) exactly there. The small-M peak is the **cache-hot working-set effect** the
prior probe already attributed to its pp32 peak: both arms ramp at small M (OFF 20→24, ON 37→47)
because the small activation fits hot in cache, and the ratio's small-M bump is the *difference of
two cache-sensitive ramps*, not a clean IME amortization curve. So the ramp is **texture, not an
IME-win upgrade** — if anything the large-M decay weakly argues AGAINST a matmul-intensity win.
**The stable, defensible "real prefill increment" is therefore the cache-cold large-M +0.18
(1.47→1.65), NOT the cache-inflated +0.52 small-M peak.** And even that +0.18 is not ISOLATED to
the IME `vmadot` unit: SPACEMIT toggles the whole GEMM-path kernel family (quant blocking / RVV
packing / inner-loop) together, so the IME matmul cannot be separated from the rest by this A/B.

## ATTRIBUTION (per composed Control A + Control B)

Codegen is cancelled by construction (identical clang flags). Decomposing the ON/OFF speedup:
- **~1.47× = the SpacemiT RVV+quant kernel FAMILY swap.** Present in M=1 decode where the IME
  `vmadot` array physically cannot run ⇒ this component is NOT the IME unit. It is the SpacemiT
  backend's whole optimized Q4_0 kernel set (gated by the SPACEMIT macro alongside, but distinct
  from, the IME matmul).
- **~+0.15-0.19 (1.47→~1.65) = a REAL prefill-only increment, but NOT cleanly isolable to IME.**
  It is several sigma above the decode floor and appears only in prefill where the IME `vmadot`
  CAN run — so it plausibly INCLUDES an IME contribution. BUT it is confounded: the M≥tile GEMM
  path differs from the M=1 GEVM path in MORE than the matmul unit (different quant blocking, RVV
  packing, inner-loop structure in the SpacemiT GEMM kernel). So the +0.18 cannot be cleanly
  attributed to the IME `vmadot` unit alone — nor dismissed. The honest limit is **ISOLATION, not
  absence.** (The small-M sweep above does NOT rescue isolation: the within-prefill ratio peaks at
  M≈32 and DECAYS to large M — a cache-hot working-set shape, not the rise-with-intensity an IME
  matmul win would show; if anything it argues weakly against a FLOP-intensity win.)

## VERDICT — no CLEANLY-ISOLATED IME-unit e2e prefill win (clean one-toolchain toggle)

The clean SPACEMIT toggle (identical clang-18 `-fno-integrated-as` flags, only the IME unit's
macro flipped, objdump-confirmed 32 vmadot ON / 0 OFF, both arms coherent) shows:
1. A real ~1.65× ON/OFF e2e speedup on tinyllama-1B-Q4_0 prefill — BUT
2. ~1.47× of it survives in M=1 decode where the IME array cannot run ⇒ that 1.47× is the
   **SpacemiT RVV kernel-family swap** (227 SpacemiT symbols, decode-reaching), NOT the IME unit;
3. a real prefill-only increment of ~+0.18 (1.47→~1.65) remains, several sigma above the decode
   floor and present only where IME can run — but it is **confounded** with GEMM-path-vs-GEVM-path
   kernel-family differences and so **cannot be cleanly isolated to the IME `vmadot` unit** (and
   the pp256-1024 flatness does not settle it — the IME tile saturates before M=256).

⇒ **No cleanly-isolated IME-matrix-unit e2e prefill win demonstrated on this model/regime** — and,
symmetrically, NOT a proof of zero IME effect (the +0.18 prefill increment is real but
unisolable). This is the conservative HONEST-hypothesis outcome (the prior probe's decode-control
NULL, now sharpened by the clean toggle the prior probe deferred). It is CONSISTENT WITH the
memory-wall finding: tinyllama-1B-Q4_0 prefill is dequant/memory-dominated, so any IME contribution
stays buried under the kernel-family floor and cannot be teased out at this model size.

**What IS defensible / shippable from this:**
- The SpacemiT backend (SPACEMIT=ON) gives a real ~1.65× e2e speedup over the same-toolchain
  RVV-only build on Q4_0 — but it is a **kernel-family** win (RVV+quant+IME together), with the
  IME-unit component bounded by the flat ~+0.15-0.19 prefill increment, NOT a scaling matmul win.
- The IME perf claim stays the **kernel-level micro** (IME-PERF-FINDING 5.51× vs hand-rolled, or
  the proper ggml-RVV Win-B baseline), with an HONEST e2e null at the unit level.
- Improvement over the prior cross-build probe: codegen is cancelled by construction here, so the
  attribution is sharper — the surviving decode speedup is now provably a *kernel-family* effect
  (227 SpacemiT symbols, decode-reaching), not a compiler-codegen artifact.

**Scope (disclosed):** K1's ~7GB caps the model at ~1B; a larger compute-bound prefill (or a
GEMM-heavy workload) could still surface an M-scaling IME win that this memory-bound 1B model
cannot. Absence here is the memory-wall null, not a disproof of the IME unit in principle.

