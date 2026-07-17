# IME mma_slide (vmadot1) — sliding-window MAC, silicon bit-exact on real K1 (X60) — FINDING

**Date:** 2026-06-25
**Hardware under test:** `ssh k1` = SpacemiT(R) X60, harts 0-3 (IME1), via `taskset -c 0-3` (hart 4 lacks `_ime`).
**Invariant in force:** I8 — only a real `ssh` run on the X60 counts as runtime evidence. **No QEMU.**
**Op under test:** the 5th IME op `tcrv.ime.mma_slide` (`vmadot1`/`vmadot2`/`vmadot3`,
Xsmti8i32mm_slide), int8 SIGNED A (even VS1:VS1+1 8x8 pair) × int8 B → int32, slide window.

---

## VERDICT (TL;DR)

> **PASS — slide semantics CONFIRMED on silicon, 16/16 bit-exact, 0 core LOC.**
> STEP 0 (de-risk): a 4-way window discriminator on real X60 confirmed the spec semantics:
> `vmadot`==A-rows-0..3 (anchor, pins low-half + base) AND `vmadot1`==A-rows-1..4 (slide-1) AND
> ≠ all other windows (rows 2..5, 3..6). STEP 3: the PIPELINE-EMITTED `vmadot1` slide kernel
> reproduced the slide-1 scalar oracle bit-exactly in all 16 int32 elements, ≠ all other windows.
> objdump confirms `smt.vmadot1 v4,v0,v2` = **0xe620322b** (funct7 111001 / `e6...`, DISTINCT from
> the non-slide MAC's `e2...`) ENGAGED inside the exported wrapper. hart-4 SIGILL (exit 132) =
> native IME, not emulation. 3/3 deterministic. This is the live N2 rapid-add: a genuinely-NEW-SHAPE
> IME kernel slots in at ~0 CORE LINES, ~360 plugin/test LOC, ZERO core family-branches.

---

## STEP 0 — slide-semantics discriminator FIRST (de-risk before building the emitter)

Mirrors the mma_su 4-way test. A tiny asm probe drove plain `vmadot` AND `vmadot1` on the SAME
loaded 8x8 A-pair (v0=rows0..3, v1=rows4..7) with row-DISTINCT data (incl. negatives), against four
window oracles computed from the spec pseudocode (`/home/kingdom/spacemit-ime/instruction-func.adoc`
line 159: `C[i*N+j] += int32(A[slide*K + i*K + k] * B[j*K + k])`). Cross-built with SpacemiT GCC15.2
(`-O2 -static -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d`), run `taskset -c 0-3`:

```
OK: four slide-window oracles pairwise distinct (discriminator load-bearing)
ANCHOR vmadot==o0(rows0..3): 1
SLIDE1 vmadot1==o1(rows1..4): 1  (!=o0:1 !=o2:1 !=o3:1)
VERDICT: PASS  vmadot=rows0..3 (anchor) AND vmadot1=slide-1 rows1..4 (!= all other windows)
```

The **plain-vmadot anchor is load-bearing**: it pins that v0 is the low half and the base window is
rows 0..3, so `vmadot1`==rows1..4 discriminates *direction + half*, not just magnitude. The spec
semantics (A from even VS1:VS1+1 pair; slide shifts the read-window DOWN by `slide` rows) hold on
silicon — safe to build the emitter. (Had it differed, the task mandate was STOP + report; it did not.)

---

## Encoding (objdump, SpacemiT GCC15.2 fork)

| mnemonic (registers)     | objdump encoding | funct7 | note |
|--------------------------|------------------|--------|------|
| `vmadot   v4,v0,v2`      | `e220322b`       | 111000 | non-slide MAC (anchor) |
| **`vmadot1 v4,v0,v2`**   | **`e620322b`**   | **111001** | **the slide-1 kernel's actual leaf** |
| `vmadot1 v2,v0,v1`       | `e610312b`       | 111001 | the design-doc's probe form (B=v1 OVERLAPS the A-pair — an illegal register choice for the real kernel) |
| `vmadot2 v4,v0,v2`       | `e620722b`       | 111001 | slide-2 follow-on |
| `vmadot3 v4,v0,v2`       | `e620b22b`       | 111001 | slide-3 follow-on |

**Encoding-assertion note:** the real kernel needs A=even-pair v0:v1, B=v2, C=even-pair v4:v5 ⇒
`vmadot1 v4,v0,v2` = **e620322b**, NOT the task line's `e610312b` (which is `v2,v0,v1`, putting B=v1
inside the A pair). The STEP-3 gate asserts on the **mnemonic `smt.vmadot1`** + the **funct7 nibble
flip (`e6` vs vmadot's `e2`, bit26 111000→111001)**, which is the discriminating fact regardless of
registers — NOT the literal word e610312b (asserting that would false-fail a correct kernel).

---

## STEP 1 — the op + plugin (0 CORE, capability-fact dispatch, fail-closed)

| piece | file | added LOC (logic+comment) | core? |
|-------|------|--------------------------|-------|
| ODS op `MMASlideOp` (clone MMASUOp + `$slide` I64Attr) | `include/TianChenRV/Dialect/IME/IR/IMEOps.td` | 55 | no (plugin dialect) |
| verifier `MMASlideOp::verify()` (reuse `verifyIMEMACBoundary` + slide∈{1,2,3} + slide→mnemonic) | `lib/Dialect/IME/IR/IMEDialect.cpp` | 41 | no (plugin dialect) |
| capability derive (`ime_slide` fact) + variant routing + `ime.slide` stamp + boundary route | `lib/Plugin/IME/IMEExtensionPlugin.cpp` | +94 / −8 | no (plugin) |
| `macSlideHelperBody` (A even-pair load) + `IMEMACSlideToEmitCFunc` + illegal-op/patterns | `lib/Plugin/IME/IMEBackendEmissionDriver.cpp` | +171 / −3 | no (plugin) |
| lit: round-trip (slide 1/2) + 6 fail-closed neg + materialization | `test/Dialect/IME/mma-slide.mlir`, `test/Conversion/EmitC/ime-mma-slide-materialization.mlir` | 199 | no (test) |

**0-CORE CONFIRMED.** `git diff --name-only` touches exactly 4 source files — all under
`include/TianChenRV/Dialect/IME/` + `lib/Plugin/IME/` + `lib/Dialect/IME/`. **NOTHING** in
`lib/Transforms/`, `lib/Conversion/EmitC/Builtin/`, or any core conversion. Verified the same way the
mma_su precedent was (`grep -rln MMASUOp lib/Transforms lib/Conversion/EmitC/Builtin` = none). The
slide is selected by a capability-DERIVED fact (`ime_slide` property → `derived.slide` → `ime.slide`
variant attr → `MMASlideOp` boundary + `slide` op attr), NOT a family-name string and NOT a second
capability id. Fail-closed: slide ∉ {1,2,3} rejected at derive AND verify; slide with unsigned/mixed
or tiled-matmul shape rejected (no emitter); ime_op must match the slide stride.

**Non-comment logic delta** across the 4 source files: 263 added lines (much is the EmitC emit
pattern boilerplate that mirrors the existing MAC pattern; the logic-unique delta is the slide helper
body, the `ime_slide` derivation, and the fact routing).

---

## STEP 2 — lit PASS

- `test/Dialect/IME/mma-slide.mlir`: round-trip slide=1 (ime_op=vmadot1) + slide=2 (vmadot2), plus 6
  fail-closed negatives — slide=4 rejected, slide=0 rejected, ime_op/slide mismatch (vmadot, vmadot1
  vs slide=2), wrong elem width, unknown attr. `--verify-diagnostics` RC=0; FileCheck **PASS**.
- `test/Conversion/EmitC/ime-mma-slide-materialization.mlir`: full pipeline from `ime_slide="1"`
  capability property → `ime_vmadot1_mma_slide_slice` variant → `tcrv_ime.mma_slide` boundary →
  `vmadot1` slide helper, with `--implicit-check-not` guards that NO non-slide IME helper/variant
  (vmadot/vmadotu/vmadotsu) and NO other family dialect leaks. FileCheck **PASS**.
- **No regression:** the existing mma / mma_u / mma_su / capability-absent materialization + dialect
  tests all still PASS.

---

## STEP 3 — K1 full verify (emit → cross-build → objdump → X60 bit-exact)

- **Emit (OUR pipeline, not a hand probe):** `tcrv-opt ...materialize... --mlir-print-op-generic |
  mlir-translate --mlir-to-cpp` produced the `extern "C"` wrapper +
  `static inline tcrv_ime_vmadot1_mma_slide_4x4x8` helper (A even-pair load, `vmadot1 v4,v0,v2`).
- **Cross-build:** SpacemiT GCC15.2 fork, `-O2 -static -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii
  -mabi=lp64d`. The gate `#include`s the pipeline-emitted C and calls the EXPORTED wrapper.
- **objdump:** `smt.vmadot1 v4,v0,v2` = `e620322b` (funct7 111001 / `e6`) ENGAGED inside
  `tcrv_emitc_ime_mma_slide_kernel_ime_vmadot1_mma_slide_slice`.
- **Run X60 `taskset -c 0-3`:** vlenb=32 / vl(e8,m1)=32 pinned; four window oracles pairwise-distinct
  asserted; `IME(slide1) == o1(rows1..4)` in all 16 int32, ≠ slide0/2/3. **bit-exact 16/16.**
- **Negative control:** `taskset -c 4` → exit 132 (SIGILL) ⇒ native IME on harts 0-3, not emulation.
- **Determinism:** 3/3 independent runs → PASS, identical.

```
$ taskset -c 0-3 ./ime_mma_slide_gate
IME(slide1)  7488 5652 3816 1980 4420 3520 2620 1720 1352 1388 1424 1460 -1716 -744 228 1200
MATCH: slide1=1 (!=slide0:1 !=slide2:1 !=slide3:1)
VERDICT: PASS  emitted vmadot1 kernel == slide-1 ref (rows1..4), != all other windows -> bit-exact 16/16
EXIT_CODE=0
```

---

## Win-A / Win-B / Win-C mapping (HONEST, scoped)

- **Win-B (algorithm-change vs ggml's shipped RVV slide/conv path — the MANDATED baseline, NOT
  scalar):** the slide primitive computes 4 overlapping 4x8 windows from ONE loaded 8x8 A-pair with
  NO A re-load — structural A-reuse a plain-RVV strided-dot/im2col path cannot express. **GAP, not a
  number:** no ggml RVV *slide/conv* baseline kernel was located on this box to A/B against, so per the
  task discipline (and the repeated over-optimism corrections in memory) NO Win-B speedup number is
  manufactured here. The defensible claim is the **algorithm-change capability + bit-exact silicon
  correctness**; a Win-B micro number is owed against a real ggml RVV slide baseline if/when one exists
  (build-confound-free A/B, same toolchain both arms — the prefill-probe lesson).
- **Win-A (an IME tune knob on/off):** the slide stride `slide ∈ {1,2,3}` (window stride / dilation)
  is a genuine IME tune knob, AND the A-pair-reuse factor (one loaded 8x8 pair feeds up to 4 windows
  vs naive per-window reload) is a second knob. All under the SAME `spacemit.ime` capability. slide
  2/3 assemble (`e620722b`/`e620b22b`) but were not silicon-run here — silicon-gated follow-ons.
- **Win-C (a pass that changes structure):** **NULL** (honest, per [[winc-structural-null]]). This is
  a boundary-op + emitter; the A-pair-reuse lives in the emitted kernel body, not a separable pass.

**Perf regime caveat (per [[kernel-wins-dont-transplant-to-e2e]]):** the IME-suitable regime is
compute-bound PREFILL / batched-window conv, NOT M=1 decode (memory wall). Decode-e2e transplant is
expected NULL. The primary deliverable here is **STRUCTURAL (0-core rapid-add) + bit-exact silicon**;
perf is secondary and was NOT campaigned (no idle-box A/B run) to avoid manufacturing a transient win.

---

## Reproduce
Local scratch (NOT in git): `/home/kingdom/spacemit-ime/mma-slide/` holds `slide_probe.c` (STEP 0
discriminator), `emitted_full.c` (the pipeline-emitted wrapper+helper), `ime_mma_slide_gate.cpp`
(`#include "emitted_full.c"`), and the static binaries. Toolchain root:
`/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4`. On K1 the binaries are at
`~/n2-ime-probe/{slide_probe,ime_mma_slide_gate}`.
```
tcrv-opt test/Conversion/EmitC/ime-mma-slide-materialization.mlir \
  --tcrv-materialize-plugin-variants --tcrv-select-variants \
  --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emitc-lowerable-routes \
  --mlir-print-op-generic | mlir-translate --mlir-to-cpp > emitted_full.c
```
