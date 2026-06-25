# Option-2 M1 — EMIT-IDENTITY on the SUPPORTED RVV1.0 regime (FINDING, 2026-06-24)

> **Scope: EMIT-IDENTITY ONLY.** M1 proves the compiler now **AUTO-SELECTS** the repack
> (previously hand-chosen by authoring the concrete repack op directly in the input IR)
> and auto-emits the **byte-identical** mf2 RVV1.0 repack kernel that hand-authoring the
> repack op produces. **NO e2e / NO 2.6× claim** is made or inherited here — see §5.

## 1. RESULT (the gate)

**`diff kernelA.cpp kernelB.cpp` is EMPTY — byte-identical.** Both files SHA256
`9da66267adcf67d109e2a83ee1868ad5a88d3318f24cceedf0974adebab4880c`.

The byte-diff isolates exactly one question: *are the C1 audit attrs emitter-inert AND
does the auto-selected path emit exactly what hand-authoring the repack op produces?*
Answer: **yes** (empty diff). The selection-pass decision (algorithm/reason/
materialization/weight_layout_contract) leaks zero bytes into the emitted kernel.

## 2. THE TWO EMIT PATHS (both on march=rv64gcv = RVV1.0 / Zvl128b / VLEN128)

**Kernel A — AUTO-SELECTED.** The committed C1 fixture's q4_0@decode ABSTRACT
`tcrv_rvv.quant_contraction` op (`weight_layout="plain"`, stride-18) is lowered by the
in-compiler selection pass at `march=rv64gcv` → `deriveMinimumVLEN=128` →
`selectContractionAlgorithm` picks **REPACK** → the C1 bridge realizes the real
`tcrv_rvv.repack_gemv_q4_0_q8_0` op carrying the block_q4_0x16 x16 facts
(288/16/32, half_lanes=8 ⇒ **mf2**, NO `integer_core_lmul`) **PLUS** the 4 emitter-inert
audit attrs (`tcrv_rvv.contraction_algorithm="repack"`, `path_materialization="realized"`,
`path_selection_reason="repack-kept-q4_0-vlen128-decode"`, `weight_layout_contract="x16"`):

```
build/bin/tcrv-opt test/Conversion/RVV/rvv-lower-quant-contraction-stage-b-selection.mlir \
  --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc \
  | mlir-translate-20 --mlir-to-cpp > kernelA.cpp
```

**Kernel B — DIRECT.** Hand-authored `kernelB.mlir`: the C1 fixture's **EXACT** ABI
wrapper (kernel `@ggml_vec_dot_q4_0_q8_0_kernel`, variant `@ggml_vec_dot_q4_0_q8_0`, the
SAME 8 `runtime_abi_value`s n,s,bs,vx,bx,vy,by,nrc, column_count wired from `%bs` —
mirroring A's `nc`-from-`%bs` wiring) with the abstract op REPLACED by the DIRECT
`tcrv_rvv.repack_gemv_q4_0_q8_0` op carrying the **SAME** x16 facts
(`weight_block_stride=288`, `weight_interleave=16`, `weight_quant_byte_offset=32`,
`half_lanes=8`, mf2 / NOT isRVV0p7; `activation_block_stride=34`,
`activation_quant_byte_offset=2`) **but WITHOUT the 4 audit attrs**:

```
build/bin/tcrv-opt kernelB.mlir --tcrv-rvv-lower-to-emitc \
  | mlir-translate-20 --mlir-to-cpp > kernelB.cpp
```

**B was HAND-AUTHORED** (`kernelB.mlir`), not an existing artifact. *Why not the existing
`test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-0-q8-0.mlir`*: that committed fixture
proves the direct repack op lowers to the canonical mf2/half_lanes=8 shape (it is cited
as a **secondary witness** that this IS the RVV1.0 mf2 form), but its ABI wrapper differs
from A's — kernel `@ggml_repack_gemv_q4_0_q8_0_kernel` with only 5 abi values
(n,s,vx,vy,nc). A raw diff against it would be non-empty for a **wrapper-naming** reason
(different exported function name + parameter list), NOT for audit-attr leakage — a false
negative. B must mirror A's wrapper so the diff isolates only the audit attrs.

**NOT circular (B is anchored to an independent canonical fixture, not "built to match
A").** B's repack-op facts (`288`/`16`/`32`, `half_lanes=8`, mf2 / no `integer_core_lmul`,
act `34`/`2`) are **byte-for-byte the SAME facts** as the committed canonical direct-repack
fixture `rvv-to-emitc-repack-gemv-q4-0-q8-0.mlir:33` — a fixture that PREDATES this work
and is not authored to match A. So the chain is: **independently-canonical repack facts ==
the facts the C1 bridge stamps in A == B's facts**, and `diff(A,B)=∅`. The empty diff
therefore says not merely "B matches A" but "**A's auto-selected emit matches the
independently-canonical RVV1.0 repack form**." The only thing B drops vs A is the 4 audit
attrs — which is exactly what the gate proves inert.

## 3. THE EMITTED KERNEL (shape confirmation — it is the SUPPORTED RVV1.0 mf2 form)

Exported signature (identical in A and B):
```
extern "C" void tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
    size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5,
    const uint8_t* v6, size_t v7, int32_t v8)
```
(8-value ABI wrapper preserved; unused values bx/by/nrc stay as params.)

Structural checks on the emitted cpp — **RVV1.0 mf2 / half_lanes=8 two-8-lane-halves**:
- `i8mf2` PRESENT (16×), `f32m2` accumulators, two `vfmv_v_f_f32m2` seeds, two
  `vse32_v_f32m2` stores, `vwmacc_vx_i16m1` lane-wise accumulate, stride `* 288`
  (block_q4_0x16), **NO** `redsum` (block-as-lane, no cross-lane wall).
- Whole-LMUL `f32m4` / `i8m1` / `vfmv_v_f_f32m4` spellings ABSENT — i.e. this is **NOT**
  the dropped RVV0.7 / xtheadvector whole-LMUL m1 regime (the over-claim #12 guard: the
  RVV0.7 reference was the WRONG, null-decode kernel; M1 is grounded entirely on RVV1.0).

## 4. THE LIT TEST (committed)

`test/Conversion/RVV/rvv-emit-identity-quant-contraction-q4-0-repack-vlen128.mlir` — three
RUN lines lower the ABSTRACT op on `march=rv64gcv` through
`--tcrv-rvv-lower-quant-contraction` + `--tcrv-rvv-lower-to-emitc` and FileCheck:
- `CHECK`: the repack-GEMV export signature + repack kernel shape (stride 288, two
  `vfmv_v_f_f32m2`, two `vle8_v_i8mf2`, `vwmacc_vx_i16m1`, two `vse32_v_f32m2`); plus
  `CHECK-NOT` for the abstract op / a leftover repack-as-op / block-dot / unrealized casts.
- `MF2`: the supported RVV1.0 fractional `i8mf2` strip is PRESENT.
- `NOWHOLE-NOT`: the dropped RVV0.7 whole-LMUL `f32m4`/`i8m1` + any `redsum` are ABSENT.

Verified PASS through the real lit harness (1/1; and 6/6 with the C1 selection fixture,
both direct repack-emit fixtures, and the block-dot identity fixture — no regressions).

## 5. HONEST FRAMING (over-claim #12 guard — MANDATORY)

- M1 proves **EMIT-IDENTITY ONLY**: the compiler now AUTO-SELECTS the repack (previously
  the path was hand-chosen by authoring the concrete op in the input IR) and auto-emits
  the byte-identical mf2 RVV1.0 kernel that hand-authoring the repack op produces. This
  closes the **auto-SELECTION** gap noted in [[repack-winA-always-mf2]] — the repack kernel
  was already auto-emitted, but the repack-vs-block-dot SELECTION was hand-done in the IR.
- M1 does **NOT** measure or inherit the 1.22→2.6× decode e2e. That number is the **PRIOR
  mf2/RVV1.0 result** (rvv SG2044 VLEN128, Win-B vs ggml block-dot). A fresh RVV1.0
  measurement = **M2** (`ssh rvv`, NOT done here). Even with identical bytes, the prior
  e2e came through a hand-placed `.inc`-swap integration (= C4) that M1 does not exercise.
- HOST-only, no hardware. The kernel is lit-emitted, NOT run. Grounded entirely on RVV1.0
  (rv64gcv VLEN128); the RVV0.7 / xtheadvector / whole-LMUL-m1 form (the dropped,
  null-decode regime) is explicitly NOT used.
- M1 proves emit-IDENTITY, **not correctness-on-real-weights**: the emitted kernel reads
  block_q4_0x16 (x16) weights per the *asserted* `weight_layout_contract="x16"`, which the
  C3/C4 system layer must MATERIALIZE (load-time x16 store). The abstract op carries PLAIN
  stride-18 weights and has no real producer outside lit, so this is a fixture-only emit —
  no miscompile, but also no claim that the kernel runs correctly on plain weights here.

## 6. ARTIFACTS (all paths absolute under the repo root)

- `.../kernel-coverage/M1-emit-identity/kernelB.mlir` — hand-authored Kernel B source.
- `.../kernel-coverage/M1-emit-identity/kernelA.cpp`, `kernelB.cpp` — the emitted kernels
  (byte-identical; SHA256 above).
- `.../kernel-coverage/M1-emit-identity/A-post-selection.mlir`, `A-emitc.mlir` — Path A
  intermediate MLIR (post-selection + post-lower-to-emitc) for inspection.
- `.../kernel-coverage/M1-emit-identity/run-emit-identity-gate.sh` — reproducible gate
  harness (re-runs both emits + diff; forced from a fresh `tcrv-opt` invocation).
- `test/Conversion/RVV/rvv-emit-identity-quant-contraction-q4-0-repack-vlen128.mlir` — the
  committed emit-identity LIT test.

## 7. BLOCKER

None. Gate empty, lit green, no regressions. M2 (single-cell e2e via the producer,
reproducing the prior 1.22→2.6× on `ssh rvv`) remains the next, hardware-gated milestone —
out of M1 scope.
