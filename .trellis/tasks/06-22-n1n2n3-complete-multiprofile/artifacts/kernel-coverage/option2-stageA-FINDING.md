# Option-2 STAGE A — abstract quant-contraction op + IDENTITY-default lowering (FINDING, 2026-06-24)

Implements stage A of `option2-stageA-abstract-op-DESIGN.md`: an ABSTRACT,
algorithm-UNCOMMITTED quantized-contraction op + an IDENTITY-DEFAULT lowering
pass that rewrites it to today's concrete block-dot op, BYTE-EXACT, ZERO behavior
change. LIT-only (no hardware). The lead commits; this agent did NOT commit.

**Verdict: STAGE A PROVEN behavior-neutral.** Existing-set emission byte-identical
before vs after (forced/clean rebuilds, both halves); new abstract fixture emits
byte-identical C to the hand-authored block-dot; 705/708 lit pass (the 3 fails are
a PRE-EXISTING Python e2e self-test failure, confirmed at clean HEAD, unrelated).

---

## A1 — the abstract op + fail-closed verifier

- **Op:** `GgmlQuantContractionOp` (`tcrv_rvv.quant_contraction`),
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:3975`, modeled on
  `GgmlBlockDotQ40Q80Op` (:3873) + `GgmlRepackGemvQ40Q80Op` (:4266). Carries
  `[TCRVEmitCLowerableOpInterface]` (both interface methods have default impls →
  no method bodies required; it is NOT added to `kBlockDotKernels[]` — lowered
  before the EmitC pass). Operands: `weight_base, activation_base, output,
  element_count, column_count(nc), vl` — **nc carried ALWAYS** (so the later
  repack branch can reach it). Attrs: `quant, scale_model, m_regime, qk,
  weight_layout, weight_block_stride, activation_block_stride, quant_byte_offset,
  activation_high_byte_offset, OptionalAttr min_vlen`. Deliberately OMITS the
  repack-only `weight_interleave / half_lanes / x16` facts (stage C).
- **Verifier:** `GgmlQuantContractionOp::verify()`,
  `RVVDialectWideningOps.cpp:1058`. Fail-closed allow-list (`:1069`) +
  **`weight_layout` PINNED `"plain"` fail-closed (`:1101`)** — rejects `"x16"`
  and any repack-only attr (`half_lanes` rejected by the allow-list). Pins
  `quant=="q4_0"`, `scale_model`, `m_regime∈{decode,prefill}`, and the q4_0 plain
  byte facts IDENTICALLY to the block-dot verifier (qk=32, stride 18/34, offset
  2/16). `min_vlen` advisory-only (never gates correctness).

## A2 — the identity lowering pass (registered, abstract→block-dot, nc-dropped)

- **Pass:** `RVVLowerQuantContraction` (`--tcrv-rvv-lower-quant-contraction`),
  declared `Passes.td:876`, constructor `Passes.h`, registered in `tcrv-opt.cpp`
  next to `createRVVLowerToEmitCPass`. Impl
  `lib/Conversion/RVV/RVVLowerQuantContraction.cpp` (`createRVVLowerQuantContractionPass`
  :122), CMake-wired in `lib/Conversion/RVV/CMakeLists.txt`.
- **Identity branch = block-dot** (`lowerToBlockDot`, :85):
  `module.walk` → `OpBuilder.create<GgmlBlockDotQ40Q80Op>` with operands
  `(weight_base, activation_base, output, element_count, vl)` — **DROPPING
  column_count (nc)** (block-dot delegates M/N to ggml's mul_mat; the bare
  4-operand vec_dot) — and attrs reconstructed verbatim
  (`kind="ggml_q4_0_q8_0_block_dot"`, scale_model, qk=32, strides 18/34, offsets
  2/16), with the three schedule knobs passed nullptr (left for
  MaterializeRVVQ40Schedule downstream, exactly as today). `replaceAllUsesWith` +
  `erase`.
- **identity gate = `quant=="q4_0" && m_regime=="decode"`** (DESIGN §2.2);
  **every other cell = ERROR STUB:** a non-decode (prefill/GEMM) or non-q4_0
  request `emitError`s ("wires ONLY the q4_0 decode block-dot identity branch;
  the (quant, m_regime) cell is the stage-C/repack lowering target") — the repack
  lowering is stage C; never reached on the wired q4_0-decode path. (A q4_0+prefill
  op therefore fails fail-closed rather than silently taking block-dot.)
- **Structural no-op when no abstract op exists** (proven below): running the
  pass on the hand-authored block-dot fixture leaves emission byte-identical.

## A3 — one wired fixture (q4_0-decode block-dot path only)

`test/Conversion/RVV/rvv-to-emitc-quant-contraction-q4-0-block-dot.mlir`, RUN:
`tcrv-opt %s --tcrv-rvv-lower-quant-contraction --tcrv-rvv-lower-to-emitc | FileCheck %s`.
Authors the abstract op; CHECK lines are the SAME as the hand-authored block-dot
test. The hand-authored `rvv-to-emitc-q4-0-q8-0-block-dot.mlir` is UNTOUCHED.

- **Signature-preserving nc stand-in (load-bearing detail):** every
  `runtime_abi_value` in the variant body becomes a C function parameter (in
  declaration order; `RVVToEmitC.cpp:133-143`). The fixture therefore keeps the
  EXACT runtime-ABI set of the hand-authored test (`n, s, bs, vx, bx, vy, by,
  nrc`) and binds the abstract op's `column_count` to the existing `bs`
  (output-stride) index value — a **signature-preserving stand-in**, NOT a real
  column count. nc is DROPPED by the block-dot lowering, so its only role here is
  to satisfy the nc-carrying contract WITHOUT adding a parameter that would
  change the emitted signature. (Stage B must bind a REAL nc; it must not inherit
  this `bs` reuse.)
- Also added a fail-closed verifier test `test/Dialect/RVV/quant-contraction-dataflow.mlir`
  (`--split-input-file --verify-diagnostics`): accepts plain q4_0-decode + advisory
  min_vlen; REJECTS `weight_layout="x16"`, the repack-only `half_lanes` attr, a
  wrong `m_regime`, and the repacked 288 stride. FileCheck PASS.
- **Check-agent addition:** `test/Conversion/RVV/rvv-lower-quant-contraction-prefill-stub.mlir`
  (`--tcrv-rvv-lower-quant-contraction --verify-diagnostics`) covers the
  verifier-vs-PASS split the dataflow test misses: the verifier ACCEPTS q4_0+prefill
  (a valid abstract request) but the PASS error-stub REJECTS it fail-closed (it is the
  repack/GEMM cell, not wired in A) rather than silently lowering to block-dot.
  Reproduced empirically: verifier accepts prefill; pass emits the stage-A stub error
  and fails (non-zero) — confirmed NOT a silent miscompile.

## A4 — byte-exact identity proof (forced/clean rebuilds, before/after equality)

Per MEMORY (`build-incremental-unreliable`): incremental builds are unsound for
fingerprint gates → used a `git stash -u` before/after protocol with CONFIRMED
compile+link both halves. **The recorded absolutes `f810ce6b`/`cb04b219` are
STALE** (they are the older/smaller block-dot glob; the 06-22→06-24 coverage
commits grew the `*block-dot*` glob 36→40, so a set hash cannot match — the
DESIGN docs copy them forward as planning text, not re-measurements). The SOUND
gate is before/after EQUALITY of same-build-state measurements (MEMORY-endorsed).

Fingerprint = `cksum` over the sorted concatenation of each fixture's
`--tcrv-rvv-lower-to-emitc` output (the canonical text the f810ce6b/cb04b219
fingerprints hash, per MEMORY "for --tcrv-rvv-lower-to-emitc").

1. **Identity (new == hand-authored):** the new abstract fixture's emitted EmitC
   is **byte-identical** to the hand-authored block-dot's:
   `cksum 87607893, 11847 bytes` for BOTH. (`diff` empty.)
2. **Existing set unchanged (zero bytes added on every existing path):** emitted
   the SAME existing file lists (my new fixture excluded) on a clean HEAD build
   (BEFORE) and a clean build WITH all changes (AFTER); `diff -r` EMPTY on both:
   - block-dot/gemm glob (40 existing files): `cksum 1440756685, 5465094 bytes`
     BEFORE == AFTER.
   - full `test/Conversion/RVV/*.mlir` (107 existing files):
     `cksum 1294733381, 7259519 bytes` BEFORE == AFTER.
   (The new pass only fires on a `quant_contraction` op; no existing fixture has
   one, and the new op adds no instances to existing fixtures — so the existing
   emission is provably untouched.)
3. **Structural no-op confirmed:** the hand-authored block-dot fixture run
   THROUGH `--tcrv-rvv-lower-quant-contraction` then `--tcrv-rvv-lower-to-emitc`
   is byte-identical (`cksum 87607893`) to without the new pass.
4. **lit:** 705/708 pass, including both new fixtures and the entire
   Conversion/RVV + Dialect/RVV suites. The 3 failures
   (`Scripts/rvv-generated-bundle-abi-e2e-*`) are a PRE-EXISTING Python e2e
   self-test `AssertionError` (fake-bundle dequant metadata) — **confirmed
   identical at clean HEAD with all changes stashed**; no causal path to the
   MLIR op/pass/fixture changes (zero Python touched).

## ZERO behavior change + boundary

Stage A is pure scaffolding: NO runtime behavior, NO perf claim, NO selection
logic, NO repack lowering, NO weight-packing, does NOT touch
`low_precision_resource`. It is the foundation that makes "the compiler selects
the contraction algorithm" expressible in-compiler (stage B adds the
selection-logic + repack branch — which is WHY A carries nc; stage C adds the
plain→x16 weight materialization). The abstract op + identity pass produce
EXACTLY today's emit, byte-for-byte.

**Blocker:** none for stage A. (Deferred, B/C: the repack branch is
layout-incompatible from plain bytes without the stage-C plain→x16
materialization — sidestepped here by the plain→plain identity.)

## Files

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (+73) — op def :3975
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (+157) — verifier :1058
- `include/TianChenRV/Transforms/Passes.td` (+37) — pass decl :876
- `include/TianChenRV/Transforms/Passes.h` (+1) — constructor decl
- `lib/Conversion/RVV/RVVLowerQuantContraction.cpp` (NEW) — identity pass
- `lib/Conversion/RVV/CMakeLists.txt` (+1)
- `tools/tcrv-opt/tcrv-opt.cpp` (+3) — pass registration
- `test/Conversion/RVV/rvv-to-emitc-quant-contraction-q4-0-block-dot.mlir` (NEW) — A3 fixture
- `test/Dialect/RVV/quant-contraction-dataflow.mlir` (NEW) — fail-closed verifier test
