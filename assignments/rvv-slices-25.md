# RVV Slice Module Backlog

This backlog gives 25 bounded RVV slice modules. Each module is intended to be
large enough for roughly two weeks of focused work and small enough to review as
one PR. The point is to enrich the compiler's typed RVV surface and route
coverage, not to build frontends or workflow tooling.

Every slice must preserve the project route contract:

```text
typed tcrv_rvv body/config
  + explicit runtime ABI values
  + target capability facts
  -> RVV plugin-owned legality and route derivation
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> RVV intrinsic C/C++
```

Do not implement a slice through legacy `tcrv_rvv.i32_*` helpers, `RVVI32M1*`
tables, route-id-driven semantics, source-front-door markers, or common EmitC
branches that decide RVV semantics.

## Common Work Plan

For each module:

1. Find the closest existing pattern in:

```text
include/TianChenRV/Dialect/RVV/IR/RVVOps.td
lib/Dialect/RVV/IR/RVVDialect.cpp
include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h
lib/Plugin/RVV/RVVSelectedBodyRealization.cpp
include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h
lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp
lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp
test/Dialect/RVV/
test/Target/RVV/
```

2. Extend the generic typed body surface only where needed.

3. Add or extend an `RVVSelectedBodyOperationKind`,
   `RVVSelectedBodyMemoryForm`, or route family description only when the
   provider needs a new operation/memory class.

4. Add verifier failures for unsupported dtype, LMUL, policy, operand shape,
   mask source, ABI role, or memory form.

5. Add FileCheck tests that prove the generated C/C++ uses the expected RVV
   type, header, setvl, load/store, and intrinsic family.

6. Add optional local QEMU proof when claiming runtime correctness.

## 1. Compress Store

Feature: support mask-controlled dense packing with `vcompress`.

Likely files:

```text
RVVOps.td
RVVDialect.cpp
RVVSelectedBodyRealization.cpp
RVVEmitCRouteProvider.h
RVVEmitCRoutePlanning.cpp
RVVEmitCRouteProvider.cpp
test/Dialect/RVV/
test/Target/RVV/
```

Required:

- Add a typed body form for `compress(mask, value, vl)`.
- Derive the RVV compress intrinsic from element type, LMUL, policy, and mask
  type.
- Define the output memory rule: compressed lanes are stored contiguously and
  inactive lanes do not write.
- Add negative coverage for masks not produced/imported inside the selected
  body.

Advanced:

- Add a runtime output-count/result boundary when the slice needs to expose the
  number of compressed lanes.
- Add QEMU cases with sparse masks and sentinel-filled output buffers.

## 2. Slide Down

Feature: support `vslidedown` movement inside a vector body.

Required:

- Add generic typed slide op or pre-realized selected-body slice with direction
  `down` and runtime/immediate offset.
- Route to the correct RVV slide-down intrinsic family.
- Preserve explicit policy behavior for filled lanes.

Advanced:

- Support both vector-immediate and vector-scalar offset forms.
- Add QEMU cases where offset is 0, 1, and greater than one hardware VL chunk.

## 3. Slide Up

Feature: support `vslideup` movement inside a vector body.

Required:

- Mirror the slide-down structure with direction `up`.
- Validate lane fill and destination overlap rules.
- Emit the correct `vslideup` intrinsic family.

Advanced:

- Add `slide1up` as a scalar-insert special case.
- Prove inactive/tail lane behavior with sentinels in QEMU.

## 4. Register Gather

Feature: support register-indexed gather with `vrgather`.

Required:

- Add a typed index-vector operand form separate from memory indexed load.
- Derive index vector C type and gather intrinsic from index element type and
  data element type.
- Add verifier coverage for mismatched data/index LMUL or unsupported index
  element type.

Advanced:

- Support immediate gather and scalar-index gather forms.
- Add duplicate-index and out-of-range-index tests.

## 5. Mask Logical Operations

Feature: support mask dataflow operations such as mask and/or/xor/not.

Required:

- Add generic mask ops or extend existing mask-composition support with
  `and`, `or`, `xor`, `not`.
- Route to RVV mask logical intrinsics.
- Allow the result mask to drive select, masked store, masked reduction, or
  compress.

Advanced:

- Add short-circuit-friendly selected-body examples that combine two compares.
- Add negative tests for mixing masks with different element/LMUL config.

## 6. Mask Population And First-Set Queries

Feature: support mask scalar queries such as population count and first-set.

Required:

- Add a selected-body operation that consumes a typed mask and returns/imports a
  scalar runtime result.
- Route to `vcpop` or `vfirst` style intrinsic families.
- Define the runtime ABI boundary for the scalar result.

Advanced:

- Add before-first/including-first mask generation if it fits the same route
  family.
- Add QEMU cases for empty mask, first lane set, and last lane set.

## 7. Unsigned Compare And Select

Feature: complete unsigned integer predicate coverage.

Required:

- Add unsigned predicate kinds such as `ltu`, `leu`, `gtu`, `geu`.
- Derive unsigned compare intrinsic spelling from typed element kind, not from
  route id or test name.
- Reuse the existing select/mask body shape where possible.

Advanced:

- Add cross-check tests where signed and unsigned results intentionally differ.

## 8. Signed Predicate Completion

Feature: fill signed predicate gaps beyond the currently exercised cases.

Required:

- Support signed `ne`, `sgt`, `sge`, and equivalent canonical forms.
- Ensure route planning canonicalizes predicates without changing the typed
  body contract.
- Add negative tests for unknown predicate strings.

Advanced:

- Share predicate canonicalization between compare-select and computed-mask
  store/reduction families.

## 9. Integer Min/Max

Feature: support vector min/max operations.

Required:

- Add generic `binary {kind = min/max/minu/maxu}` or equivalent selected body.
- Route to signed and unsigned min/max intrinsic families.
- Add FileCheck for i32 and one smaller integer width.

Advanced:

- Add masked min/max variants using existing computed-mask route conventions.

## 10. Bitwise Logical Arithmetic

Feature: support integer bitwise `and`, `or`, `xor`.

Required:

- Extend generic `binary {kind}` with bitwise operation kinds.
- Route to RVV bitwise intrinsic families for supported integer dtypes.
- Reject floating element types.

Advanced:

- Add vector-scalar and immediate variants when they share the same operation
  contract cleanly.

## 11. Shift Operations

Feature: support logical/arithmetic shifts.

Required:

- Add shift operation kinds: `sll`, `srl`, `sra`.
- Model shift amount as vector, scalar, or immediate only when the chosen form
  is explicit in the typed body.
- Reject unsupported shift-width and dtype combinations.

Advanced:

- Add masked shift variants and QEMU tests with boundary shift amounts.

## 12. Vector-Scalar VX Arithmetic

Feature: add a generic vector-scalar operand form for arithmetic.

Required:

- Avoid creating operation-specific scalar wrappers. Represent the scalar as an
  explicit runtime ABI value or MLIR scalar operand bound into the body.
- Route add/sub/mul/min/max/bitwise forms through provider-owned VX intrinsic
  derivation.
- Add verifier coverage for scalar dtype mismatch.

Advanced:

- Reuse the same infrastructure for compare and shift scalar operands.

## 13. Immediate VI Arithmetic

Feature: add immediate operand forms where RVV supports VI intrinsics.

Required:

- Add immediate attributes only for operations with valid RVV immediate ranges.
- Verify the immediate range and signedness.
- Route to VI intrinsic forms without using route ids as authority.

Advanced:

- Add canonical fallback diagnostics when an immediate is out of VI range and a
  scalar form should be used instead.

## 14. Reverse Subtract And Negation

Feature: support operand-order-sensitive arithmetic such as reverse subtract
and vector negation.

Required:

- Add explicit operation kind or operand-order attribute.
- Make route planning preserve left/right operand meaning.
- Add tests that would fail if operands are accidentally swapped.

Advanced:

- Add vector-scalar reverse-subtract immediate coverage.

## 15. Widening Add/Subtract

Feature: add widening add/sub operation families.

Required:

- Support at least one source/destination relation such as i16 to i32 or i32 to
  i64.
- Verify source vector type, destination vector type, SEW relation, and LMUL
  relation.
- Route to RVV widening add/sub intrinsic families.

Advanced:

- Add mixed signed/unsigned widening forms if the type contract can represent
  them cleanly.

## 16. Widening Multiply

Feature: add widening multiply without reducing into an accumulator.

Required:

- Add a pure widening multiply body distinct from the existing widening
  dot/reduce and widening MAcc families.
- Verify destination element width is twice source width.
- Route signed, unsigned, or mixed signedness explicitly.

Advanced:

- Add masked widening multiply with inactive lane behavior.

## 17. Narrowing Conversion

Feature: support narrowing conversions such as truncate or narrowing shift.

Required:

- Add a source/destination typed conversion relation.
- Route to RVV narrowing conversion or narrowing shift intrinsic family.
- Verify rounding/saturation/policy attributes if the operation needs them.

Advanced:

- Add saturating/rounding variants as explicit attributes with negative tests.

## 18. F32 Elementwise Arithmetic

Feature: support f32 add/sub/mul/div on the typed RVV surface.

Required:

- Extend dtype legality and C type mapping for f32 vector types.
- Route floating elementwise intrinsics from typed body/config facts.
- Reject integer-only operation kinds for f32.

Advanced:

- Add QEMU tests with NaN/Inf and signed-zero cases when feasible.

## 19. F32 FMA

Feature: support floating fused multiply-add.

Required:

- Add a typed ternary arithmetic body form or selected-body pre-realized FMA
  slice.
- Derive accumulator/result C vector types and FMA intrinsic spelling.
- Verify operand/result element type consistency.

Advanced:

- Add fused negative multiply-add or multiply-subtract variants.

## 20. Floating Compare And Select

Feature: support f32 comparison masks and select.

Required:

- Add floating predicate kinds with explicit ordered/unordered behavior where
  needed.
- Route compare and select through mask type derived from f32 typed body facts.
- Add negative tests that reject integer-only predicates on f32.

Advanced:

- Add QEMU cases for NaN behavior.

## 21. Segment3 Load/Store

Feature: extend segment memory movement from segment2 to segment3.

Required:

- Add segment count 3 route metadata and tuple C type mapping.
- Add field roles for three runtime ABI values.
- Route load/store/extract/insert forms to segment3 RVV intrinsic families.

Advanced:

- Add computed-mask segment3 update if the unmasked segment3 path is stable.

## 22. Segment4 Load/Store

Feature: extend segment memory movement from segment2 to segment4.

Required:

- Mirror the segment3 work with segment count 4.
- Keep field roles explicit; do not encode them in artifact names.
- Add verifier checks for missing or duplicate field ABI roles.

Advanced:

- Add QEMU deinterleave/interleave cases for four fields.

## 23. Whole-Register Load/Store

Feature: support whole-register memory operations.

Required:

- Add a memory form that models whole-register load/store separately from
  ordinary unit-stride vector load/store.
- Route to whole-register RVV intrinsic families where supported by the target
  toolchain.
- Add legality diagnostics if the current compiler route cannot lower the
  chosen element/LMUL form.

Advanced:

- Add multi-register variants with explicit register group count.

## 24. Fault-Only-First Load

Feature: support fault-only-first load for bounded scan-like memory cases.

Required:

- Add a typed memory op or selected-body slice whose result includes both data
  vector and effective VL/update semantics.
- Define how the updated VL or loaded count is represented in the selected
  body/runtime boundary.
- Route to fault-only-first intrinsic families and reject unsupported forms.

Advanced:

- Add QEMU tests with an early stop case when a local QEMU environment can
  model the memory condition reliably.

## 25. Indexed EEW Variants

Feature: extend indexed gather/scatter beyond the current default index width.

Required:

- Make index EEW explicit in the typed index vector/config.
- Route indexed load/store intrinsic spelling from data type, data LMUL, index
  EEW, and memory form.
- Add negative tests for unsupported index/data LMUL combinations.

Advanced:

- Add ordered vs unordered indexed memory variants as explicit policy/config
  when the route provider can distinguish them.

## Review Priority

Prefer modules that fill a missing operation or memory class over modules that
only clone an existing i32/m1 fixture to another spelling. A retained i32
add/sub/mul example is useful only when it is an ordinary instance of the
generic typed RVV surface and the provider derives the route from typed facts.
