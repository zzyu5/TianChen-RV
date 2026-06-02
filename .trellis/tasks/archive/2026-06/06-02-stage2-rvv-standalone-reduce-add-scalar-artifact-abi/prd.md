# Stage2 RVV standalone reduce-add scalar artifact ABI boundary

## Goal

Close one bounded Stage 2 RVV end-to-end artifact/runtime ABI boundary for
exactly the plain `standalone_reduce_add` selected-body path. The task proves
that a selected `tcrv.exec` RVV variant with a typed
`typed_standalone_reduce_pre_realized_body` is realized by RVV plugin-local
owners before route construction, that scalar seed/result ownership is carried
structurally through the standalone reduction provider facts, and that the
generated target artifact ABI executes correctly on real `ssh rvv`.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV standalone reduce-add scalar artifact ABI boundary`

## What I Already Know

* The repository starts clean from commit `667a08bd rvv: prove runtime scalar
  masked macc m2 abi`.
* There was no active `.trellis/.current-task`, so this task was created from
  the Hermes direction brief before editing source files.
* The relevant specs require the chain:
  selected RVV variant -> typed/realized `tcrv_rvv` body -> RVV plugin
  legality/realization/provider -> `TCRVEmitCLowerableRoute` -> common EmitC
  -> target artifact -> `ssh rvv` evidence for runtime claims.
* `.trellis/spec/extension-plugins/rvv-plugin.md` already defines the
  standalone reduction scalar channel boundary. Source/work vector facts and
  scalar accumulator/result facts must be distinct provider-derived route
  facts; route ids, artifact names, ABI strings, exact intrinsic spellings,
  and common EmitC cannot decide the boundary.
* Existing positive files already cover:
  `test/Target/RVV/pre-realized-selected-body-artifact-standalone-reduce-add.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-standalone-reduce-add.mlir`,
  dry-run script tests, direct pre-realized fail-closed script tests, and
  target artifact validator positive/negative scaffolding.
* The active plain standalone reduction operand binding summary still uses the
  old `runtime-abi-mirror|...|header-mirror` spelling. It also does not mark
  `lhs` and `acc` as exported header/prototype participants even though the
  generated prototype exports `lhs`, `acc`, `out`, and `n`.
* Computed-mask/runtime-scalar standalone reduction and recent MAcc routes
  already use compact `abi|...|hdr` route operand binding facts. This task
  should bring the plain standalone reduction path into the same provider
  binding contract.

## Requirements

* Scope exactly one production-positive behavior:
  plain `standalone_reduce_add` selected body through generated artifact and
  runtime ABI evidence.
* The selected pre-realized body must be consumed by the RVV selected-body
  realization owner before route construction. Direct pre-realized route-entry
  support must remain fail-closed.
* The provider-owned binding summary for `lhs`, `acc`, `out`, and `n` must be
  derived from the route operand binding plan and must mark every exported
  prototype parameter with `abi` and `hdr` participation.
* The route and target artifact validator must preserve and validate:
  runtime ABI order `lhs,acc,out,n`, scalar accumulator seed boundary,
  `standalone_reduce` add operation, runtime AVL/VL chunk carry, lane-0 scalar
  result layout, output store, SEW/LMUL/policy facts, standalone reduction
  route-family plan, provider-supported mirror, C type mapping, required
  `riscv_vector.h` header, and scalar-result runtime boundary.
* Common EmitC/export may only carry provider-built route facts. It must not
  infer standalone reduction semantics, ABI roles, dtype/config, scalar
  seed/result ownership, or intrinsic mapping.
* The generated-bundle harness must run counts `0`, `1`, one VL-boundary count,
  one tail count, and one larger count with at least two seed/input patterns.
  It must prove seed contribution, multi-VL accumulation, final scalar output,
  source/seed preservation, runtime `n` behavior, and no unintended writes
  beyond the scalar result contract.
* Any stale or missing provider mirror, binding plan, ABI order, scalar
  seed/result layout, runtime VL carry, type mapping, header fact, route-family
  fact, or direct pre-realized shortcut in the touched boundary must fail
  closed before artifact acceptance.

## Acceptance Criteria

* [x] The plain standalone reduction provider binding summary uses compact
      provider-derived entries with `abi|...|hdr` for `lhs`, `acc`, `out`, and
      `n`, and the generated diagnostics/header metadata mirror the exact
      provider summary.
* [x] `TargetArtifactExportTest` or equivalent focused C++ validation proves
      the target artifact consumer rejects stale/missing standalone reduction
      binding/header facts before bundle acceptance.
* [x] The explicit and pre-realized standalone reduce-add target fixtures check
      selected-body realization, route-family/provider facts, header/prototype
      facts, scalar seed/result layout, runtime AVL/VL carry, required headers,
      C type mapping, provider-supported mirror, and route operand binding
      summary.
* [x] Generated-bundle dry-run tests for explicit and pre-realized
      `standalone_reduce_add` check representative evidence JSON, generated
      C/C++, and harness facts for `lhs,acc,out,n`, seed/store behavior,
      runtime counts, two seed/input patterns, required headers, route
      operand binding, and direct pre-realized route-entry unsupported status.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts `0`, `1`,
      `16`, `17`, and `257` with at least two seed/input patterns, proving
      scalar seed contribution, multi-VL accumulation, lane-0 scalar output,
      source/seed preservation, runtime `n`, and tail/output sentinel
      preservation.
* [x] A bounded old-authority scan over touched files finds no new positive
      reliance on legacy `i32m1`, descriptors, source-front-door positive
      routes, direct-C/source-export compute, exact-intrinsic authority, or
      common EmitC RVV semantic inference.
* [x] Focused build/test/script commands pass, the task is finished/archived,
      one coherent commit is created, and `git status --short` is clean.

## Definition Of Done

* PRD, implement/check context, and journal entries are truthful.
* The production/default provider/validator path is updated where required;
  this is not closed by adding helper-only evidence.
* Runtime correctness is claimed only with real `ssh rvv` evidence.
* No computed-mask reduction, runtime-scalar masked reduction, min/max scope
  expansion, new dtype, broad reduction framework, frontend lowering, source
  front door, or tuning/performance database work is introduced.

## Technical Approach

1. Replace the plain standalone reduction operand binding summary with compact
   provider tokens that include `abi` and `hdr` for every exported runtime ABI
   parameter.
2. Update provider operand-binding fact checks to require the new header
   markers, and update target artifact expected summaries to consume the exact
   new provider summary.
3. Update MLIR fixtures and generated-bundle script FileCheck expectations to
   check the new binding summary and already-required scalar reduction facts.
4. Add or strengthen focused target artifact stale-binding/header negative
   tests for plain `standalone_reduce_add`.
5. Run focused MLIR/FileCheck, generated-bundle dry-run, script self-test,
   target artifact C++ test, real `ssh rvv` evidence, authority scan, Trellis
   validation, and `git diff --check`.

## Out Of Scope

* Computed-mask standalone reduction, runtime-scalar masked standalone
  reduction, min/max behavior changes, widening dot reduction, strided input,
  LMUL m2 as a new proof target, new dtypes, broad reduction frameworks,
  frontend lowering, source-front-door positive routes, or tuning/performance
  databases.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, test names, C strings,
  descriptors, exact intrinsic spellings, or mirror metadata as standalone
  reduction semantics or scalar ABI authority.

## Technical Notes

Specs and prior evidence read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-masked-macc-lmul-m2-artifact-abi/prd.md`

Likely implementation/test files:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-standalone-reduce-add.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-standalone-reduce-add.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-standalone-reduce-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-standalone-reduce-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-standalone-reduce-add-fail-closed.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Completion Evidence

Implemented behavior:

* Plain standalone reduction route operand binding now uses compact
  provider-derived `abi|...|hdr` entries for every exported runtime ABI
  parameter: `lhs`, `acc`, `out`, and `n`.
* Provider-side math operand binding facts now require the standalone
  reduction `lhs` header marker, `acc` seed/state/header markers, `out`
  state/store/header markers, and runtime `n` `setvl-avl|loop|hdr` markers
  before `TCRVEmitCLowerableRoute` construction.
* Target artifact validation consumes the new exact standalone reduction
  binding summary and `TargetArtifactExportTest` rejects missing/stale
  provider and candidate binding/header mirrors.
* The generated-bundle standalone reduction harness now runs two signed input
  patterns and two scalar seeds, checks source preservation, seed preservation,
  scalar `out[0]`, and non-scalar output sentinels.
* Explicit/pre-realized standalone reduce-add dry-run tests check the new
  binding summary, source pattern contract, runtime counts, and harness marker
  `seeds=-11,17 patterns=0,1`.

Focused checks run:

* `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `rtk ./build/bin/tianchenrv-target-artifact-export-test`
* Direct `tcrv-opt` / `tcrv-translate` / `FileCheck-20` checks for explicit and pre-realized `standalone_reduce_add`.
* Direct FileCheck loop over changed standalone add/min/max Target/RVV fixtures.
* Explicit, pre-realized, and pre-realized LMUL m2 generated-bundle dry-run checks for standalone reduce-add.
* Direct pre-realized route-entry fail-closed script check for `standalone_reduce_add` and `standalone_reduce_add_lmul_m2`.
* Standalone reduce min/max dry-run generation and FileCheck over changed shared golden expectations.
* Real `ssh rvv` generated-bundle compile/run for explicit and pre-realized `standalone_reduce_add`, counts `0,1,16,17,257`, seeds `-11,17`, patterns `0,1`.
* `rtk git diff --check`
* Bounded added-line authority scan over touched files.

Runtime evidence:

* Explicit artifact:
  `artifacts/tmp/06-02-standalone-reduce-add-ssh-rvv/explicit/standalone_reduce_add/evidence.json`
* Pre-realized artifact:
  `artifacts/tmp/06-02-standalone-reduce-add-ssh-rvv/pre-realized/standalone_reduce_add/evidence.json`
* PASS marker:
  `PASS op=standalone_reduce_add counts=0,1,16,17,257 seeds=-11,17 patterns=0,1`
* Remote compile/run: success for both explicit and pre-realized bundles.

Spec update review:

* No `.trellis/spec/` update was required. Existing RVV plugin, EmitC route,
  and testing specs already require provider-owned standalone reduction
  scalar channel facts, `hdr` participation for exported route operands,
  fail-closed target artifact validation, and real `ssh rvv` evidence for
  runtime correctness claims.

## Current Phase

Finish/archive.
