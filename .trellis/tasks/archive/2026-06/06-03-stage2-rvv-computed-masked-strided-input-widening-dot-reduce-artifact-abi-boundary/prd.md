# Stage2 RVV computed-masked strided-input widening dot-reduce artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV compiler path fail-closed and executable:

```text
selected tcrv.exec RVV variant
  -> typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / compare-source loads /
     i16mf2 strided source loads / compare-produced mask /
     masked_widening_dot_reduce / lane0 scalar result store
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation
  -> generated bundle/header/object
  -> ssh rvv correctness evidence
```

Route authority must remain the typed/realized `tcrv_rvv` body and RVV
plugin/provider facts. Route ids, artifact names, metadata mirrors, exact
intrinsic spellings, descriptors, C strings, scripts, runtime counts, test
names, direct route-entry support, or source-front-door markers are
mirrors/evidence only.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-masked strided-input widening dot-reduce artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no entries.
* Initial `git log --oneline -8` started at
  `b2bc90c5 rvv: validate runtime scalar dual cmp select artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require selected typed RVV
  body -> RVV plugin-local realization/provider facts -> common EmitC ->
  target artifact -> real `ssh rvv` evidence for runtime/correctness claims.
* The archived runtime-scalar dual-cmp mask-and-select task provides the
  previous completed production pattern: provider-owned facts, target
  validation consumption, generated-bundle dry-run evidence, real `ssh rvv`
  execution, task archive, and one coherent commit.
* Current repository evidence already names this route family in ODS, provider
  planning, target validation, generated-bundle scripts, explicit/pre-realized
  fixtures, and target artifact tests. This task is therefore a bounded
  production hardening/evidence task, not a broad dot/reduction/mask/stride
  feature matrix.

## Requirements

* Support exactly
  `computed_masked_strided_input_widening_dot_reduce_add`.
* Keep support rooted in selected typed/pre-realized `tcrv_rvv` body,
  RVV plugin-local realization, compare-produced mask facts, strided source
  load facts, masked widening dot reduction facts, route-family plan,
  route operand binding facts, provider-built route facts, and target
  validator consumption.
* Validate runtime ABI order:
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`.
* Validate predicate and mask facts:
  `slt`, `predicate-mask-produced-by-compare`,
  `compare-produced-mask-same-vl-scope`, and `compare-produced-mask`.
* Validate source facts:
  i16 source SEW, `mf2` source LMUL, runtime element strides from
  `lhs_stride` and `rhs_stride`, source memory form `strided-load`, and
  destination memory form `unit-stride-store`.
* Validate accumulator/result facts:
  i32 accumulator/result SEW, `m1` accumulator/result LMUL,
  `signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32`,
  `scalar-i32-seed-lane0-from-accumulator-input`,
  `store-dot-reduction-lane0-to-output-scalar`, and inactive-lane zeroing
  before reduction.
* Provider route description must carry route operand binding plan/summary,
  contraction route-family plan, runtime AVL/VL contract, required headers,
  C type mapping, target leaf/profile facts, provider target validation facts,
  and explicit `provider_supported_mirror` or equivalent mirror labeling.
* Route operand binding summary must represent every generated header/prototype
  ABI parameter as:
  `<logical>=<role>:<c-name>:abi|...|hdr`.
* Keep route binding metadata within the target bounded single-line metadata
  contract by shortening only the provider plan label or use-token spellings;
  do not drop logical operands, role/C-name facts, `abi`, `hdr`, stride,
  compare, dot, accumulator, result, or loop tokens.
* Target artifact validation must consume provider-owned facts for this route
  family and reject stale candidate/provider mirrors before bundle acceptance.
* Unsupported or stale combinations must fail closed with targeted diagnostics
  rather than falling back to names, strings, descriptors, artifact metadata,
  intrinsic spelling authority, or common EmitC RVV semantic inference.
* Generated-bundle dry-run evidence must record provider-derived route facts,
  binding order, stride roles, source/accumulator/result SEW-LMUL, predicate
  and mask facts, dot relation, inactive-lane zeroing, route-family plan,
  header/type facts, capability/profile facts, target validator consumption,
  provider mirror, and harness coverage contract.
* Runtime correctness must use real `ssh rvv` generated-bundle evidence over
  counts `0,1,16,17,257`, stride pairs `2:3` and `3:2`, and at least two
  mask/input patterns proving masked lanes skipped, strided source indexing
  honored, accumulator seed added, scalar lane0 output stored, source/tail
  sentinels preserved, and runtime `n`/AVL honored.

## Acceptance Criteria

* [x] Focused production diff strengthens the RVV plugin/provider and target
      validator fact surface for
      `computed_masked_strided_input_widening_dot_reduce_add`; no
      metadata-only closeout.
* [x] Provider-owned computed-mask strided-input widening dot-reduce facts are
      consumed by target validation rather than relying on target-local stale
      route truth.
* [x] Route operand binding plan/summary includes all nine ABI parameters with
      role, C name, `abi`, required use tokens, and `hdr`, while staying under
      the bounded target metadata value limit.
* [x] Selected-body realization/FileCheck proves pre-realized body consumption
      into setvl/with_vl/compare loads/compare/strided source loads/
      masked_widening_dot_reduce/store.
* [x] Target artifact tests fail closed for stale or missing provider mirror,
      target leaf/profile, ABI order/roles, binding plan/summary, stride
      roles, source/accumulator/result SEW-LMUL, predicate/mask facts, dot
      relation, inactive-lane zeroing, route-family plan, header/type facts,
      direct route-entry residue, and stale non-strided or non-masked dot
      facts.
* [x] Existing or tightened REALIZED/PLAN/HEADER checks for the pre-realized
      and explicit selected-body fixtures pass.
* [x] Generated-bundle dry-run records provider-derived boundary facts,
      complete route operand binding, target validator consumption, target
      leaf/profile, headers/types, `provider_supported_mirror`, strided source
      indexing, masked dot relation, inactive-lane zeroing, and runtime AVL/VL
      facts.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,16,17,257`, stride pairs `2:3` and `3:2`, and patterns `0,1`.
* [x] Smallest relevant build/test/script commands, direct FileCheck
      equivalents if lit is unavailable, `git diff --check`, and a bounded
      old-authority scan over touched files pass.
* [x] Trellis task is finished/archived and one coherent commit is created if
      the task completes.

## Completion Evidence

Completed on 2026-06-03 by continuing the existing in-progress dirty diff.

* Strengthened the RVV math operand-binding fact path so computed-mask strided
  widening dot compare inputs, dot source inputs, and stride inputs all require
  exported header/prototype participation through `hdr`.
* Updated the explicit selected-body artifact and generated-bundle dry-run
  expectations to mirror the complete nine-parameter binding summary.
* Added focused C++ coverage proving computed-mask strided widening dot math
  binding facts expose compare, strided dot payload, stride, and scalar seed
  roles, and fail closed when an exported stride parameter loses `hdr`.
* Confirmed the existing target artifact validator consumes the provider-owned
  widening-dot route description and exact candidate mirrors for binding plan,
  binding operands, provider support, target leaf/profile, headers/types,
  mask facts, strided source facts, and dot-reduction facts.
* `trellis-update-spec` decision: no spec change needed. The existing
  `emitc-route.md` operand binding summary contract and
  `mlir-testing-contract.md` computed-mask widening dot-reduce evidence
  contract already require this `hdr` and exact provider/target mirror behavior.

Checks run:

* `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-strided-input-widening-dot-reduce` from `build/test` (`5` passed, `472` excluded)
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --ssh-target rvv --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20`
* `git diff --check`
* Bounded touched-diff old-authority scan for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, exact `__riscv_*_i32m1`, descriptor/direct-C/source-export,
  source-front-door, and source-artifact residue.

Real RVV evidence:

```text
tcrv_rvv_generated_bundle_abi_computed_masked_strided_input_widening_dot_reduce_add_ok counts=0,1,16,17,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2
PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,1,16,17,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2
```

## Out Of Scope

* Broad dot/reduction/mask/stride matrix, unsigned/dtype/LMUL clone batch,
  high-level Linalg/Vector frontend, source-front-door positive route,
  one-intrinsic wrapper dialect, report-only commit, common EmitC RVV
  semantics, dual-cmp select redo, widening MAcc redo, computed MAcc redo,
  reductions, segment/indexed memory, dashboards, descriptor routes,
  direct-C/source exporters, or direct route-entry positive support.
* Treating exact intrinsic spellings, route ids, artifact names, metadata
  mirrors, generated C strings, scripts, runtime count values, test names,
  fixture names, or harness constants as the source of predicate, mask, stride,
  accumulator/result layout, dtype/config, runtime ABI, policy, route support,
  or evidence authority.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-runtime-scalar-dual-cmp-mask-and-select-artifact-abi-boundary/prd.md`

Repository files to inspect while deriving implementation:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`
* Matching dry-run and direct pre-realized fail-closed script tests if present
* `test/Target/TargetArtifactExportTest.cpp`
