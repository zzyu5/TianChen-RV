# Stage2 RVV Pre-Realized Contraction-Dequant Executable Closure

## Goal

Close one bounded executable correctness path for the newly added RVV
plugin-local realization route:

```text
selected tcrv.exec RVV variant with pre-realized typed contraction-dequant body
  -> RVV plugin-local realization
  -> realized signed i8 widening-product / i32 reduction / runtime-scale f32 dequant body
  -> existing RVV route planning and common EmitC materialization
  -> generated header/object bundle
  -> external ABI harness
  -> ssh rvv correctness evidence
```

This task is an executable-closure owner for the production realization path
added by `7d748bff`. It must prove the generated artifact came from the
pre-realized selected-body input path, not only from the older hand-authored
fully realized fixture.

## What I Already Know

* `7d748bff` made the pre-realized contraction-dequant selected body
  route-supported and artifact/header-valid, but explicitly did not claim
  executable correctness through that realization path.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-contraction-dequant-realization/`
  added the bounded pre-realized op surface, plugin-local realization into the
  realized op chain, fail-closed verifier diagnostics, route-family validation,
  and focused route/artifact tests.
* The existing explicit selected-body fixture already proves the fully realized
  chain can reach route planning and artifact validation.
* `tcrv.exec` binds ABI/runtime roles and selected variants; RVV computation
  semantics must live in the selected `tcrv_rvv` body and RVV plugin-owned
  realization/provider code.
* Common EmitC/export must remain neutral: it carries the provider-built route
  payload and must not infer dtype, operation kind, schedule, policy, or
  realization semantics from route ids, artifact names, ABI strings, metadata
  mirrors, or op-kind strings.

## Requirements

* Select or generate one pre-realized selected-body contraction-dequant fixture
  and use it as the primary artifact source for this task.
* Prove materialization exercises the plugin-local realization path before
  emission and realizes the expected typed chain:
  signed i8 input loads, signed widening product, i16 product value, signed
  i32 reduction/carry boundary, runtime f32 scale, f32 dequantized output, and
  f32 output store.
* Bind the external ABI roles for `lhs`, `rhs`, accumulator seed/carry where
  applicable, runtime `scale`, `out`, and `n`/AVL.
* Run dry-run generated-bundle evidence before remote execution.
* Compile and run on `ssh rvv` before claiming executable correctness.
* Compare against a host/reference calculation over multiple counts, signed
  input patterns, nonzero scale values, source and accumulator preservation,
  output tail sentinel preservation, and explicit f32 tolerance.
* If harness support is missing, add only minimal neutral fixture/op selection
  support. Harness changes must not choose RVV semantics from artifact names,
  route ids, ABI strings, metadata mirrors, or op-kind strings.

## Acceptance Criteria

* [x] The task fixture/artifact source is demonstrably pre-realized selected
      body input, not the fully realized fixture.
* [x] Dry-run generated-bundle evidence passes for the pre-realized path.
* [x] The evidence logs or artifact checks prove plugin-local realization ran
      before route planning/emission and produced the expected realized chain.
* [x] `ssh rvv` compile/run correctness passes for the generated
      pre-realized-path artifact.
* [x] The harness/reference covers multiple `n` values, signed input patterns,
      nonzero scale variants, f32 tolerance, source preservation, accumulator
      preservation, and tail sentinel preservation.
* [x] If any production code changes, focused lit/C++ checks cover the touched
      behavior and `tianchenrv-rvv-extension-plugin-test` plus
      `tianchenrv-target-artifact-export-test` pass as relevant.
      No production C++/MLIR code changed in this task; focused lit still
      passed for the pre-realized fixture.
* [x] If the harness script changes, `py_compile` and a focused dry-run/self-test
      pass.
* [x] A bounded old-authority and q-name-authority scan over touched files
      finds no positive legacy i32m1 route authority or q8/q4 route drift.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Trellis task metadata is truthful, the task is finished/archived when
      complete, one coherent commit is created, and the final worktree is clean.

## Definition Of Done

* The PRD matches the bounded executable-closure owner.
* The primary evidence is generated from the pre-realized selected-body path.
* Any implementation needed for this closure is small, neutral, and attached to
  the production path or harness entry point being validated.
* Checks are focused on the changed behavior and rerun after self-repair.
* No runtime/correctness claim is made without `ssh rvv` evidence.

## Out Of Scope

* New RVV route coverage.
* Zero-point, clamp, or saturation expansion.
* New dtype/LMUL batches.
* A second realization family.
* High-level Linalg, Vector, or StableHLO frontend work.
* q8/q4/llama benchmark routes.
* Compatibility wrappers that preserve legacy i32m1 route authority.
* Broad smoke matrices, dashboards, reports-only work, or repeated
  fully-realized-only ssh evidence as the primary deliverable.
* Any common EmitC/export change that chooses RVV compute, dtype, schedule,
  policy, or realization semantics.

## Technical Notes

Read and use these constraints before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/variant-pipeline/index.md`
* `.trellis/spec/testing/index.md`
* `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-contraction-dequant-realization/prd.md`

Primary code/test surfaces to inspect:

* `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`

Initial evidence targets:

* Dry-run generated bundle under
  `artifacts/tmp/06-05-pre-realized-contraction-dequant-executable-closure/`.
* Remote executable run using the same script and artifact root after dry-run
  passes.

## Implementation Summary

This task made one neutral evidence-tooling change in
`scripts/rvv_generated_bundle_abi_e2e.py`: `--pre-realized-selected-body
--op-kind widening_product_reduce_dequantize_f32` now selects
`test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
with selected variant `@pre_realized_body_rvv_product_reduce_dequantize` and
the generated pre-realized symbol
`tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize`.

The change reuses the existing provider-derived ABI, route metadata,
reference-oracle, product-reduction/dequantization boundary summaries, and
f32 tolerance logic. It does not add RVV production route coverage and does not
teach the script to infer semantics from artifact names, route ids, ABI
strings, metadata mirrors, or op-kind strings.

The script self-test now also proves deprecated direct pre-realized route entry
fails closed for `widening_product_reduce_dequantize_f32`, forcing the public
selected lowering-boundary materialization path before bundle export.

No production C++/MLIR/TableGen/CMake code changed.

## Evidence

Dry-run generated bundle:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run \
  --pre-realized-selected-body \
  --artifact-root artifacts/tmp/06-05-pre-realized-contraction-dequant-executable-closure \
  --run-id pre-realized-product-dequant-dry --overwrite \
  --op-kind widening_product_reduce_dequantize_f32 \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 23 --runtime-count 257 \
  --dequant-scale -0.125 --dequant-scale 0.375 \
  --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result: `dry_run_success`.

Evidence path:

```text
artifacts/tmp/06-05-pre-realized-contraction-dequant-executable-closure/pre-realized-product-dequant-dry/
```

The dry-run evidence records:

* `input_mode = "pre-realized-selected-body"`.
* Selected input:
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`.
* Pipeline:
  `tcrv-opt <pre-realized fixture> --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`
  followed by RVV EmitC C++ emission and target artifact bundle export.
* Materialized selected body contains
  `tcrv_rvv.widening_product`, `tcrv_rvv.standalone_reduce`, and
  `tcrv_rvv.dequantize`, with selected variant
  `@pre_realized_body_rvv_product_reduce_dequantize`.

Remote executable evidence:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root artifacts/tmp/06-05-pre-realized-contraction-dequant-executable-closure \
  --run-id pre-realized-product-dequant-ssh-rvv --overwrite \
  --op-kind widening_product_reduce_dequantize_f32 \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 23 --runtime-count 257 \
  --dequant-scale -0.125 --dequant-scale 0.375 \
  --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
```

Result: `success`.

Evidence path:

```text
artifacts/tmp/06-05-pre-realized-contraction-dequant-executable-closure/pre-realized-product-dequant-ssh-rvv/
```

Remote compile/run summary:

* Remote architecture: `riscv64`.
* Remote compiler: `/usr/bin/clang`, Ubuntu clang `18.1.3`.
* `remote_compile_succeeded = true`.
* `remote_run_succeeded = true`.
* Final pass marker:
  `PASS op=widening_product_reduce_dequantize_f32 counts=0,1,7,16,23,257 patterns=0,1 scales=-0.125,0.375 tolerance=1e-05`.

Reference coverage:

* Counts: `0, 1, 7, 16, 23, 257`.
* Signed input patterns: `0, 1`.
* Runtime f32 scale values: `-0.125`, `0.375`.
* Absolute f32 tolerance: `1e-05`.
* Every remote case prints `source_preserved accumulator_preserved tail_preserved`.
* Multi-lane cases include positive and negative signed products and widened
  products, distinguishing the intended signed i8 widening-product reduction
  from add-only, mul-only/no-seed, non-widening, and wrong sign-extension
  behavior.

## Checks Run

* `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* Dry-run generated-bundle command listed above.
* `ssh rvv` generated-bundle compile/run command listed above.
* `rtk proxy bash -lc 'cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32'`
* Bounded old-authority/q-name scan over touched files. The broad scan found
  only PRD non-goal text and the script's pre-existing legacy `--source-seed`
  help text. A diff-only scan found no newly added positive legacy i32m1,
  source-front-door/source-artifact, q8/q4, or llama route authority.
* `rtk git diff --check`
* `rtk git diff --cached --check`

## Spec Update Decision

No `.trellis/spec/` update is needed. This task instantiated existing testing
and RVV plugin contracts: executable runtime claims require `ssh rvv`; selected
pre-realized bodies must pass through plugin-local realization before route
planning; common EmitC/export remains neutral; metadata and names are mirrors,
not authority. The only code change is a bounded harness expectation mapping
for an already specified and implemented route surface.
