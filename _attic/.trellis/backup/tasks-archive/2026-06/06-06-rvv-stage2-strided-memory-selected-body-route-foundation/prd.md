# Stage2 RVV strided memory selected-body route foundation

## Goal

Establish one bounded Stage 2 route-supported typed RVV strided memory
movement proof route. A selected `tcrv.exec` RVV variant must carry either an
explicit typed `tcrv_rvv` body or a pre-realized selected body where one
runtime byte-strided side is moved through explicit typed RVV structure:

```text
strided_load_unit_store:
  out[i] = *(src_raw + i * stride_bytes)

unit_load_strided_store:
  *(dst_raw + i * dst_stride_bytes) = src[i]
```

The route must flow through RVV plugin-owned selected-body realization and
route planning, provider-built `TCRVEmitCLowerableRoute`, Common EmitC neutral
materialization, target artifact validation, and generated artifact evidence.
Executable `ssh rvv` evidence is in scope only if the generated-bundle path is
already reachable without expanding into a broad memory matrix.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV strided memory selected-body route repair/foundation`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `4c186b1a rvv: close typed widening conversion route foundation`.
- No `.trellis/.current-task` existed at the start of the round, so this task
  was created before source edits.
- Specs require this route to derive stride, memory form, element dtype,
  SEW/LMUL/policy, runtime ABI order, header/type/intrinsic mirrors, and
  provider support from selected typed `tcrv_rvv` body/config/runtime facts.
  Common EmitC/export may only carry provider-built route payloads and mirrors.
- The previous widening conversion task reported full-suite failures in:
  - `Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir`
  - `Target/RVV/pre-realized-selected-body-artifact-unit-load-strided-store.mlir`
- Focused lit reproduction in `build/test` fails only in the `HEADER`
  FileCheck pass for both strided memory fixtures. The exported headers do
  carry `tcrv_rvv.strided_memory_layout` and the source/destination stride
  source mirror, but current fixture expectations check those fields after
  `tcrv_rvv.c_type_mapping` while the target header emits them earlier as part
  of provider-derived strided memory evidence.
- Current source inspection shows production surfaces for this route family:
  `typed_strided_memory_pre_realized_body`,
  `typed_strided_store_memory_pre_realized_body`, explicit `strided_load`,
  `strided_store`, `move`, base memory selected-body realization owner, base
  memory route-family plan owners, provider preflight, target artifact route
  family validation, target header bundle mirrors, C++ target artifact
  negative tests, and generated-bundle dry-run script tests.
- Existing C++ target artifact coverage already rejects stale base memory
  route-family plan, provider support mirror, runtime ABI order, source memory
  form, destination memory form, source stride mirror, route operand binding,
  target profile, header/type facts, and non-base-family stale residue.

## Requirements

- Keep scope to one base strided memory movement boundary:
  `strided_load_unit_store` and its paired destination-strided store shape
  `unit_load_strided_store`.
- Preserve the authority chain:
  selected typed `tcrv_rvv` body/config/runtime facts -> RVV selected-body
  realization/validation -> RVV route-family provider facts -> lowerable route
  -> Common EmitC/export -> target artifact mirrors -> optional generated
  bundle runtime evidence.
- The provider/target route path must structurally carry or derive:
  - ABI roles and order `src,out,n,stride_bytes` for strided load and
    `src,dst,n,dst_stride_bytes` for strided store;
  - source/destination stride role, stride unit, and byte-stride source;
  - source/destination memory form and strided memory layout;
  - element type, SEW, LMUL, policy, vector C type mapping, and target leaf
    profile;
  - VL loop control and runtime AVL/VL plan;
  - route operand-binding plan/summary with exported header/prototype
    participation markers;
  - required headers, provider support mirror, and target header mirrors.
- Common EmitC/export may only carry provider-built route payloads and
  metadata mirrors. It must not infer stride, memory form, dtype, SEW/LMUL,
  runtime ABI roles, intrinsic spelling, or route support.
- Stale or missing facts must fail closed before provider route construction
  or target artifact acceptance: stride role/source, memory form/layout,
  runtime ABI order, operand binding, dtype/config, selected operation kind,
  required header/type mapping, target leaf profile, and
  `provider_supported_mirror`.
- Resolve the known header/FileCheck drift by aligning fixture evidence to the
  provider/target route truth. Do not count a standalone expectation tweak as
  sufficient unless focused provider/target validation proves the route facts
  are already production-derived and fail closed.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` describe this bounded
      strided memory route foundation and cite the relevant specs.
- [x] Focused code inspection identifies whether production provider/route/
      target/generated-bundle support is complete or where it fails.
- [x] The known header drift in the two pre-realized strided memory target
      fixtures is reproduced and resolved.
- [x] Positive route/artifact/header evidence proves provider-derived runtime
      ABI order, stride role/source, memory form/layout, element dtype,
      SEW/LMUL/policy, route operand binding, required headers/type mapping,
      target leaf profile, and `provider_supported_mirror`.
- [x] Negative evidence proves stale or missing stride source, memory form,
      runtime ABI/order, route operand binding, selected operation kind,
      header/type mirrors, and provider support mirror fail closed before
      target artifact acceptance.
- [x] Common EmitC/export remains neutral and does not choose strided memory
      semantics.
- [x] Focused `tcrv-opt` / `tcrv-translate` / RVV provider / target artifact
      checks for changed behavior pass.
- [x] If executable behavior is claimed, generated artifact plus real
      `ssh rvv` correctness passes over counts including `0`, `1`, a VL
      boundary, a tail case, and a multi-chunk case, with scalar oracle
      coverage and tail/sentinel preservation across at least two runtime byte
      strides.
- [x] If executable closure is too large, finish route-supported plus
      target-validation closure and record generated-bundle `ssh rvv` as the
      exact next continuation point. Not applicable: executable closure passed
      for explicit and pre-realized strided load/store generated-bundle modes.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, dtype-prefixed
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor, source-front-door,
      source-artifact, route-string/artifact-name/ABI-string/test-name,
      exact-intrinsic-spelling, or Common EmitC semantic authority.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      final clean worktree, archive, and one coherent commit complete the round
      if the module behavior is complete.

## Evidence Plan

- Run focused lit reproduction for the two known failing target fixtures from
  `build/test`.
- Inspect provider/target route-family validation for base strided memory
  facts before deciding whether production code or fixture evidence needs
  repair.
- If production support is already real, repair the header-order drift and add
  focused stale mirror checks in the strided memory target fixtures.
- Run generated-bundle dry-run for explicit and pre-realized strided load/store
  modes if the route remains reachable.
- Run real `ssh rvv` generated-bundle correctness for strided load/store only
  if dry-run and harness coverage already support the required count/stride
  matrix.

## Definition Of Done

- Current HEAD either has route-supported typed strided memory provider and
  target artifact validation closure, or records the precise production
  blocker and next continuation point without false executable claims.
- Any implementation changes are production-path changes or focused evidence
  for already-existing production support; helper/test/spec-only work is not a
  completion unless production support was already real.
- Specs are updated only if this round discovers a durable rule not already
  captured in `.trellis/spec/`.
- The task is finished/archived and one coherent commit is created when the
  module behavior is complete.

## Out Of Scope

- Broad memory matrices beyond the bounded base strided memory movement route.
- Gather/scatter expansion unless needed only as stale-residue rejection for
  this route.
- High-level Linalg, Vector, StableHLO, or source-front-door lowering.
- Reduction, MAcc, contraction, compare/select, segment2, computed-mask memory,
  dtype/LMUL clone batches, IME, Offload, performance benchmarking, autotuning,
  or dashboards.
- One-intrinsic wrapper routes.
- Descriptor-driven computation or Common EmitC semantic inference.
- Treating route ids, artifact names, ABI strings, test names, intrinsic
  spellings, status fields, manifests, or metadata mirrors as route authority.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/index.md`
- `.trellis/spec/guides/capability-first-design-guide.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant archived task read:

- `.trellis/tasks/archive/2026-06/06-06-06-06-rvv-stage2-typed-widening-conversion-route-foundation/prd.md`
- `.trellis/tasks/archive/2026-06/06-06-06-06-rvv-stage2-typed-widening-conversion-route-foundation/check.jsonl`

Primary inspection surfaces from the direction brief:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/`
- `lib/Plugin/RVV/Construction/`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/*strided*memory*.mlir`
- `test/Target/RVV/*strided-load-unit-store*.mlir`
- `test/Target/RVV/*unit-load-strided-store*.mlir`
- `test/Target/TargetArtifactExportTest.cpp`
- `test/Scripts/rvv-generated-bundle-abi-e2e-*strided*dry-run.test`

## Completion Evidence

Completed as a bounded Stage 2 base strided memory selected-body route repair
and evidence closure. Production support was already real in the inspected
provider and target path: the RVV selected-body realization owner consumes the
pre-realized strided load/store bodies into `setvl`, `with_vl`,
`strided_load` or `strided_store`, `move`, and unit-stride load/store
structure; the base memory route-family provider derives the runtime ABI,
stride, memory-form, layout, route operand binding, header/type, and provider
support facts; target artifact validation rejects stale mirrors before header
acceptance.

Focused fixture changes:

- `test/Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir`
  now treats target header metadata order as non-authority while still checking
  provider-derived `memory_form`, `strided_memory_layout`,
  `source_stride_source`, source/destination memory forms, route operand
  binding, headers/type mapping, target leaf profile, and
  `provider_supported_mirror`.
- `test/Target/RVV/pre-realized-selected-body-artifact-unit-load-strided-store.mlir`
  now checks the same provider-derived target header facts for destination
  byte-strided store.
- Both fixtures now mutate stale target-artifact mirrors for runtime ABI order,
  route operand binding, provider support mirror, strided memory layout, and
  source/destination stride source, proving target export fails closed instead
  of accepting stale metadata.

Provider and target route facts proved:

- Operations: `strided_load_unit_store` and `unit_load_strided_store`.
- Typed compute op: `tcrv_rvv.move`.
- Runtime ABI order: `src,out,n,stride_bytes` for strided load;
  `src,dst,n,dst_stride_bytes` for strided store.
- Memory layout: `byte-strided-source-unit-stride-output-runtime-abi` for
  strided load; `unit-stride-source-byte-strided-destination-runtime-abi` for
  strided store.
- Stride sources: `runtime_abi:stride_bytes` and
  `runtime_abi:dst_stride_bytes`.
- Target mirrors: `rvv-base-memory-movement-route-family-plan.v1`, route
  operand binding plans/summaries, `stddef.h,stdint.h,riscv_vector.h`, route
  type mappings, route-local runtime AVL/VL plan, target leaf profiles, and
  `provider_supported_mirror`.

Generated artifact evidence:

- Explicit strided load dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-explicit-strided-load-unit-store-dryrun`
- Pre-realized strided load dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-pre-realized-strided-load-unit-store-dryrun`
- Explicit strided store dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-explicit-unit-load-strided-store-dryrun`
- Pre-realized strided store dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-pre-realized-unit-load-strided-store-dryrun`

Real RVV executable evidence:

```text
PASS op=strided_load_unit_store counts=0,1,16,23,257 stride_bytes=4,8 source_preserved
PASS op=strided_load_unit_store counts=0,1,16,23,257 stride_bytes=4,8 source_preserved
PASS op=unit_load_strided_store counts=0,1,16,23,257 stride_bytes=4,8 source_preserved
PASS op=unit_load_strided_store counts=0,1,16,23,257 stride_bytes=4,8 source_preserved
```

The first load/store pair used pre-realized selected-body input; the second
load/store pair used explicit selected-body input. Strided load evidence
checked byte-strided source reads, contiguous output, source preservation, and
tail preservation. Strided store evidence checked selected destination writes,
sentinel preservation for skipped destination bytes, tail preservation, and
source preservation.

Self-repair:

- The initial focused lit reproduction used the correct `build/test` working
  directory after direct repo-root lit invocation could not discover the test
  suite.
- Generated-bundle dry-run and `ssh rvv` checks were run with
  `--llvm-readobj ''` because `llvm-readobj` is not installed on this host;
  this skips only optional local object header/symbol inspection.

Focused checks:

- Focused lit reproduction initially failed for both strided memory target
  fixtures at the `HEADER` FileCheck pass.
- Focused lit for both repaired target fixtures passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Focused lit filter for `pre-realized-strided-memory-negative`, generated
  bundle strided load/store dry-run tests, and direct pre-realized route-entry
  fail-closed tests passed.
- Four generated-bundle dry-runs passed for explicit/pre-realized strided
  load/store.
- Four real `ssh rvv` generated-bundle runs passed for explicit/pre-realized
  strided load/store with counts `0,1,16,23,257` and stride bytes `4,8`.
- `ninja -C build check-tianchenrv` completed 491/494 passing. The two prior
  strided memory failures are resolved; remaining failures are unrelated
  existing tests:
  `Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-negative.mlir`,
  `Plugin/construction-protocol-common.test`, and
  `Plugin/template-extension-plugin.test`.
- `git diff --check` passed.
- Trellis task context validation passed.

Spec update judgment:

- No `.trellis/spec/` update was needed. The existing RVV plugin, EmitC route,
  and MLIR testing contracts already require provider-derived typed
  `tcrv_rvv` facts, Common EmitC neutrality, target artifact fail-closed
  mirror validation, and runtime `ssh rvv` evidence for executable claims. This
  round implemented focused fixture/evidence closure under those existing
  contracts rather than introducing a new durable rule.

Old-authority scan:

- Added-line scan over the touched strided memory target fixtures produced no
  matches for legacy `RVVI32M1`, `rvv-i32m1`, dtype-prefixed
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-artifact,
  descriptor, direct-C/source-export, exact intrinsic spelling, or Common
  EmitC semantic authority patterns.
