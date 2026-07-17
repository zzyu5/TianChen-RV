# RVV i32 binary intrinsic descriptor registry across family and vector shape

## Goal

Make RVV i32 add/sub/mul intrinsic emission a structured, target-owned
descriptor path across arithmetic family and selected vector shape. The
descriptor must be consumed by the RVV plugin/target lowering, emission, and
artifact export path so future RVV i32 growth is not encoded as scattered
string switches or vsub-specific evidence trails.

## Background

The previous task
`05-10-linalg-i32m2-vsub-rvv-scalar-ssh-external-abi-evidence` is complete and
archived at commit `f8fc43d`. This task must not reopen it and must not become
another evidence-only vsub task. Existing vsub i32m2 behavior is a regression
guard; the new module work is descriptor ownership and consumption for the
i32 add/sub/mul family across supported selected vector shapes.

## Boundaries

- Compiler implementation stays in C++ / MLIR / LLVM / TableGen / CMake / lit
  / FileCheck.
- Python may only orchestrate tools, parse artifacts, or run bounded evidence
  checks.
- Keep `tcrv.exec` compute-free.
- Keep RVV-specific descriptor logic under RVV plugin/target ownership.
- Do not add dtype families, floating point, masked/tail policy expansion, or
  performance claims.
- Do not commit generated artifacts, build outputs, raw ssh logs, secrets, or
  transient files.

## Functional Requirements

1. Identify and reuse the current source of truth for i32 add/sub/mul family
   metadata and RVV vector-shape metadata. If `I32BinaryFamilyRegistry` and
   `RVVVectorShape` already exist, extend/connect them instead of duplicating
   them.
2. Add or repair a structured C++ descriptor API under target/RVV ownership.
   It must answer at least:
   - arithmetic family: `i32-vadd`, `i32-vsub`, `i32-vmul`
   - MLIR RVV op spelling or typed op kind
   - generated C arithmetic check expression
   - intrinsic basename/suffix and full intrinsic spelling
   - vector type, vector suffix, and `vsetvl` suffix
   - selected shape capability strings
   - route and component-group names
   - callable ABI names
   - success markers where those are target-export concepts
3. Make the RVV microkernel source/header/object export path consume the
   descriptor for add/sub/mul and the supported selected vector shapes,
   including `i32m1` and `i32m2`.
4. Make the RVV+scalar dispatch bundle/export path consume the same descriptor
   or a direct projection so dispatch artifacts cannot reuse stale vadd/vsub
   identities when `i32-vmul` or `i32m2` is selected.
5. Add a focused front-door proof for marked-linalg `i32-vmul` + `i32m2`
   through the existing plan-and-export bundle path. It must check:
   - descriptor-derived intrinsic spelling `__riscv_vmul_vv_i32m2`
   - selected vector-shape metadata
   - route/component identity
   - external ABI name
   - caller arithmetic check `lhs * rhs`
   - absence of stale vadd/vsub identity
6. Preserve existing focused `i32-vsub` + `i32m2` and `i32-vadd` behavior.

## Acceptance Criteria

- [ ] Descriptor API exists in target/RVV-owned C++ headers/sources and is
      backed by existing family and vector-shape registries where present.
- [ ] RVV microkernel emission/export uses the descriptor for i32 add/sub/mul
      and selected shapes, including `i32m1` and `i32m2`.
- [ ] RVV+scalar dispatch artifact export uses descriptor-derived route,
      callable ABI, operation label, selected shape metadata, and arithmetic
      check content.
- [ ] Focused tests prove `i32-vmul` + `i32m2` through the plan/export bundle
      front door and reject stale vadd/vsub identity.
- [ ] Existing `i32-vsub` + `i32m2` bundle/front-door proof still passes.
- [ ] Relevant build/lit/script checks pass.
- [ ] If real `ssh rvv` is available and runtime artifacts changed, one
      bounded `scripts/rvv_scalar_dispatch_e2e.py` evidence run is recorded.

## Minimal Validation

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Focused C++/lit/FileCheck tests for changed descriptor and export behavior.
- Focused linalg/plan-and-export bundle test for `i32-vmul` + `i32m2`.
- Existing focused `i32-vsub` + `i32m2` bundle/front-door regression test.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if practical after focused checks.

## Out Of Scope

- New high-level linalg generality beyond bounded i32 add/sub/mul patterns.
- New dtype families, floating point, or broader vector policy semantics.
- New `tcrv.exec` compute semantics.
- Extension-specific branches in core orchestration.
- Performance benchmarking or broad smoke matrices.

## Technical Notes

- Required specs are listed in `implement.jsonl` and `check.jsonl`.
- Previous task PRD is included as historical regression context only.
- Source paths to inspect include:
  - `include/TianChenRV/Target/RVV/`
  - `lib/Target/RVV/`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `scripts/rvv_scalar_dispatch_e2e.py`
  - focused tests under `test/Target/`, `test/Transforms/`, and `test/Scripts/`

## Current Status

Completed. The RVV target layer now has a structured
`RVVI32BinaryIntrinsicDescriptor` that composes the existing
`I32BinaryFamilyRegistry` add/sub/mul family descriptors with the existing
finite `RVVVectorShape` i32m1/i32m2 descriptors.

Implementation changes:
- Added `include/TianChenRV/Target/RVV/RVVI32BinaryDescriptor.h`.
- Routed RVV microkernel source/self-check emission through descriptor-derived
  full intrinsic names, shape metadata comments, and arithmetic check
  expressions.
- Routed RVV+scalar dispatch embedded-RVV-source validation and self-check
  arithmetic through the same descriptor projection.
- Added descriptor-level C++ coverage proving `i32-vmul` + `i32m2` derives
  `__riscv_vmul_vv_i32m2`, selected shape metadata, route/component ABI
  identity, and `lhs[index] * rhs[index]`.
- Added focused plan-and-export bundle front-door coverage for marked-linalg
  `i32-vmul` + `i32m2`.
- Extended `scripts/rvv_scalar_dispatch_e2e.py` fixture routing so the runner
  can select the compiler-owned `i32-vmul` + `i32m2` plan-and-export fixture.

Bounded real `ssh rvv` evidence:

```text
python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vmul --vector-shape=i32m2 --run-id codex-i32m2-vmul-ssh-20260510T0001 --overwrite --timeout 120
```

Evidence artifact directory:

```text
artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-i32m2-vmul-ssh-20260510T0001/
```

Evidence result:
- top-level `status = success`, `pass_fail_result = pass`, and
  `runtime_success = true`;
- `ssh_evidence.runtime_success = true`;
- source-built and bundle-object external ABI paths both ran with exit code 0;
- branch coverage records `rvv_available=0` and `rvv_available=1`;
- runtime counts record `7` and `16`;
- selected family is `i32-vmul`;
- selected vector shape is `i32m2`;
- required RVV intrinsic list includes `__riscv_vmul_vv_i32m2`.

Validation completed:
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- focused lit filter:
  `I32BinaryFamilyRegistry|plan-linalg-i32m2-vmul|plan-linalg-i32m2-vsub|rvv-scalar-dispatch-bundle-e2e`
- focused lit filter:
  `rvv-microkernel-family-mul|rvv-microkernel-i32m2-family-sub|rvv-microkernel-i32m2-object|rvv-scalar-i32-vmul-dispatch-generic-route|rvv-scalar-i32-vsub-dispatch-i32m2-generic-route`
- real `ssh rvv` evidence command above
- read-only evidence JSON assertion for runtime success, branch/count coverage,
  selected family/shape, required intrinsic, source/object handoff markers,
  and obvious secret-like text absence
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  (173/173 passed)

Spec update judgment:
- No `.trellis/spec/` update is needed. The existing specs already require
  descriptor-derived i32 binary ABI identity, selected vector-shape metadata,
  plan-and-export bundle coverage, and bounded `ssh rvv` evidence. This task
  implements that contract without changing the long-term architecture rule.
