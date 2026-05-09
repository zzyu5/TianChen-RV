# i32-vmul RVV scalar dispatch bundle ssh rvv evidence

## Goal

Carry descriptor-backed `i32-vmul` from bounded linalg/frontend planning through
the existing RVV-primary plus scalar-fallback dispatch bundle path, then capture
fresh `ssh rvv` runtime correctness evidence for both explicit dispatch guard
cases.

## Why Now

HEAD `dce6999` already completed the standalone descriptor-backed `i32-vmul`
artifact path: registry facts, RVV/scalar plugin materialization, bounded
frontend `arith.muli` lowering, and standalone RVV/scalar source artifacts are
present. The remaining module-sized bottleneck is dispatch participation:
`I32BinaryFamilyRegistry` already defines vmul dispatch route facts, but the
active RVV+scalar dispatch exporter and evidence runner still only recognize
add/sub dispatch families. This task closes that gap without widening compiler
semantics or claiming performance.

## Requirements

- Extend RVV+scalar dispatch target/export code so `i32-vmul` is accepted as a
  descriptor-backed dispatch family wherever add/sub dispatch families are
  accepted.
- Preserve distinct vmul route ids, ABI names, component group, function stems,
  header guards, self-check marker, RVV intrinsic, scalar operator, and emitted
  multiply semantics.
- Register vmul dispatch source/header/object composite routes through the
  target artifact exporter registry.
- Add direct vmul dispatch self-check source/object translate routes only as
  needed for the evidence path.
- Extend `scripts/rvv_scalar_dispatch_e2e.py` so
  `--arithmetic-family=i32-vmul` can generate, validate, and consume vmul
  dispatch artifacts through the existing compiler pipeline and bundle index
  metadata.
- Add a bounded linalg/frontend-to-dispatch fixture proving an `arith.muli`
  body reaches the vmul RVV+scalar dispatch source/header/object bundle path.
- Add local tests for vmul dispatch route distinctness, stale family mismatch
  rejection, dry-run evidence tooling, and add/sub regressions touched by this
  migration.
- Run the vmul dispatch bundle workflow on `ssh rvv`, compiling/linking/running
  generated artifacts on the RVV host and recording concise evidence under
  `artifacts/tmp/...`.

## Acceptance Criteria

- [x] `i32-vmul` dispatch source/header/object routes are registered and visible
      in focused target/export tests, with route ids and ABI names distinct from
      add/sub.
- [x] Generic target source/header/object or bundle export for a planned vmul
      dispatch path emits multiply semantics:
      `__riscv_vmul_vv_i32m1` and `out[index] = lhs[index] * rhs[index]`.
- [x] Direct vmul dispatch self-check route emits
      `tcrv_rvv_scalar_i32_vmul_dispatch_self_check_ok` and checks
      `lhs[index] * rhs[index]`.
- [x] Bounded frontend linalg input with `tcrv_frontend_lowering = "i32-vmul"`
      and `arith.muli` reaches RVV+scalar dispatch bundle export through
      `--tcrv-plan-and-export-target-artifact-bundle`.
- [x] `scripts/rvv_scalar_dispatch_e2e.py --self-test` covers vmul family
      selection, bundle external caller multiply semantics, and stale-family
      validation behavior.
- [x] Focused dry-run script tests cover `--arithmetic-family=i32-vmul` for
      direct self-check and bundle modes without making runtime claims.
- [x] Focused add/sub dispatch regression tests touched by this migration still
      pass.
- [x] `git diff --check` passes.
- [x] CMake configure with repository LLVM/MLIR paths passes.
- [x] Focused lit/C++/script checks for the changed behavior pass.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes unless a narrower blocker is recorded truthfully.
- [x] Fresh `ssh rvv` evidence for vmul dispatch bundle records remote compile
      commands, selected march/mabi, generated source/header/object/caller
      artifact names, source-built path result, bundle-object path result,
      stdout showing scalar and RVV guard cases, and pass/fail summary.
- [x] Trellis task validation/finish/archive and workspace journal update are
      completed if the module is finished.
- [x] One coherent git commit is created if the module is complete.

## Completion Evidence

- Local build root: `artifacts/tmp/tianchenrv-build`
- Remote evidence root:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/vmul-dispatch-ssh-rvv-20260509`
- Evidence JSON:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/vmul-dispatch-ssh-rvv-20260509/evidence.json`
- Remote target: `ssh rvv`
- Selected flags: `-O2 -march=rv64gcv -mabi=lp64d`
- Remote host facts: `riscv64`, `/usr/bin/clang`, Ubuntu clang 18.1.3,
  Linux `ubuntu` 6.12.23 on riscv64.
- Generated bundle artifacts:
  - source:
    `artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vmul-dispatch-c.c`
  - header:
    `artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vmul-dispatch-header.h`
  - object:
    `artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vmul-dispatch-object.o`
  - caller: `rvv_bundle_dispatch_external_caller.c`
- Bundle metadata used route
  `tcrv-export-rvv-scalar-i32-vmul-dispatch-c`, component group
  `rvv-scalar-i32-vmul-dispatch-external-abi.v1`, and external ABI name
  `rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1`.
- Source-built path: compile/link/run exit codes all `0`; stdout marker
  `tcrv_rvv_scalar_i32_vmul_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`
  observed.
- Bundle-object path: link/run exit codes all `0`; the same stdout marker was
  observed.
- Evidence result: `status = success`, `pass_fail_result = pass`.

## Scope Boundary

This task is limited to the already implemented finite i32 add/sub/vmul family
surface. It may generalize local add/sub dispatch helper code to consume the
descriptor registry, but it must not move compiler semantics into Python or add
generic arithmetic/compute interpretation to `tcrv.exec`.

Parameter layering remains explicit:

- hardware facts and selected march/mabi stay in capability/toolchain evidence;
- compile-time family facts stay descriptor-backed;
- runtime `n` and `rvv_available` stay runtime ABI/control values;
- descriptor-local element counts stay plugin/export-local metadata.

## Non-goals

- No performance benchmarking or ratio claims.
- No arithmetic families beyond `i32-vadd`, `i32-vsub`, and `i32-vmul`.
- No i64/e64, masks, widening/narrowing, new RVV policy families, dynamic-shape
  frontend expansion, StableHLO/TOSA lowering, or generic RVV lowering.
- No new `tcrv.exec` compute ops.
- No extension-specific semantic branches in generic core passes.
- No Python implementation of compiler registry, plugin proposal, lowering,
  emission, route selection, runtime ABI decisions, or source generation.
- No docs-only, smoke-only, wrapper-only, report-only, metadata-only, or
  helper-only closeout.

## Technical Notes

- Repo root: `/home/kingdom/phdworks/TianchenRV`
- Starting HEAD: `dce6999 feat(rvv): add descriptor-backed i32 vmul artifacts`
- Starting worktree: clean
- No open Trellis task existed before this task was created.
- Prior PRDs read:
  - `.trellis/tasks/archive/2026-05/05-09-descriptor-backed-i32-vmul-standalone-artifact-path/prd.md`
  - `.trellis/tasks/archive/2026-05/05-09-rvv-scalar-vsub-dispatch-ssh-rvv-runtime-evidence/prd.md`
- Current code finding:
  - `include/TianChenRV/Target/I32BinaryFamilyRegistry.h` already contains vmul
    dispatch descriptor facts.
  - `lib/Target/Builtin/RVVScalarDispatch.cpp` still matches/registers only
    add/sub dispatch callable pairs.
  - `tools/tcrv-translate/tcrv-translate.cpp` has vadd source/header/object
    direct routes and add/sub self-check routes, but no vmul dispatch
    self-check route.
  - `scripts/rvv_scalar_dispatch_e2e.py` supports add/sub families only.

## Continuation Rule If Unfinished

Keep this task open and record the exact incomplete layer: dispatch routes,
translate registration, script workflow, local lit checks, remote compile,
remote link, scalar guard run, RVV guard run, evidence capture, task archive, or
commit. Do not archive or claim vmul runtime correctness until fresh `ssh rvv`
compile/link/run evidence passes for both guard paths.
