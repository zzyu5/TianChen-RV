# Journal - codex (Part 7)

> Continuation from `journal-6.md` (archived at ~2000 lines)
> Started: 2026-05-15

---



## Session 76: Delete support-layer I32 RVV runtime ABI residue

**Date**: 2026-05-15
**Task**: Delete support-layer I32 RVV runtime ABI residue
**Branch**: `main`

### Summary

Deleted Support-owned I32/RVV runtime ABI contracts and helpers; rewired tests/specs to explicit finite-binary ABI primitives and explicit dispatch guard names; focused build/tests/ref-scans passed.

### Main Changes

- Removed `I32BinaryRuntimeABIContract`, `getI32BinaryRuntimeABIContract`, and
  I32 helper wrappers/defaults from Support headers and implementation.
- Removed Support-owned RVV/scalar dispatch runtime C ABI identity strings and
  direct-route labels from Support/Target tests.
- Kept generic finite-binary ABI primitives: explicit caller-owned contracts,
  role binding, mem-window/runtime-param mirror validation, and invocation
  contract formatting.
- Switched dispatch guard materialization to an explicit generic
  `dispatch_available` C name instead of the Support default `rvv_available`.
- Updated lowering/runtime spec to make plugin/target-owned ABI construction
  the only future source of runtime ABI identity.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | chore(support): erase i32 rvv runtime abi residue |

### Testing

- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-runtime-abi-callable-plan-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/DispatchRuntimeGuard Target/RVVScalarDispatch` from `artifacts/tmp/tianchenrv-build/test`
- [OK] Focused ref-scans over Support and directly affected Target tests for deleted I32/RVV ABI authority terms
- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-delete-support-i32-rvv-runtime-abi`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 77: Delete common EmitC source-authority exporter residue

**Date**: 2026-05-15
**Task**: Delete common EmitC source-authority exporter residue
**Branch**: `main`

### Summary

Deleted common Conversion/EmitC C++ source-authority APIs and tests; kept only generic in-memory EmitC materialization; focused build/test/ref-scan passed.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
