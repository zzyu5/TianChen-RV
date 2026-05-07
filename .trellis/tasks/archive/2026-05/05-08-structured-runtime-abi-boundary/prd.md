# structured runtime ABI parameter boundary

## Goal

Add a minimal C++ structured representation for runtime-callable C ABI parameters used by the existing bounded i32-vadd RVV, scalar, and RVV+scalar dispatch C source exports, so object/link/run integration can later consume typed parameter roles instead of comments or string hints.

## Requirements

* Extend existing target/emission data structures rather than adding a parallel registry.
* Record C parameter name, C type spelling, semantic role, and whether each parameter is IR-modeled or target/export ABI-owned.
* Bound roles to the current i32-vadd slice: lhs input buffer, rhs input buffer, output buffer, and runtime element count.
* For current RVV and scalar bounded microkernel exports, mark buffer pointers and runtime n as target/export ABI-owned unless real MLIR IR operands/attributes/SSA values/region arguments exist.
* Derive generated C comments/checks from structured metadata where comments are emitted.
* Keep RVV/scalar details plugin-local or target-owned and keep target-neutral checks target-neutral.
* Add focused tests for RVV i32-vadd export, scalar fallback export, and RVV+scalar dispatch export metadata.
* Add a negative/coherence test only if it directly validates the structured ABI boundary.

## Out of Scope

* No object generation, linking, runtime execution, ssh rvv compile/run, remote probes, broad smoke fixtures, or evidence packaging.
* No Python implementation of compiler/runtime ABI decisions.
* No generic compute ops in `tcrv.exec`.
* No broad target selection logic or new target-family branches in core orchestration.
* No conflation of descriptor-local `element_count` with high-level tensor shape or runtime AVL/n.
* No RVV runtime/correctness/performance claim.

## Acceptance Criteria

* Existing bounded RVV, scalar, and RVV+scalar dispatch C source export routes populate structured ABI parameter metadata.
* Supported artifact routes missing required ABI parameter roles are rejected or surfaced by a focused coherence/preflight check.
* Focused lit/FileCheck and/or C++ tests prove the metadata.
* `git diff --check`, CMake configure, and `check-tianchenrv` pass locally.
* Worktree is clean after one coherent commit.

## Technical Notes

* Primary implementation stack remains C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck.
* Python is allowed only for scripts/utilities, not core ABI/compiler decisions.
