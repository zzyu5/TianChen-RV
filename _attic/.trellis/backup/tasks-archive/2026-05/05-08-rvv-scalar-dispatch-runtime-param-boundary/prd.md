# runtime param boundary for RVV scalar dispatch

## Goal

Make the bounded RVV+scalar i32-vadd dispatch path consume real `tcrv.exec`
runtime-parameter IR for scalar runtime ABI values before emission/export claims
those values are IR-modeled. This continues the previous mem-window boundary work:
lhs/rhs/out buffer meanings are already backed by `mem_window`; this task adds
IR-backed meanings for runtime element count `n` and explicit dispatch
availability guard `rvv_available`.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Mandatory precondition passed: worktree is clean, supervisor-policy files are
  clean, and HEAD is `8435a26 feat: back RVV scalar dispatch ABI buffers with
  mem windows`.
* The previous round made buffer ABI meanings real execution IR via
  `tcrv.exec.mem_window`, without making new RVV runtime/correctness/performance
  claims.
* Remaining compiler gap: scalar runtime ABI values for `n` / AVL and
  `rvv_available` are still likely represented mainly through metadata/export
  defaults rather than an execution IR boundary.
* User requires one serial worker, no subagents, no parallel agents, and no
  Python implementation of compiler IR/modeling/decisions.

## Requirements

* Inventory whether an execution-layer runtime scalar parameter op, attribute,
  or equivalent already exists.
* If absent, add the smallest compute-free ODS operation or structured attribute
  under `tcrv.exec` to express runtime ABI scalar values for execution
  organization.
* Required bounded roles:
  * `runtime-element-count` for C parameter `n`.
  * `dispatch-availability-guard` for C parameter `rvv_available`.
* Runtime parameter representation may describe ABI role/name/type/ownership and
  purpose/kind, but must not encode computation, high-level tensor shape, RVV
  vector math, hardware probing, or target-family legality.
* Core verifier behavior must stay target-neutral: non-empty single-line string
  metadata, duplicate-role rejection in the same direct kernel scope when export
  would be ambiguous, and simple C identifier checking if implemented
  generically.
* Planning/lowering/emission materialization must preserve or emit runtime
  parameter IR alongside existing mem-window IR for the bounded RVV+scalar
  i32-vadd path.
* RVV+scalar target export must validate and consume runtime parameter IR for
  `n` and `rvv_available`, failing closed on missing, duplicate, unexpected, or
  inconsistent metadata.
* Generated source metadata should make IR-backed consumption visible, for
  example with `dispatch_runtime_param[...]` comments. Comments are evidence,
  not source of truth.
* Preserve parameter layering:
  * VLEN/vlenb are hardware facts / target capability data.
  * SEW/LMUL/policy are compile-time variant or RVV dialect configuration.
  * runtime `n`, AVL/vl flow, and dispatch availability are runtime
    SSA/control/ABI values.
  * lhs/rhs/out buffer meanings remain backed by `mem_window`.
  * `element_count` remains descriptor-local bounded fixture metadata.
  * `required_march`/`selected_march` remain bounded compile/toolchain metadata
    for this slice.

## Out Of Scope

* Broad ABI subsystem.
* Tensor shape model.
* Dynamic runtime integration.
* Automatic hardware probing or performance path.
* New evidence bridge or test-harness-only work.
* Generic compute operations inside `tcrv.exec`.
* Target-family/vendor branches in core dialect verification or generic routing.
* Hand-authored C fixture as proof of compiler behavior.

## Acceptance Criteria

* Positive lit/FileCheck or C++ coverage proves runtime parameter IR
  parses/prints/verifies for the two bounded roles.
* Execution-planning pipeline emits or preserves runtime parameters for the
  bounded RVV+scalar dispatch path.
* Emission/readiness or target export diagnostics prove runtime ABI scalar
  values are backed by real IR rather than detached comments/exporter defaults.
* RVV+scalar source export still produces the expected dispatcher signature and
  uses IR-backed `n` and `rvv_available` roles.
* Negative tests cover missing/duplicate `runtime-element-count`, missing or
  duplicate `dispatch-availability-guard`, and stale/inconsistent
  `c_name`/`c_type`/`ownership`/`role` metadata with clear diagnostics.
* Existing source/object/executable evidence tests continue to pass.
* Local checks run: `git diff --check`, `cmake --build build --target tcrv-opt
  tcrv-translate -j2`, `cmake --build build --target check-tianchenrv -j2`,
  plus focused tests for changed exec/planning/export behavior.

## Definition Of Done

* Implementation remains C++/MLIR/TableGen/CMake/lit/FileCheck.
* Python, if used, is limited to allowed evidence orchestration.
* Trellis task validates and is archived before the final compiler commit.
* Worktree is clean at the end.
* One coherent compiler commit is created for runtime-parameter boundary code,
  focused tests, and necessary spec/doc updates.
