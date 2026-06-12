# Stage2 RVV Selected Exec-Envelope ABI Closure

## Goal

Prove one bounded selected `tcrv.exec` RVV envelope carries explicit ABI
organization into an already route-supported typed RVV body without making
`tcrv.exec`, common EmitC, artifact metadata, ABI strings, or fixture names
compute or route authority. The bounded kernel is `scalar_broadcast_macc_add`
because it exercises input buffers, a scalar runtime ABI value, an accumulator
input, an output buffer, and runtime `n`/AVL.

## What I Already Know

- Current HEAD is `8a4c72f9`, with a clean worktree before this task was
  created.
- The previous archived task closed plugin-local selected-body realization for
  `scalar_broadcast_macc_add`: a pre-realized typed body realizes to
  `setvl`/`with_vl`/`load`/`splat`/`load`/`macc`/`store`, reaches provider route
  construction, emits the expected RVV sequence, and has `ssh rvv` correctness
  evidence.
- Existing scalar-broadcast MAcc fixtures bind ABI values directly with
  `tcrv_rvv.runtime_abi_value`, but the selected `tcrv.exec` envelope does not
  yet prove those values are explicitly linked to sibling
  `tcrv.exec.mem_window` / `tcrv.exec.runtime_param` ABI declarations.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` already exist and verify
  generic ABI role metadata; `tcrv_rvv.runtime_abi_value` already verifies role,
  C name/type, ownership, and result type.
- `RuntimeABIMemWindow` and `RuntimeABIParam` support helpers exist, but
  `RuntimeABIMemWindow` is not currently consumed by the RVV route path.

## Requirements

- Add a bounded, structural link from each selected RVV
  `tcrv_rvv.runtime_abi_value` to a direct same-kernel `tcrv.exec.mem_window`
  or `tcrv.exec.runtime_param` symbol.
- Keep the link as ABI/envelope provenance only. It must not define operation
  kind, dtype, SEW/LMUL, memory form, accumulator layout, scalar broadcast
  semantics, intrinsic spelling, route support, or correctness.
- For buffer ABI roles, the link must resolve to a `tcrv.exec.mem_window` with
  matching ABI role, C type, ownership, purpose, binding, memory space, and
  read/write access.
- For scalar/control ABI roles such as `rhs_scalar` and `n`, the link must
  resolve to a `tcrv.exec.runtime_param` with matching ABI role, C name, C type,
  ownership, and scalar purpose.
- Add an opt-in selected-variant requirement so the bounded
  `scalar_broadcast_macc_add` fixture fails closed if any consumed
  `tcrv_rvv.runtime_abi_value` lacks the exec-envelope binding.
- Preserve existing RVV route authority: typed RVV body/config/runtime facts
  are consumed by RVV realization/planning/provider; common EmitC materializes
  only the provider-built `TCRVEmitCLowerableRoute`; target artifact metadata is
  mirror-only.
- Surface a mirror-only exec ABI binding summary in generated RVV artifact
  metadata/header records for the bounded route.

## Acceptance Criteria

- A focused positive `scalar_broadcast_macc_add` selected-body artifact shows
  kernel-level `tcrv.exec.mem_window` / `tcrv.exec.runtime_param` declarations,
  `tcrv_rvv.runtime_abi_value exec_binding` links, realized typed RVV body
  structure, provider route metadata, and emitted header signature.
- Direct pre-realized route-entry evidence for `scalar_broadcast_macc_add`
  still consumes the pre-realized body, then reaches the provider route and
  generated bundle path with the same exec ABI binding mirror.
- Negative coverage fails closed when a selected RVV variant requires exec ABI
  bindings but a consumed runtime ABI value has no `exec_binding`.
- Negative coverage fails closed when an `exec_binding` points to the wrong
  exec op kind, wrong ABI role, wrong C type/name, or wrong ownership.
- Generated bundle dry-run evidence checks the exec ABI binding mirror, runtime
  ABI order, route operand binding, provider-supported mirror label, absence of
  descriptor/direct-C/source-export residue, and emitted callable signature.
- If executable behavior is claimed, run real `ssh rvv` generated-bundle
  compile/run evidence for at least three runtime counts including a tail case;
  otherwise report the exact blocker and make no runtime correctness claim.
- Run focused lit/script checks, `git diff --check`, an authority scan over
  touched exec/RVV/provider/target/script/fixture files, and `check-tianchenrv`
  or document the exact blocker.

## Non-Goals

- No new RVV operation families, dtype/LMUL matrix expansion, high-level
  Linalg/Vector/StableHLO frontend lowering, source-front-door positive route,
  global autotuning, dashboards, broad dispatch matrices, or one-op-per-
  intrinsic wrappers.
- No legacy `i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, `rvv-i32m1` route
  authority, descriptor-driven computation, direct-C/source-export route, or
  artifact-name/test-name route authority.
- No movement of RVV dtype/config/schedule/intrinsic choices into
  `tcrv.exec`, common EmitC, or target artifact plumbing.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-selected-body-realization-closure/prd.md`.
- Likely implementation files:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  selected scalar-broadcast MAcc fixtures under `test/Target/RVV`, and
  generated-bundle script tests under `test/Scripts`.
