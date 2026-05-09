# RVV i32-vadd microkernel exporter body hardening

## Goal

Harden the selected RVV i32-vadd microkernel export route so the target-owned C
intrinsic emission steps are derived from the verified `tcrv_rvv` typed
dataflow body, not from route assumptions alone.

## Scope

- Keep the slice bounded to `tcrv_rvv.i32_vadd_microkernel`.
- Preserve the route:
  selected `tcrv.exec.variant` -> RVV plugin legality/lowering boundary ->
  `tcrv_rvv.setvl` / `with_vl` / `i32_load` / `i32_add` / `i32_store` ->
  target-owned C source/header/object export using `riscv_vector.h`.
- Do not add generic intrinsic IR, generic vector lowering, new `tcrv.exec`
  compute ops, or offload/IME/Sophgo work.

## Acceptance

- The RVV exporter builds its load/add/store emission plan from the actual
  typed `tcrv_rvv` body order, SSA chain, and `buffer_role` attributes.
- Positive lit coverage proves the selected RVV route reaches C export and
  emits `riscv_vector.h` plus the expected RVV C intrinsics.
- At most one focused negative check covers the enforced typed-dataflow route
  invariant.
- Required local checks pass:
  `git diff --check`, `cmake --build build --target tcrv-opt`,
  `cmake --build build --target tcrv-translate`, and
  `cmake --build build --target check-tianchenrv`.
- No RVV runtime/correctness/performance claim is made without real `ssh rvv`
  compile/link/run evidence.
