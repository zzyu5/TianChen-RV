# RVV setvl control-plane IR

## Goal

Add the first plugin-local RVV runtime vector-length control-plane IR surface:
a bounded `tcrv_rvv.setvl` operation that models runtime AVL to VL without
adding RVV arithmetic, memory operations, generic lowering, runtime ABI glue,
or hardware evidence claims.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial repository inspection showed clean worktree at `c1d2019 chore: codify RVV parameter flow policy`.
* The three supervisor-policy files were clean at the start and are out of
  scope for this compiler commit.
* Primary implementation stack is C++ / MLIR / LLVM / TableGen / ODS / CMake /
  lit / FileCheck.
* Python is allowed only for runner, supervisor, probe, artifact parsing,
  fixture generation, and small support utilities.
* `tcrv.exec` must remain compute-free and focused on execution, capability,
  variant, dispatch, and fallback organization.
* `tcrv_rvv` owns RVV plugin-local semantics. The current dialect already has
  `!tcrv_rvv.vl`, `#tcrv_rvv.policy`, `tcrv_rvv.lowering_boundary`, and
  `tcrv_rvv.i32_vadd_microkernel`.

## Requirements

* Add a bounded ODS op `tcrv_rvv.setvl`.
* The op must take one runtime AVL SSA operand, preferably `index`.
* The op must return one `!tcrv_rvv.vl`.
* The op must carry bounded compile-time first-slice config: SEW `32`, LMUL
  `m1`, and `#tcrv_rvv.policy`.
* The op must not accept VLEN/vlenb, element count, required march,
  required capabilities, pointer/memory operands, arithmetic semantics, or
  target capability facts.
* Add plugin-local C++ verifier diagnostics for runtime AVL/value layering,
  result type, bounded SEW/LMUL/policy config, and known forbidden layering
  mistakes.
* Update the RVV dialect and VL type descriptions so they describe `setvl` as
  the first bounded runtime VL control-plane surface rather than saying the
  dialect cannot model vsetvl at all.
* Update the minimum relevant spec text, especially
  `.trellis/spec/extension-plugins/rvv-plugin.md`, without duplicating the
  broad parameter policy from `c1d2019`.
* Add focused lit/FileCheck coverage under `test/Dialect/RVV`.

## Acceptance Criteria

* Positive lit coverage shows parsing/printing or round-trip of runtime AVL to
  `!tcrv_rvv.vl` with explicit SEW/LMUL/policy metadata.
* Negative lit coverage checks missing/non-runtime AVL representation, unsupported
  SEW/LMUL, and forbidden `vlen`/`vlenb`, `element_count`, or `required_march`
  attributes.
* Existing RVV dialect and microkernel tests continue to pass.
* Required validation commands pass:
  * `git diff --check`
  * `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
  * `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
* No `ssh rvv` run is required or claimed for this compiler-only IR surface.

## Out Of Scope

* RVV arithmetic ops, load/store ops, reductions, masks, or generic vector
  lowering framework.
* `tcrv_rvv.with_vl`, unless `setvl` cannot be represented or tested coherently
  without it.
* RVV C exporter, scalar exporter, host dispatch exporter, target artifact
  routes, smoke probes, dashboards, or evidence packaging.
* Expanding `required_march` string-comparison dependence.
* Any RVV runtime, correctness, hardware execution, or performance claim.

## Technical Notes

* `VLEN`/`vlenb` remain hardware facts / target capability evidence.
* `SEW`/`LMUL` remain compile-time variant configuration for the selected slice.
* `AVL` and `vl` remain runtime SSA/control values.
* `element_count` remains descriptor-local for bounded fixture/export slices.
* This task intentionally returns to real MLIR/C++/ODS compiler progress after
  the previous policy-only round.
