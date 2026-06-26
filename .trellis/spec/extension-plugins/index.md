# Extension Plugin Specs

This layer defines planned TCRV extension families and their plugin boundaries.
RVV, IME, TensorExt, Offload, scalar fallback, and future vendor/custom
families are parts of one unified TCRV RISC-V MLIR system, not independent
backend dialects.

## Pre-Development Checklist

- [ ] Is the target plugin classified correctly as ISA-vector, matrix-like ISA extension, runtime-offload, or future custom plugin?
- [ ] Are extension family ops execution ops, not high-level tensor ops?
- [ ] Does the plugin declare capability, legality, tuning, cost, and emission behavior?
- [ ] Does the plugin rely on `tcrv.exec` for variant/dispatch/fallback structure?
- [ ] Does the plugin use common TCRV interfaces and the common EmitC route where possible?
- [ ] Is the work advancing the real RVV trunk rather than a not-yet-built family? (见 [../guides/trunk-discipline.md](../guides/trunk-discipline.md))
- [ ] If this is RVV, does executable support start from typed `tcrv_rvv` body and plugin route provider rather than legacy `i32m1` route tables?
- [ ] If this is scalar fallback, is there no active executable scalar body unless a later rebuild task adds one?
- [ ] Are current hardware claims limited to verified environments?

## Guidelines Index

| Spec | Description |
|---|---|
| [RVV Plugin](./rvv-plugin.md) | Current primary real hardware plugin |
| [Scalar Fallback Plugin](./scalar-fallback-plugin.md) | Portable fallback plugin for coverage-oriented execution |
| [IME Plugin](./ime-plugin.md) | IME matrix-extension plugin |
| [Offload Runtime Plugin](./offload-runtime-plugin.md) | Sophgo/vendor runtime-offload capability |
| [Future Plugins](./future-plugins.md) | AME/custom ISA/vendor extension slots |

## Quality Check

- RVV is the first full plugin and the current real/mature mainline.
- Legacy `RVVI32M1*` / `rvv-i32m1-*` executable routes fail closed; route
  authority is the corrected typed `tcrv_rvv` surface (见 core-invariants I7).
- Coverage and selected-body realization build on that typed `tcrv_rvv` surface.
  For performance-sensitive RVV work, claim tuning or parity with handwritten
  kernels only with resource-aware selected-body realization or measured
  same-target evidence.
- Scalar fallback has no active executable scalar body unless rebuilt later.
- IME is the second non-RVV family integrated through the same common pipeline
  with zero core family-name branches (I3); offload routes fail closed until a
  real producer exists (I7). RVV is the first and broadest realized family;
  don't strand the RVV trunk for other families（见 [../guides/trunk-discipline.md](../guides/trunk-discipline.md)）。
- Future plugins are future slots: add one only when actual target facts and an
  explicit task make it real.
