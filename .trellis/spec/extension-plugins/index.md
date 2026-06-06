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
- [ ] Is RVV Stage 1/2 work prioritized before IME/offload/future-family positive workflows?
- [ ] If this is RVV, does executable support start from typed `tcrv_rvv` body and plugin route provider rather than legacy `i32m1` route tables?
- [ ] If this is scalar fallback, is there no active executable scalar body unless a later rebuild task adds one?
- [ ] Are current hardware claims limited to verified environments?

## Guidelines Index

| Spec | Description |
|---|---|
| [RVV Plugin](./rvv-plugin.md) | Current primary real hardware plugin |
| [Scalar Fallback Plugin](./scalar-fallback-plugin.md) | Portable fallback plugin for coverage-oriented execution |
| [IME Plugin](./ime-plugin.md) | Later K3/IME matrix-extension plugin |
| [Offload Runtime Plugin](./offload-runtime-plugin.md) | Sophgo/vendor runtime-offload capability |
| [Future Plugins](./future-plugins.md) | AME/custom ISA/vendor extension slots |

## Quality Check

- RVV is the first full plugin and current mainline.
- Stage 1 resets RVV route authority and fail-closes legacy `RVVI32M1*` /
  `rvv-i32m1-*` executable routes.
- Stage 2 expands coverage and selected-body realization on the corrected typed
  `tcrv_rvv` surface. For performance-sensitive RVV work, Stage 2 also requires
  resource-aware selected-body realization or measured same-target evidence
  before claiming tuning or parity with handwritten kernels.
- Scalar fallback has no active executable scalar body unless rebuilt later.
- IME and Offload executable integration are Stage3/later unless explicitly
  selected after RVV maturity.
- Future plugins remain Stage3/later slots unless actual target facts and an
  explicit task promote them.
