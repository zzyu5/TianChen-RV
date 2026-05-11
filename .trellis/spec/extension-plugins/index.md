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
- Scalar fallback is a coverage path and must not become the primary
  performance story without real lowering/runtime evidence.
- IME is a later extension plugin validation path.
- Offload is runtime capability, not custom ISA.
- Future plugins remain slots unless actual target facts exist.
