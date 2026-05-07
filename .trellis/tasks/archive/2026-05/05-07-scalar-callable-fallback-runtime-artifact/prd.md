# scalar callable fallback runtime artifact

## Goal

Promote the finite scalar fallback i32 vector-add microkernel export from a
default standalone self-check source into a target-owned runtime-callable C
library source artifact. This is the upstream prerequisite for later host-side
RVV runtime dispatch glue: the RVV selected path already has a callable ABI, so
the scalar fallback path must expose a compatible callable ABI before generic
dispatch glue can call both sides cleanly.

## Requirements

- Default scalar microkernel source export emits a deterministic callable C
  function:
  `void <generated_name>(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n)`.
- The callable function performs bounded scalar i32 addition over the provided
  arrays and does not include `main`, stdio self-check machinery, or the scalar
  success marker in the default artifact.
- Metadata comments include selected kernel, selected variant, selected role,
  lowering boundary, scalar executable microkernel, artifact kind, element
  count, required capabilities, runtime ABI kind/name, and runtime glue role.
- Scalar plugin emission readiness/planning reports supported
  runtime-callable source metadata only for the finite selected scalar
  fallback path with a matching `tcrv_scalar.lowering_boundary` and
  `tcrv_scalar.i32_vadd_microkernel`.
- Unsupported scalar paths remain metadata-only or fail closed with diagnostics.
- RVV behavior remains unchanged except for target-neutral expectations needed
  to allow RVV callable and scalar callable source artifact kinds to coexist.
- Generic target artifact export remains target-neutral. Target-specific route
  facts stay in scalar/RVV/offload target-owned registration.

## Acceptance Criteria

- Focused lit/FileCheck coverage proves default scalar source is
  library-style callable C with no embedded self-check harness.
- Scalar emission-plan tests expect runtime-callable ABI fields and
  `runtime-callable-c-source` artifact kind for the finite scalar microkernel.
- Existing RVV callable artifact, emission manifest, artifact export, execution
  coherence, and `tcrv-translate` tests remain passing.
- Required validation commands pass:
  `git diff --check`, CMake configure, and `check-tianchenrv`.

## Out Of Scope

- Host-side runtime dispatch glue.
- Generic scalar lowering, arbitrary scalar kernels, object generation, linking,
  benchmarks, performance measurement, IME/AME/Sophgo/offload execution, RVV
  hardware compile/run, or unsupported RVV runtime/correctness/performance
  claims.

## Technical Notes

- Initial repo state was clean at `418ae7d feat: emit RVV callable library artifact`.
- Scalar exporter currently emits a standalone self-check `main`; RVV exporter
  already has separate default callable and explicit self-check helper modes.
- Relevant specs: `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
