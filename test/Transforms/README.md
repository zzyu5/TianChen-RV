# Transform Tests

This directory groups transform tests by the compiler stage or public pass
surface they exercise. Lit recursively discovers `.mlir` and `.test` files
under these subdirectories; C++ harnesses remain explicitly owned by
`test/CMakeLists.txt`.

| Directory | Coverage |
|---|---|
| `CapabilityRequires/` | capability requirement verification diagnostics |
| `PluginVariantLegality/` | public plugin variant legality pass behavior |
| `VariantSelection/` | capability-aware selected variant decisions |
| `DispatchSynthesis/` | dispatch and fallback structure synthesis |
| `VariantMaterialization/` | plugin variant metadata materialization |
| `LoweringBoundary/` | selected RVV/scalar lowering-boundary metadata |
| `EmissionReadiness/` | emission readiness and metadata plan diagnostics |
| `ExecutionPlanning/` | end-to-end public execution-planning pipeline |

Keep regression inputs in `test/` when they assert compiler behavior. Generated
runner evidence and exploratory scratch output belongs under ignored artifact
directories such as `artifacts/tmp/`, not in this test tree.
