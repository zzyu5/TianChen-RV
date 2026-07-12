#ifndef WEFT_PLUGIN_SCALAR_SCALARBACKENDEMISSIONDRIVER_H
#define WEFT_PLUGIN_SCALAR_SCALARBACKENDEMISSIONDRIVER_H

namespace weft {
namespace conversion {
namespace emitc {
class BackendEmissionRegistry;
} // namespace emitc
} // namespace conversion

namespace plugin {
namespace scalar {

/// Registers the portable scalar typed-emission backend (the
/// `ScalarBackendEmissionDriver`, which lowers a selected
/// `weft_scalar.compute_skeleton` boundary directly into a standalone,
/// pure-scalar EmitC module with no __riscv_ intrinsics) into `registry`. The
/// driver is a function-local static owned by this translation unit, so it
/// outlives the registry. Mirrors `registerToyBackendEmitter`: the builtin
/// backend table calls this; the scalar family lowers via the shared
/// `TypedBackendEmissionDriver` harness with zero core edits.
void registerScalarBackendEmitter(
    conversion::emitc::BackendEmissionRegistry &registry);

} // namespace scalar
} // namespace plugin
} // namespace weft

#endif // WEFT_PLUGIN_SCALAR_SCALARBACKENDEMISSIONDRIVER_H
