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
/// `ScalarBackendEmissionDriver`, which mechanically projects an exact
/// Scalar final body and its family-local typed computation-plan nodes into a
/// standalone, pure-scalar EmitC module with no __riscv_ intrinsics) into
/// `registry`. The driver is a function-local static owned by this translation
/// unit, so it outlives the registry. The builtin backend table calls this;
/// the scalar family lowers via the shared `TypedBackendEmissionDriver`
/// harness with zero core edits.
void registerScalarBackendEmitter(
    conversion::emitc::BackendEmissionRegistry &registry);

} // namespace scalar
} // namespace plugin
} // namespace weft

#endif // WEFT_PLUGIN_SCALAR_SCALARBACKENDEMISSIONDRIVER_H
