#ifndef TIANCHENRV_CONVERSION_RVV_RVVBACKENDEMISSIONDRIVER_H
#define TIANCHENRV_CONVERSION_RVV_RVVBACKENDEMISSIONDRIVER_H

namespace tianchenrv {
namespace conversion {
namespace emitc {
class BackendEmissionRegistry;
} // namespace emitc

namespace rvv {

/// Registers the RVV typed-emission backend (the `RVVBackendEmissionDriver`
/// wrapping the existing RVV->emitc DialectConversion) into `registry`. The
/// driver is a function-local static owned by this translation unit, so it
/// outlives the registry. Mirrors `registerRVVExtensionPlugin` on the plugin
/// side: the builtin backend table calls this; a future RVM family adds its own
/// `registerRVMBackendEmitter` alongside it with zero core edits.
void registerRVVBackendEmitter(emitc::BackendEmissionRegistry &registry);

} // namespace rvv
} // namespace conversion
} // namespace tianchenrv

#endif // TIANCHENRV_CONVERSION_RVV_RVVBACKENDEMISSIONDRIVER_H
