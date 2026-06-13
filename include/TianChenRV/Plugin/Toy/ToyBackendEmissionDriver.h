#ifndef TIANCHENRV_PLUGIN_TOY_TOYBACKENDEMISSIONDRIVER_H
#define TIANCHENRV_PLUGIN_TOY_TOYBACKENDEMISSIONDRIVER_H

namespace tianchenrv {
namespace conversion {
namespace emitc {
class BackendEmissionRegistry;
} // namespace emitc
} // namespace conversion

namespace plugin {
namespace toy {

/// Registers the Toy typed-emission backend (the `ToyBackendEmissionDriver`,
/// which lowers a selected `tcrv_toy.compute_skeleton` boundary directly into a
/// standalone EmitC module) into `registry`. The driver is a function-local
/// static owned by this translation unit, so it outlives the registry. Mirrors
/// `registerRVVBackendEmitter`: the builtin backend table calls this; the Toy
/// family lowers via the shared `TypedBackendEmissionDriver` harness with zero
/// core edits, so the legacy route-builder/materializer is never reached.
void registerToyBackendEmitter(
    conversion::emitc::BackendEmissionRegistry &registry);

} // namespace toy
} // namespace plugin
} // namespace tianchenrv

#endif // TIANCHENRV_PLUGIN_TOY_TOYBACKENDEMISSIONDRIVER_H
