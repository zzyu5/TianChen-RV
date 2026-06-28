#ifndef TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITEBACKENDEMISSIONDRIVER_H
#define TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITEBACKENDEMISSIONDRIVER_H

namespace tianchenrv {
namespace conversion {
namespace emitc {
class BackendEmissionRegistry;
} // namespace emitc
} // namespace conversion

namespace plugin {
namespace tensorext_lite {

/// Registers the TensorExtLite typed-emission backend (the
/// `TensorExtLiteBackendEmissionDriver`, which lowers a selected
/// configure->load_frag->tile_mma->store_frag role sequence directly into a
/// standalone EmitC module) into `registry`. Mirrors `registerToyBackendEmitter`:
/// the builtin backend table calls this; the TensorExtLite family lowers via the
/// shared `TypedBackendEmissionDriver` harness with zero core edits.
void registerTensorExtLiteBackendEmitter(
    conversion::emitc::BackendEmissionRegistry &registry);

} // namespace tensorext_lite
} // namespace plugin
} // namespace tianchenrv

#endif // TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITEBACKENDEMISSIONDRIVER_H
