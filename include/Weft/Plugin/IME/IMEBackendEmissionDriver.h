#ifndef WEFT_PLUGIN_IME_IMEBACKENDEMISSIONDRIVER_H
#define WEFT_PLUGIN_IME_IMEBACKENDEMISSIONDRIVER_H

namespace weft {
namespace conversion {
namespace emitc {
class BackendEmissionRegistry;
} // namespace emitc
} // namespace conversion

namespace plugin {
namespace ime {

/// Registers the IME typed-emission backend (the `IMEBackendEmissionDriver`,
/// which lowers a selected `weft.ime.mma` boundary into a standalone EmitC
/// module: the FOUNDATION-validated int8->int32 `vmadot` MAC kernel). Mirrors
/// `registerTemplateBackendEmitter`: the builtin backend table calls this; the
/// IME family lowers via the shared `TypedBackendEmissionDriver` harness with
/// zero core edits (one row in the builtin backend table).
void registerIMEBackendEmitter(
    conversion::emitc::BackendEmissionRegistry &registry);

} // namespace ime
} // namespace plugin
} // namespace weft

#endif // WEFT_PLUGIN_IME_IMEBACKENDEMISSIONDRIVER_H
