#ifndef WEFT_PLUGIN_TEMPLATE_TEMPLATEBACKENDEMISSIONDRIVER_H
#define WEFT_PLUGIN_TEMPLATE_TEMPLATEBACKENDEMISSIONDRIVER_H

namespace weft {
namespace conversion {
namespace emitc {
class BackendEmissionRegistry;
} // namespace emitc
} // namespace conversion

namespace plugin {
namespace template_ext {

/// Registers the Template typed-emission backend (the
/// `TemplateBackendEmissionDriver`, which lowers a selected
/// `weft_template.compute_skeleton` boundary directly into a standalone EmitC
/// module) into `registry`. Mirrors `registerToyBackendEmitter`: the builtin
/// backend table calls this; the Template family lowers via the shared
/// `TypedBackendEmissionDriver` harness with zero core edits.
void registerTemplateBackendEmitter(
    conversion::emitc::BackendEmissionRegistry &registry);

} // namespace template_ext
} // namespace plugin
} // namespace weft

#endif // WEFT_PLUGIN_TEMPLATE_TEMPLATEBACKENDEMISSIONDRIVER_H
