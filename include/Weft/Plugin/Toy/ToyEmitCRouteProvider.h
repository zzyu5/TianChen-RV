#ifndef WEFT_PLUGIN_TOY_TOYEMITCROUTEPROVIDER_H
#define WEFT_PLUGIN_TOY_TOYEMITCROUTEPROVIDER_H

#include "llvm/Support/Error.h"

namespace weft::conversion::emitc {
struct WEFTEmitCSourceOpProvenance;
} // namespace weft::conversion::emitc

namespace weft::plugin {
class VariantEmitCLowerableRequest;
} // namespace weft::plugin

namespace weft::plugin::toy {

// Stage 1 (description-engine retirement): the selected Toy EmitC route is no
// longer materialized as a string `WEFTEmitCLowerableRoute`. This readiness
// probe runs the same fail-closed validation gates the former route builder ran
// (construction-protocol ready, exactly one selected compute_skeleton boundary,
// valid source provenance, consistent construction route mapping) and yields the
// genuine source provenance the emission plan consumes. The route id is the
// construction constant `getToyTemplateEmitCConstructionRoute().routeID`.
llvm::Error validateToyTemplateEmitCRouteReadiness(
    const weft::plugin::VariantEmitCLowerableRequest &request,
    weft::conversion::emitc::WEFTEmitCSourceOpProvenance &outSource);

} // namespace weft::plugin::toy

#endif // WEFT_PLUGIN_TOY_TOYEMITCROUTEPROVIDER_H
