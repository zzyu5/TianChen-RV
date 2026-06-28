#ifndef TIANCHENRV_PLUGIN_TOY_TOYEMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_TOY_TOYEMITCROUTEPROVIDER_H

#include "llvm/Support/Error.h"

namespace tianchenrv::conversion::emitc {
struct TCRVEmitCSourceOpProvenance;
} // namespace tianchenrv::conversion::emitc

namespace tianchenrv::plugin {
class VariantEmitCLowerableRequest;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::toy {

// Stage 1 (description-engine retirement): the selected Toy EmitC route is no
// longer materialized as a string `TCRVEmitCLowerableRoute`. This readiness
// probe runs the same fail-closed validation gates the former route builder ran
// (construction-protocol ready, exactly one selected compute_skeleton boundary,
// valid source provenance, consistent construction route mapping) and yields the
// genuine source provenance the emission plan consumes. The route id is the
// construction constant `getToyTemplateEmitCConstructionRoute().routeID`.
llvm::Error validateToyTemplateEmitCRouteReadiness(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCSourceOpProvenance &outSource);

} // namespace tianchenrv::plugin::toy

#endif // TIANCHENRV_PLUGIN_TOY_TOYEMITCROUTEPROVIDER_H
