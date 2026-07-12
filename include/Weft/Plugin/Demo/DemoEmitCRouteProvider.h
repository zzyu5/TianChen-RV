#ifndef WEFT_PLUGIN_DEMO_DEMOEMITCROUTEPROVIDER_H
#define WEFT_PLUGIN_DEMO_DEMOEMITCROUTEPROVIDER_H

#include "llvm/Support/Error.h"

namespace weft::conversion::emitc {
struct WEFTEmitCSourceOpProvenance;
} // namespace weft::conversion::emitc

namespace weft::plugin {
class VariantEmitCLowerableRequest;
} // namespace weft::plugin

namespace weft::plugin::demo_ext {

// Stage 1 (description-engine retirement): the selected Demo EmitC route is
// no longer materialized as a string `WEFTEmitCLowerableRoute`. This readiness
// probe runs the same fail-closed validation gates the former route builder ran
// (construction-protocol ready, exactly one selected compute_skeleton boundary,
// valid source provenance, consistent construction route mapping) and yields the
// genuine source provenance the emission plan consumes. The route id is the
// construction constant `getDemoEmitCConstructionRoute().routeID`.
llvm::Error validateDemoComputeSkeletonEmitCRouteReadiness(
    const VariantEmitCLowerableRequest &request,
    weft::conversion::emitc::WEFTEmitCSourceOpProvenance &outSource);

} // namespace weft::plugin::demo_ext

#endif // WEFT_PLUGIN_DEMO_DEMOEMITCROUTEPROVIDER_H
