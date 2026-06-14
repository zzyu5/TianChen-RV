#ifndef TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEMITCROUTEPROVIDER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::conversion::emitc {
struct TCRVEmitCSourceOpProvenance;
} // namespace tianchenrv::conversion::emitc

namespace tianchenrv::plugin {
class VariantEmitCLowerableRequest;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::tensorext_lite {

// Stage 1 (description-engine retirement): the selected TensorExtLite EmitC
// route is no longer materialized as a string `TCRVEmitCLowerableRoute`. This
// readiness probe runs the same fail-closed validation gates the former route
// builder ran (construction-protocol ready, the selected role sequence is
// present and well-formed, the construction route mapping is consistent, and
// every role op exposes valid source provenance) and yields the genuine source
// provenance list the emission plan consumes. The route id is the construction
// constant `getTensorExtLiteFragmentMmaEmitCConstructionRoute().routeID`.
llvm::Error validateTensorExtLiteFragmentMmaEmitCRouteReadiness(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    llvm::SmallVectorImpl<
        tianchenrv::conversion::emitc::TCRVEmitCSourceOpProvenance>
        &outSources);

} // namespace tianchenrv::plugin::tensorext_lite

#endif // TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEMITCROUTEPROVIDER_H
