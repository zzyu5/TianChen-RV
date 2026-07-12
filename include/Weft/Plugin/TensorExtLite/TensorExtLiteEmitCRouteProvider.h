#ifndef WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEMITCROUTEPROVIDER_H
#define WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEMITCROUTEPROVIDER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace weft::conversion::emitc {
struct WEFTEmitCSourceOpProvenance;
} // namespace weft::conversion::emitc

namespace weft::plugin {
class VariantEmitCLowerableRequest;
} // namespace weft::plugin

namespace weft::plugin::tensorext_lite {

// Stage 1 (description-engine retirement): the selected TensorExtLite EmitC
// route is no longer materialized as a string `WEFTEmitCLowerableRoute`. This
// readiness probe runs the same fail-closed validation gates the former route
// builder ran (construction-protocol ready, the selected role sequence is
// present and well-formed, the construction route mapping is consistent, and
// every role op exposes valid source provenance) and yields the genuine source
// provenance list the emission plan consumes. The route id is the construction
// constant `getTensorExtLiteFragmentMmaEmitCConstructionRoute().routeID`.
llvm::Error validateTensorExtLiteFragmentMmaEmitCRouteReadiness(
    const weft::plugin::VariantEmitCLowerableRequest &request,
    llvm::SmallVectorImpl<
        weft::conversion::emitc::WEFTEmitCSourceOpProvenance>
        &outSources);

} // namespace weft::plugin::tensorext_lite

#endif // WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEMITCROUTEPROVIDER_H
