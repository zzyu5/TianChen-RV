#ifndef TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEMITCROUTEPROVIDER_H

#include "llvm/Support/Error.h"

namespace tianchenrv::conversion::emitc {
class TCRVEmitCLowerableRoute;
} // namespace tianchenrv::conversion::emitc

namespace tianchenrv::plugin {
class VariantEmitCLowerableRequest;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::tensorext_lite {

llvm::Error buildTensorExtLiteFragmentMmaEmitCLowerableRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);

} // namespace tianchenrv::plugin::tensorext_lite

#endif // TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEMITCROUTEPROVIDER_H
