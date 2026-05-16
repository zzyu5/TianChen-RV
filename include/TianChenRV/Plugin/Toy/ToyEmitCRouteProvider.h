#ifndef TIANCHENRV_PLUGIN_TOY_TOYEMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_TOY_TOYEMITCROUTEPROVIDER_H

#include "llvm/Support/Error.h"

namespace tianchenrv::conversion::emitc {
class TCRVEmitCLowerableRoute;
} // namespace tianchenrv::conversion::emitc

namespace tianchenrv::plugin {
class VariantEmitCLowerableRequest;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::toy {

llvm::Error buildToyTemplateEmitCLowerableRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);

} // namespace tianchenrv::plugin::toy

#endif // TIANCHENRV_PLUGIN_TOY_TOYEMITCROUTEPROVIDER_H
