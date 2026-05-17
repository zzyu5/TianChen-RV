#ifndef TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATEEMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATEEMITCROUTEPROVIDER_H

#include "llvm/Support/Error.h"

namespace tianchenrv::conversion::emitc {
class TCRVEmitCLowerableRoute;
} // namespace tianchenrv::conversion::emitc

namespace tianchenrv::plugin {
class VariantEmitCLowerableRequest;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::template_ext {

llvm::Error buildTemplateComputeSkeletonEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);

} // namespace tianchenrv::plugin::template_ext

#endif // TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATEEMITCROUTEPROVIDER_H
