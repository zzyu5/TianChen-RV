#include "TianChenRV/Plugin/Scalar/ScalarEmitCRouteProvider.h"

namespace tianchenrv {
namespace plugin {
namespace scalar {

const ScalarEmitCConstructionRoute &getScalarEmitCConstructionRoute() {
  static const ScalarEmitCConstructionRoute route{
      /*translateRouteID=*/"tcrv-scalar-emitc-to-cpp",
      /*translateRouteDescription=*/
      "export the selected portable-scalar materialized EmitC module through "
      "the MLIR EmitC C/C++ emitter",
      /*callee=*/"tcrv_scalar_compute_skeleton",
      /*sourceOpName=*/"tcrv_scalar.compute_skeleton",
      /*sourceRole=*/"compute",
      /*opInterface=*/"TCRVEmitCLowerableOpInterface",
  };
  return route;
}

} // namespace scalar
} // namespace plugin
} // namespace tianchenrv
