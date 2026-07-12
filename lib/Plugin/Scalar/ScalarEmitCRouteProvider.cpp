#include "Weft/Plugin/Scalar/ScalarEmitCRouteProvider.h"

namespace weft {
namespace plugin {
namespace scalar {

const ScalarEmitCConstructionRoute &getScalarEmitCConstructionRoute() {
  static const ScalarEmitCConstructionRoute route{
      /*translateRouteID=*/"weft-scalar-emitc-to-cpp",
      /*translateRouteDescription=*/
      "export the selected portable-scalar materialized EmitC module through "
      "the MLIR EmitC C/C++ emitter",
      /*callee=*/"weft_scalar_compute_skeleton",
      /*sourceOpName=*/"weft_scalar.compute_skeleton",
      /*sourceRole=*/"compute",
      /*opInterface=*/"WEFTEmitCLowerableOpInterface",
  };
  return route;
}

} // namespace scalar
} // namespace plugin
} // namespace weft
