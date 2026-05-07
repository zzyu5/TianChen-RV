#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"

#include "TianChenRV/Target/Offload/OffloadRuntimeDescriptor.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVVScalarDispatch.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"

namespace tianchenrv::target {

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error =
          rvv::registerRVVMicrokernelTargetExporters(registry))
    return error;
  if (llvm::Error error =
          scalar::registerScalarMicrokernelTargetExporters(registry))
    return error;
  if (llvm::Error error =
          rvv_scalar::registerRVVScalarDispatchTargetExporters(registry))
    return error;
  return offload::registerOffloadRuntimeDescriptorTargetExporters(registry);
}

} // namespace tianchenrv::target
