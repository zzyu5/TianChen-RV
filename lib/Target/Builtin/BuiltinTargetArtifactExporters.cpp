#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"

#include "TianChenRV/Target/Offload/OffloadRuntimeDescriptor.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVV/RVVSmokeProbe.h"
#include "TianChenRV/Target/RVVScalarDispatch.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"
#include "TianChenRV/Target/Toy/ToyMetadataArtifact.h"

namespace tianchenrv::target {

llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = rvv::registerRVVSmokeProbeTargetExporters(registry))
    return error;
  if (llvm::Error error =
          rvv::registerRVVMicrokernelTargetExporters(registry))
    return error;
  if (llvm::Error error =
          scalar::registerScalarMicrokernelTargetExporters(registry))
    return error;
  if (llvm::Error error =
          rvv_scalar::registerRVVScalarDispatchTargetExporters(registry))
    return error;
  if (llvm::Error error =
          offload::registerOffloadRuntimeDescriptorTargetExporters(registry))
    return error;
  return toy::registerToyMetadataArtifactTargetExporters(registry);
}

} // namespace tianchenrv::target
