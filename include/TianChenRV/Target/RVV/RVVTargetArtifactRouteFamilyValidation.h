#ifndef TIANCHENRV_TARGET_RVV_RVVTARGETARTIFACTROUTEFAMILYVALIDATION_H
#define TIANCHENRV_TARGET_RVV_RVVTARGETARTIFACTROUTEFAMILYVALIDATION_H

#include "llvm/Support/Error.h"

namespace tianchenrv {
namespace target {
struct TargetArtifactCandidate;
} // namespace target

namespace conversion::emitc {
class TCRVEmitCLowerableRoute;
} // namespace conversion::emitc

namespace plugin::rvv {
struct RVVSelectedBodyEmitCRouteDescription;
} // namespace plugin::rvv

namespace target::rvv {

struct RVVTargetArtifactRouteFamilyValidationContext {
  const TargetArtifactCandidate &candidate;
  const conversion::emitc::TCRVEmitCLowerableRoute &route;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description;
};

llvm::Error validateRVVTargetArtifactRouteFamilyProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context);

llvm::Error validateRVVTargetArtifactRouteFamilyCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context);

} // namespace target::rvv
} // namespace tianchenrv

#endif // TIANCHENRV_TARGET_RVV_RVVTARGETARTIFACTROUTEFAMILYVALIDATION_H
