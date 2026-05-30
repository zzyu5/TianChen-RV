#include "TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h"

#include "TianChenRV/Plugin/RVV/RVVBaseMemoryMovementSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVReductionSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Errc.h"

#include <string>

namespace tianchenrv::plugin::rvv {

namespace {

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::ArrayRef<RVVSelectedBodyRealizationOwner>
getRVVSelectedBodyRealizationOwnerRegistry() {
  static const RVVSelectedBodyRealizationOwner owners[] = {
      {"elementwise/compare-select",
       isPreRealizedRVVElementwiseCompareSelectClusterOp,
       realizePreRealizedRVVElementwiseCompareSelectOwner},
      {"runtime scalar splat-store",
       isPreRealizedRVVRuntimeScalarSplatStoreOwnerOp,
       realizePreRealizedRVVRuntimeScalarSplatStoreOwner},
      {"runtime scalar computed-mask store",
       isPreRealizedRVVRuntimeScalarComputedMaskStoreOwnerOp,
       realizePreRealizedRVVRuntimeScalarComputedMaskStoreOwner},
      {"runtime scalar computed-mask load-store",
       isPreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwnerOp,
       realizePreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwner},
      {"reduction", isPreRealizedRVVReductionOwnerOp,
       realizePreRealizedRVVReductionOwner},
      {"standalone reduction", isPreRealizedRVVStandaloneReductionClusterOp,
       realizePreRealizedRVVStandaloneReductionOwner},
      {"MAcc", isPreRealizedRVVMAccClusterOp,
       realizePreRealizedRVVMAccOwner},
      {"computed-mask MAcc", isPreRealizedRVVComputedMaskMAccClusterOp,
       realizePreRealizedRVVComputedMaskMAccOwner},
      {"contraction", isPreRealizedRVVContractionClusterOp,
       realizePreRealizedRVVContractionOwner},
      {"widening conversion", isPreRealizedRVVWideningConversionOwnerOp,
       realizePreRealizedRVVWideningConversionOwner},
      {"base memory movement", isPreRealizedRVVBaseMemoryMovementOwnerOp,
       realizePreRealizedRVVBaseMemoryMovementOwner},
      {"computed-mask memory", isPreRealizedRVVComputedMaskMemoryClusterOp,
       realizePreRealizedRVVComputedMaskMemoryOwner},
      {"segment2 memory", isPreRealizedRVVSegment2MemoryOwnerOp,
       realizePreRealizedRVVSegment2MemoryOwner}};
  return owners;
}

llvm::Expected<const RVVSelectedBodyRealizationOwner *>
getUniqueRVVSelectedBodyRealizationOwner(mlir::Operation *bodyOp,
                                         llvm::StringRef context) {
  if (!bodyOp)
    return makeRVVPluginError(llvm::Twine(context) +
                              " requires a pre-realized RVV body op");

  llvm::SmallVector<const RVVSelectedBodyRealizationOwner *, 2> matches;
  for (const RVVSelectedBodyRealizationOwner &owner :
       getRVVSelectedBodyRealizationOwnerRegistry()) {
    if (owner.isConsumer && owner.isConsumer(bodyOp))
      matches.push_back(&owner);
  }

  if (matches.empty())
    return makeRVVPluginError(
        llvm::Twine(context) +
        " has no selected-body realization owner for pre-realized op '" +
        bodyOp->getName().getStringRef() + "'");
  if (matches.size() > 1) {
    std::string owners;
    llvm::raw_string_ostream os(owners);
    for (const RVVSelectedBodyRealizationOwner *owner : matches) {
      if (!owners.empty())
        os << ", ";
      os << owner->familyName;
    }
    os.flush();
    return makeRVVPluginError(
        llvm::Twine(context) +
        " found ambiguous selected-body realization owners for pre-realized op '" +
        bodyOp->getName().getStringRef() + "': " + owners);
  }
  return matches.front();
}

llvm::Expected<mlir::Operation *>
findUniquePreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "selected RVV realization requires a materialized tcrv.exec.variant");

  llvm::SmallVector<mlir::Operation *, 2> bodies;
  variant.getBody().walk([&](mlir::Operation *op) {
    for (const RVVSelectedBodyRealizationOwner &owner :
         getRVVSelectedBodyRealizationOwnerRegistry()) {
      if (owner.isConsumer && owner.isConsumer(op)) {
        bodies.push_back(op);
        return;
      }
    }
  });

  if (bodies.size() == 1) {
    llvm::Expected<const RVVSelectedBodyRealizationOwner *> owner =
        getUniqueRVVSelectedBodyRealizationOwner(
            bodies.front(), "selected RVV realization owner registry");
    if (!owner)
      return owner.takeError();
    return bodies.front();
  }

  if (bodies.empty())
    return makeRVVPluginError(
        "selected RVV realization requires exactly one registry-owned "
        "pre-realized tcrv_rvv body when no realized setvl/with_vl body is "
        "present");

  return makeRVVPluginError(
      "selected RVV realization requires exactly one registry-owned "
      "pre-realized tcrv_rvv body when no realized setvl/with_vl body is "
      "present; multiple pre-realized bodies matched the owner registry");
}

} // namespace

llvm::ArrayRef<RVVSelectedBodyRealizationOwner>
getRVVSelectedBodyRealizationOwners() {
  return getRVVSelectedBodyRealizationOwnerRegistry();
}

llvm::Expected<const RVVSelectedBodyRealizationOwner *>
getRVVSelectedBodyRealizationOwnerForBody(mlir::Operation *bodyOp,
                                          llvm::StringRef context) {
  return getUniqueRVVSelectedBodyRealizationOwner(bodyOp, context);
}

bool variantContainsPreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found)
      return;
    for (const RVVSelectedBodyRealizationOwner &owner :
         getRVVSelectedBodyRealizationOwnerRegistry()) {
      if (owner.isConsumer && owner.isConsumer(op)) {
        found = true;
        return;
      }
    }
  });
  return found;
}


llvm::Error diagnoseRetiredPreRealizedRVVRouteEntrySelectedBody(
    const VariantLoweringBoundaryRequest &request) {
  (void)request;
  return makeRVVPluginError(
      "direct pre-realized RVV route-entry realization is retired; "
      "pre-realized RVV selected bodies must use public selected "
      "lowering-boundary materialization before provider route construction");
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization requires materialized "
        "kernel and variant");

  llvm::Expected<mlir::Operation *> bodyOp =
      findUniquePreRealizedRVVSelectedBody(variant);
  if (!bodyOp)
    return bodyOp.takeError();

  llvm::Expected<const RVVSelectedBodyRealizationOwner *> owner =
      getRVVSelectedBodyRealizationOwnerForBody(
          *bodyOp, "pre-realized RVV selected-body realization owner registry");
  if (!owner)
    return owner.takeError();
  if (!(*owner)->realize)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected-body realization owner '") +
        (*owner)->familyName + "' has no realization hook");
  return (*owner)->realize(request, *bodyOp);
}

} // namespace tianchenrv::plugin::rvv
