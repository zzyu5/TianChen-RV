#include "Weft/Plugin/RVV/RVVSelectedBodyRealization.h"

#include "Weft/Plugin/RVV/RVVBaseMemoryMovementSelectedBodyRealizationOwner.h"
#include "Weft/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.h"
#include "Weft/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.h"
#include "Weft/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.h"
#include "Weft/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.h"
#include "Weft/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.h"
#include "Weft/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.h"
#include "Weft/Plugin/RVV/RVVReductionSelectedBodyRealizationOwner.h"
#include "Weft/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.h"
#include "Weft/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h"
#include "Weft/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h"
#include "Weft/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Errc.h"

#include <string>

namespace weft::plugin::rvv {

namespace {

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV RVV extension plugin first slice failed: ") +
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
       realizePreRealizedRVVSegment2MemoryOwner},
      {"composite/gather-macc-scatter", nullptr, nullptr,
       hasPreRealizedRVVCompositeGatherMAccScatterOwnerCandidate,
       realizePreRealizedRVVCompositeGatherMAccScatterOwner}};
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
findUniquePreRealizedRVVSelectedBody(weft::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "selected RVV realization requires a materialized weft.exec.variant");

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
        "pre-realized weft_rvv body when no realized setvl/with_vl body is "
        "present");

  return makeRVVPluginError(
      "selected RVV realization requires exactly one registry-owned "
      "pre-realized weft_rvv body when no realized setvl/with_vl body is "
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

std::optional<RVVPreRealizedSelectedBodyMatch>
findFirstPreRealizedRVVSelectedBodyMatch(weft::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return std::nullopt;

  std::optional<RVVPreRealizedSelectedBodyMatch> match;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (match)
      return;
    for (const RVVSelectedBodyRealizationOwner &owner :
         getRVVSelectedBodyRealizationOwnerRegistry()) {
      if (owner.isConsumer && owner.isConsumer(op)) {
        match = RVVPreRealizedSelectedBodyMatch{op, owner.familyName};
        return;
      }
    }
  });
  return match;
}

bool variantContainsPreRealizedRVVSelectedBody(weft::exec::VariantOp variant) {
  return findFirstPreRealizedRVVSelectedBodyMatch(variant).has_value();
}

llvm::Error diagnoseRetiredPreRealizedRVVRouteEntrySelectedBody(
    const VariantLoweringBoundaryRequest &request) {
  (void)request;
  return makeRVVPluginError(
      "direct pre-realized RVV route-entry realization is retired; "
      "pre-realized RVV selected bodies must use public selected "
      "lowering-boundary materialization before provider route construction");
}

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request) {
  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization requires materialized "
        "kernel and variant");

  const RVVSelectedBodyRealizationOwner *variantOwner = nullptr;
  for (const RVVSelectedBodyRealizationOwner &owner :
       getRVVSelectedBodyRealizationOwnerRegistry()) {
    if (!owner.isVariantConsumer || !owner.isVariantConsumer(variant))
      continue;
    if (variantOwner)
      return makeRVVPluginError(
          "selected RVV variant matches multiple variant-level body "
          "realization owners");
    variantOwner = &owner;
  }
  if (variantOwner) {
    if (!variantOwner->realizeVariant)
      return makeRVVPluginError(
          llvm::Twine("variant-level selected-body owner '") +
          variantOwner->familyName + "' has no realization hook");
    return variantOwner->realizeVariant(request);
  }

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

} // namespace weft::plugin::rvv
