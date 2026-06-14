//===- RVVEmitCRouteVerification.cpp - RVV route description verification --===//
//
// Behavior-preserving split out of RVVEmitCRoutePlanning.cpp: the public
// verifyRVVSelectedBodyEmitCRouteDescription entry point -- the fail-closed
// cross-check of a derived RVVSelectedBodyEmitCRouteDescription against the
// per-family expected route facts (intrinsics, CTypes, runtime-ABI order,
// metadata payloads). Relocated byte-identical; this is the same public-API
// function (declared in RVVEmitCRouteProvider.h) it always was, moved to its own
// translation unit. Helpers it consumes are public-API or declared in the
// co-located implementation-private RVVEmitCRoutePlanningInternal.h.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "RVVEmitCRoutePlanningInternal.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"
#include "TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>



namespace tianchenrv::plugin::rvv {

llvm::Error verifyRVVSelectedBodyEmitCRouteDescription(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (context.trim().empty())
    return makeRVVEmitCRouteProviderError(
        "selected-body route description verification requires a non-empty "
        "context");

  llvm::Expected<RVVSelectedBodyRouteProfile> profile =
      deriveRVVSelectedBodyRouteProfile(description);
  if (!profile)
    return profile.takeError();
  const RVVSelectedBodyOperationProfile &operationProfile =
      profile->operation;
  const RVVSelectedBodyConfigProfile &configProfile = profile->config;
  const RVVSelectedBodyTargetLeafProfile &targetLeaves =
      profile->targetLeaves;
  const tcrv::rvv::RVVSelectedBodyConfigVLContract &configContract =
      *configProfile.configContract;
  const bool isComputedMaskWideningDotReduce =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
  const bool isComputedMaskStridedInputWideningDotReduce =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::
          ComputedMaskStridedInputWideningDotReduceAdd;
  const bool isComputedMaskSelect =
      operationProfile.operation == RVVSelectedBodyOperationKind::ComputedMaskSelect;
  const bool isF32ClampSelect =
      operationProfile.operation == RVVSelectedBodyOperationKind::F32ClampSelect;
  const bool isDequantClampF32Epilogue =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
  const bool isPlainCompareSelect =
      operationProfile.operation == RVVSelectedBodyOperationKind::CmpSelect;
  const bool isVectorReductionRoute =
      operationProfile.operation == RVVSelectedBodyOperationKind::ReduceAdd &&
      description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad;
  const bool isRuntimeScalarCompareSelect =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect;
  const bool isRuntimeScalarDualCompareMaskAndSelect =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect;
  const bool isRuntimeScalarComputedMaskStore =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
  const bool isMaskedUnitStore =
      operationProfile.operation == RVVSelectedBodyOperationKind::MaskedUnitStore;
  const bool isStridedInputWideningDotReduce =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd;
  const bool isContractionRoute =
      isRVVSelectedBodyContractionRouteOperation(operationProfile.operation);
  const bool isProductReductionDequantizationRoute =
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionChainRoute =
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      isProductReductionDequantizationRoute;
  const bool isElementwiseArithmeticRoute =
      isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
          operationProfile.operation, description.memoryForm);
  const bool isScalarBroadcastElementwiseRoute =
      isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
          operationProfile.operation);
  const bool isRuntimeScalarSplatStoreRoute =
      isRVVSelectedBodyRuntimeScalarSplatStoreRouteOperation(
          operationProfile.operation);
  const bool isDequantizationRoute =
      isRVVSelectedBodyDequantizationRouteOperation(operationProfile.operation);
  const bool isRuntimeScalarComputedMaskSelectRoute =
      isRVVSelectedBodyComputedMaskSelectRouteOperation(
          operationProfile.operation);
  const bool isComputedMaskMemoryRouteFamilyRoute =
      isRVVSelectedBodyComputedMaskMemoryRouteOperation(
          operationProfile.operation);
  const bool isComputedMaskSegment2MemoryRouteFamilyRoute =
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskSegment2LoadUnitStore ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskSegment2StoreUnitLoad ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
  const bool isNonSegmentComputedMaskMemoryRouteFamilyRoute =
      isComputedMaskMemoryRouteFamilyRoute &&
      !isComputedMaskSegment2MemoryRouteFamilyRoute;
  const bool isBaseMemoryMovementRouteFamilyRoute =
      isRVVSelectedBodyBaseMemoryMovementRouteOperation(
          operationProfile.operation);
  const bool isPlainSegment2MemoryRouteFamilyRoute =
      isRVVSelectedBodySegment2MemoryRouteOperation(operationProfile.operation);
  const bool isSegment2MemoryRouteFamilyRoute =
      isPlainSegment2MemoryRouteFamilyRoute ||
      isComputedMaskSegment2MemoryRouteFamilyRoute;
  const bool isComputedMaskIndexedGather =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
  const bool isRuntimeScalarComputedMaskIndexedGather =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
  const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isComputedMaskIndexedGatherLike =
      isComputedMaskIndexedGather || isRuntimeScalarComputedMaskIndexedGather ||
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isComputedMaskIndexedScatter =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
  const bool isRuntimeScalarComputedMaskIndexedScatter =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
  const bool isComputedMaskIndexedScatterLike =
      isComputedMaskIndexedScatter || isRuntimeScalarComputedMaskIndexedScatter ||
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isStandaloneReductionRoute =
      isRVVSelectedBodyStandaloneReductionRouteOperation(
          operationProfile.operation);
  const bool isComputedMaskStandaloneReductionRoute =
      isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          operationProfile.operation);
  const bool isRuntimeScalarComputedMaskStandaloneReductionRoute =
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          operationProfile.operation);
  const bool isRuntimeScalarComputedMaskedMAcc =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
  const bool isComputedMaskedMAcc =
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
      isRuntimeScalarComputedMaskedMAcc;
  const bool isMAccRouteFamilyRoute =
      isRVVSelectedBodyMAccRouteFamilyConsumer(operationProfile.operation);

  std::optional<RVVPlainSegment2MemoryRouteFacts> plainSegment2Facts;
  std::optional<RVVComputedMaskSegment2MemoryRouteFacts>
      computedMaskSegment2Facts;
  if (isSegment2MemoryRouteFamilyRoute) {
    plainSegment2Facts =
        getRVVPlainSegment2MemoryRouteFacts(operationProfile.operation);
    computedMaskSegment2Facts =
        getRVVComputedMaskSegment2MemoryRouteFacts(operationProfile.operation);
    if (!plainSegment2Facts && !computedMaskSegment2Facts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment2 provider requires canonical route facts for the "
          "selected operation before route-description verification");
  }

  std::optional<RVVCompareSelectRouteFacts> compareSelectFacts;
  std::optional<RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts>
      dualCompareSelectFacts;
  if (isPlainCompareSelect || isRuntimeScalarComputedMaskSelectRoute ||
      isF32ClampSelect || isDequantClampF32Epilogue) {
    compareSelectFacts = getRVVCompareSelectRouteFacts(
        operationProfile.operation, description.sew, description.lmul,
        description.comparePredicateKind, description.secondaryComparePredicateKind);
    if (!compareSelectFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " compare/select provider requires canonical compare/select route "
          "facts for the selected operation, predicate, SEW, and LMUL before "
          "provider materialization");
    if (isRuntimeScalarDualCompareMaskAndSelect)
      dualCompareSelectFacts = compareSelectFacts;
  }

  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          operationProfile.operationMnemonic);
  if (!route)
    return route.takeError();
  const RVVSelectedBodyConstructionRoute &constructionRoute = **route;

  const bool usesGenericBinary =
      description.typedComputeOpName == "tcrv_rvv.binary";
  if (usesGenericBinary && operationProfile.isCompareSelect)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " compare/select cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && operationProfile.isReduction)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && operationProfile.isMaskedArithmetic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " masked arithmetic cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && operationProfile.isMultiplyAccumulate)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " multiply-accumulate cannot use generic tcrv_rvv.binary");
  // The low-precision dequant(/clamp) routes carry a candidate-aware typed-compute
  // chain on the description (validated against the bounded legal set by the
  // construction-protocol layer), so it must not be mirrored against the single
  // static construction-route chain here.
  const bool isDequantTypedComputeRoute =
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  if (!usesGenericBinary && !isDequantTypedComputeRoute)
    if (llvm::Error error = requireRouteDescriptionField(
            context, "typed compute op", description.typedComputeOpName,
            constructionRoute.typedComputeOpName))
      return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC route id", description.emitCRouteID,
          constructionRoute.emitCRouteID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "target artifact route id",
          description.targetArtifactRouteID,
          getRVVSelectedBodyTargetArtifactRouteID()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "target artifact kind", description.targetArtifactKind,
          getRVVSelectedBodyTargetArtifactKind()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI name", description.runtimeABIName,
          constructionRoute.runtimeABIName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI contract", description.runtimeABIContractName,
          constructionRoute.runtimeABIContractName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "lowering boundary op", description.boundaryOpName,
          kRVVSelectedBodyLoweringBoundaryOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "config contract", description.configContractID,
          configContract.configContractID))
    return error;
  llvm::StringRef expectedElementTypeStorage;
  if (isF32ClampSelect || isDequantClampF32Epilogue ||
      isProductReductionDequantizationRoute) {
    expectedElementTypeStorage =
        getRVVSelectedBodyFloatElementTypeName(description.sew);
    if (expectedElementTypeStorage.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " f32 clamp/select route requires a supported f32 element type for SEW " +
          llvm::Twine(description.sew));
  } else if (description.operation ==
                 RVVSelectedBodyOperationKind::WideningProduct &&
             description.wideningProductRelation ==
                 "unsigned-u8mf4xu8mf4-to-u16mf2") {
    expectedElementTypeStorage =
        getRVVSelectedBodyUnsignedIntegerElementTypeName(description.sew);
    if (expectedElementTypeStorage.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " unsigned widening-product route requires a supported unsigned "
          "integer element type for SEW " +
          llvm::Twine(description.sew));
  } else if (isProductReductionChainRoute &&
             description.wideningProductRelation ==
                 "unsigned-u8mf4xu8mf4-to-u16mf2") {
    expectedElementTypeStorage =
        getRVVSelectedBodyUnsignedIntegerElementTypeName(description.sew);
    if (expectedElementTypeStorage.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " unsigned product-reduction route requires a supported unsigned "
          "integer result element type for SEW " +
          llvm::Twine(description.sew));
  } else {
    llvm::Expected<llvm::StringRef> expectedElementType =
        getRVVSelectedBodyElementTypeNameForSEW(description.sew, context);
    if (!expectedElementType)
      return expectedElementType.takeError();
    expectedElementTypeStorage = *expectedElementType;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "element type", description.elementTypeName,
          expectedElementTypeStorage))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime VL contract", description.runtimeVLContractID,
          configContract.runtimeVLContractID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime AVL source", description.runtimeAVLASource,
          configContract.runtimeAVLASource))
    return error;
  if (llvm::Error error =
          verifyRVVCompositeGatherMAccScatterResourceDescriptionSelection(
              description, context))
    return error;
  if (isContractionRoute) {
    if (llvm::Error error =
            verifyRVVSelectedBodyContractionRouteDescriptionMirrors(
                description, context))
      return error;
  } else if (isElementwiseArithmeticRoute ||
             isScalarBroadcastElementwiseRoute) {
    if (llvm::Error error =
            verifyRVVSelectedBodyElementwiseRouteDescriptionMirrors(
                description, context))
      return error;
  } else if (operationProfile.operation ==
             RVVSelectedBodyOperationKind::ReduceAdd) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            kRVVVectorReductionTargetLeafProfile))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            kRVVVectorReductionProviderSupportedMirror))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            kRVVVectorReductionRequiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            kRVVVectorReductionCTypeMappingSummary))
      return error;
  } else if (isRuntimeScalarSplatStoreRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            kRVVRuntimeScalarSplatStoreTargetLeafProfile))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            kRVVRuntimeScalarSplatStoreProviderSupportedMirror))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            kRVVRuntimeScalarSplatStoreRequiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            kRVVRuntimeScalarSplatStoreCTypeMappingSummary))
      return error;
  } else if (isPlainCompareSelect) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            compareSelectFacts->targetLeafProfile))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            compareSelectFacts->providerSupportedMirror))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            compareSelectFacts->requiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            compareSelectFacts->cTypeMappingSummary))
      return error;
  } else if (operationProfile.isWideningConversion) {
    std::optional<RVVWideningConversionRouteFacts> routeFacts =
        getRVVWideningConversionRouteFacts(operationProfile.operation);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " requires provider-owned widening conversion canonical route facts "
          "for the selected operation");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            routeFacts->targetLeafProfile))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            routeFacts->providerSupportedMirror))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            routeFacts->requiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            routeFacts->cTypeMappingSummary))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source element type", description.sourceElementTypeName,
            routeFacts->sourceElementTypeName))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "result element type", description.resultElementTypeName,
            routeFacts->resultElementTypeName))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "conversion kind", description.conversionKind,
            routeFacts->conversionKind))
      return error;
  } else if (isDequantizationRoute) {
    std::optional<RVVDequantizationRouteFacts> routeFacts =
        getRVVDequantizationRouteFacts(operationProfile.operation);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " requires provider-owned dequantization canonical route facts for "
          "the selected operation");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            routeFacts->targetLeafProfile))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            routeFacts->providerSupportedMirror))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            routeFacts->requiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            routeFacts->cTypeMappingSummary))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source element type", description.sourceElementTypeName,
            routeFacts->sourceElementTypeName))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "result element type", description.resultElementTypeName,
            routeFacts->resultElementTypeName))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequantization kind", description.conversionKind,
            routeFacts->dequantizationKind))
      return error;
    if (!rvvGearboxCandidateSetContains(description.gearboxCandidateSet,
                                        description.gearboxSelectedCandidate))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " selected RVV Gearbox candidate must belong to the "
          "provider-consumed legal candidate set");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox candidate set", description.gearboxCandidateSet,
            routeFacts->gearboxCandidateSet))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox selected candidate",
            description.gearboxSelectedCandidate,
            routeFacts->gearboxSelectedCandidate))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox selection reason",
            description.gearboxSelectionReason,
            routeFacts->gearboxSelectionReason))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox legality scope", description.gearboxLegalityScope,
            routeFacts->gearboxLegalityScope))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox schedule id", description.gearboxScheduleID,
            routeFacts->gearboxScheduleID))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox selector", description.gearboxSelector,
            routeFacts->gearboxSelector))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox source", description.gearboxSource,
            routeFacts->gearboxSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox operation", description.gearboxOperation,
            routeFacts->gearboxOperation))
      return error;
    if (description.gearboxUnroll != routeFacts->gearboxUnroll)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " Gearbox unroll must mirror provider-consumed Gearbox facts");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox VL policy", description.gearboxVLPolicy,
            routeFacts->gearboxVLPolicy))
      return error;
    if (description.gearboxSourceSEW != routeFacts->gearboxSourceSEW)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " Gearbox source SEW must mirror provider-consumed Gearbox facts");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox source LMUL", description.gearboxSourceLMUL,
            routeFacts->gearboxSourceLMUL))
      return error;
    if (description.gearboxDestSEW != routeFacts->gearboxDestSEW)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " Gearbox destination SEW must mirror provider-consumed Gearbox facts");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox destination LMUL", description.gearboxDestLMUL,
            routeFacts->gearboxDestLMUL))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox runtime AVL source",
            description.gearboxRuntimeAVLSource,
            routeFacts->gearboxRuntimeAVLSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox producer scope",
            description.gearboxProducerScope,
            routeFacts->gearboxProducerScope))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "Gearbox consumer scope",
            description.gearboxConsumerScope,
            routeFacts->gearboxConsumerScope))
      return error;
    if (description.gearboxProducerScope == description.gearboxConsumerScope)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " Gearbox producer and consumer scopes must be distinct provider "
          "facts");
  } else if (isRuntimeScalarComputedMaskSelectRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            compareSelectFacts->targetLeafProfile))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            compareSelectFacts->providerSupportedMirror))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            compareSelectFacts->requiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            compareSelectFacts->cTypeMappingSummary))
      return error;
  } else if (isBaseMemoryMovementRouteFamilyRoute) {
    // Base memory movement mirrors are verified by the RVV owner below.
  } else if (isNonSegmentComputedMaskMemoryRouteFamilyRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            getComputedMaskMemoryTargetLeafProfile(operationProfile.operation)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            getComputedMaskMemoryProviderSupportedMirror(
                operationProfile.operation)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            getComputedMaskMemoryRequiredHeaderDeclarations(
                operationProfile.operation)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            getComputedMaskMemoryCTypeMappingSummary(operationProfile.operation)))
      return error;
  } else if (isSegment2MemoryRouteFamilyRoute) {
    llvm::StringRef expectedTargetLeafProfile =
        plainSegment2Facts ? plainSegment2Facts->targetLeafProfile
                           : computedMaskSegment2Facts->targetLeafProfile;
    llvm::StringRef expectedProviderSupportedMirror =
        plainSegment2Facts ? plainSegment2Facts->providerSupportedMirror
                           : computedMaskSegment2Facts->providerSupportedMirror;
    llvm::StringRef expectedHeaderDeclarations =
        plainSegment2Facts ? plainSegment2Facts->requiredHeaderDeclarations
                           : computedMaskSegment2Facts->requiredHeaderDeclarations;
    llvm::StringRef expectedCTypeMapping =
        plainSegment2Facts ? plainSegment2Facts->cTypeMappingSummary
                           : computedMaskSegment2Facts->cTypeMappingSummary;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            expectedTargetLeafProfile))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror, expectedProviderSupportedMirror))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations, expectedHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            expectedCTypeMapping))
      return error;
  } else if (isRVVSelectedBodyMAccRouteFamilyConsumer(
                 operationProfile.operation)) {
    // MAcc mirrors are verified by the RVV owner below.
  } else if (isStandaloneReductionRoute) {
    std::optional<RVVStandaloneReductionRouteFacts> routeFacts =
        getRVVStandaloneReductionRouteFacts(operationProfile.operation,
                                            description.sew);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " requires provider-owned standalone reduction canonical route "
          "facts for the selected operation and SEW");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            routeFacts->targetLeafProfile))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            routeFacts->providerSupportedMirror))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            routeFacts->requiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            routeFacts->cTypeMappingSummary))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            ""))
      return error;
  }
  llvm::StringRef expectedRuntimeABIOrder = configContract.runtimeABIOrder;
  if (isContractionRoute) {
    expectedRuntimeABIOrder =
        getRVVSelectedBodyContractionRuntimeABIOrder(operationProfile.operation);
  } else if (operationProfile.operation ==
             RVVSelectedBodyOperationKind::StridedAdd) {
    expectedRuntimeABIOrder = kRVVStridedRuntimeABIOrder;
  } else if (isSegment2MemoryRouteFamilyRoute) {
    expectedRuntimeABIOrder =
        plainSegment2Facts ? plainSegment2Facts->runtimeABIOrder
                           : computedMaskSegment2Facts->runtimeABIOrder;
  } else if (operationProfile.isMemoryMovement &&
             !isBaseMemoryMovementRouteFamilyRoute) {
  } else if (isComputedMaskSelect) {
    expectedRuntimeABIOrder = kRVVComputedMaskSelectRuntimeABIOrder;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            kRVVComputedMaskSelectMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for computed-mask select routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isRuntimeScalarCompareSelect) {
    expectedRuntimeABIOrder = kRVVRuntimeScalarCompareSelectRuntimeABIOrder;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            "unit-stride-lhs-runtime-scalar-threshold-true-false-select-output-runtime-abi"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for runtime scalar compare/select routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isRuntimeScalarDualCompareMaskAndSelect) {
    expectedRuntimeABIOrder =
        kRVVRuntimeScalarDualCompareMaskAndSelectRuntimeABIOrder;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for runtime scalar dual-compare "
          "mask-and select routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isComputedMaskMemoryRouteFamilyRoute) {
    expectedRuntimeABIOrder =
        getComputedMaskMemoryRuntimeABIOrder(operationProfile.operation);
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            operationProfile.operation ==
                    RVVSelectedBodyOperationKind::ComputedMaskStridedStore
                ? llvm::StringRef(kRVVComputedMaskStridedStoreMemoryLayout)
            : operationProfile.operation ==
                    RVVSelectedBodyOperationKind::
                        ComputedMaskStridedLoadUnitStore
                ? llvm::StringRef(kRVVComputedMaskStridedLoadMemoryLayout)
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            getComputedMaskMemoryLayout(operationProfile.operation)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource,
            operationProfile.operation ==
                    RVVSelectedBodyOperationKind::ComputedMaskStridedStore
                ? llvm::StringRef(kRVVDestinationByteStrideSource)
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            operationProfile.operation ==
                    RVVSelectedBodyOperationKind::
                        ComputedMaskStridedLoadUnitStore
                ? llvm::StringRef(kRVVSourceByteStrideSource)
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            getComputedMaskMemorySourceMemoryForm(operationProfile.operation)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm,
            getComputedMaskMemoryDestinationMemoryForm(
                operationProfile.operation)))
      return error;
    if (description.indexEEW !=
        ((isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike)
             ? 32
             : 0))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must mirror computed-mask memory route-family indexed "
          "route facts");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit,
            (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike)
                ? llvm::StringRef(kRVVIndexedGatherOffsetUnit)
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource,
            (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike)
                ? llvm::StringRef(kRVVIndexSource)
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness,
            isComputedMaskIndexedScatterLike
                ? llvm::StringRef(kRVVIndexedScatterIndexUniqueness)
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm,
            isComputedMaskIndexedGatherLike
                ? llvm::StringRef(kRVVMaskedIndexedLoadSourceMemoryForm)
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm,
            isComputedMaskIndexedScatterLike
                ? llvm::StringRef(kRVVMaskedIndexedStoreDestinationMemoryForm)
                : llvm::StringRef()))
      return error;
  } else if (operationProfile.isMaskedMemoryMovement &&
             !isBaseMemoryMovementRouteFamilyRoute) {
    expectedRuntimeABIOrder =
        description.memoryForm ==
                RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore
            ? kRVVComputedMaskStridedStoreRuntimeABIOrder
        : description.memoryForm ==
                RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore
            ? kRVVComputedMaskStridedLoadRuntimeABIOrder
        : description.memoryForm ==
                RVVSelectedBodyMemoryForm::
                    ComputedMaskIndexedGatherLoadUnitStore
            ? kRVVComputedMaskIndexedGatherRuntimeABIOrder
        : description.memoryForm ==
                RVVSelectedBodyMemoryForm::
                    ComputedMaskUnitLoadIndexedScatterStore
            ? kRVVComputedMaskIndexedScatterRuntimeABIOrder
            : kRVVComputedMaskMemoryRuntimeABIOrder;
  } else if (isRuntimeScalarDualCompareMaskAndSelect) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for runtime scalar dual-compare "
          "mask-and select routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isRuntimeScalarDualCompareMaskAndSelect) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for runtime scalar dual-compare "
          "mask-and select routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isF32ClampSelect) {
    expectedRuntimeABIOrder = kRVVF32ClampSelectRuntimeABIOrder;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            "unit-stride-f32-input-runtime-lower-upper-select-output-runtime-abi"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          (isDequantClampF32Epilogue
               ? " index EEW must be empty for dequant-clamp f32 epilogue routes"
               : " index EEW must be empty for f32 clamp/select routes"));
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isDequantClampF32Epilogue) {
    expectedRuntimeABIOrder = kRVVDequantClampF32EpilogueRuntimeABIOrder;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            "unit-stride-i32-input-runtime-scale-dequant-runtime-lower-upper-select-f32-output-runtime-abi"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for dequant-clamp f32 epilogue routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isIndexedMemoryMovement &&
             !isBaseMemoryMovementRouteFamilyRoute) {
  } else if (operationProfile.isWideningConversion) {
    expectedRuntimeABIOrder = kRVVWideningConversionRuntimeABIOrder;
  } else if (isDequantizationRoute) {
    expectedRuntimeABIOrder = kRVVDequantizeI32ToF32RuntimeABIOrder;
  } else if (std::optional<llvm::StringRef> maccRuntimeABIOrder =
                 getExpectedRVVSelectedBodyMAccRuntimeABIOrder(
                     operationProfile.operation)) {
    expectedRuntimeABIOrder = *maccRuntimeABIOrder;
  } else if (isContractionRoute) {
    expectedRuntimeABIOrder =
        getRVVSelectedBodyContractionRuntimeABIOrder(operationProfile.operation);
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore) {
    expectedRuntimeABIOrder = kRVVRuntimeScalarSplatStoreRuntimeABIOrder;
  } else if (isStandaloneReductionRoute) {
    if (std::optional<RVVStandaloneReductionRouteFacts> routeFacts =
            getRVVStandaloneReductionRouteFacts(operationProfile.operation,
                                                description.sew))
      expectedRuntimeABIOrder = routeFacts->runtimeABIOrder;
  }
  if (!isElementwiseArithmeticRoute && !isScalarBroadcastElementwiseRoute &&
      !isBaseMemoryMovementRouteFamilyRoute)
    if (llvm::Error error = requireRouteDescriptionField(
            context, "runtime ABI order", description.runtimeABIOrder,
            expectedRuntimeABIOrder))
      return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyElementwiseRouteDescriptionMirrors(
              description, context))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyBaseMemoryMovementRouteDescriptionMirrors(
              description, context))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyMAccRouteDescriptionMirrors(description,
                                                          context))
    return error;
  if (isRuntimeScalarSplatStoreRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "runtime scalar splat-store route family plan",
            description.runtimeScalarSplatStoreRouteFamilyPlanID,
            kRVVRuntimeScalarSplatStoreRouteFamilyPlanID))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "runtime scalar splat-store route family plan",
            description.runtimeScalarSplatStoreRouteFamilyPlanID, ""))
      return error;
  }
  if (isPlainCompareSelect) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "plain compare-select route family plan",
            description.plainCompareSelectRouteFamilyPlanID,
            compareSelectFacts->plainCompareSelectRouteFamilyPlanID))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "select layout", description.selectLayout,
            compareSelectFacts->selectLayout))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "plain compare-select route family plan",
            description.plainCompareSelectRouteFamilyPlanID, ""))
      return error;
  }
  if (operationProfile.isWideningConversion) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening conversion route family plan",
            description.wideningConversionRouteFamilyPlanID,
            kRVVWideningConversionRouteFamilyPlanID))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening conversion route family plan",
            description.wideningConversionRouteFamilyPlanID, ""))
      return error;
  }
  if (isDequantizationRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequantization route family plan",
            description.dequantizationRouteFamilyPlanID,
            kRVVDequantizationRouteFamilyPlanID))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequantization route family plan",
            description.dequantizationRouteFamilyPlanID, ""))
      return error;
  }
  if (isRuntimeScalarComputedMaskSelectRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "computed-mask select route family plan",
            description.computedMaskSelectRouteFamilyPlanID,
            compareSelectFacts->computedMaskSelectRouteFamilyPlanID))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "computed-mask select mask producer source",
            description.computedMaskSelectMaskProducerSource,
            compareSelectFacts->computedMaskSelectMaskProducerSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "select layout", description.selectLayout,
            compareSelectFacts->selectLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask/tail policy route family plan",
            description.maskTailPolicyRouteFamilyPlanID,
            compareSelectFacts->maskTailPolicyRouteFamilyPlanID))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask/tail policy owner",
            description.maskTailPolicyOwner,
            compareSelectFacts->maskTailPolicyOwner))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "computed-mask select route family plan",
            description.computedMaskSelectRouteFamilyPlanID, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "computed-mask select mask producer source",
            description.computedMaskSelectMaskProducerSource, ""))
      return error;
    if (!isPlainCompareSelect && !isContractionRoute)
      if (llvm::Error error = requireRouteDescriptionField(
              context, "select layout", description.selectLayout, ""))
        return error;
  }
  if (compareSelectFacts) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lower bound role", description.lowerBoundRole,
            compareSelectFacts->lowerBoundRole))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "upper bound role", description.upperBoundRole,
            compareSelectFacts->upperBoundRole))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lower bound C type", description.lowerBoundCType,
            compareSelectFacts->lowerBoundCType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "upper bound C type", description.upperBoundCType,
            compareSelectFacts->upperBoundCType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "bound order", description.boundOrder,
            compareSelectFacts->boundOrder))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "clamp relation", description.clampRelation,
            compareSelectFacts->clampRelation))
      return error;
  }
  if (isComputedMaskMemoryRouteFamilyRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "computed-mask memory route family plan",
            description.computedMaskMemoryRouteFamilyPlanID,
            kRVVComputedMaskMemoryRouteFamilyPlanID))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "computed-mask memory mask producer source",
            description.computedMaskMemoryMaskProducerSource,
            getComputedMaskMemoryProducerSource(operationProfile.operation)))
      return error;
    if (isNonSegmentComputedMaskMemoryRouteFamilyRoute) {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "mask/tail policy route family plan",
              description.maskTailPolicyRouteFamilyPlanID,
              kRVVMaskTailPolicyRouteFamilyPlanID))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "mask/tail policy owner",
              description.maskTailPolicyOwner,
              kRVVComputedMaskMemoryMaskTailPolicyOwner))
        return error;
    }
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "computed-mask memory route family plan",
            description.computedMaskMemoryRouteFamilyPlanID, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "computed-mask memory mask producer source",
            description.computedMaskMemoryMaskProducerSource, ""))
      return error;
  }
  if (!isRuntimeScalarComputedMaskSelectRoute &&
      !isNonSegmentComputedMaskMemoryRouteFamilyRoute &&
      !isComputedMaskSegment2MemoryRouteFamilyRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask/tail policy route family plan",
            description.maskTailPolicyRouteFamilyPlanID, ""))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "mask/tail policy owner",
                                         description.maskTailPolicyOwner, ""))
      return error;
  }
  if (isPlainSegment2MemoryRouteFamilyRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "plain segment2 memory route family plan",
            description.segment2MemoryRouteFamilyPlanID,
            kRVVSegment2MemoryRouteFamilyPlanID))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "plain segment2 memory route family plan",
            description.segment2MemoryRouteFamilyPlanID, ""))
      return error;
  }
  if (isStandaloneReductionRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction route family plan",
            description.standaloneReductionRouteFamilyPlanID,
            kRVVStandaloneReductionRouteFamilyPlanID))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction route family plan",
            description.standaloneReductionRouteFamilyPlanID, ""))
      return error;
  }
  if (!isContractionRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "contraction route family plan",
            description.contractionRouteFamilyPlanID, ""))
      return error;
  }
  llvm::StringRef expectedOperandBindingPlanID =
      getExpectedRVVRouteOperandBindingPlanID(operationProfile.operation);
  if (std::optional<RVVWideningProductRouteFacts> wideningProductFacts =
          getRVVWideningProductRouteFacts(description))
    expectedOperandBindingPlanID =
        wideningProductFacts->routeOperandBindingPlanID;
  if (!expectedOperandBindingPlanID.empty()) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "route operand ABI binding plan",
            description.routeOperandBindingPlanID,
            expectedOperandBindingPlanID))
      return error;
    if (description.routeOperandBindingSummary.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " requires a non-empty route operand ABI binding mirror summary");
    if (compareSelectFacts &&
        description.routeOperandBindingSummary !=
            compareSelectFacts->routeOperandBindingSummary)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " compare/select provider requires route operand ABI binding "
          "summary to mirror canonical compare/select facts '" +
          compareSelectFacts->routeOperandBindingSummary + "' but was '" +
          description.routeOperandBindingSummary + "'");
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "route operand ABI binding plan",
            description.routeOperandBindingPlanID, ""))
      return error;
    if (!description.routeOperandBindingSummary.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " must not carry a route operand ABI binding summary for this "
          "unconverted route");
  }
  if (isElementwiseArithmeticRoute || isScalarBroadcastElementwiseRoute ||
      isRuntimeScalarSplatStoreRoute ||
      isPlainCompareSelect ||
      isVectorReductionRoute ||
      operationProfile.isWideningConversion ||
      isBaseMemoryMovementRouteFamilyRoute ||
      isRuntimeScalarComputedMaskSelectRoute ||
      isComputedMaskMemoryRouteFamilyRoute ||
      isRuntimeScalarComputedMaskedMAcc || isStandaloneReductionRoute ||
      isContractionRoute)
    if (llvm::Error error = requireRouteDescriptionField(
            context, "runtime control plan", description.runtimeControlPlanID,
            getRVVRuntimeAVLVLControlPlanID()))
      return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL def op", description.vlDefOpName,
          configContract.vlDefOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL scope op", description.vlScopeOpName,
          configContract.vlScopeOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL uses", description.vlUses, configContract.vlUses))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop kind", description.emitCLoopKind,
          configContract.emitCLoopKind))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop induction", description.emitCLoopInductionName,
          configContract.emitCLoopInductionName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC full-chunk VL", description.emitCFullChunkVLName,
          configContract.emitCFullChunkVLName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop VL", description.emitCLoopVLName,
          tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "remaining AVL metadata",
          description.remainingAVLMetadata, configContract.remainingAVLMetadata))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "pointer advance metadata",
          description.pointerAdvanceMetadata,
          configContract.pointerAdvanceMetadata))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "bounded slice", description.boundedSlice,
          configContract.boundedSlice))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "multi-VL support", description.multiVL,
          configContract.multiVL))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL C type", description.vlCType, configProfile.vlCType))
    return error;
  if (!isDequantizationRoute && !isProductReductionDequantizationRoute &&
      !isDequantClampF32Epilogue) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "vector type", description.vectorTypeName,
            configProfile.vectorTypeName))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "vector C type", description.vectorCType,
            configProfile.vectorCType))
      return error;
  } else if (isDequantizationRoute) {
    std::optional<RVVDequantizationRouteFacts> routeFacts =
        getRVVDequantizationRouteFacts(operationProfile.operation);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " dequantization requires provider-owned route facts before "
          "route-description verification");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant result vector type", description.vectorTypeName,
            routeFacts->resultVectorTypeName))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant result vector C type", description.vectorCType,
            routeFacts->resultVectorCType))
      return error;
  } else if (isDequantClampF32Epilogue) {
    llvm::StringRef expectedResultVectorType =
        getRVVSelectedBodyFloatVectorTypeName(
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1());
    llvm::StringRef expectedResultVectorCType =
        getRVVSelectedBodyFloatVectorCType(
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1());
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant-clamp result vector type",
            description.vectorTypeName, expectedResultVectorType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant-clamp result vector C type",
            description.vectorCType, expectedResultVectorCType))
      return error;
  }
  if (operationProfile.isIndexedMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector type", description.indexVectorTypeName,
            configProfile.indexVectorTypeName))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector C type", description.indexVectorCType,
            configProfile.indexVectorCType))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector type", description.indexVectorTypeName, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector C type", description.indexVectorCType, ""))
      return error;
  }
  if (isContractionRoute) {
    if (description.sourceSEW <= 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " source SEW must be provider-derived from typed source vectors "
          "for widening mixed-width RVV contraction routes");
    if (description.sourceLMUL.empty() ||
        description.sourceVectorTypeName.empty() ||
        description.sourceVectorCType.empty() ||
        description.sourceVectorLoadIntrinsic.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " source LMUL/type/load leaves must be provider-derived from typed "
          "source vectors for widening mixed-width RVV contraction routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "conversion relation", description.conversionRelation,
            ""))
      return error;
  } else if (operationProfile.isWideningConversion) {
    const bool isI16ToI32 =
        operationProfile.operation == RVVSelectedBodyOperationKind::WidenI16ToI32;
    std::int64_t expectedSourceSEW =
        isI16ToI32 ? tcrv::rvv::getRVVSEW16Bits()
                   : tcrv::rvv::getRVVFirstSliceSEWBits();
    llvm::StringRef expectedSourceLMUL =
        isI16ToI32 ? tcrv::rvv::getRVVLMULMF2()
                   : tcrv::rvv::getRVVLMULM1();
    llvm::StringRef expectedSourceVectorType =
        getRVVSelectedBodyVectorTypeName(expectedSourceSEW,
                                         expectedSourceLMUL);
    llvm::StringRef expectedSourceVectorCType =
        getRVVSelectedBodySignedVectorCType(expectedSourceSEW,
                                            expectedSourceLMUL);
    llvm::StringRef expectedSourceLoadIntrinsic =
        getRVVSelectedBodyVectorLoadIntrinsic(expectedSourceSEW,
                                              expectedSourceLMUL);
    llvm::StringRef expectedConversionRelation =
        isI16ToI32 ? kRVVWidenI16ToI32ConversionRelation
                   : kRVVWideningConversionRelation;
    if (description.sourceSEW != expectedSourceSEW)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " source SEW must be provider-derived from typed source vector for "
          "widening conversion");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source LMUL", description.sourceLMUL,
            expectedSourceLMUL))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector type", description.sourceVectorTypeName,
            expectedSourceVectorType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector C type", description.sourceVectorCType,
            expectedSourceVectorCType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector-load intrinsic",
            description.sourceVectorLoadIntrinsic,
            expectedSourceLoadIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "conversion relation", description.conversionRelation,
            expectedConversionRelation))
      return error;
  } else if (isDequantizationRoute) {
    std::optional<RVVDequantizationRouteFacts> routeFacts =
        getRVVDequantizationRouteFacts(operationProfile.operation);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " dequantization requires provider-owned route facts before "
          "source/result/scale route-description verification");
    if (description.sourceSEW != routeFacts->sourceSEW)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " source SEW must be provider-derived from typed source vector for "
          "dequantization");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source LMUL", description.sourceLMUL,
            routeFacts->sourceLMUL))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector type", description.sourceVectorTypeName,
            routeFacts->sourceVectorTypeName))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector C type", description.sourceVectorCType,
            routeFacts->sourceVectorCType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector-load intrinsic",
            description.sourceVectorLoadIntrinsic,
            routeFacts->sourceVectorLoadIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequantization relation",
            description.dequantizationRelation,
            routeFacts->dequantizationRelation))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant conversion kind", description.conversionKind,
            routeFacts->dequantizationKind))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant convert intrinsic",
            description.dequantizeConvertIntrinsic,
            routeFacts->convertIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant scale intrinsic",
            description.dequantizeScaleIntrinsic,
            routeFacts->scaleIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant scale role", description.dequantScaleRole,
            routeFacts->scaleRole))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant scale C type", description.dequantScaleCType,
            routeFacts->scaleCType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant scale name", description.dequantScaleName,
            routeFacts->scaleName))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "conversion relation", description.conversionRelation,
            ""))
      return error;
  } else if (isDequantClampF32Epilogue) {
    const std::int64_t expectedSourceSEW =
        tcrv::rvv::getRVVFirstSliceSEWBits();
    const llvm::StringRef expectedSourceLMUL = tcrv::rvv::getRVVLMULM1();
    llvm::StringRef expectedSourceVectorType =
        getRVVSelectedBodyVectorTypeName(expectedSourceSEW,
                                         expectedSourceLMUL);
    llvm::StringRef expectedSourceVectorCType =
        getRVVSelectedBodySignedVectorCType(expectedSourceSEW,
                                            expectedSourceLMUL);
    llvm::StringRef expectedSourceLoadIntrinsic =
        getRVVSelectedBodyVectorLoadIntrinsic(expectedSourceSEW,
                                              expectedSourceLMUL);
    if (description.sourceSEW != expectedSourceSEW)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " source SEW must be provider-derived from typed i32 source vector "
          "for dequant-clamp epilogue");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source LMUL", description.sourceLMUL,
            expectedSourceLMUL))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector type", description.sourceVectorTypeName,
            expectedSourceVectorType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector C type", description.sourceVectorCType,
            expectedSourceVectorCType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector-load intrinsic",
            description.sourceVectorLoadIntrinsic,
            expectedSourceLoadIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequantization relation",
            description.dequantizationRelation,
            kRVVDequantizeI32ToF32Relation))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant conversion kind", description.conversionKind,
            kRVVDequantizeI32ToF32Kind))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant convert intrinsic",
            description.dequantizeConvertIntrinsic,
            getRVVSelectedBodyI32ToF32DequantConvertIntrinsic()))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant scale intrinsic",
            description.dequantizeScaleIntrinsic,
            getRVVSelectedBodyF32ScalarScaleIntrinsic()))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant scale role", description.dequantScaleRole,
            "dequant-scale-value"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant scale C type", description.dequantScaleCType,
            "float"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "dequant scale name", description.dequantScaleName,
            "scale"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "conversion relation", description.conversionRelation,
            ""))
      return error;
  } else {
    if (description.sourceSEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " source SEW must be empty for non-conversion routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source LMUL", description.sourceLMUL, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector type", description.sourceVectorTypeName,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector C type", description.sourceVectorCType,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector-load intrinsic",
            description.sourceVectorLoadIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "conversion relation", description.conversionRelation,
            ""))
      return error;
  }
  if (operationProfile.isCompareSelect || operationProfile.isMaskedArithmetic ||
      operationProfile.isMaskedMemoryMovement ||
      isComputedMaskWideningDotReduce ||
      isComputedMaskStridedInputWideningDotReduce ||
      isComputedMaskedMAcc ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
      isComputedMaskStandaloneReductionRoute ||
      isRuntimeScalarComputedMaskStandaloneReductionRoute) {
    if (llvm::Error error =
            requireRouteDescriptionField(context, "mask type",
                                         description.maskTypeName,
                                         configProfile.maskTypeName))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "mask C type",
                                         description.maskCType,
                                         configProfile.maskCType))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask type", description.maskTypeName, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask C type", description.maskCType, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "setvl intrinsic", description.setVLIntrinsic,
          configProfile.setVLIntrinsic))
    return error;
  llvm::StringRef expectedDescriptionVectorLoadIntrinsic =
      operationProfile.operation ==
              RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd
          ? getRVVSelectedBodyVectorLoadIntrinsic(
                tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2())
          : configProfile.vectorLoadIntrinsic;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector-load intrinsic", description.vectorLoadIntrinsic,
          expectedDescriptionVectorLoadIntrinsic))
    return error;
  if (operationProfile.isIndexedMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-load intrinsic", description.indexLoadIntrinsic,
            configProfile.indexLoadIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-scale intrinsic", description.indexScaleIntrinsic,
            configProfile.indexScaleIntrinsic))
      return error;
    if (operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-load intrinsic",
              description.indexedLoadIntrinsic,
              configProfile.indexedLoadIntrinsic))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-store intrinsic",
              description.indexedStoreIntrinsic, ""))
        return error;
    } else if (operationProfile.operation ==
                   RVVSelectedBodyOperationKind::
                       ComputedMaskIndexedGatherLoadUnitStore ||
               operationProfile.operation ==
                   RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskIndexedGatherLoadUnitStore) {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-load intrinsic",
              description.indexedLoadIntrinsic, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-store intrinsic",
              description.indexedStoreIntrinsic, ""))
        return error;
    } else if (operationProfile.operation ==
                   RVVSelectedBodyOperationKind::
                       ComputedMaskIndexedScatterStoreUnitLoad ||
               operationProfile.operation ==
                   RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad) {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-load intrinsic",
              description.indexedLoadIntrinsic, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-store intrinsic",
              description.indexedStoreIntrinsic,
              getRVVSelectedBodyMaskedIndexedStoreIntrinsic(
                  configProfile.sew, configProfile.lmul)))
        return error;
    } else if (operationProfile.operation ==
               RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskIndexedGatherMAccScatter) {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-load intrinsic",
              description.indexedLoadIntrinsic, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-store intrinsic",
              description.indexedStoreIntrinsic,
              getRVVSelectedBodyMaskedIndexedStoreIntrinsic(
                  configProfile.sew, configProfile.lmul)))
        return error;
    } else {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-load intrinsic",
              description.indexedLoadIntrinsic, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-store intrinsic",
              description.indexedStoreIntrinsic,
              configProfile.indexedStoreIntrinsic))
        return error;
    }
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-load intrinsic", description.indexLoadIntrinsic,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-scale intrinsic", description.indexScaleIntrinsic,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed-load intrinsic",
            description.indexedLoadIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed-store intrinsic",
            description.indexedStoreIntrinsic, ""))
      return error;
  }
  if (isContractionRoute) {
    // Contraction-specific load mirrors are verified by the contraction owner.
  } else if (operationProfile.operation == RVVSelectedBodyOperationKind::StridedAdd ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::StridedLoadUnitStore) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided-load intrinsic",
            description.stridedLoadIntrinsic,
            configProfile.stridedLoadIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided-load intrinsic",
            description.stridedLoadIntrinsic, ""))
      return error;
  }
  if (operationProfile.operation ==
          RVVSelectedBodyOperationKind::MaskedUnitLoadStore ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked-load intrinsic",
            description.maskedLoadIntrinsic,
            getRVVSelectedBodyMaskedLoadIntrinsic(configProfile.sew,
                                                  configProfile.lmul)))
      return error;
  } else if (operationProfile.operation ==
             RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked-load intrinsic",
            description.maskedLoadIntrinsic,
            getRVVSelectedBodyRuntimeScalarMaskedLoadIntrinsic(configProfile)))
      return error;
  } else if (operationProfile.operation ==
             RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked-load intrinsic",
            description.maskedLoadIntrinsic,
            getRVVSelectedBodyMaskedStridedLoadIntrinsic(configProfile.sew,
                                                         configProfile.lmul)))
      return error;
  } else if (operationProfile.operation ==
                 RVVSelectedBodyOperationKind::
                     ComputedMaskIndexedGatherLoadUnitStore ||
             operationProfile.operation ==
                 RVVSelectedBodyOperationKind::
                     RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
             operationProfile.operation ==
                 RVVSelectedBodyOperationKind::
                     RuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked-load intrinsic",
            description.maskedLoadIntrinsic,
            getRVVSelectedBodyMaskedIndexedLoadIntrinsic(configProfile.sew,
                                                         configProfile.lmul)))
      return error;
  } else if (operationProfile.operation ==
                 RVVSelectedBodyOperationKind::
                     ComputedMaskSegment2LoadUnitStore ||
             operationProfile.operation ==
                 RVVSelectedBodyOperationKind::
                     RuntimeScalarComputedMaskSegment2LoadUnitStore) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked-load intrinsic",
            description.maskedLoadIntrinsic,
            getRVVSelectedBodyMaskedSegmentLoadIntrinsic(
                configProfile.sew, configProfile.lmul, 2)))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked-load intrinsic",
            description.maskedLoadIntrinsic, ""))
      return error;
  }
  if (!isElementwiseArithmeticRoute && !isScalarBroadcastElementwiseRoute)
    if (llvm::Error error = requireRouteDescriptionField(
            context, "store intrinsic", description.storeIntrinsic,
            operationProfile.operation ==
                        RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
                    operationProfile.operation ==
                        RVVSelectedBodyOperationKind::
                            ComputedMaskIndexedScatterStoreUnitLoad ||
                    operationProfile.operation ==
                        RVVSelectedBodyOperationKind::
                            RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
                    operationProfile.operation ==
                        RVVSelectedBodyOperationKind::
                            RuntimeScalarComputedMaskIndexedGatherMAccScatter
                ? llvm::StringRef()
            : isRuntimeScalarComputedMaskStore
                ? getRVVSelectedBodyRuntimeScalarMaskedStoreIntrinsic(
                      configProfile)
            : isMaskedUnitStore
                ? getRVVSelectedBodyMaskedStoreIntrinsic(configProfile.sew,
                                                         configProfile.lmul)
            : isStandaloneReductionRoute
                ? getRVVStandaloneReductionScalarResultStoreIntrinsic(
                      description.sew, description.lmul)
            : isDequantizationRoute
                ? getRVVSelectedBodyFloatStoreIntrinsic(description.sew,
                                                        description.lmul)
            : isProductReductionDequantizationRoute
                ? getRVVSelectedBodyFloatStoreIntrinsic(description.sew,
                                                        description.lmul)
                : configProfile.storeIntrinsic))
      return error;
  if (!isElementwiseArithmeticRoute && !isScalarBroadcastElementwiseRoute) {
    if (operationProfile.operation ==
            RVVSelectedBodyOperationKind::UnitLoadStridedStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedStore) {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "strided-store intrinsic",
              description.stridedStoreIntrinsic,
              operationProfile.operation ==
                      RVVSelectedBodyOperationKind::ComputedMaskStridedStore
                  ? getRVVSelectedBodyMaskedStridedStoreIntrinsic(
                        configProfile.sew, configProfile.lmul)
                  : configProfile.stridedStoreIntrinsic))
        return error;
    } else {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "strided-store intrinsic",
              description.stridedStoreIntrinsic, ""))
        return error;
    }
  }
  if (!isContractionRoute && !isElementwiseArithmeticRoute &&
      !isScalarBroadcastElementwiseRoute && !isMAccRouteFamilyRoute)
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compute intrinsic", description.intrinsic,
            targetLeaves.intrinsic))
      return error;
  if (operationProfile.isCompareSelect || isComputedMaskedMAcc) {
    if (llvm::Error error = requireRouteDescriptionText(
            context, "compare predicate kind",
            description.comparePredicateKind))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic,
            targetLeaves.compareIntrinsic))
      return error;
    if (isRuntimeScalarDualCompareMaskAndSelect || isF32ClampSelect ||
        isDequantClampF32Epilogue) {
      if (llvm::Error error = requireRouteDescriptionText(
              context, "secondary compare predicate kind",
              description.secondaryComparePredicateKind))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "secondary compare intrinsic",
              description.secondaryCompareIntrinsic,
              targetLeaves.compareIntrinsic))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "mask-and intrinsic", description.maskAndIntrinsic,
              isRuntimeScalarDualCompareMaskAndSelect
                  ? targetLeaves.maskAndIntrinsic
                  : llvm::StringRef()))
        return error;
    } else {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "secondary compare predicate kind",
              description.secondaryComparePredicateKind, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "secondary compare intrinsic",
              description.secondaryCompareIntrinsic, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "mask-and intrinsic", description.maskAndIntrinsic, ""))
        return error;
    }
  } else if (isContractionRoute) {
    // Contraction compare mirrors are verified by the owner.
  } else if (isElementwiseArithmeticRoute || isScalarBroadcastElementwiseRoute) {
    // Elementwise route-family owner verifies compare/masked-merge mirrors.
  } else if (operationProfile.isMaskedArithmetic ||
             operationProfile.isMaskedMemoryMovement ||
             isComputedMaskWideningDotReduce ||
             isComputedMaskStridedInputWideningDotReduce ||
             isComputedMaskedMAcc ||
             isComputedMaskStandaloneReductionRoute ||
             isRuntimeScalarComputedMaskStandaloneReductionRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic,
            targetLeaves.compareIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "secondary compare predicate kind",
            description.secondaryComparePredicateKind, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "secondary compare intrinsic",
            description.secondaryCompareIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask-and intrinsic", description.maskAndIntrinsic, ""))
      return error;
  }
  if (isContractionRoute || isElementwiseArithmeticRoute ||
      isScalarBroadcastElementwiseRoute || isMAccRouteFamilyRoute) {
    // Owner-local route families verify masked-merge mirrors.
  } else if (operationProfile.isMaskedArithmetic ||
      operationProfile.isMaskedMemoryMovement ||
      isComputedMaskWideningDotReduce ||
      isComputedMaskStridedInputWideningDotReduce ||
      isComputedMaskedMAcc ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
      isComputedMaskStandaloneReductionRoute ||
      isRuntimeScalarComputedMaskStandaloneReductionRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked merge intrinsic",
            description.maskedMergeIntrinsic,
            targetLeaves.maskedMergeIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked merge intrinsic",
            description.maskedMergeIntrinsic, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "result value name", description.resultName,
          operationProfile.resultName))
    return error;
  if (operationProfile.isCompareSelect || operationProfile.isMaskedArithmetic ||
      operationProfile.isMaskedMemoryMovement ||
      isComputedMaskWideningDotReduce ||
      isComputedMaskStridedInputWideningDotReduce ||
      isComputedMaskedMAcc ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
      isComputedMaskStandaloneReductionRoute ||
      isRuntimeScalarComputedMaskStandaloneReductionRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask value name", description.maskName,
            operationProfile.maskName))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask value name", description.maskName, ""))
      return error;
  }
  if (operationProfile.isMaskedArithmetic || isPlainCompareSelect ||
      isComputedMaskSelect ||
      isRuntimeScalarCompareSelect ||
      isRuntimeScalarDualCompareMaskAndSelect || isF32ClampSelect ||
      isDequantClampF32Epilogue ||
      isComputedMaskedMAcc) {
    llvm::StringRef expectedMaskMemoryForm =
        (isF32ClampSelect || isDequantClampF32Epilogue)
            ? compareSelectFacts->maskMemoryForm
        : (isPlainCompareSelect || isComputedMaskSelect ||
         isRuntimeScalarCompareSelect ||
         isComputedMaskedMAcc)
            ? llvm::StringRef(kRVVComputedMaskMemoryMaskMemoryForm)
        : isRuntimeScalarDualCompareMaskAndSelect
            ? llvm::StringRef("composed-compare-produced-mask")
            : llvm::StringRef();
    llvm::StringRef expectedInactiveLaneContract =
        (isComputedMaskSelect || isRuntimeScalarCompareSelect ||
         isRuntimeScalarDualCompareMaskAndSelect || isF32ClampSelect ||
         isDequantClampF32Epilogue)
            ? llvm::StringRef()
        : isComputedMaskedMAcc
            ? llvm::StringRef("masked-macc-false-lanes-preserve-accumulator")
            : llvm::StringRef(kRVVMaskedInactiveLaneContract);
    llvm::StringRef expectedMaskedPassthroughLayout =
        (isComputedMaskSelect || isRuntimeScalarCompareSelect ||
         isRuntimeScalarDualCompareMaskAndSelect || isF32ClampSelect ||
         isDequantClampF32Epilogue)
            ? llvm::StringRef()
        : isComputedMaskedMAcc
            ? llvm::StringRef("accumulator-vector-preserves-inactive-lanes")
            : llvm::StringRef(kRVVMaskedPassthroughLayout);
    llvm::StringRef expectedMaskRole =
        (isF32ClampSelect || isDequantClampF32Epilogue)
            ? compareSelectFacts->maskRole
        : isRuntimeScalarDualCompareMaskAndSelect
            ? llvm::StringRef("predicate-mask-produced-by-mask-and")
            : llvm::StringRef(kRVVMaskedPredicateMaskRole);
    llvm::StringRef expectedMaskSource =
        (isF32ClampSelect || isDequantClampF32Epilogue)
            ? compareSelectFacts->maskSource
        : isRuntimeScalarDualCompareMaskAndSelect
            ? llvm::StringRef(
                  "mask-and-of-two-runtime-scalar-compare-produced-masks")
            : llvm::StringRef(kRVVMaskedCompareMaskSource);
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask role", description.maskRole, expectedMaskRole))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource,
            expectedMaskSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask memory form", description.maskMemoryForm,
            expectedMaskMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask composition", description.maskComposition,
            isRuntimeScalarDualCompareMaskAndSelect ? "and" : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive lane contract",
            description.inactiveLaneContract, expectedInactiveLaneContract))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout,
            expectedMaskedPassthroughLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive-lane zeroing requirement",
            description.inactiveLaneZeroingRequirement, ""))
      return error;
  } else if (isComputedMaskWideningDotReduce ||
             isComputedMaskStridedInputWideningDotReduce ||
             isComputedMaskStandaloneReductionRoute ||
             isRuntimeScalarComputedMaskStandaloneReductionRoute) {
    llvm::StringRef expectedInactiveLaneZeroingRequirement =
        (isComputedMaskStandaloneReductionRoute ||
         isRuntimeScalarComputedMaskStandaloneReductionRoute)
            ? ((operationProfile.operation ==
                        RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd ||
                operationProfile.operation ==
                    RVVSelectedBodyOperationKind::
                        RuntimeScalarComputedMaskStandaloneReduceAdd)
                   ? llvm::StringRef(
                         kRVVStandaloneReductionMaskedInactiveLaneZeroingRequirement)
                   : llvm::StringRef(
                         kRVVStandaloneReductionMaskedInactiveLaneNeutralRequirement))
            : getRVVSelectedBodyContractionExpectedMaskedInactiveLaneZeroingRequirement();
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask role", description.maskRole,
            kRVVMaskedPredicateMaskRole))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource,
            kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask memory form", description.maskMemoryForm,
            kRVVComputedMaskMemoryMaskMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask composition", description.maskComposition, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive lane contract",
            description.inactiveLaneContract, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive-lane zeroing requirement",
            description.inactiveLaneZeroingRequirement,
            expectedInactiveLaneZeroingRequirement))
      return error;
  } else if (operationProfile.isMaskedMemoryMovement) {
    const bool isComputedMask =
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::
                ComputedMaskIndexedGatherLoadUnitStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::
                ComputedMaskIndexedScatterStoreUnitLoad ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskIndexedGatherMAccScatter ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskSegment2LoadUnitStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskSegment2StoreUnitLoad ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
    const bool isMaskedStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::MaskedUnitStore;
    const bool isRuntimeScalarComputedMaskStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
    const bool isRuntimeScalarComputedMaskLoadStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
    const bool isComputedMaskStridedStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
    const bool isComputedMaskIndexedScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
    const bool isRuntimeScalarComputedMaskIndexedScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
    const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedGatherMAccScatter;
    const bool isComputedMaskIndexedScatterLike =
        isComputedMaskIndexedScatter || isRuntimeScalarComputedMaskIndexedScatter ||
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
	    const bool isComputedMaskSegment2Store =
	        operationProfile.operation ==
	        RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
	    const bool isRuntimeScalarComputedMaskSegment2Store =
	        operationProfile.operation ==
	        RVVSelectedBodyOperationKind::
	            RuntimeScalarComputedMaskSegment2StoreUnitLoad;
	    const bool isComputedMaskSegment2Update =
	        operationProfile.operation ==
	        RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
	    const bool isComputedMaskSegment2StoreLike =
	        isComputedMaskSegment2Store ||
	        isRuntimeScalarComputedMaskSegment2Store ||
	        isComputedMaskSegment2Update;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask role", description.maskRole,
            isComputedMask ? kRVVMaskedPredicateMaskRole : kRVVMaskRole))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource,
            isComputedMask ? kRVVMaskedCompareMaskSource : kRVVMaskSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask memory form", description.maskMemoryForm,
            isComputedMask ? kRVVComputedMaskMemoryMaskMemoryForm
                           : kRVVMaskMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask composition", description.maskComposition, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive lane contract",
            description.inactiveLaneContract,
            isComputedMaskStridedStore
                ? kRVVMaskedStridedStoreInactiveLaneContract
            : isComputedMaskIndexedScatterLike
                ? kRVVMaskedIndexedStoreInactiveLaneContract
            : isComputedMaskSegment2StoreLike
                ? kRVVMaskedStoreInactiveLaneContract
            : isRuntimeScalarComputedMaskStore
                ? kRVVMaskedStoreInactiveLaneContract
            : isRuntimeScalarComputedMaskLoadStore
                ? kRVVMaskedMemoryInactiveLaneContract
                : (isMaskedStore ? kRVVMaskedStoreInactiveLaneContract
                                 : kRVVMaskedMemoryInactiveLaneContract)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout,
            isComputedMaskStridedStore
                ? kRVVMaskedStridedStorePassthroughLayout
            : isRuntimeScalarComputedMaskIndexedGatherMAccScatter
                ? kRVVMaskedMemoryPassthroughLayout
            : isComputedMaskIndexedScatterLike
                ? kRVVMaskedIndexedStorePassthroughLayout
            : isComputedMaskSegment2StoreLike
                ? kRVVMaskedStorePassthroughLayout
            : isRuntimeScalarComputedMaskStore
                ? kRVVMaskedStorePassthroughLayout
            : isRuntimeScalarComputedMaskLoadStore
                ? kRVVMaskedMemoryPassthroughLayout
                : (isMaskedStore ? kRVVMaskedStorePassthroughLayout
                                 : kRVVMaskedMemoryPassthroughLayout)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive-lane zeroing requirement",
            description.inactiveLaneZeroingRequirement, ""))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask role", description.maskRole, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask memory form", description.maskMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask composition", description.maskComposition, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive lane contract",
            description.inactiveLaneContract, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive-lane zeroing requirement",
            description.inactiveLaneZeroingRequirement, ""))
      return error;
  }
  if (operationProfile.operation == RVVSelectedBodyOperationKind::ReduceAdd) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction accumulator layout",
            description.reductionAccumulatorLayout,
            kRVVReductionAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction result layout",
            description.reductionResultLayout, kRVVReductionResultLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction store VL", description.reductionStoreVL,
            kRVVReductionStoreVL))
      return error;
  } else if (isStandaloneReductionRoute) {
    llvm::StringRef expectedAccumulatorLayout =
        getRVVStandaloneReductionAccumulatorLayoutForSEW(description.sew);
    llvm::StringRef expectedScalarResultVectorType =
        getRVVStandaloneReductionScalarResultVectorTypeName(description.sew,
                                                            description.lmul);
    llvm::StringRef expectedScalarResultVectorCType =
        getRVVStandaloneReductionScalarResultVectorCType(description.sew,
                                                         description.lmul);
    llvm::StringRef expectedScalarCType =
        getRVVSelectedBodySignedScalarCType(description.sew);
    llvm::StringRef expectedStandaloneSourceVectorType =
        description.vectorTypeName;
    llvm::StringRef expectedStandaloneSourceVectorCType =
        description.vectorCType;
    if (operationProfile.operation ==
        RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd) {
      expectedStandaloneSourceVectorType = getRVVSelectedBodyVectorTypeName(
          tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2());
      expectedStandaloneSourceVectorCType = getRVVSelectedBodySignedVectorCType(
          tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2());
    }
    if (expectedAccumulatorLayout.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " requires a supported typed scalar accumulator layout for the "
          "selected standalone reduction SEW");
    if (expectedScalarResultVectorType.empty() ||
        expectedScalarResultVectorCType.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " requires a supported typed scalar accumulator/result vector "
          "channel for the selected standalone reduction SEW/LMUL");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction source vector type",
            description.standaloneReductionSourceVectorTypeName,
            expectedStandaloneSourceVectorType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction source vector C type",
            description.standaloneReductionSourceVectorCType,
            expectedStandaloneSourceVectorCType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction scalar C type",
            description.standaloneReductionScalarCType, expectedScalarCType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction scalar result vector type",
            description.standaloneReductionScalarResultVectorTypeName,
            expectedScalarResultVectorType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction scalar result vector C type",
            description.standaloneReductionScalarResultVectorCType,
            expectedScalarResultVectorCType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction accumulator layout",
            description.reductionAccumulatorLayout,
            expectedAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction result layout",
            description.reductionResultLayout,
            kRVVStandaloneReductionResultLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction store VL", description.reductionStoreVL,
            kRVVStandaloneReductionStoreVL))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction scalar result runtime boundary",
            description.standaloneReductionScalarResultRuntimeBoundary,
            kRVVStandaloneReductionScalarResultRuntimeBoundary))
      return error;
  } else {
    if (!isProductReductionChainRoute) {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "reduction accumulator layout",
              description.reductionAccumulatorLayout, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "reduction result layout",
              description.reductionResultLayout, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "standalone reduction scalar result runtime boundary",
              description.standaloneReductionScalarResultRuntimeBoundary, ""))
        return error;
    }
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction source vector type",
            description.standaloneReductionSourceVectorTypeName, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction source vector C type",
            description.standaloneReductionSourceVectorCType, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction scalar C type",
            description.standaloneReductionScalarCType, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction scalar result vector type",
            description.standaloneReductionScalarResultVectorTypeName, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "standalone reduction scalar result vector C type",
            description.standaloneReductionScalarResultVectorCType, ""))
      return error;
    if (operationProfile.operation !=
            RVVSelectedBodyOperationKind::WideningDotReduceAdd &&
        !isProductReductionChainRoute &&
        !isStridedInputWideningDotReduce &&
        !isComputedMaskWideningDotReduce &&
        !isComputedMaskStridedInputWideningDotReduce)
      if (llvm::Error error = requireRouteDescriptionField(
              context, "reduction store VL", description.reductionStoreVL, ""))
        return error;
  }
  if (!isContractionRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening multiply-accumulate accumulator layout",
            description.wideningMAccAccumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening multiply-accumulate result layout",
            description.wideningMAccResultLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening multiply-accumulate relation",
            description.wideningMAccRelation, ""))
      return error;
  }
  if (!isContractionRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening dot-product accumulator layout",
            description.wideningDotProductAccumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening dot-product result layout",
            description.wideningDotProductResultLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening dot-product relation",
            description.wideningDotProductRelation, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening product intrinsic",
            description.wideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked widening product intrinsic",
            description.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "scalar seed splat intrinsic",
            description.scalarSeedSplatIntrinsic,
            isStandaloneReductionRoute
                ? getRVVStandaloneReductionScalarSeedSplatIntrinsic(
                      description.sew, description.lmul)
                : ""))
      return error;
  }
  if (isContractionRoute) {
    // Contraction-specific memory mirrors are verified by the owner.
  } else if (isRVVSelectedBodyComputedMaskMAccAccumulationRouteFamilyConsumer(
                 operationProfile.operation)) {
    // Computed-mask MAcc memory mirrors are verified by the RVV owner.
  } else if (operationProfile.operation ==
             RVVSelectedBodyOperationKind::StridedAdd) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout, kRVVStridedMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource,
            kRVVOutStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         "strided-store"))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for non-indexed routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isComputedMaskSelect) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            kRVVComputedMaskSelectMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for computed-mask select routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isRuntimeScalarCompareSelect) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            "unit-stride-lhs-runtime-scalar-threshold-true-false-select-output-runtime-abi"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for runtime scalar compare/select routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isRuntimeScalarDualCompareMaskAndSelect) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for runtime scalar dual-compare "
          "mask-and select routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isF32ClampSelect || isDequantClampF32Epilogue) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            isDequantClampF32Epilogue
                ? llvm::StringRef(
                      "unit-stride-i32-input-runtime-scale-dequant-runtime-lower-upper-select-f32-output-runtime-abi")
                : compareSelectFacts->indexedMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            compareSelectFacts->sourceMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm,
            compareSelectFacts->destinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          (isDequantClampF32Epilogue
               ? " index EEW must be empty for dequant-clamp f32 epilogue routes"
               : " index EEW must be empty for f32 clamp/select routes"));
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isMemoryMovement) {
    const bool isUnitLoadStridedStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::UnitLoadStridedStore;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout,
            isUnitLoadStridedStore ? kRVVUnitLoadStridedStoreMemoryLayout
                                   : kRVVStridedLoadUnitStoreMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (isUnitLoadStridedStore) {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "out stride source", description.outStrideSource,
              kRVVDestinationByteStrideSource))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "source stride source", description.sourceStrideSource,
              ""))
        return error;
    } else {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "out stride source", description.outStrideSource, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "source stride source", description.sourceStrideSource,
              kRVVSourceStrideSource))
        return error;
    }
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            isUnitLoadStridedStore ? kRVVUnitStrideSourceMemoryForm
                                   : kRVVSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm,
            isUnitLoadStridedStore ? "strided-store"
                                   : kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for non-indexed routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isSegmentedMemoryMovement) {
    llvm::StringRef expectedIndexedMemoryLayout =
        computedMaskSegment2Facts ? computedMaskSegment2Facts->segmentMemoryLayout
                                  : llvm::StringRef();
    llvm::StringRef expectedSourceMemoryForm =
        plainSegment2Facts ? plainSegment2Facts->sourceMemoryForm
                           : computedMaskSegment2Facts->sourceMemoryForm;
    llvm::StringRef expectedDestinationMemoryForm =
        plainSegment2Facts ? plainSegment2Facts->destinationMemoryForm
                           : computedMaskSegment2Facts->destinationMemoryForm;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            expectedIndexedMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            expectedSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         expectedDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for segment2 memory routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isMaskedMemoryMovement) {
    const bool isComputedMask =
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::
                ComputedMaskIndexedScatterStoreUnitLoad ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskIndexedGatherMAccScatter ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
    const bool isMaskedStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::MaskedUnitStore;
    const bool isRuntimeScalarComputedMaskStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
    const bool isRuntimeScalarComputedMaskLoadStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
    const bool isComputedMaskStridedStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
    const bool isComputedMaskStridedLoad =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore;
    const bool isComputedMaskIndexedGather =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
    const bool isRuntimeScalarComputedMaskIndexedGather =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
    const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedGatherMAccScatter;
    const bool isComputedMaskIndexedGatherLike =
        isComputedMaskIndexedGather || isRuntimeScalarComputedMaskIndexedGather ||
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
    const bool isComputedMaskIndexedScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
    const bool isRuntimeScalarComputedMaskIndexedScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
    const bool isComputedMaskIndexedScatterLike =
        isComputedMaskIndexedScatter || isRuntimeScalarComputedMaskIndexedScatter ||
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            isComputedMaskStridedStore
                ? kRVVComputedMaskStridedStoreMemoryLayout
            : isComputedMaskStridedLoad
                ? kRVVComputedMaskStridedLoadMemoryLayout
                : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            isMaskedStore
                ? kRVVMaskedStoreMemoryLayout
            : isRuntimeScalarComputedMaskStore
                ? kRVVRuntimeScalarComputedMaskStoreMemoryLayout
            : isRuntimeScalarComputedMaskLoadStore
                ? kRVVRuntimeScalarComputedMaskLoadStoreMemoryLayout
            : isComputedMaskStridedStore
                ? kRVVComputedMaskStridedStoreMemoryLayout
            : isComputedMaskStridedLoad
                ? kRVVComputedMaskStridedLoadMemoryLayout
            : isRuntimeScalarComputedMaskIndexedGather
                ? kRVVRuntimeScalarComputedMaskIndexedGatherMemoryLayout
            : isComputedMaskIndexedGather
                ? kRVVComputedMaskIndexedGatherMemoryLayout
            : isComputedMaskIndexedScatter
                ? kRVVComputedMaskIndexedScatterMemoryLayout
            : isRuntimeScalarComputedMaskIndexedScatter
                ? kRVVRuntimeScalarComputedMaskIndexedScatterMemoryLayout
            : isRuntimeScalarComputedMaskIndexedGatherMAccScatter
                ? kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterMemoryLayout
                : (isComputedMask ? kRVVComputedMaskMemoryLayout
                                  : kRVVMaskedMemoryLayout)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource,
            isComputedMaskStridedStore ? kRVVDestinationByteStrideSource
                                       : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            isComputedMaskStridedLoad ? kRVVSourceByteStrideSource : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            isRuntimeScalarComputedMaskStore
                ? kRVVUnitStrideSourceMemoryForm
            : isComputedMaskIndexedGatherLike
                ? kRVVMaskedIndexedLoadSourceMemoryForm
            : isComputedMaskIndexedScatterLike
                ? kRVVUnitStrideSourceMemoryForm
            : isComputedMaskStridedLoad ? kRVVMaskedStridedLoadSourceMemoryForm
                                        : kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         (isMaskedStore ||
                                          isRuntimeScalarComputedMaskStore)
                                             ? kRVVMaskedStoreDestinationMemoryForm
                                         : isComputedMaskStridedStore
                                             ? kRVVMaskedStridedStoreDestinationMemoryForm
                                         : isComputedMaskIndexedScatterLike
                                             ? kRVVMaskedIndexedStoreDestinationMemoryForm
                                             : kRVVDestinationMemoryForm))
      return error;
    if ((isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike) &&
        description.indexEEW != 32)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be provider-derived as 32 for masked indexed "
          "memory routes");
    if (!isComputedMaskIndexedGatherLike && !isComputedMaskIndexedScatterLike &&
        description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for masked memory routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit,
            (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike)
                ? kRVVIndexedGatherOffsetUnit
                : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource,
            (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike)
                ? kRVVIndexSource
                : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness,
            isComputedMaskIndexedScatterLike
                ? kRVVIndexedScatterIndexUniqueness
                : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm,
            isComputedMaskIndexedGatherLike
                ? kRVVMaskedIndexedLoadSourceMemoryForm
                : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm,
            isComputedMaskIndexedScatterLike
                ? kRVVMaskedIndexedStoreDestinationMemoryForm
                : ""))
      return error;
  } else if (operationProfile.isIndexedMemoryMovement) {
    const bool isScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            isScatter ? kRVVIndexedScatterMemoryLayout
                      : kRVVIndexedGatherMemoryLayout))
      return error;
    if (description.indexEEW != 32)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be provider-derived as 32 for indexed memory");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit,
            kRVVIndexedGatherOffsetUnit))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource,
            kRVVIndexSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness,
            isScatter ? kRVVIndexedScatterIndexUniqueness : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm,
            isScatter ? "" : kRVVIndexedDataMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm,
            isScatter ? kRVVIndexedDestinationMemoryForm : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            isScatter ? kRVVUnitStrideSourceMemoryForm : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm,
            isScatter ? kRVVIndexedDestinationMemoryForm
                      : kRVVDestinationMemoryForm))
      return error;
  } else {
    const bool isUnitStrideSourceStoreRoute =
        isElementwiseArithmeticRoute || isPlainCompareSelect ||
        operationProfile.isWideningConversion || isDequantizationRoute;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            isUnitStrideSourceStoreRoute ? kRVVUnitStrideSourceMemoryForm : ""))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         isUnitStrideSourceStoreRoute
                                             ? kRVVDestinationMemoryForm
                                             : ""))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for non-indexed routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  }
  if (operationProfile.isSegmentedMemoryMovement) {
    const auto requireSegment2FactField =
        [&](llvm::StringRef label, llvm::StringRef actual,
            llvm::StringRef plainExpected,
            llvm::StringRef computedExpected) -> llvm::Error {
      return requireRouteDescriptionField(
          context, label, actual,
          plainSegment2Facts ? plainExpected : computedExpected);
    };
    if (llvm::Error error = requireSegment2FactField(
            "segment memory layout", description.segmentMemoryLayout,
            plainSegment2Facts ? plainSegment2Facts->segmentMemoryLayout
                               : llvm::StringRef(),
            computedMaskSegment2Facts
                ? computedMaskSegment2Facts->segmentMemoryLayout
                : llvm::StringRef()))
      return error;
    std::int64_t expectedSegmentCount =
        plainSegment2Facts ? plainSegment2Facts->segmentCount
                           : computedMaskSegment2Facts->segmentCount;
    if (description.segmentCount != expectedSegmentCount)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment count must be provider-derived from canonical segment2 "
          "route facts before route-description verification");
    if (llvm::Error error = requireSegment2FactField(
            "segment tuple C type", description.segmentTupleCType,
            plainSegment2Facts ? plainSegment2Facts->segmentTupleCType
                               : llvm::StringRef(),
            computedMaskSegment2Facts
                ? computedMaskSegment2Facts->segmentTupleCType
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireSegment2FactField(
            "segment-load intrinsic", description.segmentLoadIntrinsic,
            plainSegment2Facts ? plainSegment2Facts->segmentLoadIntrinsic
                               : llvm::StringRef(),
            computedMaskSegment2Facts
                ? computedMaskSegment2Facts->segmentLoadIntrinsic
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireSegment2FactField(
            "segment-store intrinsic", description.segmentStoreIntrinsic,
            plainSegment2Facts ? plainSegment2Facts->segmentStoreIntrinsic
                               : llvm::StringRef(),
            computedMaskSegment2Facts
                ? computedMaskSegment2Facts->segmentStoreIntrinsic
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireSegment2FactField(
            "segment field extract intrinsic",
            description.segmentFieldExtractIntrinsic,
            plainSegment2Facts ? plainSegment2Facts->segmentFieldExtractIntrinsic
                               : llvm::StringRef(),
            computedMaskSegment2Facts
                ? computedMaskSegment2Facts->segmentFieldExtractIntrinsic
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireSegment2FactField(
            "field0 role", description.field0Role,
            plainSegment2Facts ? plainSegment2Facts->field0Role
                               : llvm::StringRef(),
            computedMaskSegment2Facts ? computedMaskSegment2Facts->field0Role
                                      : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireSegment2FactField(
            "field1 role", description.field1Role,
            plainSegment2Facts ? plainSegment2Facts->field1Role
                               : llvm::StringRef(),
            computedMaskSegment2Facts ? computedMaskSegment2Facts->field1Role
                                      : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireSegment2FactField(
            "field0 result name", description.field0Name,
            plainSegment2Facts ? plainSegment2Facts->field0Name
                               : llvm::StringRef(),
            computedMaskSegment2Facts ? computedMaskSegment2Facts->field0Name
                                      : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireSegment2FactField(
            "field1 result name", description.field1Name,
            plainSegment2Facts ? plainSegment2Facts->field1Name
                               : llvm::StringRef(),
            computedMaskSegment2Facts ? computedMaskSegment2Facts->field1Name
                                      : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireSegment2FactField(
            "field0 source memory form", description.field0SourceMemoryForm,
            plainSegment2Facts ? plainSegment2Facts->field0SourceMemoryForm
                               : llvm::StringRef(),
            computedMaskSegment2Facts
                ? computedMaskSegment2Facts->field0SourceMemoryForm
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireSegment2FactField(
            "field1 source memory form", description.field1SourceMemoryForm,
            plainSegment2Facts ? plainSegment2Facts->field1SourceMemoryForm
                               : llvm::StringRef(),
            computedMaskSegment2Facts
                ? computedMaskSegment2Facts->field1SourceMemoryForm
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireSegment2FactField(
            "field0 destination memory form",
            description.field0DestinationMemoryForm,
            plainSegment2Facts ? plainSegment2Facts->field0DestinationMemoryForm
                               : llvm::StringRef(),
            computedMaskSegment2Facts
                ? computedMaskSegment2Facts->field0DestinationMemoryForm
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireSegment2FactField(
            "field1 destination memory form",
            description.field1DestinationMemoryForm,
            plainSegment2Facts ? plainSegment2Facts->field1DestinationMemoryForm
                               : llvm::StringRef(),
            computedMaskSegment2Facts
                ? computedMaskSegment2Facts->field1DestinationMemoryForm
                : llvm::StringRef()))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment memory layout",
            description.segmentMemoryLayout, ""))
      return error;
    if (description.segmentCount != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment count must be empty for non-segmented routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment tuple C type", description.segmentTupleCType,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment-load intrinsic",
            description.segmentLoadIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment-store intrinsic",
            description.segmentStoreIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment field extract intrinsic",
            description.segmentFieldExtractIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 role", description.field0Role, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 role", description.field1Role, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 result name", description.field0Name, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 result name", description.field1Name, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 source memory form",
            description.field0SourceMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 source memory form",
            description.field1SourceMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 destination memory form",
            description.field0DestinationMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 destination memory form",
            description.field1DestinationMemoryForm, ""))
      return error;
  }
  if (!isElementwiseArithmeticRoute && !isScalarBroadcastElementwiseRoute) {
    if (description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad ||
        description.memoryForm == RVVSelectedBodyMemoryForm::RHSScalarBroadcast ||
        description.memoryForm ==
            RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore ||
        description.memoryForm ==
            RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect ||
        description.memoryForm ==
            RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore ||
        description.memoryForm ==
            RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskUnitStrideMAcc ||
        description.memoryForm == RVVSelectedBodyMemoryForm::
                                      RuntimeScalarComputedMaskUnitStrideStandaloneReduction)
      if (llvm::Error error = requireRouteDescriptionText(
              context, "RHS broadcast intrinsic",
              description.rhsBroadcastIntrinsic))
        return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "RHS broadcast intrinsic",
            description.rhsBroadcastIntrinsic,
            targetLeaves.rhsBroadcastIntrinsic))
      return error;
  }

  if (llvm::Error error = verifyRVVSelectedBodyConstructionRuntimeABIParameters(
          description.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));

  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
