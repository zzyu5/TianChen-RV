#include "TianChenRV/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

void copyLowPrecisionResourceAttrs(mlir::Operation *source,
                                   mlir::Operation *destination) {
  for (mlir::NamedAttribute attr : source->getAttrs())
    if (isRVVLowPrecisionResourceAttrName(attr.getName().getValue()))
      destination->setAttr(attr.getName(), attr.getValue());
}

llvm::Expected<std::string>
requireLowPrecisionResourceStringFact(mlir::Operation *op,
                                      llvm::StringRef attrName) {
  if (!op)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "body carrying pass-produced low-precision direct-contraction resource "
        "facts before selected-body resource realization");
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV contraction selected-body realization "
                    "requires pass-produced low-precision direct-contraction "
                    "resource fact '") +
        attrName + "' before selected-body resource realization");
  return attr.getValue().str();
}

llvm::Expected<std::int64_t>
requireLowPrecisionResourceIntegerFact(mlir::Operation *op,
                                       llvm::StringRef attrName) {
  if (!op)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "body carrying pass-produced low-precision direct-contraction resource "
        "facts before selected-body resource realization");
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV contraction selected-body realization "
                    "requires pass-produced low-precision direct-contraction "
                    "resource fact '") +
        attrName + "' before selected-body resource realization");
  return attr.getInt();
}

llvm::Error requireLowPrecisionResourceExpectedStringFact(
    mlir::Operation *op, llvm::StringRef attrName,
    llvm::StringRef expected) {
  llvm::Expected<std::string> value =
      requireLowPrecisionResourceStringFact(op, attrName);
  if (!value)
    return value.takeError();
  if (*value == expected)
    return llvm::Error::success();
  return makeRVVPluginError(
      llvm::Twine("pre-realized RVV contraction selected-body realization "
                  "cannot consume stale or unsupported low-precision "
                  "direct-contraction resource fact '") +
      attrName + "': expected '" + expected + "' but found '" + *value + "'");
}

llvm::Error requireLowPrecisionResourceExpectedIntegerFact(
    mlir::Operation *op, llvm::StringRef attrName, std::int64_t expected) {
  llvm::Expected<std::int64_t> value =
      requireLowPrecisionResourceIntegerFact(op, attrName);
  if (!value)
    return value.takeError();
  if (*value == expected)
    return llvm::Error::success();
  return makeRVVPluginError(
      llvm::Twine("pre-realized RVV contraction selected-body realization "
                  "cannot consume stale or unsupported low-precision "
                  "direct-contraction resource fact '") +
      attrName + "': expected " + llvm::Twine(expected) + " but found " +
      llvm::Twine(*value));
}

llvm::Error requireLowPrecisionResourceExpectedIntegerFact(
    mlir::Operation *op, llvm::StringRef attrName,
    std::int64_t actual, std::int64_t expected) {
  llvm::Expected<std::int64_t> value =
      requireLowPrecisionResourceIntegerFact(op, attrName);
  if (!value)
    return value.takeError();
  if (actual != expected)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV contraction selected-body realization "
                    "cannot consume stale or unsupported low-precision "
                    "direct-contraction resource plan for '") +
        attrName + "': selected body expects " + llvm::Twine(actual) +
        " but provider primitive facts require " + llvm::Twine(expected));
  if (*value == expected)
    return llvm::Error::success();
  return makeRVVPluginError(
      llvm::Twine("pre-realized RVV contraction selected-body realization "
                  "cannot consume stale or unsupported low-precision "
                  "direct-contraction resource fact '") +
      attrName + "': expected " + llvm::Twine(expected) +
      " matching the selected primitive/resource plan but found " +
      llvm::Twine(*value));
}

llvm::Error requireLowPrecisionResourceCandidatePrimitiveStringMatch(
    llvm::StringRef field, llvm::StringRef candidateValue,
    llvm::StringRef primitiveValue) {
  if (!primitiveValue.empty() && candidateValue == primitiveValue)
    return llvm::Error::success();
  return makeRVVPluginError(
      llvm::Twine("pre-realized RVV contraction selected-body realization "
                  "cannot consume stale or unsupported low-precision "
                  "direct-contraction resource candidate for '") +
      field + "': candidate selected '" + candidateValue +
      "' but provider primitive facts require '" + primitiveValue + "'");
}

mlir::Operation *createRealizedSetVL(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value nValue,
                                     std::int64_t sew, llvm::StringRef lmul,
                                     tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.setvl");
  state.addOperands(nValue);
  state.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  tcrv::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  return builder.create(state);
}

tcrv::rvv::WithVLOp createRealizedWithVL(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value vlValue,
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role, mlir::ArrayAttr requires, std::int64_t sew,
    llvm::StringRef lmul, tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.with_vl");
  state.addOperands(vlValue);
  tcrv::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  state.addAttribute(rvv::getRVVSourceKernelAttrName(),
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(rvv::getRVVSelectedVariantAttrName(),
                     symbolRef(builder, variant.getSymName()));
  state.addAttribute(rvv::getRVVOriginAttrName(),
                     builder.getStringAttr(kRVVPluginName));
  state.addAttribute(rvv::getRVVSelectedPathRoleAttrName(),
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(rvv::getRVVStatusAttrName(),
                     builder.getStringAttr(rvv::getRVVLoweringBoundaryStatus()));
  state.addAttribute(rvv::getRVVRequiredCapabilitiesAttrName(), requires);
  state.addAttribute(rvv::getRVVConstructionProtocolMetadataName(),
                     builder.getStringAttr(
                         rvv::getRVVConstructionProtocolVersion()));
  state.addRegion();
  auto withVL = llvm::cast<tcrv::rvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

llvm::Expected<RVVLowPrecisionContractionResourceCandidate>
materializeLowPrecisionResourceRealizationAttrs(
    mlir::OpBuilder &builder, mlir::Operation *source,
    mlir::Operation *destination, bool usesProductReductionDequantClamp,
    const RVVLowPrecisionWideningReductionPrimitiveFacts &primitiveFacts,
    llvm::StringRef tailPolicy, llvm::StringRef maskPolicy,
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t productSEW, llvm::StringRef productLMUL,
    std::int64_t reductionResultSEW,
    llvm::StringRef reductionResultLMUL) {
  if (!destination)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "realized with_vl operation before consuming low-precision resource "
        "facts");

  if (llvm::Expected<std::string> candidateSet =
          requireLowPrecisionResourceStringFact(
              source, kRVVLowPrecisionResourceCandidateSetAttrName)) {
    (void)*candidateSet;
  } else {
    return candidateSet.takeError();
  }
  llvm::Expected<std::int64_t> vectorRegisterBudget =
      requireLowPrecisionResourceIntegerFact(
          source, kRVVLowPrecisionResourceVectorRegisterBudgetAttrName);
  if (!vectorRegisterBudget)
    return vectorRegisterBudget.takeError();

  const RVVLowPrecisionContractionResourceOperation operation =
      usesProductReductionDequantClamp
          ? RVVLowPrecisionContractionResourceOperation::
                ProductReductionDequantClampF32
          : RVVLowPrecisionContractionResourceOperation::
                ProductReductionDequantizeF32;
  llvm::SmallVector<RVVLowPrecisionContractionResourceCandidate, 3>
      candidates = buildRVVLowPrecisionProductReductionResourceCandidates(
          operation, tailPolicy, maskPolicy, sourceSEW, sourceLMUL,
          productSEW, productLMUL, reductionResultSEW, reductionResultLMUL,
          reductionResultSEW, reductionResultLMUL, *vectorRegisterBudget);
  std::optional<RVVLowPrecisionContractionResourceCandidate> selected =
      selectRVVLowPrecisionProductReductionResourceCandidate(candidates);
  if (llvm::Expected<std::string> selectedCandidateID =
          requireLowPrecisionResourceStringFact(
              source, kRVVLowPrecisionResourceSelectedCandidateAttrName)) {
    selected = findRVVLowPrecisionProductReductionResourceCandidate(
        candidates, *selectedCandidateID);
    if (!selected)
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV contraction selected-body "
                      "realization cannot match explicit selected "
                      "low-precision resource candidate '") +
          *selectedCandidateID +
          "' in the provider-owned product-reduction candidate set");
  } else {
    return selectedCandidateID.takeError();
  }
  if (!selected) {
    llvm::StringRef rejection =
        candidates.empty() ? llvm::StringRef("no-resource-candidates-built")
                           : candidates.front().rejectionReason;
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV contraction selected-body realization "
                    "pruned every low-precision direct-contraction resource "
                    "candidate before selected-body realization: ") +
        rejection + " for vector register budget " +
        llvm::Twine(*vectorRegisterBudget));
  }
  const llvm::StringRef realizationDecision =
      getRVVLowPrecisionContractionResourceRealizationDecision(
          selected->candidateID);
  if (realizationDecision.empty())
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV contraction selected-body realization "
                    "cannot derive a resource decision for selected candidate '") +
        selected->candidateID + "'");

  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "source dtype", selected->sourceElementTypeName,
              primitiveFacts.sourceElementTypeName))
    return std::move(error);
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "product dtype", selected->productElementTypeName,
              primitiveFacts.productElementTypeName))
    return std::move(error);
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "accumulator dtype", selected->accumulatorElementTypeName,
              primitiveFacts.accumulatorElementTypeName))
    return std::move(error);
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "final result dtype", selected->resultElementTypeName,
              primitiveFacts.finalResultElementTypeName))
    return std::move(error);

  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceCandidateSetAttrName,
          selected->candidateSetID))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceSelectedCandidateAttrName,
          selected->candidateID))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceSelectionReasonAttrName,
          selected->selectionReason))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceLegalityScopeAttrName,
          selected->legalityScope))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceSourceDTypeAttrName,
          selected->sourceElementTypeName))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedIntegerFact(
          source, kRVVLowPrecisionResourceSourceSEWAttrName, sourceSEW,
          primitiveFacts.sourceSEW))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceSourceLMULAttrName,
          selected->sourceLMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceOperandFormAttrName,
          selected->operandForm))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceSourceSignednessAttrName,
          selected->sourceSignedness))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedIntegerFact(
          source, kRVVLowPrecisionResourceStorageElementWidthAttrName,
          selected->storageElementWidth))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedIntegerFact(
          source, kRVVLowPrecisionResourceEffectiveElementWidthAttrName,
          selected->effectiveElementWidth))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourcePackingLayoutAttrName,
          selected->packingLayout))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceUnpackIntentAttrName,
          selected->unpackIntent))
    return std::move(error);
  const bool isPackedI4Resource =
      isRVVLowPrecisionResourcePackedI4CandidateID(selected->candidateID);
  if (isPackedI4Resource) {
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source, kRVVLowPrecisionResourcePerformanceFeedbackAttrName,
            kRVVLowPrecisionResourcePackedI4PerformanceFeedback))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source, kRVVLowPrecisionResourcePerformanceBaselineAttrName,
            kRVVLowPrecisionResourcePackedI4PerformanceBaseline))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source,
            kRVVLowPrecisionResourcePerformanceBestSpeedupRangeAttrName,
            kRVVLowPrecisionResourcePackedI4PerformanceBestSpeedupRange))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source, kRVVLowPrecisionResourcePerformanceActionAttrName,
            kRVVLowPrecisionResourcePackedI4PerformanceAction))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source, kRVVLowPrecisionResourcePerformanceMaturityAttrName,
            kRVVLowPrecisionResourcePackedI4PerformanceMaturity))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source,
            kRVVLowPrecisionResourcePerformanceMaturityEvidenceAttrName,
            kRVVLowPrecisionResourcePackedI4PerformanceMaturityEvidence))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source,
            kRVVLowPrecisionResourcePerformanceMaturityOutcomeAttrName,
            kRVVLowPrecisionResourcePackedI4PerformanceMaturityOutcome))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source,
            kRVVLowPrecisionResourcePerformanceSelectionEligibleAttrName,
            kRVVLowPrecisionResourcePackedI4PerformanceSelectionEligible))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source, kRVVLowPrecisionResourceDispatchPreferenceAttrName,
            kRVVLowPrecisionResourcePackedI4DispatchPreference))
      return std::move(error);
  }
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceProductDTypeAttrName,
          selected->productElementTypeName))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedIntegerFact(
          source, kRVVLowPrecisionResourceProductSEWAttrName, productSEW,
          primitiveFacts.productSEW))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceProductLMULAttrName,
          selected->productLMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceProductEMULAttrName,
          selected->productEMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceAccumulatorDTypeAttrName,
          selected->accumulatorElementTypeName))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedIntegerFact(
          source, kRVVLowPrecisionResourceAccumulatorSEWAttrName,
          reductionResultSEW, primitiveFacts.accumulatorSEW))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceAccumulatorLMULAttrName,
          selected->accumulatorLMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceAccumulatorEMULAttrName,
          selected->accumulatorEMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceResultDTypeAttrName,
          selected->resultElementTypeName))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedIntegerFact(
          source, kRVVLowPrecisionResourceResultSEWAttrName,
          reductionResultSEW, primitiveFacts.reductionResultSEW))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceResultLMULAttrName,
          selected->resultLMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceMemoryFormAttrName,
          selected->memoryForm))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceTailPolicyAttrName,
          selected->tailPolicy))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceMaskPolicyAttrName,
          selected->maskPolicy))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedIntegerFact(
          source, kRVVLowPrecisionResourceUnrollFactorAttrName,
          selected->unrollFactor))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedIntegerFact(
          source, kRVVLowPrecisionResourceAccumulatorCountAttrName,
          selected->accumulatorCount))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceReductionLayoutAttrName,
          selected->reductionLayout))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedIntegerFact(
          source, kRVVLowPrecisionResourceVSetVLRegionCountAttrName,
          selected->vsetvlRegionCount))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedIntegerFact(
          source, kRVVLowPrecisionResourcePeakLiveVectorGroupsAttrName,
          selected->peakLiveVectorGroups))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedIntegerFact(
          source, kRVVLowPrecisionResourceVectorRegisterBudgetAttrName,
          selected->vectorRegisterBudget))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceRuntimeAVLSourceAttrName,
          selected->runtimeAVLSource))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVGearboxProducerScopeAttrName, selected->producerScope))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVGearboxConsumerScopeAttrName, selected->consumerScope))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceRuntimeABIOrderAttrName,
          selected->runtimeABIOrder))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourcePrimitiveContractAttrName,
          selected->primitiveContractID))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourcePrimitiveKindAttrName,
          selected->primitiveKind))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourcePrimitiveChainContractAttrName,
          selected->primitiveChainContractID))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourcePrimitiveChainKindAttrName,
          selected->primitiveChainKind))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source,
          kRVVLowPrecisionResourcePrimitiveWideningProductRelationAttrName,
          selected->primitiveWideningProductRelation))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source,
          kRVVLowPrecisionResourcePrimitiveProductReductionChainRelationAttrName,
          selected->primitiveProductReductionChainRelation))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source,
          kRVVLowPrecisionResourcePrimitiveWideningProductIntrinsicAttrName,
          selected->primitiveWideningProductIntrinsic))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourcePrimitiveReductionIntrinsicAttrName,
          selected->primitiveReductionIntrinsic))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source,
          kRVVLowPrecisionResourcePrimitiveScalarSeedSplatIntrinsicAttrName,
          selected->primitiveScalarSeedSplatIntrinsic))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourcePrimitiveAccumulatorLayoutAttrName,
          selected->primitiveAccumulatorLayout))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourcePrimitiveResultLayoutAttrName,
          selected->primitiveResultLayout))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourcePrimitiveReductionStoreVLAttrName,
          selected->primitiveReductionStoreVL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceLegalityAttrName,
          selected->isLegal ? llvm::StringRef(kRVVLowPrecisionResourceLegal)
                            : llvm::StringRef("rejected")))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceRejectionReasonAttrName,
          selected->rejectionReason))
    return std::move(error);

  destination->setAttr(
      kRVVLowPrecisionResourceRealizationProducerAttrName,
      builder.getStringAttr(kRVVLowPrecisionResourceRealizationProducer));
  destination->setAttr(
      kRVVLowPrecisionResourceRealizationDecisionAttrName,
      builder.getStringAttr(realizationDecision));
  destination->setAttr(kRVVGearboxProducerScopeAttrName,
                       builder.getStringAttr(selected->producerScope));
  destination->setAttr(kRVVGearboxConsumerScopeAttrName,
                       builder.getStringAttr(selected->consumerScope));
  destination->setAttr(
      kRVVLowPrecisionResourceRealizedUnrollFactorAttrName,
      builder.getI64IntegerAttr(selected->unrollFactor));
  destination->setAttr(
      kRVVLowPrecisionResourceRealizedVSetVLRegionCountAttrName,
      builder.getI64IntegerAttr(selected->vsetvlRegionCount));
  destination->setAttr(
      kRVVLowPrecisionResourceRealizedPeakLiveVectorGroupsAttrName,
      builder.getI64IntegerAttr(selected->peakLiveVectorGroups));
  if (isPackedI4Resource) {
    destination->setAttr(
        kRVVLowPrecisionResourcePerformanceFeedbackAttrName,
        builder.getStringAttr(kRVVLowPrecisionResourcePackedI4PerformanceFeedback));
    destination->setAttr(
        kRVVLowPrecisionResourcePerformanceBaselineAttrName,
        builder.getStringAttr(kRVVLowPrecisionResourcePackedI4PerformanceBaseline));
    destination->setAttr(
        kRVVLowPrecisionResourcePerformanceBestSpeedupRangeAttrName,
        builder.getStringAttr(
            kRVVLowPrecisionResourcePackedI4PerformanceBestSpeedupRange));
    destination->setAttr(
        kRVVLowPrecisionResourcePerformanceActionAttrName,
        builder.getStringAttr(kRVVLowPrecisionResourcePackedI4PerformanceAction));
  }
  return *selected;
}

mlir::Type getGenericVectorType(mlir::OpBuilder &builder, std::int64_t sew,
                                llvm::StringRef lmul) {
  mlir::Type elementType = builder.getIntegerType(sew);
  return tcrv::rvv::VectorType::get(builder.getContext(), elementType, lmul);
}

mlir::Type getGenericF32VectorType(mlir::OpBuilder &builder,
                                   llvm::StringRef lmul) {
  return tcrv::rvv::VectorType::get(builder.getContext(),
                                    builder.getF32Type(), lmul);
}

mlir::Type getStage1GenericMaskType(mlir::OpBuilder &builder) {
  return tcrv::rvv::MaskType::get(builder.getContext(), builder.getI32Type(),
                                  tcrv::rvv::getRVVLMULM1());
}

mlir::Type getGenericMaskTypeForVector(mlir::OpBuilder &builder,
                                       mlir::Value vector) {
  auto vectorType = llvm::dyn_cast<tcrv::rvv::VectorType>(vector.getType());
  if (!vectorType)
    return getStage1GenericMaskType(builder);
  return tcrv::rvv::MaskType::get(builder.getContext(),
                                  vectorType.getElementType(),
                                  vectorType.getLmul());
}

mlir::Operation *createRealizedGenericLoad(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value buffer,
                                           mlir::Value vl, std::int64_t sew,
                                           llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.load");
  state.addOperands({buffer, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericStridedLoad(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value buffer,
                                                  mlir::Value stride,
                                                  mlir::Value vl,
                                                  std::int64_t sew,
                                                  llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.strided_load");
  state.addOperands({buffer, stride, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericCompare(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value lhs,
                                              mlir::Value rhs,
                                              mlir::Value vl,
                                              llvm::StringRef kind) {
  mlir::OperationState state(loc, "tcrv_rvv.compare");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addTypes(getGenericMaskTypeForVector(builder, lhs));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericWideningMAccCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    llvm::StringRef maccRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulator, mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.widening_macc");
  state.addOperands({lhs, rhs, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addAttribute("macc_relation", builder.getStringAttr(maccRelation));
  state.addTypes(getGenericVectorType(builder,
                                      tcrv::rvv::getRVVFirstSliceSEWBits(),
                                      tcrv::rvv::getRVVLMULM1()));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericWideningDotReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    llvm::StringRef dotProductRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulatorSeed, mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.widening_dot_reduce");
  state.addOperands({lhs, rhs, accumulatorSeed, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addAttribute("dot_product_relation",
                     builder.getStringAttr(dotProductRelation));
  state.addTypes(getGenericVectorType(builder,
                                      tcrv::rvv::getRVVFirstSliceSEWBits(),
                                      tcrv::rvv::getRVVLMULM1()));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericMaskedWideningDotReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef maskRole, llvm::StringRef maskSource,
    llvm::StringRef maskMemoryForm, llvm::StringRef accumulatorLayout,
    llvm::StringRef resultLayout, llvm::StringRef dotProductRelation,
    mlir::Value mask, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulatorSeed, mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_widening_dot_reduce");
  state.addOperands({mask, lhs, rhs, accumulatorSeed, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("mask_role", builder.getStringAttr(maskRole));
  state.addAttribute("mask_source", builder.getStringAttr(maskSource));
  state.addAttribute("mask_memory_form",
                     builder.getStringAttr(maskMemoryForm));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addAttribute("dot_product_relation",
                     builder.getStringAttr(dotProductRelation));
  state.addTypes(getGenericVectorType(builder,
                                      tcrv::rvv::getRVVFirstSliceSEWBits(),
                                      tcrv::rvv::getRVVLMULM1()));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericWideningProductCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef productRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value vl, std::int64_t productSEW, llvm::StringRef productLMUL) {
  mlir::OperationState state(loc, "tcrv_rvv.widening_product");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("product_relation",
                     builder.getStringAttr(productRelation));
  state.addTypes(getGenericVectorType(builder, productSEW, productLMUL));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericStandaloneWideningReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    mlir::Value input, mlir::Value accumulatorSeed, mlir::Value vl,
    std::int64_t resultSEW, llvm::StringRef resultLMUL) {
  mlir::OperationState state(loc, "tcrv_rvv.standalone_reduce");
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("signed_widening_reduce_add"));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(getGenericVectorType(builder, resultSEW, resultLMUL));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericDequantizeCompute(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef dequantizationRelation, mlir::Value source,
    mlir::Value scale, mlir::Value vl, llvm::StringRef resultLMUL) {
  mlir::OperationState state(loc, "tcrv_rvv.dequantize");
  state.addOperands({source, scale, vl});
  state.addAttribute("kind", builder.getStringAttr("i32_to_f32_scaled"));
  state.addAttribute("dequant_relation",
                     builder.getStringAttr(dequantizationRelation));
  state.addTypes(getGenericF32VectorType(builder, resultLMUL));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericF32Splat(mlir::OpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::Value scalar,
                                               mlir::Value vl,
                                               llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.splat");
  state.addOperands({scalar, vl});
  state.addTypes(getGenericF32VectorType(builder, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericSelect(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value mask,
                                             mlir::Value trueValue,
                                             mlir::Value falseValue,
                                             mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.select");
  state.addOperands({mask, trueValue, falseValue, vl});
  state.addTypes(trueValue.getType());
  return builder.create(state);
}

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

void createRealizedVSetVLRegionMarker(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value vl,
                                      llvm::StringRef phase,
                                      std::int64_t regionIndex,
                                      std::int64_t regionCount,
                                      llvm::StringRef resourceDecision) {
  mlir::OperationState state(loc, "tcrv_rvv.vsetvl_region_marker");
  state.addOperands(vl);
  state.addAttribute("phase", builder.getStringAttr(phase));
  state.addAttribute("region_index", builder.getI64IntegerAttr(regionIndex));
  state.addAttribute("region_count", builder.getI64IntegerAttr(regionCount));
  state.addAttribute("resource_decision",
                     builder.getStringAttr(resourceDecision));
  (void)builder.create(state);
}

mlir::Operation *createRealizedGearboxCrossRegionHandoff(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value input,
    mlir::Value vl, mlir::Value runtimeAVL,
    const RVVLowPrecisionContractionResourceCandidate &selectedCandidate) {
  mlir::OperationState state(loc, "tcrv_rvv.gearbox_cross_region_handoff");
  const llvm::StringRef resourceDecision =
      getRVVLowPrecisionContractionResourceRealizationDecision(
          selectedCandidate.candidateID);
  state.addOperands({input, vl, runtimeAVL});
  state.addAttribute(
      "contract",
      builder.getStringAttr(
          "gearbox-product-reduce-to-dequant-cross-region-handoff.v1"));
  state.addAttribute(
      "from_phase",
      builder.getStringAttr(
          getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
              resourceDecision)));
  state.addAttribute("to_phase", builder.getStringAttr("dequant-store"));
  state.addAttribute(
      "region_count",
      builder.getI64IntegerAttr(selectedCandidate.vsetvlRegionCount));
  state.addAttribute("runtime_avl_source",
                     builder.getStringAttr(selectedCandidate.runtimeAVLSource));
  state.addAttribute("resource_decision",
                     builder.getStringAttr(resourceDecision));
  state.addAttribute("producer_scope",
                     builder.getStringAttr(selectedCandidate.producerScope));
  state.addAttribute("consumer_scope",
                     builder.getStringAttr(selectedCandidate.consumerScope));
  state.addAttribute("primitive_chain_contract",
                     builder.getStringAttr(
                         selectedCandidate.primitiveChainContractID));
  state.addAttribute("primitive_chain_kind",
                     builder.getStringAttr(selectedCandidate.primitiveChainKind));
  state.addAttribute("primitive_widening_product_relation",
                     builder.getStringAttr(
                         selectedCandidate.primitiveWideningProductRelation));
  state.addAttribute(
      "primitive_product_reduction_chain_relation",
      builder.getStringAttr(
          selectedCandidate.primitiveProductReductionChainRelation));
  state.addAttribute("primitive_widening_product_intrinsic",
                     builder.getStringAttr(
                         selectedCandidate.primitiveWideningProductIntrinsic));
  state.addAttribute("primitive_reduction_intrinsic",
                     builder.getStringAttr(
                         selectedCandidate.primitiveReductionIntrinsic));
  state.addAttribute("primitive_scalar_seed_splat_intrinsic",
                     builder.getStringAttr(
                         selectedCandidate.primitiveScalarSeedSplatIntrinsic));
  state.addAttribute("primitive_accumulator_layout",
                     builder.getStringAttr(
                         selectedCandidate.primitiveAccumulatorLayout));
  state.addAttribute("primitive_result_layout",
                     builder.getStringAttr(selectedCandidate.primitiveResultLayout));
  state.addAttribute("primitive_reduction_store_vl",
                     builder.getStringAttr(
                         selectedCandidate.primitiveReductionStoreVL));
  state.addTypes(input.getType());
  return builder.create(state);
}

struct RVVSelectedBodyContractionRealizationPlan {
  mlir::Operation *preRealizedBody = nullptr;

  bool usesWideningMAcc = false;
  bool usesDotReduction = false;
  bool usesProductReductionDequantization = false;
  bool usesProductReductionDequantClamp = false;
  bool usesComputedMask = false;
  bool usesStridedInputs = false;

  llvm::StringRef opKind;
  llvm::StringRef productKind;
  llvm::StringRef accumulatorLayout;
  llvm::StringRef resultLayout;
  llvm::StringRef contractionRelation;
  llvm::StringRef productRelation;
  llvm::StringRef productReductionChainRelation;
  llvm::StringRef dequantizationRelation;
  llvm::StringRef scaleRole;
  llvm::StringRef dequantStoreBoundary;
  llvm::StringRef lowerPredicateKind;
  llvm::StringRef upperPredicateKind;
  llvm::StringRef boundOrder;
  llvm::StringRef selectLayout;
  llvm::StringRef predicateKind;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;

  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t productSEW = 0;
  llvm::StringRef productLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
  tcrv::rvv::PolicyAttr policy;

  mlir::Value compareLHS;
  mlir::Value compareRHS;
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value acc;
  mlir::Value scale;
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value out;
  mlir::Value n;
  mlir::Value lhsStride;
  mlir::Value rhsStride;
};

llvm::StringRef stringifyLowPrecisionRealizationTailPolicy(
    tcrv::rvv::TailPolicy policy) {
  switch (policy) {
  case tcrv::rvv::TailPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::TailPolicy::Undisturbed:
    return "undisturbed";
  }
  return {};
}

llvm::StringRef stringifyLowPrecisionRealizationMaskPolicy(
    tcrv::rvv::MaskPolicy policy) {
  switch (policy) {
  case tcrv::rvv::MaskPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  return {};
}

std::optional<RVVSelectedBodyOperationKind>
getLowPrecisionProductReductionRealizationOperation(
    const RVVSelectedBodyContractionRealizationPlan &plan) {
  if (!plan.usesProductReductionDequantization)
    return std::nullopt;
  return plan.usesProductReductionDequantClamp
             ? RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32
             : RVVSelectedBodyOperationKind::
                   WideningProductReduceDequantizeF32;
}

llvm::Error requireLowPrecisionPrimitiveStringField(
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (!expected.empty() && actual == expected)
    return llvm::Error::success();
  return makeRVVPluginError(
      llvm::Twine("pre-realized RVV contraction selected-body realization "
                  "cannot consume stale or unsupported low-precision "
                  "widening-reduction primitive fact '") +
      field + "': expected '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireLowPrecisionPrimitiveIntegerField(
    llvm::StringRef field, std::int64_t actual, std::int64_t expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVPluginError(
      llvm::Twine("pre-realized RVV contraction selected-body realization "
                  "cannot consume stale or unsupported low-precision "
                  "widening-reduction primitive fact '") +
      field + "': expected " + llvm::Twine(expected) + " but found " +
      llvm::Twine(actual));
}

llvm::Error requireLowPrecisionPrimitiveNonEmptyField(
    llvm::StringRef field, llvm::StringRef value) {
  if (!value.empty())
    return llvm::Error::success();
  return makeRVVPluginError(
      llvm::Twine("pre-realized RVV contraction selected-body realization "
                  "requires provider-owned low-precision widening-reduction "
                  "primitive fact '") +
      field + "' before resource-aware realization planning");
}

llvm::Error validateLowPrecisionPrimitiveFactsForRealization(
    const RVVSelectedBodyContractionRealizationPlan &plan,
    const RVVLowPrecisionWideningReductionPrimitiveFacts &primitiveFacts) {
  if (!primitiveFacts.hasFacts)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "provider-owned low-precision widening-reduction primitive facts "
        "before resource-aware realization planning");

  if (llvm::Error error = requireLowPrecisionPrimitiveNonEmptyField(
          "contract", primitiveFacts.contractID))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveNonEmptyField(
          "low-precision primitive contract",
          primitiveFacts.lowPrecisionPrimitiveContractID))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveNonEmptyField(
          "low-precision primitive kind",
          primitiveFacts.lowPrecisionPrimitiveKind))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveNonEmptyField(
          "widening product intrinsic",
          primitiveFacts.wideningProductIntrinsic))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveNonEmptyField(
          "widening reduction intrinsic", primitiveFacts.reductionIntrinsic))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveNonEmptyField(
          "scalar seed splat intrinsic",
          primitiveFacts.scalarSeedSplatIntrinsic))
    return std::move(error);

  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "product kind", plan.productKind, "signed_widening_product"))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "source LMUL", plan.sourceLMUL, primitiveFacts.sourceLMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveIntegerField(
          "source SEW", plan.sourceSEW, primitiveFacts.sourceSEW))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "product LMUL", plan.productLMUL, primitiveFacts.productLMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveIntegerField(
          "product SEW", plan.productSEW, primitiveFacts.productSEW))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "reduction result LMUL", plan.resultLMUL,
          primitiveFacts.reductionResultLMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveIntegerField(
          "reduction result SEW", plan.resultSEW,
          primitiveFacts.reductionResultSEW))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "widening product relation", plan.productRelation,
          primitiveFacts.wideningProductRelation))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "product-reduction chain relation",
          plan.productReductionChainRelation,
          primitiveFacts.productReductionChainRelation))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "accumulator layout", plan.accumulatorLayout,
          primitiveFacts.accumulatorLayout))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "result layout", plan.resultLayout, primitiveFacts.resultLayout))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "reduction store VL", primitiveFacts.reductionStoreVL, "1"))
    return std::move(error);
  return llvm::Error::success();
}

void populateWideningDotContractionRealizationPlan(
    RVVSelectedBodyContractionRealizationPlan &plan, mlir::Operation *bodyOp,
    llvm::StringRef opKind, llvm::StringRef accumulatorLayout,
    llvm::StringRef resultLayout, llvm::StringRef dotProductRelation,
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    tcrv::rvv::PolicyAttr policy, mlir::Value lhs, mlir::Value rhs,
    mlir::Value acc, mlir::Value out, mlir::Value n) {
  plan.preRealizedBody = bodyOp;
  plan.usesDotReduction = true;
  plan.opKind = opKind;
  plan.accumulatorLayout = accumulatorLayout;
  plan.resultLayout = resultLayout;
  plan.contractionRelation = dotProductRelation;
  plan.sourceSEW = sourceSEW;
  plan.sourceLMUL = sourceLMUL;
  plan.resultSEW = resultSEW;
  plan.resultLMUL = resultLMUL;
  plan.policy = policy;
  plan.lhs = lhs;
  plan.rhs = rhs;
  plan.acc = acc;
  plan.out = out;
  plan.n = n;
}

void populateComputedMaskContractionRealizationPlan(
    RVVSelectedBodyContractionRealizationPlan &plan, mlir::Value compareLHS,
    mlir::Value compareRHS, llvm::StringRef predicateKind,
    llvm::StringRef maskRole, llvm::StringRef maskSource,
    llvm::StringRef maskMemoryForm) {
  plan.usesComputedMask = true;
  plan.compareLHS = compareLHS;
  plan.compareRHS = compareRHS;
  plan.predicateKind = predicateKind;
  plan.maskRole = maskRole;
  plan.maskSource = maskSource;
  plan.maskMemoryForm = maskMemoryForm;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedWideningMAccPreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
  plan.usesWideningMAcc = true;
  plan.opKind = body.getOpKind();
  plan.accumulatorLayout = body.getAccumulatorLayout();
  plan.resultLayout = body.getResultLayout();
  plan.contractionRelation = body.getMaccRelation();
  plan.sourceSEW = static_cast<std::int64_t>(body.getSourceSew());
  plan.sourceLMUL = body.getSourceLmul();
  plan.resultSEW = static_cast<std::int64_t>(body.getResultSew());
  plan.resultLMUL = body.getResultLmul();
  plan.policy = body.getPolicy();
  plan.lhs = body.getLhs();
  plan.rhs = body.getRhs();
  plan.acc = body.getAcc();
  plan.out = body.getOut();
  plan.n = body.getN();
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body.getOpKind(), body.getAccumulatorLayout(),
      body.getResultLayout(), body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body.getOpKind(), body.getAccumulatorLayout(),
      body.getResultLayout(), body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  plan.usesStridedInputs = true;
  plan.lhsStride = body.getLhsStride();
  plan.rhsStride = body.getRhsStride();
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body.getOpKind(), body.getAccumulatorLayout(),
      body.getResultLayout(), body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  populateComputedMaskContractionRealizationPlan(
      plan, body.getCompareLhs(), body.getCompareRhs(), body.getPredicateKind(),
      body.getMaskRole(), body.getMaskSource(), body.getMaskMemoryForm());
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::
        TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body.getOpKind(), body.getAccumulatorLayout(),
      body.getResultLayout(), body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  populateComputedMaskContractionRealizationPlan(
      plan, body.getCompareLhs(), body.getCompareRhs(), body.getPredicateKind(),
      body.getMaskRole(), body.getMaskSource(), body.getMaskMemoryForm());
  plan.usesStridedInputs = true;
  plan.lhsStride = body.getLhsStride();
  plan.rhsStride = body.getRhsStride();
  return plan;
}

template <typename BodyOp>
RVVSelectedBodyContractionRealizationPlan
makeWideningProductReduceDequantClampF32RealizationPlan(BodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
  plan.usesProductReductionDequantization = true;
  plan.usesProductReductionDequantClamp = true;
  plan.opKind = body.getOpKind();
  plan.productKind = "signed_widening_product";
  plan.accumulatorLayout = body.getAccumulatorLayout();
  plan.resultLayout = body.getResultLayout();
  plan.contractionRelation = body.getProductReductionChainRelation();
  plan.productRelation = body.getProductRelation();
  plan.productReductionChainRelation = body.getProductReductionChainRelation();
  plan.dequantizationRelation = body.getDequantRelation();
  plan.scaleRole = body.getScaleRole();
  plan.dequantStoreBoundary = body.getDequantStoreBoundary();
  plan.lowerPredicateKind = body.getLowerPredicateKind();
  plan.upperPredicateKind = body.getUpperPredicateKind();
  plan.boundOrder = body.getBoundOrder();
  plan.selectLayout = body.getSelectLayout();
  plan.sourceSEW = static_cast<std::int64_t>(body.getSourceSew());
  plan.sourceLMUL = body.getSourceLmul();
  plan.productSEW = static_cast<std::int64_t>(body.getProductSew());
  plan.productLMUL = body.getProductLmul();
  plan.resultSEW = static_cast<std::int64_t>(body.getResultSew());
  plan.resultLMUL = body.getResultLmul();
  plan.policy = body.getPolicy();
  plan.lhs = body.getLhs();
  plan.rhs = body.getRhs();
  plan.acc = body.getAcc();
  plan.scale = body.getScale();
  plan.lowerBound = body.getLowerBound();
  plan.upperBound = body.getUpperBound();
  plan.out = body.getOut();
  plan.n = body.getN();
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedWideningProductReduceDequantClampF32PreRealizedBodyOp
        body) {
  return makeWideningProductReduceDequantClampF32RealizationPlan(body);
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedWideningProductReduceDequantClampF32BodyOp body) {
  return makeWideningProductReduceDequantClampF32RealizationPlan(body);
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedWideningProductReduceDequantizePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
  plan.usesProductReductionDequantization = true;
  plan.opKind = body.getOpKind();
  plan.productKind = "signed_widening_product";
  plan.accumulatorLayout = body.getAccumulatorLayout();
  plan.resultLayout = body.getResultLayout();
  plan.contractionRelation = body.getProductReductionChainRelation();
  plan.productRelation = body.getProductRelation();
  plan.productReductionChainRelation = body.getProductReductionChainRelation();
  plan.dequantizationRelation = body.getDequantRelation();
  plan.scaleRole = body.getScaleRole();
  plan.dequantStoreBoundary = body.getDequantStoreBoundary();
  plan.sourceSEW = static_cast<std::int64_t>(body.getSourceSew());
  plan.sourceLMUL = body.getSourceLmul();
  plan.productSEW = static_cast<std::int64_t>(body.getProductSew());
  plan.productLMUL = body.getProductLmul();
  plan.resultSEW = static_cast<std::int64_t>(body.getResultSew());
  plan.resultLMUL = body.getResultLmul();
  plan.policy = body.getPolicy();
  plan.lhs = body.getLhs();
  plan.rhs = body.getRhs();
  plan.acc = body.getAcc();
  plan.scale = body.getScale();
  plan.out = body.getOut();
  plan.n = body.getN();
  return plan;
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedContractionFamily(
    const VariantLoweringBoundaryRequest &request, mlir::ArrayAttr requires,
    const RVVSelectedBodyContractionRealizationPlan &plan) {
  if (!plan.preRealizedBody)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "contraction family pre-realized body op");
  if (!plan.usesWideningMAcc && !plan.usesDotReduction &&
      !plan.usesProductReductionDequantization)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "widening macc, widening dot reduction, or product-reduction "
        "dequantization family operation");
  if (plan.usesProductReductionDequantization &&
      (!plan.scale || plan.productRelation.empty() ||
       plan.productReductionChainRelation.empty() ||
       plan.dequantizationRelation.empty() || plan.scaleRole.empty() ||
       plan.dequantStoreBoundary.empty()))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "runtime scale, product relation, reduction relation, "
        "dequantization relation, scale role, and f32 store boundary for "
        "product-reduction-dequantization routes");
  if (plan.usesProductReductionDequantClamp &&
      (!plan.lowerBound || !plan.upperBound || plan.lowerPredicateKind.empty() ||
       plan.upperPredicateKind.empty() || plan.boundOrder.empty() ||
       plan.selectLayout.empty()))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "runtime lower/upper bounds, clamp predicates, bound order, and "
        "select layout for product-reduction-dequant-clamp routes");
  if (plan.usesStridedInputs && (!plan.lhsStride || !plan.rhsStride))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires lhs "
        "and rhs stride runtime ABI values for strided-input routes");
  if (plan.usesComputedMask && (!plan.compareLHS || !plan.compareRHS))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "compare lhs/rhs runtime ABI values for computed-mask routes");

  std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
      lowPrecisionPrimitiveFacts;
  if (plan.usesProductReductionDequantization) {
    std::optional<RVVSelectedBodyOperationKind> operation =
        getLowPrecisionProductReductionRealizationOperation(plan);
    if (!operation)
      return makeRVVPluginError(
          "pre-realized RVV contraction selected-body realization requires "
          "a provider-owned product-reduction operation before consuming "
          "low-precision primitive facts");
    std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts> facts =
        getRVVLowPrecisionWideningReductionPrimitiveFacts(*operation);
    if (!facts)
      return makeRVVPluginError(
          "pre-realized RVV contraction selected-body realization requires "
          "provider-owned low-precision widening-reduction primitive facts "
          "for the product-reduction operation");
    if (llvm::Error error =
            validateLowPrecisionPrimitiveFactsForRealization(plan, *facts))
      return std::move(error);
    lowPrecisionPrimitiveFacts = std::move(*facts);
  }

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = plan.preRealizedBody->getLoc();

  builder.setInsertionPoint(plan.preRealizedBody);
  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, plan.n, plan.resultSEW,
                          plan.resultLMUL, plan.policy));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires, plan.resultSEW,
                           plan.resultLMUL, plan.policy);
  copyLowPrecisionResourceAttrs(plan.preRealizedBody, withVL.getOperation());
  std::optional<RVVLowPrecisionContractionResourceCandidate>
      selectedResourceCandidate;
  if (plan.usesProductReductionDequantization) {
    if (!lowPrecisionPrimitiveFacts)
      return makeRVVPluginError(
          "pre-realized RVV contraction selected-body realization lost "
          "provider-owned low-precision primitive facts before materializing "
          "resource-aware realization attributes");
    llvm::Expected<RVVLowPrecisionContractionResourceCandidate> candidate =
        materializeLowPrecisionResourceRealizationAttrs(
            builder, plan.preRealizedBody, withVL.getOperation(),
            plan.usesProductReductionDequantClamp, *lowPrecisionPrimitiveFacts,
            stringifyLowPrecisionRealizationTailPolicy(plan.policy.getTail()),
            stringifyLowPrecisionRealizationMaskPolicy(plan.policy.getMask()),
            plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
            plan.productLMUL, plan.resultSEW, plan.resultLMUL);
    if (!candidate)
      return candidate.takeError();
    selectedResourceCandidate = *candidate;
  }

  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Value compareLHSValue;
  mlir::Value compareRHSValue;
  if (plan.usesProductReductionDequantization) {
    const llvm::StringRef resourceDecision =
        getRVVLowPrecisionContractionResourceRealizationDecision(
            selectedResourceCandidate->candidateID);
    const bool usesGroupedLowPrecisionProductReduction =
        isRVVLowPrecisionResourceGroupedRealizationDecision(resourceDecision);
    createRealizedVSetVLRegionMarker(
        builder, loc, setvl.getVl(),
        usesGroupedLowPrecisionProductReduction
            ? llvm::StringRef("grouped-product-reduce-main")
            : getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
                  resourceDecision),
        1, selectedResourceCandidate->vsetvlRegionCount, resourceDecision);
    if (usesGroupedLowPrecisionProductReduction)
      createRealizedVSetVLRegionMarker(
          builder, loc, setvl.getVl(), "tail-product-reduce", 2,
          selectedResourceCandidate->vsetvlRegionCount, resourceDecision);
  }
  if (plan.usesComputedMask) {
    auto compareLHSLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, plan.compareLHS, setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto compareRHSLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, plan.compareRHS, setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    compareLHSValue = compareLHSLoad.getLoaded();
    compareRHSValue = compareRHSLoad.getLoaded();
  }

  auto realizeContractionSourceLoad =
      [&](mlir::Value buffer, mlir::Value stride) -> mlir::Value {
    if (plan.usesStridedInputs) {
      auto load = llvm::cast<tcrv::rvv::StridedLoadOp>(
          createRealizedGenericStridedLoad(builder, loc, buffer, stride,
                                           setvl.getVl(), plan.sourceSEW,
                                           plan.sourceLMUL));
      return load.getLoaded();
    }
    auto load = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, buffer, setvl.getVl(), plan.sourceSEW,
        plan.sourceLMUL));
    return load.getLoaded();
  };

  mlir::Value lhsValue =
      realizeContractionSourceLoad(plan.lhs, plan.lhsStride);
  mlir::Value rhsValue =
      realizeContractionSourceLoad(plan.rhs, plan.rhsStride);
  mlir::Value compareMask;
  if (plan.usesComputedMask) {
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(builder, loc, compareLHSValue,
                                     compareRHSValue, setvl.getVl(),
                                     plan.predicateKind));
    compareMask = compare.getMask();
  }

  if (plan.usesProductReductionDequantization) {
    auto product = llvm::cast<tcrv::rvv::WideningProductOp>(
        createRealizedGenericWideningProductCompute(
            builder, loc, plan.productKind,
            selectedResourceCandidate->primitiveWideningProductRelation,
            lhsValue, rhsValue, setvl.getVl(), plan.productSEW,
            plan.productLMUL));
    auto reduced = llvm::cast<tcrv::rvv::StandaloneReduceOp>(
        createRealizedGenericStandaloneWideningReduceCompute(
            builder, loc, selectedResourceCandidate->primitiveAccumulatorLayout,
            selectedResourceCandidate->primitiveResultLayout,
            product.getResult(), plan.acc, setvl.getVl(), plan.resultSEW,
            plan.resultLMUL));
    mlir::Value dequantSource = reduced.getResult();
    tcrv::rvv::WithVLOp consumerWithVL;
    auto handoff = llvm::cast<tcrv::rvv::GearboxCrossRegionHandoffOp>(
        createRealizedGearboxCrossRegionHandoff(
            builder, loc, reduced.getResult(), setvl.getVl(), plan.n,
            *selectedResourceCandidate));
    dequantSource = handoff.getOutput();
    consumerWithVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, plan.resultSEW,
                             plan.resultLMUL, plan.policy);
    copyLowPrecisionResourceAttrs(plan.preRealizedBody,
                                  consumerWithVL.getOperation());
    llvm::Expected<RVVLowPrecisionContractionResourceCandidate>
        consumerCandidate = materializeLowPrecisionResourceRealizationAttrs(
            builder, plan.preRealizedBody, consumerWithVL.getOperation(),
            plan.usesProductReductionDequantClamp, *lowPrecisionPrimitiveFacts,
            stringifyLowPrecisionRealizationTailPolicy(plan.policy.getTail()),
            stringifyLowPrecisionRealizationMaskPolicy(plan.policy.getMask()),
            plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
            plan.productLMUL, plan.resultSEW, plan.resultLMUL);
    if (!consumerCandidate)
      return consumerCandidate.takeError();
    if (consumerCandidate->candidateID != selectedResourceCandidate->candidateID)
      return makeRVVPluginError(
          "pre-realized RVV contraction selected-body realization requires "
          "producer and consumer scopes to consume the same selected "
          "low-precision resource candidate");
    builder.setInsertionPointToStart(&consumerWithVL.getBody().front());
    createRealizedVSetVLRegionMarker(
        builder, loc, setvl.getVl(), "dequant-store",
        getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
            getRVVLowPrecisionContractionResourceRealizationDecision(
                consumerCandidate->candidateID)),
        consumerCandidate->vsetvlRegionCount,
        getRVVLowPrecisionContractionResourceRealizationDecision(
            consumerCandidate->candidateID));
    auto dequantized = llvm::cast<tcrv::rvv::DequantizeOp>(
        createRealizedGenericDequantizeCompute(
            builder, loc, plan.dequantizationRelation, dequantSource,
            plan.scale, setvl.getVl(), plan.resultLMUL));
    mlir::Value valueToStore = dequantized.getResult();
    if (plan.usesProductReductionDequantClamp) {
      auto lowerSplat = llvm::cast<tcrv::rvv::SplatOp>(
          createRealizedGenericF32Splat(builder, loc, plan.lowerBound,
                                        setvl.getVl(), plan.resultLMUL));
      auto upperSplat = llvm::cast<tcrv::rvv::SplatOp>(
          createRealizedGenericF32Splat(builder, loc, plan.upperBound,
                                        setvl.getVl(), plan.resultLMUL));
      auto lowerCompare = llvm::cast<tcrv::rvv::CompareOp>(
          createRealizedGenericCompare(builder, loc, dequantized.getResult(),
                                       lowerSplat.getBroadcast(), setvl.getVl(),
                                       plan.lowerPredicateKind));
      auto lowerSelect = llvm::cast<tcrv::rvv::SelectOp>(
          createRealizedGenericSelect(builder, loc, lowerCompare.getMask(),
                                      lowerSplat.getBroadcast(),
                                      dequantized.getResult(), setvl.getVl()));
      auto upperCompare = llvm::cast<tcrv::rvv::CompareOp>(
          createRealizedGenericCompare(builder, loc, upperSplat.getBroadcast(),
                                       lowerSelect.getSelected(), setvl.getVl(),
                                       plan.upperPredicateKind));
      auto upperSelect = llvm::cast<tcrv::rvv::SelectOp>(
          createRealizedGenericSelect(builder, loc, upperCompare.getMask(),
                                      upperSplat.getBroadcast(),
                                      lowerSelect.getSelected(),
                                      setvl.getVl()));
      valueToStore = upperSelect.getSelected();
    }
    createRealizedGenericStore(builder, loc, plan.out, valueToStore,
                               setvl.getVl());
    if (consumerWithVL)
      builder.setInsertionPointAfter(consumerWithVL);
  } else if (plan.usesWideningMAcc) {
    auto accumulatorLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, plan.acc, setvl.getVl(), plan.resultSEW,
            plan.resultLMUL));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericWideningMAccCompute(
            builder, loc, plan.opKind, plan.accumulatorLayout,
            plan.resultLayout, plan.contractionRelation, lhsValue, rhsValue,
            accumulatorLoad.getLoaded(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, plan.out,
                               (*compute)->getResult(0), setvl.getVl());
  } else if (plan.usesComputedMask) {
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericMaskedWideningDotReduceCompute(
            builder, loc, plan.opKind, plan.maskRole, plan.maskSource,
            plan.maskMemoryForm, plan.accumulatorLayout, plan.resultLayout,
            plan.contractionRelation, compareMask, lhsValue, rhsValue, plan.acc,
            setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, plan.out,
                               (*compute)->getResult(0), setvl.getVl());
  } else {
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericWideningDotReduceCompute(
            builder, loc, plan.opKind, plan.accumulatorLayout,
            plan.resultLayout, plan.contractionRelation, lhsValue, rhsValue,
            plan.acc, setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, plan.out,
                               (*compute)->getResult(0), setvl.getVl());
  }
  plan.preRealizedBody->erase();
  return withVL;
}

} // namespace

bool isPreRealizedRVVContractionClusterOp(mlir::Operation *op) {
  return llvm::isa<
      tcrv::rvv::TypedWideningMAccPreRealizedBodyOp,
      tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp,
      tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp,
      tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp,
      tcrv::rvv::
          TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp,
      tcrv::rvv::TypedWideningProductReduceDequantizePreRealizedBodyOp,
      tcrv::rvv::
          TypedWideningProductReduceDequantClampF32PreRealizedBodyOp,
      tcrv::rvv::TypedWideningProductReduceDequantClampF32BodyOp>(op);
}

llvm::Expected<tcrv::rvv::WithVLOp> realizePreRealizedRVVContractionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!isPreRealizedRVVContractionClusterOp(bodyOp))
    return makeRVVPluginError(
        "contraction selected-body realization owner received a body outside "
        "its RVV-owned realization family");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "materialized kernel and variant");

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto wideningMAccBody =
          llvm::dyn_cast<tcrv::rvv::TypedWideningMAccPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningMAccBody(request,
                                                           wideningMAccBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, makeContractionRealizationPlan(wideningMAccBody));
  }

  if (auto dotReduceBody = llvm::dyn_cast<
          tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningDotReduceBody(
                request, dotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, makeContractionRealizationPlan(dotReduceBody));
  }

  if (auto stridedDotReduceBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedStridedInputWideningDotReducePreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedStridedInputWideningDotReduceBody(
                request, stridedDotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(stridedDotReduceBody));
  }

  if (auto maskedDotReduceBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedComputedMaskWideningDotReducePreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskWideningDotReduceBody(
                request, maskedDotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, makeContractionRealizationPlan(maskedDotReduceBody));
  }

  if (auto maskedStridedDotReduceBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskStridedInputWideningDotReduceBody(
                request, maskedStridedDotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(maskedStridedDotReduceBody));
  }

  if (auto productReduceDequantBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedWideningProductReduceDequantizePreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningProductReduceDequantizeBody(
                request, productReduceDequantBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(productReduceDequantBody));
  }

  if (auto productReduceDequantClampBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedWideningProductReduceDequantClampF32PreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningProductReduceDequantClampF32Body(
                request, productReduceDequantClampBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(productReduceDequantClampBody));
  }

  if (auto explicitProductReduceDequantClampBody =
          llvm::dyn_cast<
              tcrv::rvv::TypedWideningProductReduceDequantClampF32BodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validateExplicitRVVSelectedWideningProductReduceDequantClampF32Body(
                request, explicitProductReduceDequantClampBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(explicitProductReduceDequantClampBody));
  }

  return makeRVVPluginError(
      "contraction selected-body realization owner found an unsupported "
      "pre-realized body op");
}

} // namespace tianchenrv::plugin::rvv
