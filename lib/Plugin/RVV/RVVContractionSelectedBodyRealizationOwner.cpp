#include "TianChenRV/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"

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

void materializeLowPrecisionRealizationAdmissionAttrs(
    mlir::OpBuilder &builder, mlir::Operation *destination,
    const RVVLowPrecisionSelectedBodyRealizationAdmission &admission) {
  destination->setAttr(
      kRVVLowPrecisionResourceRealizationAdmissionContractAttrName,
      builder.getStringAttr(admission.admissionContract));
  destination->setAttr(
      kRVVLowPrecisionResourceRealizationAdmissionDecisionAttrName,
      builder.getStringAttr(stringifyRVVLowPrecisionRealizationAdmissionDecision(
          admission.decision)));
  destination->setAttr(
      kRVVLowPrecisionResourceRealizationAdmissionEvidenceAttrName,
      builder.getStringAttr(admission.measurementEvidenceID));
  destination->setAttr(
      kRVVLowPrecisionResourceRealizationAdmissionDispatchPolicyAttrName,
      builder.getStringAttr(admission.dispatchPolicyPath));
  destination->setAttr(
      kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionContractAttrName,
      builder.getStringAttr(admission.scheduleDecisionContract));
  destination->setAttr(
      kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionAttrName,
      builder.getStringAttr(admission.scheduleDecision));
  destination->setAttr(
      kRVVLowPrecisionResourceRealizationAdmissionScheduleDecisionReasonAttrName,
      builder.getStringAttr(admission.scheduleDecisionReason));
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

llvm::Expected<std::optional<std::int64_t>>
readLowPrecisionResourceIntegerFact(mlir::Operation *op,
                                    llvm::StringRef attrName) {
  if (!op)
    return std::nullopt;
  mlir::Attribute attr = op->getAttr(attrName);
  if (!attr)
    return std::nullopt;
  auto integerAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr);
  if (!integerAttr)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV contraction selected-body realization "
                    "requires integer low-precision direct-contraction "
                    "resource fact '") +
        attrName + "'");
  return integerAttr.getInt();
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

llvm::Error validateLowPrecisionResourceCandidatePrimitiveFacts(
    const RVVLowPrecisionContractionResourceCandidate &candidate,
    const RVVLowPrecisionWideningReductionPrimitiveFacts &primitiveFacts) {
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "low-precision primitive contract",
              candidate.primitiveContractID,
              primitiveFacts.lowPrecisionPrimitiveContractID))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "low-precision primitive kind", candidate.primitiveKind,
              primitiveFacts.lowPrecisionPrimitiveKind))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive chain contract", candidate.primitiveChainContractID,
              primitiveFacts.contractID))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive chain kind", candidate.primitiveChainKind,
              primitiveFacts.kind))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive source load", candidate.primitiveSourceLoadKind,
              primitiveFacts.sourceLoadKind))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive source extension",
              candidate.primitiveSourceExtensionKind,
              primitiveFacts.sourceExtensionKind))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "widening product multiplicand roles",
              candidate.wideningProductMultiplicandRoleSummary,
              kRVVLowPrecisionResourceWideningProductMultiplicandRoles))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "widening product extension policy",
              candidate.wideningProductExtensionPolicy,
              kRVVLowPrecisionResourceWideningProductExtensionPolicy))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "widening product candidate fact",
              candidate.wideningProductCandidateFact,
              primitiveFacts.wideningProductCandidateFact))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "widening reduction candidate fact",
              candidate.reductionCandidateFact,
              primitiveFacts.reductionCandidateFact))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive widening product relation",
              candidate.primitiveWideningProductRelation,
              primitiveFacts.wideningProductRelation))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive product-reduction chain relation",
              candidate.primitiveProductReductionChainRelation,
              primitiveFacts.productReductionChainRelation))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive widening product intrinsic",
              candidate.primitiveWideningProductIntrinsic,
              primitiveFacts.wideningProductIntrinsic))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive reduction intrinsic",
              candidate.primitiveReductionIntrinsic,
              primitiveFacts.reductionIntrinsic))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive scalar seed splat intrinsic",
              candidate.primitiveScalarSeedSplatIntrinsic,
              primitiveFacts.scalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive accumulator layout",
              candidate.primitiveAccumulatorLayout,
              primitiveFacts.accumulatorLayout))
    return error;
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive result layout", candidate.primitiveResultLayout,
              primitiveFacts.resultLayout))
    return error;
  return requireLowPrecisionResourceCandidatePrimitiveStringMatch(
      "primitive reduction store VL", candidate.primitiveReductionStoreVL,
      primitiveFacts.reductionStoreVL);
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
  const std::int64_t candidateCount =
      getRVVLowPrecisionProductReductionResourceCandidateCount(candidates);
  const std::int64_t legalCandidateCount =
      getRVVLowPrecisionProductReductionLegalResourceCandidateCount(candidates);
  std::optional<std::int64_t> selectedCandidateIndex =
      getRVVLowPrecisionProductReductionSelectedCandidateIndex(
          candidates, selected->candidateID);
  if (!selectedCandidateIndex)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV contraction selected-body realization "
                    "cannot find selected low-precision resource candidate '") +
        selected->candidateID + "' in the provider-built candidate "
        "enumeration");
  if (candidateCount < 2 || legalCandidateCount < 2)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV contraction selected-body realization "
                    "requires at least two legal provider-built low-precision "
                    "resource candidates before selecting a resource plan"));
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
              "source signedness", selected->sourceSignedness,
              primitiveFacts.sourceSignedness))
    return std::move(error);
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive source load", selected->primitiveSourceLoadKind,
              primitiveFacts.sourceLoadKind))
    return std::move(error);
  if (llvm::Error error =
          requireLowPrecisionResourceCandidatePrimitiveStringMatch(
              "primitive source extension",
              selected->primitiveSourceExtensionKind,
              primitiveFacts.sourceExtensionKind))
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
  if (llvm::Error error = validateLowPrecisionResourceCandidatePrimitiveFacts(
          *selected, primitiveFacts))
    return std::move(error);

  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceCandidateSetAttrName,
          selected->candidateSetID))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceSelectedCandidateAttrName,
          selected->candidateID))
    return std::move(error);
  llvm::Expected<std::optional<std::int64_t>> sourceCandidateCountOr =
      readLowPrecisionResourceIntegerFact(
          source, kRVVLowPrecisionResourceCandidateCountAttrName);
  if (!sourceCandidateCountOr)
    return sourceCandidateCountOr.takeError();
  llvm::Expected<std::optional<std::int64_t>> sourceLegalCandidateCountOr =
      readLowPrecisionResourceIntegerFact(
          source, kRVVLowPrecisionResourceLegalCandidateCountAttrName);
  if (!sourceLegalCandidateCountOr)
    return sourceLegalCandidateCountOr.takeError();
  llvm::Expected<std::optional<std::int64_t>> sourceSelectedCandidateIndexOr =
      readLowPrecisionResourceIntegerFact(
          source, kRVVLowPrecisionResourceSelectedCandidateIndexAttrName);
  if (!sourceSelectedCandidateIndexOr)
    return sourceSelectedCandidateIndexOr.takeError();
  const std::optional<std::int64_t> sourceCandidateCount =
      *sourceCandidateCountOr;
  const std::optional<std::int64_t> sourceLegalCandidateCount =
      *sourceLegalCandidateCountOr;
  const std::optional<std::int64_t> sourceSelectedCandidateIndex =
      *sourceSelectedCandidateIndexOr;
  const bool hasAnyCandidateEnumeration =
      sourceCandidateCount.has_value() || sourceLegalCandidateCount.has_value() ||
      sourceSelectedCandidateIndex.has_value();
  const bool hasFullCandidateEnumeration =
      sourceCandidateCount.has_value() && sourceLegalCandidateCount.has_value() &&
      sourceSelectedCandidateIndex.has_value();
  if (hasAnyCandidateEnumeration && !hasFullCandidateEnumeration)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV contraction selected-body realization "
                    "requires low-precision resource candidate enumeration "
                    "facts '") +
        kRVVLowPrecisionResourceCandidateCountAttrName + "', '" +
        kRVVLowPrecisionResourceLegalCandidateCountAttrName + "', and '" +
        kRVVLowPrecisionResourceSelectedCandidateIndexAttrName +
        "' to be carried together");
  if (hasFullCandidateEnumeration) {
    if (*sourceCandidateCount != candidateCount)
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV contraction selected-body realization "
                      "cannot consume stale low-precision resource candidate "
                      "count: expected ") +
          llvm::Twine(candidateCount) + " but found " +
          llvm::Twine(*sourceCandidateCount));
    if (*sourceLegalCandidateCount != legalCandidateCount)
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV contraction selected-body realization "
                      "cannot consume stale low-precision legal resource "
                      "candidate count: expected ") +
          llvm::Twine(legalCandidateCount) + " but found " +
          llvm::Twine(*sourceLegalCandidateCount));
    if (*sourceSelectedCandidateIndex != *selectedCandidateIndex)
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV contraction selected-body realization "
                      "cannot consume stale low-precision selected candidate "
                      "index: expected ") +
          llvm::Twine(*selectedCandidateIndex) + " but found " +
          llvm::Twine(*sourceSelectedCandidateIndex));
  }
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceSelectionReasonAttrName,
          selected->selectionReason))
    return std::move(error);
  if (auto planningContract =
          source->getAttrOfType<mlir::StringAttr>(
              kRVVLowPrecisionResourcePlanningContractAttrName)) {
    if (planningContract.getValue() != selected->planningContract)
      return makeRVVPluginError(
          llvm::Twine("pre-realized RVV contraction selected-body "
                      "realization cannot consume stale low-precision "
                      "resource planning contract: expected '") +
          selected->planningContract + "' but found '" +
          planningContract.getValue() + "'");
  }
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
            source,
            kRVVLowPrecisionResourcePackedLoadUnpackContractAttrName,
            selected->packedLoadUnpackContract))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source, kRVVLowPrecisionResourcePackedStorageLoadAttrName,
            selected->packedStorageLoad))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source, kRVVLowPrecisionResourcePackedUnpackPlanAttrName,
            selected->packedUnpackPlan))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source, kRVVLowPrecisionResourcePackedUnpackedSourceAttrName,
            selected->packedUnpackedSource))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source,
            kRVVLowPrecisionResourceScheduleDecisionContractAttrName,
            kRVVLowPrecisionResourcePackedI4ScheduleDecisionContract))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source, kRVVLowPrecisionResourceScheduleDecisionAttrName,
            kRVVLowPrecisionResourcePackedI4ScheduleDecision))
      return std::move(error);
    if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
            source, kRVVLowPrecisionResourceScheduleDecisionReasonAttrName,
            kRVVLowPrecisionResourcePackedI4ScheduleDecisionReason))
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
          kRVVLowPrecisionResourceWideningProductMultiplicandRolesAttrName,
          selected->wideningProductMultiplicandRoleSummary))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source,
          kRVVLowPrecisionResourceWideningProductExtensionPolicyAttrName,
          selected->wideningProductExtensionPolicy))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source,
          kRVVLowPrecisionResourceWideningProductCandidateFactAttrName,
          selected->wideningProductCandidateFact))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourceReductionCandidateFactAttrName,
          selected->reductionCandidateFact))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourcePrimitiveSourceLoadAttrName,
          selected->primitiveSourceLoadKind))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionResourceExpectedStringFact(
          source, kRVVLowPrecisionResourcePrimitiveSourceExtensionAttrName,
          selected->primitiveSourceExtensionKind))
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
  destination->setAttr(
      kRVVLowPrecisionResourcePlanningContractAttrName,
      builder.getStringAttr(selected->planningContract));
  destination->setAttr(kRVVLowPrecisionResourceCandidateCountAttrName,
                       builder.getI64IntegerAttr(candidateCount));
  destination->setAttr(kRVVLowPrecisionResourceLegalCandidateCountAttrName,
                       builder.getI64IntegerAttr(legalCandidateCount));
  destination->setAttr(
      kRVVLowPrecisionResourceSelectedCandidateIndexAttrName,
      builder.getI64IntegerAttr(*selectedCandidateIndex));
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
  destination->setAttr(
      kRVVLowPrecisionResourceProductRegionIndexAttrName,
      builder.getI64IntegerAttr(
          getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
              realizationDecision)));
  destination->setAttr(
      kRVVLowPrecisionResourceDequantRegionIndexAttrName,
      builder.getI64IntegerAttr(
          getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
              realizationDecision)));
  destination->setAttr(
      kRVVLowPrecisionResourceProductPhaseAttrName,
      builder.getStringAttr(
          getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
              realizationDecision)));
  destination->setAttr(kRVVLowPrecisionResourceDequantPhaseAttrName,
                       builder.getStringAttr("dequant-store"));
  if (isRVVLowPrecisionResourceDequantClampCandidateID(
          selected->candidateID)) {
    destination->setAttr(
        kRVVLowPrecisionResourceClampRegionIndexAttrName,
        builder.getI64IntegerAttr(
            getRVVLowPrecisionResourceClampRegionIndexForCandidate(
                selected->candidateID)));
    destination->setAttr(
        kRVVLowPrecisionResourceClampPhaseAttrName,
        builder.getStringAttr(
            getRVVLowPrecisionResourceClampPhaseForCandidate(
                selected->candidateID)));
    destination->setAttr(
        kRVVLowPrecisionResourceClampCompareSelectPhaseAttrName,
        builder.getStringAttr(
            getRVVLowPrecisionResourceClampCompareSelectPhaseForCandidate(
                selected->candidateID)));
    destination->setAttr(
        kRVVLowPrecisionResourceClampSelectLayoutAttrName,
        builder.getStringAttr(
            getRVVLowPrecisionResourceClampSelectLayoutForCandidate(
                selected->candidateID)));
  }
  if (isPackedI4Resource) {
    destination->setAttr(
        kRVVLowPrecisionResourcePackedLoadUnpackContractAttrName,
        builder.getStringAttr(selected->packedLoadUnpackContract));
    destination->setAttr(
        kRVVLowPrecisionResourcePackedStorageLoadAttrName,
        builder.getStringAttr(selected->packedStorageLoad));
    destination->setAttr(
        kRVVLowPrecisionResourcePackedUnpackPlanAttrName,
        builder.getStringAttr(selected->packedUnpackPlan));
    destination->setAttr(
        kRVVLowPrecisionResourcePackedUnpackedSourceAttrName,
        builder.getStringAttr(selected->packedUnpackedSource));
    destination->setAttr(
        kRVVLowPrecisionResourceScheduleDecisionContractAttrName,
        builder.getStringAttr(
            kRVVLowPrecisionResourcePackedI4ScheduleDecisionContract));
    destination->setAttr(
        kRVVLowPrecisionResourceScheduleDecisionAttrName,
        builder.getStringAttr(kRVVLowPrecisionResourcePackedI4ScheduleDecision));
    destination->setAttr(
        kRVVLowPrecisionResourceScheduleDecisionReasonAttrName,
        builder.getStringAttr(
            kRVVLowPrecisionResourcePackedI4ScheduleDecisionReason));
  }
  return *selected;
}

mlir::Type getGenericVectorType(mlir::OpBuilder &builder, std::int64_t sew,
                                llvm::StringRef lmul,
                                bool isUnsigned = false) {
  mlir::Type elementType =
      isUnsigned
          ? mlir::IntegerType::get(
                builder.getContext(), sew,
                mlir::IntegerType::SignednessSemantics::Unsigned)
          : builder.getIntegerType(sew);
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
                                           llvm::StringRef lmul,
                                           bool isUnsigned = false) {
  mlir::OperationState state(loc, "tcrv_rvv.load");
  state.addOperands({buffer, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul, isUnsigned));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericStridedLoad(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value buffer,
                                                  mlir::Value stride,
                                                  mlir::Value vl,
                                                  std::int64_t sew,
                                                  llvm::StringRef lmul,
                                                  bool isUnsigned = false) {
  mlir::OperationState state(loc, "tcrv_rvv.strided_load");
  state.addOperands({buffer, stride, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul, isUnsigned));
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
    mlir::Value vl, std::int64_t productSEW, llvm::StringRef productLMUL,
    bool isUnsigned = false) {
  mlir::OperationState state(loc, "tcrv_rvv.widening_product");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("product_relation",
                     builder.getStringAttr(productRelation));
  state.addTypes(
      getGenericVectorType(builder, productSEW, productLMUL, isUnsigned));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericPackedI4NibbleUnpackProductCompute(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef productRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value vl, std::int64_t productSEW, llvm::StringRef productLMUL) {
  // The packed-i4 nibble-unpack widening product carries the i4 sign-extend /
  // unpack STRUCTURE as one typed op (the fixed vsll/vsra/vwmul/vsra/vwmacc chain
  // is the op's lowering); the single-scope Stage 3 conversion walks the typed op
  // and never reads operand_form/unpack_intent mirror strings.
  mlir::OperationState state(loc, "tcrv_rvv.packed_i4_nibble_unpack_product");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute(
      "kind",
      builder.getStringAttr("signed_packed_i4_nibble_unpack_product"));
  state.addAttribute("product_relation",
                     builder.getStringAttr(productRelation));
  state.addTypes(
      getGenericVectorType(builder, productSEW, productLMUL, /*isUnsigned=*/false));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericStandaloneWideningReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    mlir::Value input, mlir::Value accumulatorSeed, mlir::Value vl,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    bool isUnsigned = false) {
  mlir::OperationState state(loc, "tcrv_rvv.standalone_reduce");
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute(
      "kind",
      builder.getStringAttr(isUnsigned ? "unsigned_widening_reduce_add"
                                       : "signed_widening_reduce_add"));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(
      getGenericVectorType(builder, resultSEW, resultLMUL, isUnsigned));
  return builder.create(state);
}

// Deferred-wide (N3 max-legal-LMUL) realization helpers. These build the
// structurally-distinct deferred-wide chain the resource-aware selector picks
// when the vreg budget admits the wide rung: a tcrv_rvv.widening_accumulate
// (i16m4 product -> loop-carried i32m8 vector accumulate) plus the single
// trailing tcrv_rvv.standalone_reduce that folds the i32m8 accumulator with one
// vredsum (kind "add", NOT the narrow per-iteration "signed_widening_reduce_add"
// vwredsum). Emission is body-determined: these ops ARE the structural markers
// the conversion (RVVToEmitC isDeferredWideDequantBody) follows (I5).
mlir::Operation *createRealizedGenericWideningAccumulate(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value product,
    mlir::Value vl, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL) {
  mlir::OperationState state(loc, "tcrv_rvv.widening_accumulate");
  state.addOperands({product, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("signed_widening_accumulate_add"));
  state.addAttribute("accumulate_relation",
                     builder.getStringAttr("signed-i16m4-into-i32m8-deferred-add"));
  state.addTypes(getGenericVectorType(builder, accumulatorSEW, accumulatorLMUL,
                                      /*isUnsigned=*/false));
  return builder.create(state);
}

// The 2nd kernel family (signed i16 dot-reduce, P-B8) deferred accumulate: the
// i16m4 x i16m4 -> i32m8 single-widening product is ALREADY the i32m8
// accumulator width, so the deferred accumulate is a SAME-width vadd.vv (NOT the
// byte path's widening vwadd.wv). tcrv_rvv.deferred_accumulate is the structural
// marker the conversion (RVVToEmitC isDeferredWideDotReduceBody) follows (I5).
mlir::Operation *createRealizedGenericDeferredAccumulate(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value product,
    mlir::Value vl, std::int64_t accumulatorSEW,
    llvm::StringRef accumulatorLMUL) {
  mlir::OperationState state(loc, "tcrv_rvv.deferred_accumulate");
  state.addOperands({product, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("signed_deferred_accumulate_add"));
  state.addAttribute(
      "accumulate_relation",
      builder.getStringAttr("signed-i32m8-into-i32m8-deferred-add"));
  state.addTypes(getGenericVectorType(builder, accumulatorSEW, accumulatorLMUL,
                                      /*isUnsigned=*/false));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericDeferredWideTrailingReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value input,
    mlir::Value accumulatorSeed, mlir::Value vl, std::int64_t resultSEW,
    llvm::StringRef resultLMUL) {
  mlir::OperationState state(loc, "tcrv_rvv.standalone_reduce");
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute("kind", builder.getStringAttr("add"));
  state.addAttribute(
      "accumulator_layout",
      builder.getStringAttr("scalar-i32-seed-lane0-from-accumulator-input"));
  state.addAttribute(
      "result_layout",
      builder.getStringAttr("store-standalone-reduction-lane0-to-output-scalar"));
  state.addTypes(getGenericVectorType(builder, resultSEW, resultLMUL,
                                      /*isUnsigned=*/false));
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
                                      llvm::StringRef planningContract,
                                      std::int64_t regionIndex,
                                      std::int64_t regionCount,
                                      llvm::StringRef resourceDecision) {
  mlir::OperationState state(loc, "tcrv_rvv.vsetvl_region_marker");
  state.addOperands(vl);
  state.addAttribute("phase", builder.getStringAttr(phase));
  state.addAttribute("planning_contract",
                     builder.getStringAttr(planningContract));
  state.addAttribute("region_index", builder.getI64IntegerAttr(regionIndex));
  state.addAttribute("region_count", builder.getI64IntegerAttr(regionCount));
  state.addAttribute("resource_decision",
                     builder.getStringAttr(resourceDecision));
  (void)builder.create(state);
}

mlir::Operation *createRealizedGearboxCrossRegionHandoff(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value input,
    mlir::Value vl, mlir::Value runtimeAVL,
    const RVVLowPrecisionContractionResourceCandidate &selectedCandidate,
    const RVVLowPrecisionSelectedBodyRealizationAdmission *admission =
        nullptr) {
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
  state.addAttribute("planning_contract",
                     builder.getStringAttr(selectedCandidate.planningContract));
  state.addAttribute("resource_candidate_set",
                     builder.getStringAttr(selectedCandidate.candidateSetID));
  state.addAttribute("resource_selected_candidate",
                     builder.getStringAttr(selectedCandidate.candidateID));
  state.addAttribute("resource_candidate_count",
                     builder.getI64IntegerAttr(
                         selectedCandidate.candidateCount));
  state.addAttribute("resource_legal_candidate_count",
                     builder.getI64IntegerAttr(
                         selectedCandidate.legalCandidateCount));
  state.addAttribute("resource_selected_candidate_index",
                     builder.getI64IntegerAttr(selectedCandidate.candidateIndex));
  state.addAttribute("operand_form",
                     builder.getStringAttr(selectedCandidate.operandForm));
  state.addAttribute("packing_layout",
                     builder.getStringAttr(selectedCandidate.packingLayout));
  state.addAttribute("unpack_intent",
                     builder.getStringAttr(selectedCandidate.unpackIntent));
  state.addAttribute(
      "peak_live_vector_groups",
      builder.getI64IntegerAttr(selectedCandidate.peakLiveVectorGroups));
  state.addAttribute(
      "vector_register_budget",
      builder.getI64IntegerAttr(selectedCandidate.vectorRegisterBudget));
  if (isRVVLowPrecisionResourcePackedI4CandidateID(
          selectedCandidate.candidateID)) {
    state.addAttribute(
        "packed_load_unpack_contract",
        builder.getStringAttr(selectedCandidate.packedLoadUnpackContract));
    state.addAttribute("packed_storage_load",
                       builder.getStringAttr(selectedCandidate.packedStorageLoad));
    state.addAttribute("packed_unpack_plan",
                       builder.getStringAttr(selectedCandidate.packedUnpackPlan));
    state.addAttribute(
        "packed_unpacked_source",
        builder.getStringAttr(selectedCandidate.packedUnpackedSource));
    state.addAttribute("resource_cost_contract",
                       builder.getStringAttr(
                           selectedCandidate.resourceCostContract));
    state.addAttribute("resource_cost_model",
                       builder.getStringAttr(selectedCandidate.resourceCostModel));
    state.addAttribute("resource_cost_loop_body_steps",
                       builder.getI64IntegerAttr(
                           selectedCandidate.resourceCostLoopBodySteps));
    state.addAttribute("resource_cost_blocker",
                       builder.getStringAttr(selectedCandidate.resourceCostBlocker));
    state.addAttribute(
        "performance_admission_decision",
        builder.getStringAttr(selectedCandidate.performanceAdmissionDecision));
    state.addAttribute(
        "performance_admission_closure",
        builder.getStringAttr(selectedCandidate.performanceAdmissionClosure));
    state.addAttribute(
        "performance_admission_reopen_requirement",
        builder.getStringAttr(
            selectedCandidate.performanceAdmissionReopenRequirement));
    state.addAttribute(
        "beyond_local_repair_admission_contract",
        builder.getStringAttr(
            selectedCandidate.beyondLocalRepairAdmissionContract));
    state.addAttribute(
        "beyond_local_repair_admission_decision",
        builder.getStringAttr(
            selectedCandidate.beyondLocalRepairAdmissionDecision));
    state.addAttribute(
        "beyond_local_repair_admission_blocker",
        builder.getStringAttr(
            selectedCandidate.beyondLocalRepairAdmissionBlocker));
    state.addAttribute(
        "beyond_local_repair_admission_reopen_requirement",
        builder.getStringAttr(
            selectedCandidate.beyondLocalRepairAdmissionReopenRequirement));
  }
  state.addAttribute(
      "product_region_index",
      builder.getI64IntegerAttr(
          getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
              resourceDecision)));
  state.addAttribute(
      "dequant_region_index",
      builder.getI64IntegerAttr(
          getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
              resourceDecision)));
  if (isRVVLowPrecisionResourceDequantClampCandidateID(
          selectedCandidate.candidateID)) {
    state.addAttribute(
        "clamp_region_index",
        builder.getI64IntegerAttr(
            getRVVLowPrecisionResourceClampRegionIndexForCandidate(
                selectedCandidate.candidateID)));
    state.addAttribute(
        "clamp_phase",
        builder.getStringAttr(getRVVLowPrecisionResourceClampPhaseForCandidate(
            selectedCandidate.candidateID)));
    state.addAttribute(
        "clamp_compare_select_phase",
        builder.getStringAttr(
            getRVVLowPrecisionResourceClampCompareSelectPhaseForCandidate(
                selectedCandidate.candidateID)));
    state.addAttribute(
        "clamp_select_layout",
        builder.getStringAttr(
            getRVVLowPrecisionResourceClampSelectLayoutForCandidate(
                selectedCandidate.candidateID)));
  }
  if (isRVVLowPrecisionResourcePackedI4CandidateID(
          selectedCandidate.candidateID)) {
    state.addAttribute(
        "remediation_plan_contract",
        builder.getStringAttr(selectedCandidate.remediationPlanContract));
    state.addAttribute("remediation_plan",
                       builder.getStringAttr(selectedCandidate.remediationPlan));
    state.addAttribute(
        "remediation_statement_strategy",
        builder.getStringAttr(selectedCandidate.remediationStatementStrategy));
    state.addAttribute(
        "remediation_vector_budget",
        builder.getStringAttr(selectedCandidate.remediationVectorBudget));
    state.addAttribute(
        "remediation_schedule_contract",
        builder.getStringAttr(selectedCandidate.remediationScheduleContract));
    state.addAttribute(
        "remediation_unpack_plan",
        builder.getStringAttr(selectedCandidate.remediationUnpackPlan));
    state.addAttribute(
        "remediation_product_plan",
        builder.getStringAttr(selectedCandidate.remediationProductPlan));
    state.addAttribute(
        "remediation_reduction_plan",
        builder.getStringAttr(selectedCandidate.remediationReductionPlan));
    state.addAttribute("remediation_vl_plan",
                       builder.getStringAttr(selectedCandidate.remediationVLPlan));
    state.addAttribute(
        "schedule_decision_contract",
        builder.getStringAttr(selectedCandidate.scheduleDecisionContract));
    state.addAttribute("schedule_decision",
                       builder.getStringAttr(selectedCandidate.scheduleDecision));
    state.addAttribute(
        "schedule_decision_reason",
        builder.getStringAttr(selectedCandidate.scheduleDecisionReason));
  }
  state.addAttribute("producer_scope",
                     builder.getStringAttr(selectedCandidate.producerScope));
  state.addAttribute("consumer_scope",
                     builder.getStringAttr(selectedCandidate.consumerScope));
  state.addAttribute("primitive_chain_contract",
                     builder.getStringAttr(
                         selectedCandidate.primitiveChainContractID));
  state.addAttribute("primitive_chain_kind",
                     builder.getStringAttr(selectedCandidate.primitiveChainKind));
  state.addAttribute("primitive_source_signedness",
                     builder.getStringAttr(selectedCandidate.sourceSignedness));
  state.addAttribute("primitive_source_load",
                     builder.getStringAttr(
                         selectedCandidate.primitiveSourceLoadKind));
  state.addAttribute("primitive_source_extension",
                     builder.getStringAttr(
                         selectedCandidate.primitiveSourceExtensionKind));
  state.addAttribute(
      "widening_product_multiplicand_roles",
      builder.getStringAttr(
          selectedCandidate.wideningProductMultiplicandRoleSummary));
  state.addAttribute("widening_product_extension_policy",
                     builder.getStringAttr(
                         selectedCandidate.wideningProductExtensionPolicy));
  state.addAttribute("widening_product_candidate_fact",
                     builder.getStringAttr(
                         selectedCandidate.wideningProductCandidateFact));
  state.addAttribute("reduction_candidate_fact",
                     builder.getStringAttr(
                         selectedCandidate.reductionCandidateFact));
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
  mlir::Operation *handoff = builder.create(state);
  if (admission)
    materializeLowPrecisionRealizationAdmissionAttrs(builder, handoff,
                                                    *admission);
  return handoff;
}

struct RVVSelectedBodyContractionRealizationPlan {
  mlir::Operation *preRealizedBody = nullptr;

  bool usesWideningMAcc = false;
  bool usesDotReduction = false;
  bool usesProductReductionChain = false;
  bool usesProductReductionDequantization = false;
  bool usesProductReductionDequantClamp = false;
  bool isUnsignedProductReduction = false;
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
  if (!plan.usesProductReductionChain)
    return std::nullopt;
  if (plan.usesProductReductionDequantClamp)
    return RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  if (plan.usesProductReductionDequantization)
    return RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32;
  return RVVSelectedBodyOperationKind::WideningProductReduceAdd;
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
          "source signedness", primitiveFacts.sourceSignedness))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveNonEmptyField(
          "primitive source load", primitiveFacts.sourceLoadKind))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveNonEmptyField(
          "primitive source extension", primitiveFacts.sourceExtensionKind))
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

  llvm::StringRef expectedProductKind =
      plan.isUnsignedProductReduction ? "unsigned_widening_product"
                                      : "signed_widening_product";
  llvm::StringRef expectedSourceSignedness =
      plan.isUnsignedProductReduction
          ? llvm::StringRef(kRVVLowPrecisionResourceSourceSignednessUnsigned)
          : llvm::StringRef(kRVVLowPrecisionResourceSourceSignednessSigned);
  llvm::StringRef expectedSourceExtension =
      plan.isUnsignedProductReduction
          ? "zero-extend-u8-to-u16-product"
          : llvm::StringRef(kRVVLowPrecisionResourcePrimitiveSourceExtension);

  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "product kind", plan.productKind, expectedProductKind))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "source LMUL", primitiveFacts.sourceLMUL, plan.sourceLMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "source signedness", primitiveFacts.sourceSignedness,
          expectedSourceSignedness))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "primitive source load", primitiveFacts.sourceLoadKind,
          kRVVLowPrecisionResourcePrimitiveSourceLoad))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "primitive source extension", primitiveFacts.sourceExtensionKind,
          expectedSourceExtension))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveIntegerField(
          "source SEW", primitiveFacts.sourceSEW, plan.sourceSEW))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "product LMUL", primitiveFacts.productLMUL, plan.productLMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveIntegerField(
          "product SEW", primitiveFacts.productSEW, plan.productSEW))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "reduction result LMUL", primitiveFacts.reductionResultLMUL,
          plan.resultLMUL))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveIntegerField(
          "reduction result SEW", primitiveFacts.reductionResultSEW,
          plan.resultSEW))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "widening product relation", primitiveFacts.wideningProductRelation,
          plan.productRelation))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "product-reduction chain relation",
          primitiveFacts.productReductionChainRelation,
          plan.productReductionChainRelation))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "accumulator layout", primitiveFacts.accumulatorLayout,
          plan.accumulatorLayout))
    return std::move(error);
  if (llvm::Error error = requireLowPrecisionPrimitiveStringField(
          "result layout", primitiveFacts.resultLayout, plan.resultLayout))
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
  plan.usesProductReductionChain = true;
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
    tcrv::rvv::TypedWideningProductReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
  plan.usesProductReductionChain = true;
  plan.opKind = body.getOpKind();
  plan.isUnsignedProductReduction = body.getSourceSignedness() == "unsigned";
  plan.productKind = plan.isUnsignedProductReduction
                         ? "unsigned_widening_product"
                         : "signed_widening_product";
  plan.accumulatorLayout = body.getAccumulatorLayout();
  plan.resultLayout = body.getResultLayout();
  plan.contractionRelation = body.getProductReductionChainRelation();
  plan.productRelation = body.getProductRelation();
  plan.productReductionChainRelation = body.getProductReductionChainRelation();
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
  plan.out = body.getOut();
  plan.n = body.getN();
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedWideningProductReduceDequantizePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
  plan.usesProductReductionChain = true;
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

//===----------------------------------------------------------------------===//
// N3 deferred-wide realization (the autotuner finale): when the resource-aware
// selector picks the wide accumulator-LMUL rung (the vreg budget admits the
// i8m2 -> i16m4 -> i32m8 chain), the realization PRODUCES the deferred-wide
// typed body -- the measured ssh-rvv winner -- instead of the narrow i8mf4
// per-iteration-vwredsum body. This is a PARALLEL realization branch: it never
// touches the narrow fact-consumption path (materializeLowPrecisionResource-
// RealizationAttrs / copyLowPrecisionResourceAttrs / the narrow primitive
// facts, all pinned to mf4/mf2/m1). The selector's budget-pruned rung is
// realized INTO the typed body's vector types (i32m8 accumulator), so the tune
// decision is structural (I5), not a constant or a mirror string. The body
// reproduces exactly the structure RVVToEmitC::isDeferredWideDequantBody
// recognizes and the wide-lmul lit/ssh-rvv evidence validated.
//
// The wide branch fires only for the plain signed product-reduce-dequantize
// (no clamp, plain-byte i8 source); packed-i4 and clamp keep the narrow path.
//===----------------------------------------------------------------------===//
llvm::Expected<tcrv::rvv::WithVLOp> realizeDeferredWideDequantBody(
    const VariantLoweringBoundaryRequest &request, mlir::ArrayAttr requires,
    const RVVSelectedBodyContractionRealizationPlan &plan,
    const RVVLowPrecisionLMULRung &rung) {
  if (!plan.preRealizedBody)
    return makeRVVPluginError(
        "deferred-wide RVV contraction realization requires a pre-realized "
        "body op");
  if (!plan.lhs || !plan.rhs || !plan.acc || !plan.scale || !plan.out ||
      !plan.n)
    return makeRVVPluginError(
        "deferred-wide RVV contraction realization requires lhs/rhs/acc/scale/"
        "out/n runtime ABI values");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = plan.preRealizedBody->getLoc();

  // The strip config is SEW8 LMUL m2 (the selector's source rung); the product
  // widens to i16m4 and the deferred accumulator to i32m8 -- all derived from
  // the budget-pruned rung, not constants.
  const std::int64_t sourceSEW = 8;
  const std::int64_t productSEW = 16;
  const std::int64_t accumulatorSEW = 32;
  const std::int64_t reductionResultSEW = 32;
  const llvm::StringRef reductionResultLMUL = tcrv::rvv::getRVVLMULM1();

  builder.setInsertionPoint(plan.preRealizedBody);
  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
      builder, loc, plan.n, sourceSEW, rung.sourceLMUL, plan.policy));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires, sourceSEW,
                           rung.sourceLMUL, plan.policy);
  // The single deferred-wide slice runs unroll_factor=1: one i8m2 strip per
  // loop iteration into the wide accumulator (matches the wide-lmul lit).
  withVL->setAttr("unroll_factor", builder.getI64IntegerAttr(1));

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, plan.lhs, setvl.getVl(), sourceSEW, rung.sourceLMUL,
      /*isUnsigned=*/false));
  auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, plan.rhs, setvl.getVl(), sourceSEW, rung.sourceLMUL,
      /*isUnsigned=*/false));
  auto product = llvm::cast<tcrv::rvv::WideningProductOp>(
      createRealizedGenericWideningProductCompute(
          builder, loc, "signed_widening_product", "signed-i8m2xi8m2-to-i16m4",
          lhsLoad.getLoaded(), rhsLoad.getLoaded(), setvl.getVl(), productSEW,
          rung.productLMUL, /*isUnsigned=*/false));
  auto accumulate = llvm::cast<tcrv::rvv::WideningAccumulateOp>(
      createRealizedGenericWideningAccumulate(builder, loc, product.getResult(),
                                              setvl.getVl(), accumulatorSEW,
                                              rung.accumulatorLMUL));
  auto reduced = llvm::cast<tcrv::rvv::StandaloneReduceOp>(
      createRealizedGenericDeferredWideTrailingReduceCompute(
          builder, loc, accumulate.getResult(), plan.acc, setvl.getVl(),
          reductionResultSEW, reductionResultLMUL));
  auto dequantized = llvm::cast<tcrv::rvv::DequantizeOp>(
      createRealizedGenericDequantizeCompute(
          builder, loc, plan.dequantizationRelation, reduced.getResult(),
          plan.scale, setvl.getVl(), reductionResultLMUL));
  createRealizedGenericStore(builder, loc, plan.out, dequantized.getResult(),
                             setvl.getVl());
  plan.preRealizedBody->erase();
  return withVL;
}

//===----------------------------------------------------------------------===//
// N3 deferred-wide DOT-REDUCE realization (P-B8, the 2nd kernel family's
// autotuner finale): when the i16 single-widening resource-aware selector picks
// the wide accumulator-LMUL rung (the vreg budget admits the i16m4 -> i32m8
// chain), the realization PRODUCES the deferred-wide dot-reduce typed body -- the
// measured ssh-rvv winner -- instead of the narrow i16mf2 per-iteration-vredsum
// body. PARALLEL to realizeDeferredWideDequantBody but: (a) a SINGLE widening
// step (the product is already i32, so the deferred accumulate is a same-width
// tcrv_rvv.deferred_accumulate vadd.vv, not the byte widening_accumulate); (b) NO
// dequant -- the trailing reduce result stores directly with a scalar acc[0]
// add. The strip config is SEW16/m4 (the dot-reduce strip the wide verifier
// branches require). The selector's budget-pruned rung is realized INTO the typed
// body's vector types (i32m8 accumulator), so the tune decision is structural
// (I5). The body reproduces exactly the structure RVVToEmitC::
// isDeferredWideDotReduceBody recognizes and the wide-lmul lit/ssh-rvv evidence
// validated (P-B7).
//===----------------------------------------------------------------------===//
llvm::Expected<tcrv::rvv::WithVLOp> realizeDeferredWideDotReduceBody(
    const VariantLoweringBoundaryRequest &request, mlir::ArrayAttr requires,
    const RVVSelectedBodyContractionRealizationPlan &plan,
    const RVVDotReduceDeferredWideLMULRung &rung) {
  if (!plan.preRealizedBody)
    return makeRVVPluginError(
        "deferred-wide RVV dot-reduce realization requires a pre-realized "
        "body op");
  if (!plan.lhs || !plan.rhs || !plan.acc || !plan.out || !plan.n)
    return makeRVVPluginError(
        "deferred-wide RVV dot-reduce realization requires lhs/rhs/acc/out/n "
        "runtime ABI values");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = plan.preRealizedBody->getLoc();

  // The strip config is SEW16 LMUL m4 (the selector's source rung, the dot-reduce
  // strip config the wide verifier branches require); the product widens ONE step
  // to i32m8, which IS the deferred accumulator (same width) -- all derived from
  // the budget-pruned rung, not constants.
  const std::int64_t sourceSEW = 16;
  const std::int64_t productSEW = 32;
  const std::int64_t accumulatorSEW = 32;
  const std::int64_t reductionResultSEW = 32;
  const llvm::StringRef reductionResultLMUL = tcrv::rvv::getRVVLMULM1();

  builder.setInsertionPoint(plan.preRealizedBody);
  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
      builder, loc, plan.n, sourceSEW, rung.sourceLMUL, plan.policy));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires, sourceSEW,
                           rung.sourceLMUL, plan.policy);
  // The single deferred-wide slice runs unroll_factor=1: one i16m4 strip per
  // loop iteration into the wide accumulator (matches the wide-lmul lit).
  withVL->setAttr("unroll_factor", builder.getI64IntegerAttr(1));

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, plan.lhs, setvl.getVl(), sourceSEW, rung.sourceLMUL,
      /*isUnsigned=*/false));
  auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, plan.rhs, setvl.getVl(), sourceSEW, rung.sourceLMUL,
      /*isUnsigned=*/false));
  auto product = llvm::cast<tcrv::rvv::WideningProductOp>(
      createRealizedGenericWideningProductCompute(
          builder, loc, "signed_widening_product",
          "signed-i16m4xi16m4-to-i32m8", lhsLoad.getLoaded(),
          rhsLoad.getLoaded(), setvl.getVl(), productSEW, rung.accumulatorLMUL,
          /*isUnsigned=*/false));
  auto accumulate = llvm::cast<tcrv::rvv::DeferredAccumulateOp>(
      createRealizedGenericDeferredAccumulate(builder, loc, product.getResult(),
                                              setvl.getVl(), accumulatorSEW,
                                              rung.accumulatorLMUL));
  auto reduced = llvm::cast<tcrv::rvv::StandaloneReduceOp>(
      createRealizedGenericDeferredWideTrailingReduceCompute(
          builder, loc, accumulate.getResult(), plan.acc, setvl.getVl(),
          reductionResultSEW, reductionResultLMUL));
  createRealizedGenericStore(builder, loc, plan.out, reduced.getResult(),
                             setvl.getVl());
  plan.preRealizedBody->erase();
  return withVL;
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedContractionFamily(
    const VariantLoweringBoundaryRequest &request, mlir::ArrayAttr requires,
    const RVVSelectedBodyContractionRealizationPlan &plan,
    const RVVLowPrecisionProductionPressureProfile *pressureProfile =
        nullptr) {
  if (!plan.preRealizedBody)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "contraction family pre-realized body op");
  if (!plan.usesWideningMAcc && !plan.usesDotReduction &&
      !plan.usesProductReductionChain)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "widening macc, widening dot reduction, or product-reduction family "
        "operation");
  if (plan.usesProductReductionChain &&
      (plan.productRelation.empty() ||
       plan.productReductionChainRelation.empty()))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "product relation and reduction relation for product-reduction "
        "routes");
  if (plan.usesProductReductionDequantization &&
      (!plan.scale || plan.dequantizationRelation.empty() ||
       plan.scaleRole.empty() || plan.dequantStoreBoundary.empty()))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "runtime scale, dequantization relation, scale role, and f32 store "
        "boundary for product-reduction-dequantization routes");
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
  if (pressureProfile && !plan.usesProductReductionDequantization)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization admission "
        "currently applies only to low-precision product-reduction "
        "dequantization families");

  std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
      lowPrecisionPrimitiveFacts;
  if (plan.usesProductReductionChain) {
    std::optional<RVVSelectedBodyOperationKind> operation =
        getLowPrecisionProductReductionRealizationOperation(plan);
    if (!operation)
      return makeRVVPluginError(
          "pre-realized RVV contraction selected-body realization requires "
          "a provider-owned product-reduction operation before consuming "
          "low-precision primitive facts");
    std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts> facts =
        getRVVLowPrecisionWideningReductionPrimitiveFacts(
            *operation, plan.isUnsignedProductReduction);
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
  std::optional<RVVLowPrecisionSelectedBodyRealizationAdmission>
      selectedAdmission;
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
    if (pressureProfile) {
      llvm::Expected<RVVLowPrecisionSelectedBodyRealizationAdmission>
          admission = admitRVVLowPrecisionSelectedBodyRealization(
              *selectedResourceCandidate, pressureProfile,
              "pre-realized RVV contraction selected-body realization "
              "admission");
      if (!admission)
        return admission.takeError();
      if (!admission->admitsRealization())
        return makeRVVPluginError(
            "pre-realized RVV contraction selected-body realization "
            "admission did not admit resource-aware realization");
      materializeLowPrecisionRealizationAdmissionAttrs(
          builder, withVL.getOperation(), *admission);
      selectedAdmission = *admission;
    }
  }

  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Value compareLHSValue;
  mlir::Value compareRHSValue;
  // Both dequant candidates realize as a SINGLE-scope typed body (Stage 3 flip):
  // the typed product/reduce slice + the dequant/clamp chain inlined in the one
  // with_vl scope, NO vsetvl_region_marker placeholders, NO
  // gearbox_cross_region_handoff, NO consumer with_vl. The packed-i4 candidate
  // emits a tcrv_rvv.packed_i4_nibble_unpack_product head with unroll_factor=1;
  // the grouped candidate emits a plain tcrv_rvv.widening_product head with
  // unroll_factor=2 -- ONE typed product/reduce slice that the conversion expands
  // unroll_factor times into the legacy unrolled grouped C. The compute structure
  // is typed; the conversion walks it without reading operand_form/unpack_intent
  // mirror strings.
  const bool realizesSingleScopePackedI4Dequant =
      plan.usesProductReductionDequantization && selectedResourceCandidate &&
      isRVVLowPrecisionResourcePackedI4CandidateID(
          selectedResourceCandidate->candidateID);
  const bool realizesSingleScopeGroupedDequant =
      plan.usesProductReductionDequantization && selectedResourceCandidate &&
      isRVVLowPrecisionResourceGroupedCandidateID(
          selectedResourceCandidate->candidateID);
  const bool realizesSingleScopeDequant =
      realizesSingleScopePackedI4Dequant || realizesSingleScopeGroupedDequant;
  if (plan.usesProductReductionDequantization && !realizesSingleScopeDequant) {
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
        selectedResourceCandidate->planningContract, 1,
        selectedResourceCandidate->vsetvlRegionCount, resourceDecision);
    if (usesGroupedLowPrecisionProductReduction)
      createRealizedVSetVLRegionMarker(
          builder, loc, setvl.getVl(), "tail-product-reduce",
          selectedResourceCandidate->planningContract, 2,
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
                                           plan.sourceLMUL,
                                           plan.isUnsignedProductReduction));
      return load.getLoaded();
    }
    auto load = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, buffer, setvl.getVl(), plan.sourceSEW,
        plan.sourceLMUL, plan.isUnsignedProductReduction));
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

  if (plan.usesProductReductionChain) {
    if (!lowPrecisionPrimitiveFacts)
      return makeRVVPluginError(
          "pre-realized RVV contraction selected-body realization lost "
          "provider-owned low-precision primitive facts before materializing "
          "product-reduction structure");
    llvm::StringRef productRelation =
        selectedResourceCandidate
            ? llvm::StringRef(
                  selectedResourceCandidate->primitiveWideningProductRelation)
            : llvm::StringRef(
                  lowPrecisionPrimitiveFacts->wideningProductRelation);
    llvm::StringRef accumulatorLayout =
        selectedResourceCandidate
            ? llvm::StringRef(
                  selectedResourceCandidate->primitiveAccumulatorLayout)
            : llvm::StringRef(lowPrecisionPrimitiveFacts->accumulatorLayout);
    llvm::StringRef resultLayout =
        selectedResourceCandidate
            ? llvm::StringRef(selectedResourceCandidate->primitiveResultLayout)
            : llvm::StringRef(lowPrecisionPrimitiveFacts->resultLayout);
    // The packed-i4 single-scope flip emits a typed nibble-unpack product head;
    // the grouped single-scope flip and every other candidate keep the typed
    // widening_product head.
    mlir::Value productResult;
    if (realizesSingleScopePackedI4Dequant) {
      auto nibbleProduct =
          llvm::cast<tcrv::rvv::PackedI4NibbleUnpackProductOp>(
              createRealizedGenericPackedI4NibbleUnpackProductCompute(
                  builder, loc, productRelation, lhsValue, rhsValue,
                  setvl.getVl(), plan.productSEW, plan.productLMUL));
      productResult = nibbleProduct.getResult();
    } else {
      auto product = llvm::cast<tcrv::rvv::WideningProductOp>(
          createRealizedGenericWideningProductCompute(
              builder, loc, plan.productKind, productRelation, lhsValue,
              rhsValue, setvl.getVl(), plan.productSEW, plan.productLMUL,
              plan.isUnsignedProductReduction));
      productResult = product.getResult();
    }
    auto reduced = llvm::cast<tcrv::rvv::StandaloneReduceOp>(
        createRealizedGenericStandaloneWideningReduceCompute(
            builder, loc, accumulatorLayout, resultLayout, productResult,
            plan.acc, setvl.getVl(), plan.resultSEW, plan.resultLMUL,
            plan.isUnsignedProductReduction));
    if (!plan.usesProductReductionDequantization) {
      createRealizedGenericStore(builder, loc, plan.out, reduced.getResult(),
                                 setvl.getVl());
    } else if (realizesSingleScopeDequant) {
      // Single-scope dequant: inline the dequant(/clamp) chain in the producer
      // with_vl -- no handoff, no consumer scope. The i32 carry feeds dequantize
      // directly. Stamp the bare structural `unroll_factor` the conversion reads
      // to size the chunk loop: 1 for packed-i4 (one slice, one plain loop), 2
      // for grouped (the conversion expands the ONE typed slice twice in the main
      // loop and adds the scalar tail loop). The factor is the selected Gearbox
      // candidate's structural unroll, not a mirror string the conversion reads.
      withVL->setAttr("unroll_factor", builder.getI64IntegerAttr(
                                           selectedResourceCandidate->unrollFactor));
      auto dequantized = llvm::cast<tcrv::rvv::DequantizeOp>(
          createRealizedGenericDequantizeCompute(
              builder, loc, plan.dequantizationRelation, reduced.getResult(),
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
                                         lowerSplat.getBroadcast(),
                                         setvl.getVl(),
                                         plan.lowerPredicateKind));
        auto lowerSelect = llvm::cast<tcrv::rvv::SelectOp>(
            createRealizedGenericSelect(builder, loc, lowerCompare.getMask(),
                                        lowerSplat.getBroadcast(),
                                        dequantized.getResult(),
                                        setvl.getVl()));
        auto upperCompare = llvm::cast<tcrv::rvv::CompareOp>(
            createRealizedGenericCompare(builder, loc,
                                         upperSplat.getBroadcast(),
                                         lowerSelect.getSelected(),
                                         setvl.getVl(),
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
    } else {
      mlir::Value dequantSource = reduced.getResult();
      tcrv::rvv::WithVLOp consumerWithVL;
      auto handoff = llvm::cast<tcrv::rvv::GearboxCrossRegionHandoffOp>(
          createRealizedGearboxCrossRegionHandoff(
              builder, loc, reduced.getResult(), setvl.getVl(), plan.n,
              *selectedResourceCandidate,
              selectedAdmission ? &*selectedAdmission : nullptr));
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
      if (consumerCandidate->candidateID !=
          selectedResourceCandidate->candidateID)
        return makeRVVPluginError(
            "pre-realized RVV contraction selected-body realization requires "
            "producer and consumer scopes to consume the same selected "
            "low-precision resource candidate");
      if (selectedAdmission)
        materializeLowPrecisionRealizationAdmissionAttrs(
            builder, consumerWithVL.getOperation(), *selectedAdmission);
      builder.setInsertionPointToStart(&consumerWithVL.getBody().front());
      createRealizedVSetVLRegionMarker(
          builder, loc, setvl.getVl(), "dequant-store",
          consumerCandidate->planningContract,
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
                                         lowerSplat.getBroadcast(),
                                         setvl.getVl(),
                                         plan.lowerPredicateKind));
        auto lowerSelect = llvm::cast<tcrv::rvv::SelectOp>(
            createRealizedGenericSelect(builder, loc, lowerCompare.getMask(),
                                        lowerSplat.getBroadcast(),
                                        dequantized.getResult(),
                                        setvl.getVl()));
        auto upperCompare = llvm::cast<tcrv::rvv::CompareOp>(
            createRealizedGenericCompare(builder, loc,
                                         upperSplat.getBroadcast(),
                                         lowerSelect.getSelected(),
                                         setvl.getVl(),
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
    }
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
      tcrv::rvv::TypedWideningProductReducePreRealizedBodyOp,
      tcrv::rvv::TypedWideningProductReduceDequantizePreRealizedBodyOp,
      tcrv::rvv::
          TypedWideningProductReduceDequantClampF32PreRealizedBodyOp,
      tcrv::rvv::TypedWideningProductReduceDequantClampF32BodyOp>(op);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVContractionOwnerImpl(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp,
    const RVVLowPrecisionProductionPressureProfile *pressureProfile) {
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
        request, requires, makeContractionRealizationPlan(wideningMAccBody),
        pressureProfile);
  }

  if (auto dotReduceBody = llvm::dyn_cast<
          tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningDotReduceBody(
                request, dotReduceBody))
      return std::move(error);
    RVVSelectedBodyContractionRealizationPlan plan =
        makeContractionRealizationPlan(dotReduceBody);
    // N3 autotuner finale for the 2nd kernel family (P-B8): ask the i16 single-
    // widening resource-aware selector whether the vreg budget admits the wide
    // accumulator-LMUL rung. The budget is the pass-stamped architectural vreg-
    // file fact (default 32; only stamped on the narrow i16mf2 dot-reduce strip
    // the selector serves); a constrained budget prunes the wide rung and falls
    // through to the narrow per-iteration-vredsum path. Same kernel realizes
    // wide at budget 32, narrow at a constrained budget -- the budget genuinely
    // drives the choice (N3). Plain (non-strided, non-masked) dot-reduce only;
    // the strided/masked variants have their own branches and stay byte-intact.
    if (!plan.usesStridedInputs && !plan.usesComputedMask && !pressureProfile) {
      llvm::Expected<std::optional<std::int64_t>> vectorRegisterBudget =
          readLowPrecisionResourceIntegerFact(
              dotReduceBody.getOperation(),
              kRVVLowPrecisionResourceVectorRegisterBudgetAttrName);
      if (!vectorRegisterBudget)
        return vectorRegisterBudget.takeError();
      if (*vectorRegisterBudget) {
        // Reserve = the load/temp headroom the strip loop keeps live BESIDES the
        // i32 accumulator (which the enumerator costs separately): the two i16
        // source loads plus slack, rounded to 8. Because the i16 product IS the
        // i32 accumulator width (one widening, the deferred vadd aliases the
        // product into the accumulator), the enumerator's prune is acc_regs +
        // reserve <= budget. At budget 32 the i32m8 rung's 8+8=16 fits -> wide;
        // a budget below 16 prunes it -> narrow. The fixed reserve is the honest
        // bound; the budget genuinely drives the wide/narrow crossover.
        constexpr std::int64_t kDeferredWideDotReduceReserveRegisterCost = 8;
        llvm::SmallVector<RVVDotReduceDeferredWideLMULRung, 4> rungs =
            enumerateRVVDotReduceDeferredWideLMULRungs(
                **vectorRegisterBudget,
                kDeferredWideDotReduceReserveRegisterCost);
        std::optional<RVVDotReduceDeferredWideLMULRung> selected =
            selectRVVDotReduceDeferredWideMaxLegalLMULRung(rungs);
        // Realize the deferred-wide winner only when the budget-pruned selection
        // is the i32m8 accumulator rung (source i16m4). Any narrower legal rung
        // (a constrained budget) falls through to the narrow realization.
        if (selected && selected->accumulatorLMUL == tcrv::rvv::getRVVLMULM8())
          return realizeDeferredWideDotReduceBody(request, requires, plan,
                                                  *selected);
      }
    }
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, plan, pressureProfile);
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
        makeContractionRealizationPlan(stridedDotReduceBody),
        pressureProfile);
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
        request, requires, makeContractionRealizationPlan(maskedDotReduceBody),
        pressureProfile);
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
        makeContractionRealizationPlan(maskedStridedDotReduceBody),
        pressureProfile);
  }

  if (auto productReduceBody = llvm::dyn_cast<
          tcrv::rvv::TypedWideningProductReducePreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningProductReduceBody(
                request, productReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, makeContractionRealizationPlan(productReduceBody),
        pressureProfile);
  }

  if (auto productReduceDequantBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedWideningProductReduceDequantizePreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningProductReduceDequantizeBody(
                request, productReduceDequantBody))
      return std::move(error);
    RVVSelectedBodyContractionRealizationPlan plan =
        makeContractionRealizationPlan(productReduceDequantBody);
    // N3 autotuner finale: ask the resource-aware selector whether the vreg
    // budget admits the wide accumulator-LMUL rung. The budget is the
    // pass-stamped architectural vreg-file fact (default 32); a constrained
    // budget prunes the wide rung and falls through to the narrow path. This is
    // what makes narrow-vs-wide selection genuinely resource-driven (N3): the
    // SAME kernel realizes wide at budget 32, narrow at a constrained budget.
    // Gated to the plain signed product-reduce-dequantize (no clamp): packed-i4
    // and clamp keep the narrow path; the wide chain assumes plain i8 source.
    // The packed-i4 vs unpacked-byte distinction is the gearbox-stamped
    // operand_form fact (the pre-realized op type is identical for both
    // encodings), so gate the wide branch on that real body fact.
    mlir::StringAttr operandForm =
        productReduceDequantBody->getAttrOfType<mlir::StringAttr>(
            kRVVLowPrecisionResourceOperandFormAttrName);
    const bool isUnpackedByteSource =
        operandForm && operandForm.getValue() ==
                           kRVVLowPrecisionResourceOperandFormUnpackedByte;
    if (!plan.usesProductReductionDequantClamp &&
        !plan.isUnsignedProductReduction && isUnpackedByteSource &&
        !pressureProfile) {
      llvm::Expected<std::optional<std::int64_t>> vectorRegisterBudget =
          readLowPrecisionResourceIntegerFact(
              productReduceDequantBody.getOperation(),
              kRVVLowPrecisionResourceVectorRegisterBudgetAttrName);
      if (!vectorRegisterBudget)
        return vectorRegisterBudget.takeError();
      if (*vectorRegisterBudget) {
        // Reserve = the load/temp headroom the strip loop keeps live BESIDES the
        // accumulator + product (which the enumerator already costs separately):
        // the two i8m2 source loads (2+2=4) plus slack, rounded to 8. The
        // enumerator's prune is acc_regs + product_regs + reserve <= budget, so
        // this fixed reserve sets the wide/narrow CROSSOVER threshold (at budget
        // 32 the i32m8 rung's 8+4+8=20 fits -> wide; a budget below 20 prunes it
        // -> narrow). The fixed reserve is the honest bound on "budget-derived":
        // the budget genuinely drives the choice, but the headroom is hand-set.
        constexpr std::int64_t kDeferredWideReserveRegisterCost = 8;
        llvm::SmallVector<RVVLowPrecisionLMULRung, 4> rungs =
            enumerateRVVLowPrecisionAccumulatorLMULRungs(
                **vectorRegisterBudget, kDeferredWideReserveRegisterCost);
        std::optional<RVVLowPrecisionLMULRung> selected =
            selectRVVLowPrecisionMaxLegalAccumulatorLMULRung(rungs);
        // Realize the deferred-wide winner only when the budget-pruned selection
        // is the i32m8 accumulator rung (source i8m2). Any narrower legal rung
        // (a constrained budget) falls through to the narrow realization.
        if (selected && selected->accumulatorLMUL == tcrv::rvv::getRVVLMULM8())
          return realizeDeferredWideDequantBody(request, requires, plan,
                                                *selected);
      }
    }
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, plan, pressureProfile);
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
        makeContractionRealizationPlan(productReduceDequantClampBody),
        pressureProfile);
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
        makeContractionRealizationPlan(explicitProductReduceDequantClampBody),
        pressureProfile);
  }

  return makeRVVPluginError(
      "contraction selected-body realization owner found an unsupported "
      "pre-realized body op");
}

llvm::Expected<tcrv::rvv::WithVLOp> realizePreRealizedRVVContractionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  return realizePreRealizedRVVContractionOwnerImpl(request, bodyOp, nullptr);
}

llvm::Expected<tcrv::rvv::WithVLOp> realizePreRealizedRVVContractionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp,
    const RVVLowPrecisionProductionPressureProfile &pressureProfile) {
  return realizePreRealizedRVVContractionOwnerImpl(request, bodyOp,
                                                  &pressureProfile);
}

} // namespace tianchenrv::plugin::rvv
