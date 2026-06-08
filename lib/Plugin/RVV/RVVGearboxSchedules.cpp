#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZERVVGEARBOXSCHEDULES
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32DestLMUL;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32DestSEW;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32CandidateSet;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32LegalityScope;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32Operation;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32ScheduleID;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32SelectedCandidate;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32SelectionReason;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32Selector;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32SourceLMUL;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32SourceSEW;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32Unroll;
using tianchenrv::plugin::rvv::kRVVGearboxCandidateSetAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxDestLMULAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxDestSEWAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxLegalityScopeAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxOperationAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxDequantizeI32ToF32SelectedVLPolicy;
using tianchenrv::plugin::rvv::kRVVGearboxConsumerScope;
using tianchenrv::plugin::rvv::kRVVGearboxConsumerScopeAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxProducerScope;
using tianchenrv::plugin::rvv::kRVVGearboxProducerScopeAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxRuntimeAVLSourceAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxRuntimeAVLSourceN;
using tianchenrv::plugin::rvv::kRVVGearboxScheduleIDAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSelectedCandidateAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSelectionReasonAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSelectorAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSourceAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSourceLMULAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxSourceSEWAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxStaticPassSource;
using tianchenrv::plugin::rvv::kRVVGearboxUnrollAttrName;
using tianchenrv::plugin::rvv::kRVVGearboxVLPolicyAttrName;
using tianchenrv::tcrv::rvv::DequantizeOp;
using tianchenrv::tcrv::rvv::GearboxCrossRegionHandoffOp;
using tianchenrv::tcrv::rvv::LoadOp;
using tianchenrv::tcrv::rvv::CompareOp;
using tianchenrv::tcrv::rvv::RuntimeABIValueOp;
using tianchenrv::tcrv::rvv::SelectOp;
using tianchenrv::tcrv::rvv::SetVLOp;
using tianchenrv::tcrv::rvv::SplatOp;
using tianchenrv::tcrv::rvv::StandaloneReduceOp;
using tianchenrv::tcrv::rvv::StoreOp;
using tianchenrv::tcrv::rvv::
    TypedWideningProductReduceDequantClampF32BodyOp;
using tianchenrv::tcrv::rvv::
    TypedWideningProductReduceDequantClampF32PreRealizedBodyOp;
using tianchenrv::tcrv::rvv::
    TypedWideningProductReduceDequantizePreRealizedBodyOp;
using tianchenrv::tcrv::rvv::VectorType;
using tianchenrv::tcrv::rvv::WithVLOp;
using tianchenrv::tcrv::rvv::WideningProductOp;

constexpr llvm::StringLiteral kDequantizationKind("i32_to_f32_scaled");
constexpr llvm::StringLiteral kDequantizationRelation(
    "signed-i32m1-to-f32m1-scale-f32");
constexpr llvm::StringLiteral kLowPrecisionProductDequantOpKind(
    "widening_product_reduce_dequantize_f32");
constexpr llvm::StringLiteral kLowPrecisionProductDequantClampOpKind(
    "widening_product_reduce_dequant_clamp_f32");
constexpr llvm::StringLiteral kLowPrecisionProductDequantMemoryForm(
    "unit-stride-widening-product-reduce-dequantize-f32");
constexpr llvm::StringLiteral kLowPrecisionProductDequantClampMemoryForm(
    "unit-stride-widening-product-reduce-dequant-clamp-f32");
constexpr llvm::StringLiteral kLowPrecisionProductRelation(
    "signed-i8mf4xi8mf4-to-i16mf2");
constexpr llvm::StringLiteral kLowPrecisionProductReductionRelation(
    "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32");
constexpr llvm::StringLiteral kLowPrecisionProductKind(
    "signed_widening_product");
constexpr llvm::StringLiteral kLowPrecisionReduceKind(
    "signed_widening_reduce_add");
constexpr llvm::StringLiteral kLowPrecisionAccumulatorRole(
    "accumulator-input-buffer");
constexpr llvm::StringLiteral kLowPrecisionAccumulatorLayout(
    "scalar-i32-seed-lane0-from-accumulator-input");
constexpr llvm::StringLiteral kLowPrecisionResultLayout(
    "store-standalone-reduction-lane0-to-output-scalar");
constexpr llvm::StringLiteral kLowPrecisionDequantStoreBoundary(
    "store-dequantized-f32-vector-to-output-buffer");
constexpr llvm::StringLiteral kLowPrecisionDequantClampStoreBoundary(
    "store-clamped-dequantized-f32-vector-to-output-buffer");
constexpr llvm::StringLiteral kLowPrecisionDequantClampLowerBoundRole(
    "lower-bound-scalar-value");
constexpr llvm::StringLiteral kLowPrecisionDequantClampUpperBoundRole(
    "upper-bound-scalar-value");
constexpr llvm::StringLiteral kLowPrecisionDequantClampPredicateKind("slt");
constexpr llvm::StringLiteral kLowPrecisionDequantClampBoundOrder(
    "lower-bound-before-upper-bound");
constexpr llvm::StringLiteral kLowPrecisionDequantClampSelectLayout(
    "clamp-lower-then-upper");
constexpr llvm::StringLiteral kLowPrecisionCrossRegionHandoffContract(
    "gearbox-product-reduce-to-dequant-cross-region-handoff.v1");
constexpr llvm::StringLiteral kLowPrecisionCrossRegionHandoffFromPhase(
    "load-product-reduce");
constexpr llvm::StringLiteral kLowPrecisionCrossRegionHandoffToPhase(
    "dequant-store");

mlir::LogicalResult requireRuntimeABIValue(RuntimeABIValueOp value,
                                           mlir::Operation *anchor,
                                           llvm::StringRef context,
                                           llvm::StringRef role,
                                           llvm::StringRef cName) {
  if (!value)
    return anchor->emitError()
           << "RVV Gearbox schedule derivation for " << context
           << " requires a visible tcrv_rvv.runtime_abi_value";
  if (value.getRole() != role || value.getCName() != cName)
    return anchor->emitError()
           << "RVV Gearbox schedule derivation for " << context
           << " requires runtime ABI role '" << role << "' named '" << cName
           << "'";
  return mlir::success();
}

mlir::LogicalResult requireStringAttr(mlir::Operation *op,
                                      mlir::OpBuilder &builder,
                                      llvm::StringRef attrName,
                                      llvm::StringRef expected) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr) {
    op->setAttr(attrName, builder.getStringAttr(expected));
    return mlir::success();
  }
  if (attr.getValue() == expected)
    return mlir::success();
  return op->emitError()
         << "RVV Gearbox pass found stale schedule fact '" << attrName
         << "': expected '" << expected << "' but found '" << attr.getValue()
         << "'";
}

mlir::LogicalResult requireIntegerAttr(mlir::Operation *op,
                                       mlir::OpBuilder &builder,
                                       llvm::StringRef attrName,
                                       std::int64_t expected) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr) {
    op->setAttr(attrName, builder.getI64IntegerAttr(expected));
    return mlir::success();
  }
  if (attr.getInt() == expected)
    return mlir::success();
  return op->emitError()
         << "RVV Gearbox pass found stale schedule fact '" << attrName
         << "': expected '" << expected << "' but found '" << attr.getInt()
         << "'";
}

mlir::LogicalResult materializeGearboxAttrs(mlir::Operation *op,
                                            mlir::OpBuilder &builder) {
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxCandidateSetAttrName,
          kRVVGearboxDequantizeI32ToF32CandidateSet)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxSelectedCandidateAttrName,
          kRVVGearboxDequantizeI32ToF32SelectedCandidate)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxSelectionReasonAttrName,
          kRVVGearboxDequantizeI32ToF32SelectionReason)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxLegalityScopeAttrName,
          kRVVGearboxDequantizeI32ToF32LegalityScope)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxScheduleIDAttrName,
          kRVVGearboxDequantizeI32ToF32ScheduleID)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxSelectorAttrName,
          kRVVGearboxDequantizeI32ToF32Selector)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(op, builder, kRVVGearboxSourceAttrName,
                                     kRVVGearboxStaticPassSource)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxOperationAttrName,
          kRVVGearboxDequantizeI32ToF32Operation)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVGearboxUnrollAttrName,
          kRVVGearboxDequantizeI32ToF32Unroll)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxVLPolicyAttrName,
          kRVVGearboxDequantizeI32ToF32SelectedVLPolicy)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVGearboxSourceSEWAttrName,
          kRVVGearboxDequantizeI32ToF32SourceSEW)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVGearboxSourceLMULAttrName,
          kRVVGearboxDequantizeI32ToF32SourceLMUL)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(op, builder, kRVVGearboxDestSEWAttrName,
                                      kRVVGearboxDequantizeI32ToF32DestSEW)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(op, builder, kRVVGearboxDestLMULAttrName,
                                     kRVVGearboxDequantizeI32ToF32DestLMUL)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(op, builder,
                                     kRVVGearboxRuntimeAVLSourceAttrName,
                                     kRVVGearboxRuntimeAVLSourceN)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(op, builder,
                                     kRVVGearboxProducerScopeAttrName,
                                     kRVVGearboxProducerScope)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(op, builder,
                                     kRVVGearboxConsumerScopeAttrName,
                                     kRVVGearboxConsumerScope)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult requireVectorType(mlir::Value value,
                                      mlir::Operation *anchor,
                                      llvm::StringRef context,
                                      mlir::Type elementType,
                                      llvm::StringRef lmul) {
  auto vectorType = llvm::dyn_cast<VectorType>(value.getType());
  if (!vectorType)
    return anchor->emitError()
           << "RVV Gearbox schedule derivation for " << context
           << " requires a typed !tcrv_rvv.vector value";
  if (vectorType.getElementType() != elementType || vectorType.getLmul() != lmul)
    return anchor->emitError()
           << "RVV Gearbox schedule derivation for " << context
           << " requires element type " << elementType << " and LMUL " << lmul;
  return mlir::success();
}

llvm::StringRef
stringifyGearboxTailPolicy(tianchenrv::tcrv::rvv::TailPolicy policy) {
  switch (policy) {
  case tianchenrv::tcrv::rvv::TailPolicy::Agnostic:
    return "agnostic";
  case tianchenrv::tcrv::rvv::TailPolicy::Undisturbed:
    return "undisturbed";
  }
  return "";
}

llvm::StringRef
stringifyGearboxMaskPolicy(tianchenrv::tcrv::rvv::MaskPolicy policy) {
  switch (policy) {
  case tianchenrv::tcrv::rvv::MaskPolicy::Agnostic:
    return "agnostic";
  case tianchenrv::tcrv::rvv::MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  return "";
}

mlir::LogicalResult requireLowPrecisionProductDequantShape(
    mlir::Operation *op, std::int64_t sourceSEW, llvm::StringRef sourceLMUL,
    std::int64_t productSEW, llvm::StringRef productLMUL,
    std::int64_t resultSEW, llvm::StringRef resultLMUL,
    llvm::StringRef memoryForm) {
  const bool supportedMemoryForm =
      memoryForm == kLowPrecisionProductDequantMemoryForm ||
      memoryForm == kLowPrecisionProductDequantClampMemoryForm;
  if (sourceSEW == 8 && sourceLMUL == "mf4" && productSEW == 16 &&
      productLMUL == "mf2" && resultSEW == 32 && resultLMUL == "m1" &&
      supportedMemoryForm)
    return mlir::success();
  return op->emitError()
         << "RVV low-precision Gearbox resource candidate derivation supports "
            "only "
         << kLowPrecisionProductDequantMemoryForm << " or "
         << kLowPrecisionProductDequantClampMemoryForm
         << " with i8mf4 sources, i16mf2 product, i32m1 accumulator, and "
            "f32m1 result";
}

mlir::LogicalResult materializeLowPrecisionResourceAttrs(
    mlir::Operation *op, mlir::OpBuilder &builder,
    tianchenrv::tcrv::rvv::PolicyAttr policy, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t productSEW,
    llvm::StringRef productLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, llvm::StringRef memoryForm) {
  if (!policy)
    return op->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires an explicit RVV policy";
  if (mlir::failed(requireLowPrecisionProductDequantShape(
          op, sourceSEW, sourceLMUL, productSEW, productLMUL, resultSEW,
          resultLMUL, memoryForm)))
    return mlir::failure();

  using namespace tianchenrv::plugin::rvv;
  std::optional<RVVLowPrecisionContractionResourceOperation> operation =
      getRVVLowPrecisionResourceOperationForMemoryForm(memoryForm);
  if (!operation)
    return op->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires a supported product-reduction memory form";
  llvm::StringRef tailPolicy = stringifyGearboxTailPolicy(policy.getTail());
  llvm::StringRef maskPolicy = stringifyGearboxMaskPolicy(policy.getMask());
  llvm::SmallVector<RVVLowPrecisionContractionResourceCandidate, 2>
      candidates = buildRVVLowPrecisionProductReductionResourceCandidates(
          *operation, tailPolicy, maskPolicy, sourceSEW, sourceLMUL,
          productSEW, productLMUL, resultSEW, resultLMUL, resultSEW,
          resultLMUL, kRVVLowPrecisionResourceVectorRegisterBudget);
  std::optional<RVVLowPrecisionContractionResourceCandidate> selected =
      selectRVVLowPrecisionProductReductionResourceCandidate(candidates);
  if (!selected) {
    llvm::StringRef rejection =
        candidates.empty() ? llvm::StringRef("no-resource-candidates-built")
                           : candidates.front().rejectionReason;
    return op->emitError()
           << "RVV low-precision Gearbox resource candidate pruning rejected "
              "all candidates for "
           << memoryForm << ": " << rejection
           << " (requires tail agnostic, mask agnostic policy and peak live "
              "vector-group estimate within the vector register budget)";
  }

  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceCandidateSetAttrName,
          selected->candidateSetID)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceSelectedCandidateAttrName,
          selected->candidateID)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceSelectionReasonAttrName,
          selected->selectionReason)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceLegalityScopeAttrName,
          selected->legalityScope)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceSourceDTypeAttrName,
          selected->sourceElementTypeName)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVLowPrecisionResourceSourceSEWAttrName,
          selected->sourceSEW)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceSourceLMULAttrName,
          selected->sourceLMUL)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceOperandFormAttrName,
          selected->operandForm)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceSourceSignednessAttrName,
          selected->sourceSignedness)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVLowPrecisionResourceStorageElementWidthAttrName,
          selected->storageElementWidth)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVLowPrecisionResourceEffectiveElementWidthAttrName,
          selected->effectiveElementWidth)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourcePackingLayoutAttrName,
          selected->packingLayout)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceUnpackIntentAttrName,
          selected->unpackIntent)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceProductDTypeAttrName,
          selected->productElementTypeName)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVLowPrecisionResourceProductSEWAttrName,
          selected->productSEW)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceProductLMULAttrName,
          selected->productLMUL)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceProductEMULAttrName,
          selected->productEMUL)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceAccumulatorDTypeAttrName,
          selected->accumulatorElementTypeName)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVLowPrecisionResourceAccumulatorSEWAttrName,
          selected->accumulatorSEW)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceAccumulatorLMULAttrName,
          selected->accumulatorLMUL)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceAccumulatorEMULAttrName,
          selected->accumulatorEMUL)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceResultDTypeAttrName,
          selected->resultElementTypeName)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVLowPrecisionResourceResultSEWAttrName,
          selected->resultSEW)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceResultLMULAttrName,
          selected->resultLMUL)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceMemoryFormAttrName,
          selected->memoryForm)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceTailPolicyAttrName,
          selected->tailPolicy)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceMaskPolicyAttrName,
          selected->maskPolicy)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVLowPrecisionResourceUnrollFactorAttrName,
          selected->unrollFactor)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVLowPrecisionResourceAccumulatorCountAttrName,
          selected->accumulatorCount)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceReductionLayoutAttrName,
          selected->reductionLayout)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVLowPrecisionResourceVSetVLRegionCountAttrName,
          selected->vsetvlRegionCount)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVLowPrecisionResourcePeakLiveVectorGroupsAttrName,
          selected->peakLiveVectorGroups)))
    return mlir::failure();
  if (mlir::failed(requireIntegerAttr(
          op, builder, kRVVLowPrecisionResourceVectorRegisterBudgetAttrName,
          selected->vectorRegisterBudget)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceRuntimeAVLSourceAttrName,
          selected->runtimeAVLSource)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(op, builder,
                                     kRVVGearboxProducerScopeAttrName,
                                     selected->producerScope)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(op, builder,
                                     kRVVGearboxConsumerScopeAttrName,
                                     selected->consumerScope)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceRuntimeABIOrderAttrName,
          selected->runtimeABIOrder)))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceLegalityAttrName,
          selected->isLegal ? llvm::StringRef(kRVVLowPrecisionResourceLegal)
                            : llvm::StringRef("rejected"))))
    return mlir::failure();
  if (mlir::failed(requireStringAttr(
          op, builder, kRVVLowPrecisionResourceRejectionReasonAttrName,
          selected->rejectionReason)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult validatePreRealizedLowPrecisionProductDequantBody(
    TypedWideningProductReduceDequantizePreRealizedBodyOp body) {
  if (body.getOpKind() != kLowPrecisionProductDequantOpKind ||
      body.getMemoryForm() != kLowPrecisionProductDequantMemoryForm)
    return body->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires op_kind '"
           << kLowPrecisionProductDequantOpKind << "' and memory_form '"
           << kLowPrecisionProductDequantMemoryForm << "'";
  if (body.getAccumulatorRole() != kLowPrecisionAccumulatorRole ||
      body.getAccumulatorLayout() != kLowPrecisionAccumulatorLayout ||
      body.getResultLayout() != kLowPrecisionResultLayout)
    return body->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires the bounded scalar i32 accumulator/result layout";
  if (body.getProductRelation() != kLowPrecisionProductRelation ||
      body.getProductReductionChainRelation() !=
          kLowPrecisionProductReductionRelation ||
      body.getDequantRelation() != kDequantizationRelation ||
      body.getScaleRole() != "dequant-scale-value" ||
      body.getDequantStoreBoundary() != kLowPrecisionDequantStoreBoundary)
    return body->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires product/reduction/dequant/store boundary facts from "
              "the selected typed body";

  if (mlir::failed(requireLowPrecisionProductDequantShape(
          body.getOperation(), body.getSourceSew(), body.getSourceLmul(),
          body.getProductSew(), body.getProductLmul(), body.getResultSew(),
          body.getResultLmul(), body.getMemoryForm())))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getLhs().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision lhs", "lhs-input-buffer", "lhs")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getRhs().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision rhs", "rhs-input-buffer", "rhs")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getAcc().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision accumulator", "accumulator-input-buffer", "acc")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getScale().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision dequant scale", "dequant-scale-value", "scale")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getOut().getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision output", "output-buffer", "out")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getN().getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision runtime AVL", "runtime-element-count", "n")))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult materializeLowPrecisionResourceForPreRealizedBody(
    TypedWideningProductReduceDequantizePreRealizedBodyOp body) {
  if (mlir::failed(validatePreRealizedLowPrecisionProductDequantBody(body)))
    return mlir::failure();
  mlir::OpBuilder builder(body.getContext());
  return materializeLowPrecisionResourceAttrs(
      body.getOperation(), builder, body.getPolicy(), body.getSourceSew(),
      body.getSourceLmul(), body.getProductSew(), body.getProductLmul(),
      body.getResultSew(), body.getResultLmul(), body.getMemoryForm());
}

template <typename BodyOp>
mlir::LogicalResult validateLowPrecisionProductDequantClampBody(BodyOp body) {
  if (body.getOpKind() != kLowPrecisionProductDequantClampOpKind ||
      body.getMemoryForm() != kLowPrecisionProductDequantClampMemoryForm)
    return body->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires op_kind '"
           << kLowPrecisionProductDequantClampOpKind
           << "' and memory_form '"
           << kLowPrecisionProductDequantClampMemoryForm << "'";
  if (body.getAccumulatorRole() != kLowPrecisionAccumulatorRole ||
      body.getAccumulatorLayout() != kLowPrecisionAccumulatorLayout ||
      body.getResultLayout() != kLowPrecisionResultLayout)
    return body->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires the bounded scalar i32 accumulator/result layout";
  if (body.getProductRelation() != kLowPrecisionProductRelation ||
      body.getProductReductionChainRelation() !=
          kLowPrecisionProductReductionRelation ||
      body.getDequantRelation() != kDequantizationRelation ||
      body.getScaleRole() != "dequant-scale-value" ||
      body.getDequantStoreBoundary() != kLowPrecisionDequantClampStoreBoundary)
    return body->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires product/reduction/dequant/clamp-store boundary facts "
              "from the selected typed body";
  if (body.getLowerPredicateKind() != kLowPrecisionDequantClampPredicateKind ||
      body.getUpperPredicateKind() != kLowPrecisionDequantClampPredicateKind ||
      body.getBoundOrder() != kLowPrecisionDequantClampBoundOrder ||
      body.getSelectLayout() != kLowPrecisionDequantClampSelectLayout)
    return body->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires lower/upper clamp predicates, bound order, and "
              "select layout from the selected typed body";

  if (mlir::failed(requireLowPrecisionProductDequantShape(
          body.getOperation(), body.getSourceSew(), body.getSourceLmul(),
          body.getProductSew(), body.getProductLmul(), body.getResultSew(),
          body.getResultLmul(), body.getMemoryForm())))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getLhs().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision lhs", "lhs-input-buffer", "lhs")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getRhs().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision rhs", "rhs-input-buffer", "rhs")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getAcc().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision accumulator", "accumulator-input-buffer", "acc")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getScale().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision dequant scale", "dequant-scale-value", "scale")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getLowerBound().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision lower clamp bound",
          kLowPrecisionDequantClampLowerBoundRole, "lower_bound")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getUpperBound().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision upper clamp bound",
          kLowPrecisionDequantClampUpperBoundRole, "upper_bound")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getOut().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision output", "output-buffer", "out")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          body.getN().template getDefiningOp<RuntimeABIValueOp>(), body,
          "low-precision runtime AVL", "runtime-element-count", "n")))
    return mlir::failure();
  return mlir::success();
}

template <typename BodyOp>
mlir::LogicalResult
materializeLowPrecisionResourceForProductDequantClampBody(BodyOp body) {
  if (mlir::failed(validateLowPrecisionProductDequantClampBody(body)))
    return mlir::failure();
  mlir::OpBuilder builder(body.getContext());
  return materializeLowPrecisionResourceAttrs(
      body.getOperation(), builder, body.getPolicy(), body.getSourceSew(),
      body.getSourceLmul(), body.getProductSew(), body.getProductLmul(),
      body.getResultSew(), body.getResultLmul(), body.getMemoryForm());
}

mlir::LogicalResult validateDequantizationGearboxBody(WithVLOp withVL,
                                                      DequantizeOp dequantize) {
  SetVLOp setvl = withVL.getVl().getDefiningOp<SetVLOp>();
  tianchenrv::tcrv::rvv::RVVConfigContractDiagnostic config =
      tianchenrv::tcrv::rvv::validateRVVSelectedBodyM1ConfigVLContract(setvl,
                                                                       withVL);
  if (!config.ok)
    return withVL->emitError()
           << "RVV Gearbox schedule derivation requires selected "
              "dequantize_i32_to_f32 SEW32 LMUL m1 config: "
           << config.message;

  if (dequantize.getKind() != kDequantizationKind ||
      dequantize.getDequantRelation() != kDequantizationRelation)
    return dequantize->emitError()
           << "RVV Gearbox schedule derivation requires dequantize kind '"
           << kDequantizationKind << "' and relation '"
           << kDequantizationRelation << "'";

  if (dequantize.getVl() != withVL.getVl())
    return dequantize->emitError()
           << "RVV Gearbox schedule derivation requires dequantize to consume "
              "the active with_vl token";

  auto *context = withVL.getContext();
  if (mlir::failed(requireVectorType(dequantize.getSource(), dequantize,
                                     "source vector", mlir::IntegerType::get(context, 32),
                                     kRVVGearboxDequantizeI32ToF32SourceLMUL)))
    return mlir::failure();
  if (mlir::failed(requireVectorType(dequantize.getResult(), dequantize,
                                     "result vector",
                                     mlir::Float32Type::get(context),
                                     kRVVGearboxDequantizeI32ToF32DestLMUL)))
    return mlir::failure();

  LoadOp sourceLoad = dequantize.getSource().getDefiningOp<LoadOp>();
  if (!sourceLoad || sourceLoad->getParentOp() != withVL.getOperation() ||
      sourceLoad.getVl() != withVL.getVl())
    return dequantize->emitError()
           << "RVV Gearbox schedule derivation requires the dequant source to "
              "come from one same-scope tcrv_rvv.load";

  if (mlir::failed(requireRuntimeABIValue(
          sourceLoad.getBuffer().getDefiningOp<RuntimeABIValueOp>(), sourceLoad,
          "source load", "lhs-input-buffer", "lhs")))
    return mlir::failure();

  if (mlir::failed(requireRuntimeABIValue(
          dequantize.getScale().getDefiningOp<RuntimeABIValueOp>(), dequantize,
          "runtime scale", "dequant-scale-value", "scale")))
    return mlir::failure();

  RuntimeABIValueOp avl = setvl.getAvl().getDefiningOp<RuntimeABIValueOp>();
  if (mlir::failed(requireRuntimeABIValue(avl, setvl, "runtime AVL",
                                          "runtime-element-count", "n")))
    return mlir::failure();

  StoreOp store;
  for (mlir::OpOperand &use : dequantize.getResult().getUses()) {
    auto candidate = llvm::dyn_cast<StoreOp>(use.getOwner());
    if (!candidate || use.getOperandNumber() != 1)
      return dequantize->emitError()
             << "RVV Gearbox schedule derivation requires the dequant result "
                "to feed only the result operand of tcrv_rvv.store";
    if (store)
      return dequantize->emitError()
             << "RVV Gearbox schedule derivation requires exactly one "
                "dequant result store";
    store = candidate;
  }
  if (!store || store->getParentOp() != withVL.getOperation() ||
      store.getVl() != withVL.getVl())
    return dequantize->emitError()
           << "RVV Gearbox schedule derivation requires one same-scope "
              "tcrv_rvv.store of the dequant result";

  if (mlir::failed(requireRuntimeABIValue(
          store.getBuffer().getDefiningOp<RuntimeABIValueOp>(), store,
          "result store", "output-buffer", "out")))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult
validateLowPrecisionProductDequantGearboxBody(WithVLOp withVL,
                                              DequantizeOp dequantize) {
  SetVLOp setvl = withVL.getVl().getDefiningOp<SetVLOp>();
  tianchenrv::tcrv::rvv::RVVConfigContractDiagnostic config =
      tianchenrv::tcrv::rvv::validateRVVSelectedBodyM1ConfigVLContract(setvl,
                                                                       withVL);
  if (!config.ok)
    return withVL->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires selected product-dequant SEW32 LMUL m1 config: "
           << config.message;

  GearboxCrossRegionHandoffOp handoff =
      dequantize.getSource().getDefiningOp<GearboxCrossRegionHandoffOp>();
  if (!handoff)
    return dequantize->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires dequantize source from one provider-collected "
              "tcrv_rvv.gearbox_cross_region_handoff";
  if (handoff.getVl() != withVL.getVl() || handoff.getRuntimeAvl() != setvl.getAvl())
    return handoff->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires tcrv_rvv.gearbox_cross_region_handoff to consume the "
              "active with_vl token and runtime AVL";
  auto producerWithVL =
      llvm::dyn_cast_or_null<WithVLOp>(handoff->getParentOp());
  if (!producerWithVL)
    return handoff->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires handoff to be nested directly in a producer "
              "tcrv_rvv.with_vl";
  if (producerWithVL == withVL)
    return handoff->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires a true multi-with_vl producer/consumer boundary";
  if (!producerWithVL->isProperAncestor(withVL.getOperation()))
    return handoff->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires producer tcrv_rvv.with_vl to enclose the consumer "
              "tcrv_rvv.with_vl scope";

  tianchenrv::tcrv::rvv::RVVConfigContractDiagnostic producerConfig =
      tianchenrv::tcrv::rvv::validateRVVSelectedBodyM1ConfigVLContract(
          setvl, producerWithVL);
  if (!producerConfig.ok)
    return producerWithVL->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires producer product/reduction SEW32 LMUL m1 config: "
           << producerConfig.message;

  StandaloneReduceOp reduce = handoff.getInput().getDefiningOp<StandaloneReduceOp>();
  if (!reduce || reduce->getParentOp() != producerWithVL.getOperation() ||
      reduce.getVl() != withVL.getVl())
    return handoff->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires handoff input from one producer-scope "
              "tcrv_rvv.standalone_reduce";

  WideningProductOp product =
      reduce.getInput().getDefiningOp<WideningProductOp>();
  if (!product || product->getParentOp() != producerWithVL.getOperation() ||
      product.getVl() != withVL.getVl())
    return reduce->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires standalone_reduce input from one producer-scope "
              "tcrv_rvv.widening_product";

  auto *context = withVL.getContext();
  if (mlir::failed(requireVectorType(product.getLhs(), product, "lhs source",
                                     mlir::IntegerType::get(context, 8),
                                     "mf4")))
    return mlir::failure();
  if (mlir::failed(requireVectorType(product.getRhs(), product, "rhs source",
                                     mlir::IntegerType::get(context, 8),
                                     "mf4")))
    return mlir::failure();
  if (mlir::failed(requireVectorType(product.getResult(), product,
                                     "widening product result",
                                     mlir::IntegerType::get(context, 16),
                                     "mf2")))
    return mlir::failure();
  if (mlir::failed(requireVectorType(reduce.getResult(), reduce,
                                     "reduction accumulator/result",
                                     mlir::IntegerType::get(context, 32),
                                     "m1")))
    return mlir::failure();
  if (mlir::failed(requireVectorType(handoff.getOutput(), handoff,
                                     "cross-region handoff result",
                                     mlir::IntegerType::get(context, 32),
                                     "m1")))
    return mlir::failure();
  if (mlir::failed(requireVectorType(dequantize.getResult(), dequantize,
                                     "dequantized result",
                                     mlir::Float32Type::get(context), "m1")))
    return mlir::failure();

  if (product.getKind() != kLowPrecisionProductKind ||
      product.getProductRelation() != kLowPrecisionProductRelation)
    return product->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires widening_product kind '"
           << kLowPrecisionProductKind << "' and relation '"
           << kLowPrecisionProductRelation << "'";
  if (reduce.getKind() != kLowPrecisionReduceKind ||
      reduce.getAccumulatorLayout() != kLowPrecisionAccumulatorLayout ||
      reduce.getResultLayout() != kLowPrecisionResultLayout)
    return reduce->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires signed widening reduce and scalar i32 accumulator "
              "layout";
  if (dequantize.getKind() != kDequantizationKind ||
      dequantize.getDequantRelation() != kDequantizationRelation ||
      dequantize.getVl() != withVL.getVl())
    return dequantize->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires dequantize kind '"
           << kDequantizationKind << "', relation '"
           << kDequantizationRelation << "', and the active with_vl token";
  const bool usesGroupedLowPrecisionHandoff =
      handoff.getResourceDecision() ==
      tianchenrv::plugin::rvv::
          kRVVLowPrecisionResourceGroupedRealizationDecision;
  const std::int64_t expectedLowPrecisionRegionCount =
      usesGroupedLowPrecisionHandoff
          ? tianchenrv::plugin::rvv::
                kRVVLowPrecisionResourceGroupedVSetVLRegions
          : tianchenrv::plugin::rvv::kRVVLowPrecisionResourceVSetVLRegions;
  const llvm::StringRef expectedLowPrecisionFromPhase =
      usesGroupedLowPrecisionHandoff
          ? llvm::StringRef("tail-product-reduce")
          : llvm::StringRef(kLowPrecisionCrossRegionHandoffFromPhase);
  const bool hasSupportedLowPrecisionDecision =
      handoff.getResourceDecision() ==
          tianchenrv::plugin::rvv::kRVVLowPrecisionResourceRealizationDecision ||
      usesGroupedLowPrecisionHandoff;
  if (handoff.getContract() != kLowPrecisionCrossRegionHandoffContract ||
      handoff.getFromPhase() != expectedLowPrecisionFromPhase ||
      handoff.getToPhase() != kLowPrecisionCrossRegionHandoffToPhase ||
      static_cast<std::int64_t>(handoff.getRegionCount()) !=
          expectedLowPrecisionRegionCount ||
      handoff.getRuntimeAvlSource() !=
          tianchenrv::plugin::rvv::kRVVGearboxRuntimeAVLSourceN ||
      !hasSupportedLowPrecisionDecision ||
      handoff.getProducerScope() !=
          tianchenrv::plugin::rvv::kRVVGearboxProducerScope ||
      handoff.getConsumerScope() !=
          tianchenrv::plugin::rvv::kRVVGearboxConsumerScope ||
      handoff.getProducerScope() == handoff.getConsumerScope())
    return handoff->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires tcrv_rvv.gearbox_cross_region_handoff to carry the "
              "RVV-owned product/reduction-to-dequant contract, phases, "
              "region count, runtime AVL source, resource decision, and "
              "distinct producer/consumer scopes";

  LoadOp lhsLoad = product.getLhs().getDefiningOp<LoadOp>();
  LoadOp rhsLoad = product.getRhs().getDefiningOp<LoadOp>();
  if (!lhsLoad || lhsLoad->getParentOp() != producerWithVL.getOperation() ||
      lhsLoad.getVl() != withVL.getVl() || !rhsLoad ||
      rhsLoad->getParentOp() != producerWithVL.getOperation() ||
      rhsLoad.getVl() != withVL.getVl())
    return product->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires producer-scope lhs/rhs unit-stride loads";

  if (mlir::failed(requireRuntimeABIValue(
          lhsLoad.getBuffer().getDefiningOp<RuntimeABIValueOp>(), lhsLoad,
          "low-precision lhs load", "lhs-input-buffer", "lhs")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          rhsLoad.getBuffer().getDefiningOp<RuntimeABIValueOp>(), rhsLoad,
          "low-precision rhs load", "rhs-input-buffer", "rhs")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          reduce.getAccumulatorSeed().getDefiningOp<RuntimeABIValueOp>(),
          reduce, "low-precision accumulator", "accumulator-input-buffer",
          "acc")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          dequantize.getScale().getDefiningOp<RuntimeABIValueOp>(), dequantize,
          "low-precision dequant scale", "dequant-scale-value", "scale")))
    return mlir::failure();
  RuntimeABIValueOp avl = setvl.getAvl().getDefiningOp<RuntimeABIValueOp>();
  if (mlir::failed(requireRuntimeABIValue(avl, setvl, "low-precision AVL",
                                          "runtime-element-count", "n")))
    return mlir::failure();

  StoreOp directStore;
  for (mlir::OpOperand &use : dequantize.getResult().getUses()) {
    auto candidate = llvm::dyn_cast<StoreOp>(use.getOwner());
    if (!candidate)
      continue;
    if (use.getOperandNumber() != 1)
      return dequantize->emitError()
             << "RVV low-precision Gearbox resource candidate derivation "
                "requires the dequantized f32 value to feed the value "
                "operand of tcrv_rvv.store";
    if (directStore)
      return dequantize->emitError()
             << "RVV low-precision Gearbox resource candidate derivation "
                "requires exactly one dequantized result store";
    directStore = candidate;
  }
  if (directStore) {
    if (directStore->getParentOp() != withVL.getOperation() ||
        directStore.getVl() != withVL.getVl())
      return dequantize->emitError()
             << "RVV low-precision Gearbox resource candidate derivation "
                "requires one consumer-scope store of the dequantized result";
    if (mlir::failed(requireRuntimeABIValue(
            directStore.getBuffer().getDefiningOp<RuntimeABIValueOp>(),
            directStore, "low-precision output", "output-buffer", "out")))
      return mlir::failure();
    return mlir::success();
  }

  SelectOp lowerSelect;
  SelectOp upperSelect;
  for (mlir::Operation &op : withVL.getBody().front()) {
    auto select = llvm::dyn_cast<SelectOp>(op);
    if (!select || select.getVl() != withVL.getVl())
      continue;
    if (select.getFalseValue() == dequantize.getResult()) {
      if (lowerSelect)
        return select->emitError()
               << "RVV low-precision Gearbox resource candidate derivation "
                  "requires exactly one lower clamp select consuming the "
                  "dequantized result";
      lowerSelect = select;
      continue;
    }
    if (lowerSelect && select.getFalseValue() == lowerSelect.getSelected()) {
      if (upperSelect)
        return select->emitError()
               << "RVV low-precision Gearbox resource candidate derivation "
                  "requires exactly one upper clamp select consuming the "
                  "lower-clamped result";
      upperSelect = select;
    }
  }
  if (!lowerSelect || !upperSelect)
    return dequantize->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires the handoff dequant result to feed a lower-then-upper "
              "clamp compare/select chain or a direct store";

  CompareOp lowerCompare =
      lowerSelect.getMask().getDefiningOp<CompareOp>();
  CompareOp upperCompare =
      upperSelect.getMask().getDefiningOp<CompareOp>();
  SplatOp lowerSplat =
      lowerSelect.getTrueValue().getDefiningOp<SplatOp>();
  SplatOp upperSplat =
      upperSelect.getTrueValue().getDefiningOp<SplatOp>();
  if (!lowerCompare || !upperCompare || !lowerSplat || !upperSplat)
    return lowerSelect->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires explicit lower/upper splat and compare producers for "
              "the clamp selects";
  if (lowerSplat.getVl() != withVL.getVl() ||
      upperSplat.getVl() != withVL.getVl() ||
      lowerCompare.getVl() != withVL.getVl() ||
      upperCompare.getVl() != withVL.getVl() ||
      lowerCompare.getKind() != kLowPrecisionDequantClampPredicateKind ||
      upperCompare.getKind() != kLowPrecisionDequantClampPredicateKind ||
      lowerCompare.getLhs() != dequantize.getResult() ||
      lowerCompare.getRhs() != lowerSplat.getBroadcast() ||
      lowerSelect.getMask() != lowerCompare.getMask() ||
      lowerSelect.getTrueValue() != lowerSplat.getBroadcast() ||
      upperCompare.getLhs() != upperSplat.getBroadcast() ||
      upperCompare.getRhs() != lowerSelect.getSelected() ||
      upperSelect.getMask() != upperCompare.getMask() ||
      upperSelect.getTrueValue() != upperSplat.getBroadcast())
    return lowerSelect->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires lower-then-upper f32 clamp compare/select structure "
              "in the consumer with_vl";

  if (mlir::failed(requireRuntimeABIValue(
          lowerSplat.getScalar().getDefiningOp<RuntimeABIValueOp>(),
          lowerSplat, "low-precision lower clamp bound",
          kLowPrecisionDequantClampLowerBoundRole, "lower_bound")))
    return mlir::failure();
  if (mlir::failed(requireRuntimeABIValue(
          upperSplat.getScalar().getDefiningOp<RuntimeABIValueOp>(),
          upperSplat, "low-precision upper clamp bound",
          kLowPrecisionDequantClampUpperBoundRole, "upper_bound")))
    return mlir::failure();

  StoreOp clampStore;
  for (mlir::OpOperand &use : upperSelect.getSelected().getUses()) {
    auto candidate = llvm::dyn_cast<StoreOp>(use.getOwner());
    if (!candidate || use.getOperandNumber() != 1)
      return upperSelect->emitError()
             << "RVV low-precision Gearbox resource candidate derivation "
                "requires the upper-clamped value to feed only the value "
                "operand of tcrv_rvv.store";
    if (clampStore)
      return upperSelect->emitError()
             << "RVV low-precision Gearbox resource candidate derivation "
                "requires exactly one clamp output store";
    clampStore = candidate;
  }
  if (!clampStore || clampStore->getParentOp() != withVL.getOperation() ||
      clampStore.getVl() != withVL.getVl())
    return upperSelect->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires one consumer-scope store of the clamped result";
  if (mlir::failed(requireRuntimeABIValue(
          clampStore.getBuffer().getDefiningOp<RuntimeABIValueOp>(),
          clampStore, "low-precision output", "output-buffer", "out")))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult materializeGearboxForWithVL(WithVLOp withVL) {
  llvm::SmallVector<DequantizeOp, 2> dequantizeOps;
  for (mlir::Operation &op : withVL.getBody().front())
    if (auto dequantize = llvm::dyn_cast<DequantizeOp>(op))
      dequantizeOps.push_back(dequantize);
  if (dequantizeOps.empty())
    return mlir::success();
  if (dequantizeOps.size() != 1)
    return withVL->emitError()
           << "RVV Gearbox MVP supports exactly one dequantize_i32_to_f32 op "
              "per with_vl body";

  DequantizeOp dequantize = dequantizeOps.front();
  if (dequantize.getSource().getDefiningOp<GearboxCrossRegionHandoffOp>()) {
    if (mlir::failed(validateLowPrecisionProductDequantGearboxBody(
            withVL, dequantize)))
      return mlir::failure();
    mlir::OpBuilder builder(withVL.getContext());
    std::optional<tianchenrv::tcrv::rvv::PolicyAttr> policy =
        withVL.getPolicy();
    if (!policy)
      return withVL->emitError()
             << "RVV low-precision Gearbox resource candidate derivation "
                "requires an explicit with_vl policy";
    auto handoff =
        dequantize.getSource().getDefiningOp<GearboxCrossRegionHandoffOp>();
    auto producerWithVL =
        llvm::dyn_cast_or_null<WithVLOp>(handoff->getParentOp());
    if (!producerWithVL)
      return handoff->emitError()
             << "RVV low-precision Gearbox resource candidate derivation "
                "requires handoff producer tcrv_rvv.with_vl";
    if (mlir::failed(materializeLowPrecisionResourceAttrs(
            producerWithVL.getOperation(), builder, *policy, 8, "mf4", 16,
            "mf2", 32, "m1", kLowPrecisionProductDequantMemoryForm)))
      return mlir::failure();
    return materializeLowPrecisionResourceAttrs(
        withVL.getOperation(), builder, *policy, 8, "mf4", 16, "mf2", 32,
        "m1", kLowPrecisionProductDequantMemoryForm);
  }
  if (dequantize.getSource().getDefiningOp<StandaloneReduceOp>())
    return dequantize->emitError()
           << "RVV low-precision Gearbox resource candidate derivation "
              "requires source-producing Gearbox handoff before dequantize";

  if (mlir::failed(validateDequantizationGearboxBody(withVL, dequantize)))
    return mlir::failure();

  mlir::OpBuilder builder(withVL.getContext());
  if (mlir::failed(materializeGearboxAttrs(withVL.getOperation(), builder)))
    return mlir::failure();
  if (mlir::failed(materializeGearboxAttrs(dequantize.getOperation(), builder)))
    return mlir::failure();
  return mlir::success();
}

class MaterializeRVVGearboxSchedulesPass final
    : public impl::MaterializeRVVGearboxSchedulesBase<
          MaterializeRVVGearboxSchedulesPass> {
public:
  void runOnOperation() override {
    llvm::SmallVector<TypedWideningProductReduceDequantizePreRealizedBodyOp, 8>
        preRealizedProductDequantBodies;
    llvm::SmallVector<
        TypedWideningProductReduceDequantClampF32PreRealizedBodyOp, 8>
        preRealizedProductDequantClampBodies;
    llvm::SmallVector<TypedWideningProductReduceDequantClampF32BodyOp, 8>
        explicitProductDequantClampBodies;
    getOperation()->walk(
        [&](TypedWideningProductReduceDequantizePreRealizedBodyOp body) {
          preRealizedProductDequantBodies.push_back(body);
        });
    getOperation()->walk(
        [&](TypedWideningProductReduceDequantClampF32PreRealizedBodyOp body) {
          preRealizedProductDequantClampBodies.push_back(body);
        });
    getOperation()->walk(
        [&](TypedWideningProductReduceDequantClampF32BodyOp body) {
          explicitProductDequantClampBodies.push_back(body);
        });
    for (TypedWideningProductReduceDequantizePreRealizedBodyOp body :
         preRealizedProductDequantBodies) {
      if (mlir::failed(materializeLowPrecisionResourceForPreRealizedBody(
              body))) {
        signalPassFailure();
        return;
      }
    }
    for (TypedWideningProductReduceDequantClampF32PreRealizedBodyOp body :
         preRealizedProductDequantClampBodies) {
      if (mlir::failed(
              materializeLowPrecisionResourceForProductDequantClampBody(
                  body))) {
        signalPassFailure();
        return;
      }
    }
    for (TypedWideningProductReduceDequantClampF32BodyOp body :
         explicitProductDequantClampBodies) {
      if (mlir::failed(
              materializeLowPrecisionResourceForProductDequantClampBody(
                  body))) {
        signalPassFailure();
        return;
      }
    }

    llvm::SmallVector<WithVLOp, 8> withVLOps;
    getOperation()->walk([&](WithVLOp withVL) { withVLOps.push_back(withVL); });

    for (WithVLOp withVL : withVLOps) {
      if (mlir::failed(materializeGearboxForWithVL(withVL))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeRVVGearboxSchedulesPass() {
  return std::make_unique<MaterializeRVVGearboxSchedulesPass>();
}

} // namespace tianchenrv::transforms
