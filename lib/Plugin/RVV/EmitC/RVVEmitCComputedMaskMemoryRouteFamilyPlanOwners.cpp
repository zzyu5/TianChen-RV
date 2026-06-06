#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"

#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {
namespace {

bool isComputedMaskSegment2MemoryRouteOperation(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return true;
  default:
    return false;
  }
}

bool isPreRealizedRuntimeScalarComputedMaskStoreOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_store";
}

bool isPreRealizedRuntimeScalarComputedMaskStoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-computed-mask-store";
}

bool isPreRealizedRuntimeScalarComputedMaskStorePredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isPreRealizedRuntimeScalarComputedMaskLoadStoreOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_load_store";
}

bool isPreRealizedRuntimeScalarComputedMaskLoadStoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-computed-mask-load-store";
}

bool isPreRealizedRuntimeScalarComputedMaskLoadStorePredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isPreRealizedRuntimeScalarComputedMaskMemoryConfig(std::int64_t sew,
                                                        llvm::StringRef lmul) {
  if (tcrv::rvv::isRVVSelectedBodyM1Config(sew, lmul))
    return true;
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      lmul == tcrv::rvv::getRVVLMULM2())
    return true;
  return tcrv::rvv::isRVVSelectedBodyI64M1Config(sew, lmul);
}

bool isPreRealizedComputedMaskStridedStoreMemoryMovementStrideUnit(
    llvm::StringRef strideUnit) {
  return strideUnit == "byte";
}

bool isPreRealizedComputedMaskIndexedMemoryMovementIndexEEW(
    std::int64_t indexEEW) {
  return indexEEW == 32;
}

bool isPreRealizedComputedMaskIndexedMemoryMovementOffsetUnit(
    llvm::StringRef offsetUnit) {
  return offsetUnit == "element";
}

bool isPreRealizedComputedMaskIndexedScatterIndexUniqueness(
    llvm::StringRef indexUniqueness) {
  return indexUniqueness == "unique";
}

bool isPreRealizedComputedMaskMemoryMovementInactiveLanePolicy(
    llvm::StringRef policy) {
  return policy == "preserve-old-destination" ||
         policy == "preserve-output-on-false-lanes";
}

bool isPreRealizedComputedMaskMemoryMovementOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_unit_load_store";
}

bool isPreRealizedComputedMaskStridedStoreOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_strided_store";
}

bool isPreRealizedComputedMaskStridedLoadOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_strided_load_unit_store";
}

bool isPreRealizedComputedMaskIndexedGatherOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_indexed_gather_load_unit_store";
}

bool isPreRealizedComputedMaskIndexedScatterOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_indexed_scatter_store_unit_load";
}

bool isPreRealizedComputedMaskMemoryMovementPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "slt";
}

bool isPreRealizedComputedMaskMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-store";
}

bool isPreRealizedComputedMaskStridedStoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-strided-store";
}

bool isPreRealizedComputedMaskStridedLoadMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-strided-load-unit-store";
}

bool isPreRealizedComputedMaskIndexedGatherMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-indexed-gather-load-unit-store";
}

bool isPreRealizedComputedMaskIndexedScatterMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-indexed-scatter-store";
}

bool isPreRealizedComputedMaskMemoryMovementMaskRole(llvm::StringRef role) {
  return role == "predicate-mask-produced-by-compare";
}

bool isPreRealizedComputedMaskMemoryMovementMaskSource(
    llvm::StringRef source) {
  return source == "compare-produced-mask-same-vl-scope";
}

bool isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "compare-produced-mask";
}

llvm::Expected<tcrv::rvv::RuntimeABIValueOp>
requirePreRealizedComputedMaskMemoryRuntimeABIValue(
    mlir::Value value, llvm::StringRef context,
    support::RuntimeABIParameterRole expectedRole) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVEmitCRouteProviderError(llvm::Twine(context) +
                                          " must be defined by explicit "
                                          "tcrv_rvv.runtime_abi_value");

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVEmitCRouteProviderError(llvm::Twine(context) +
                                          " carries unsupported runtime ABI "
                                          "role '" +
                                          binding.getRole() + "'");
  if (*role != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " must bind runtime ABI role '" +
        support::stringifyRuntimeABIParameterRole(expectedRole) +
        "' before RVV selected-body realization");
  return binding;
}

template <typename BodyOpT>
llvm::Error rejectMixedPreRealizedComputedMaskMemoryBody(
    tcrv::exec::VariantOp variant, llvm::StringRef bodyDescription) {
  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp, BodyOpT>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected ") + bodyDescription +
        " body must not be mixed with already realized RVV route body op '" +
        unexpectedRVVOp->getName().getStringRef() + "'");
  return llvm::Error::success();
}

llvm::Error requireComputedMaskMemorySelectedVariantRequires(
    tcrv::exec::VariantOp variant, llvm::StringRef context) {
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requires || requires.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected ") + context +
        " realization requires non-empty selected variant requires metadata");
  return llvm::Error::success();
}

llvm::StringRef getRuntimeScalarComputedMaskElementCType(std::int64_t sew) {
  return sew == tcrv::rvv::getRVVSEW64Bits() ? "int64_t" : "int32_t";
}

} // namespace

llvm::Error validatePreRealizedRVVSelectedRuntimeScalarComputedMaskStoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV runtime scalar computed-mask store realization requires "
        "a pre-realized runtime scalar computed-mask store body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedRuntimeScalarComputedMaskStoreOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only op_kind 'runtime_scalar_cmp_masked_store'");
  if (!isPreRealizedRuntimeScalarComputedMaskStorePredicateKind(
          body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only predicate_kind 'sle'");
  if (!isPreRealizedRuntimeScalarComputedMaskStoreMemoryForm(
          body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only memory_form "
        "'runtime-scalar-computed-mask-store'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "currently supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "requires inactive_lane_policy 'preserve-output-on-false-lanes'");
  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  if (!isPreRealizedRuntimeScalarComputedMaskMemoryConfig(sew,
                                                          body.getLmul()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "requires SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 LMUL m1 data/mask "
        "config");
  if (!tcrv::rvv::isRVVUndisturbedPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask store body "
        "requires tail undisturbed, mask undisturbed policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV runtime scalar computed-mask store lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime scalar computed-mask store rhs scalar "
          "operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV runtime scalar computed-mask store payload source "
          "operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV runtime scalar computed-mask store destination "
          "operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime scalar computed-mask store runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  llvm::StringRef expectedElementCType =
      getRuntimeScalarComputedMaskElementCType(sew);
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedElementCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedElementCType) + " *").str();
  if ((*lhs).getCType() != expectedConstPointer ||
      (*rhsScalar).getCType() != expectedElementCType ||
      (*source).getCType() != expectedConstPointer ||
      (*destination).getCType() != expectedMutablePointer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected runtime scalar computed-mask "
                    "store body requires lhs/source ") +
        expectedConstPointer + ", rhs scalar " + expectedElementCType +
        ", and destination " + expectedMutablePointer +
        " runtime ABI bindings");

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskMemoryBody<
              tcrv::rvv::TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp>(
              variant, "runtime scalar computed-mask store"))
    return error;
  return requireComputedMaskMemorySelectedVariantRequires(
      variant, "runtime scalar computed-mask store");
}

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskLoadStoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV runtime scalar computed-mask load-store realization "
        "requires a pre-realized runtime scalar computed-mask load-store body "
        "op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedRuntimeScalarComputedMaskLoadStoreOpKind(
          body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only op_kind "
        "'runtime_scalar_cmp_masked_load_store'");
  if (!isPreRealizedRuntimeScalarComputedMaskLoadStorePredicateKind(
          body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only predicate_kind 'sle'");
  if (!isPreRealizedRuntimeScalarComputedMaskLoadStoreMemoryForm(
          body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only memory_form "
        "'runtime-scalar-computed-mask-load-store'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body currently supports only mask_memory_form "
        "'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-old-destination")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body requires inactive_lane_policy 'preserve-old-destination'");
  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  if (!isPreRealizedRuntimeScalarComputedMaskMemoryConfig(sew,
                                                          body.getLmul()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body requires SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 LMUL m1 "
        "data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask load-store "
        "body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV runtime scalar computed-mask load-store lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime scalar computed-mask load-store rhs "
          "scalar operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV runtime scalar computed-mask load-store source "
          "operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV runtime scalar computed-mask load-store "
          "destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime scalar computed-mask load-store runtime "
          "n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  llvm::StringRef expectedElementCType =
      getRuntimeScalarComputedMaskElementCType(sew);
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedElementCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedElementCType) + " *").str();
  if ((*lhs).getCType() != expectedConstPointer ||
      (*rhsScalar).getCType() != expectedElementCType ||
      (*source).getCType() != expectedConstPointer ||
      (*destination).getCType() != expectedMutablePointer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected runtime scalar computed-mask "
                    "load-store body requires lhs/source ") +
        expectedConstPointer + ", rhs scalar " + expectedElementCType +
        ", and destination " + expectedMutablePointer +
        " runtime ABI bindings");

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskMemoryBody<
              tcrv::rvv::
                  TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp>(
              variant, "runtime scalar computed-mask load-store"))
    return error;
  return requireComputedMaskMemorySelectedVariantRequires(
      variant, "runtime scalar computed-mask load-store");
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask memory realization requires a "
        "pre-realized computed-mask memory body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask memory body must be a "
        "direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskMemoryMovementOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only op_kind 'computed_masked_unit_load_store'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskMemoryMovementMemoryForm(
          body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only memory_form 'computed-mask-unit-load-store'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask memory body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedComputedMaskMemoryMovementInactiveLanePolicy(
          body.getInactiveLanePolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask memory body requires "
        "inactive_lane_policy 'preserve-old-destination'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask memory body requires SEW32 "
        "LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask memory body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask memory compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask memory compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask memory active source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV computed-mask memory destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask memory runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskMemoryBody<
              tcrv::rvv::TypedComputedMaskMemoryPreRealizedBodyOp>(
              variant, "computed-mask memory"))
    return error;
  return requireComputedMaskMemorySelectedVariantRequires(
      variant, "computed-mask memory");
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskStridedStoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskStridedStorePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask strided-store realization requires a "
        "pre-realized computed-mask strided-store body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-store body must be "
        "a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskStridedStoreOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only op_kind 'computed_masked_strided_store'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskStridedStoreMemoryForm(body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only memory_form "
        "'computed-mask-unit-load-strided-store'");
  if (!isPreRealizedComputedMaskStridedStoreMemoryMovementStrideUnit(
          body.getStrideUnit()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only stride_unit 'byte'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-store body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (!isPreRealizedComputedMaskMemoryMovementInactiveLanePolicy(
          body.getInactiveLanePolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-store body requires "
        "inactive_lane_policy 'preserve-old-destination'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-store body requires "
        "SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-store body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask strided-store compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask strided-store compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask strided-store active source "
          "operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV computed-mask strided-store destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask strided-store runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destinationStride =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getDestinationStride(),
          "pre-realized RVV computed-mask strided-store destination byte stride "
          "operand",
          support::RuntimeABIParameterRole::DestinationByteStride);
  if (!destinationStride)
    return destinationStride.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskMemoryBody<
              tcrv::rvv::TypedComputedMaskStridedStorePreRealizedBodyOp>(
              variant, "computed-mask strided-store"))
    return error;
  return requireComputedMaskMemorySelectedVariantRequires(
      variant, "computed-mask strided-store");
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskStridedLoadBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskStridedLoadPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask strided-load realization requires a "
        "pre-realized computed-mask strided-load body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-load body must be "
        "a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskStridedLoadOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only op_kind 'computed_masked_strided_load_unit_store'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskStridedLoadMemoryForm(body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only memory_form "
        "'computed-mask-strided-load-unit-store'");
  if (!isPreRealizedComputedMaskStridedStoreMemoryMovementStrideUnit(
          body.getStrideUnit()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only stride_unit 'byte'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-load body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() !=
      "preserve-passthrough-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-load body requires "
        "inactive_lane_policy 'preserve-passthrough-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-load body requires "
        "SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-load body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask strided-load compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask strided-load compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask strided-load source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV computed-mask strided-load destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask strided-load runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> sourceStride =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getSourceStride(),
          "pre-realized RVV computed-mask strided-load source byte stride "
          "operand",
          support::RuntimeABIParameterRole::SourceByteStride);
  if (!sourceStride)
    return sourceStride.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskMemoryBody<
              tcrv::rvv::TypedComputedMaskStridedLoadPreRealizedBodyOp>(
              variant, "computed-mask strided-load"))
    return error;
  return requireComputedMaskMemorySelectedVariantRequires(
      variant, "computed-mask strided-load");
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskIndexedGatherBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskIndexedGatherPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask indexed gather-load realization requires "
        "a pre-realized computed-mask indexed gather body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskIndexedGatherOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only op_kind "
        "'computed_masked_indexed_gather_load_unit_store'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskIndexedGatherMemoryForm(body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only memory_form "
        "'computed-mask-indexed-gather-load-unit-store'");
  if (!isPreRealizedComputedMaskIndexedMemoryMovementIndexEEW(
          static_cast<std::int64_t>(body.getIndexEew())))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only index_eew 32");
  if (!isPreRealizedComputedMaskIndexedMemoryMovementOffsetUnit(
          body.getOffsetUnit()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only offset_unit 'element'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "currently supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() !=
      "preserve-passthrough-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "requires inactive_lane_policy "
        "'preserve-passthrough-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "requires SEW32 LMUL m1 data/mask/index config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed gather-load body "
        "requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask indexed gather-load compare lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask indexed gather-load compare rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask indexed gather-load source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> index =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getIndex(),
          "pre-realized RVV computed-mask indexed gather-load index operand",
          support::RuntimeABIParameterRole::IndexInputBuffer);
  if (!index)
    return index.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV computed-mask indexed gather-load destination "
          "operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask indexed gather-load runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskMemoryBody<
              tcrv::rvv::TypedComputedMaskIndexedGatherPreRealizedBodyOp>(
              variant, "computed-mask indexed gather-load"))
    return error;
  return requireComputedMaskMemorySelectedVariantRequires(
      variant, "computed-mask indexed gather-load");
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskIndexedScatterBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskIndexedScatterPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask indexed scatter-store realization "
        "requires a pre-realized computed-mask indexed scatter body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskIndexedScatterOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only op_kind "
        "'computed_masked_indexed_scatter_store_unit_load'");
  if (!isPreRealizedComputedMaskMemoryMovementPredicateKind(
          body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskIndexedScatterMemoryForm(body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only memory_form "
        "'computed-mask-unit-load-indexed-scatter-store'");
  if (!isPreRealizedComputedMaskIndexedMemoryMovementIndexEEW(
          static_cast<std::int64_t>(body.getIndexEew())))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only index_eew 32");
  if (!isPreRealizedComputedMaskIndexedMemoryMovementOffsetUnit(
          body.getOffsetUnit()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only offset_unit 'element'");
  if (!isPreRealizedComputedMaskIndexedScatterIndexUniqueness(
          body.getIndexUniqueness()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "requires index_uniqueness 'unique' because duplicate-index masked "
        "scatter policy is unsupported");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "currently supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "requires inactive_lane_policy 'preserve-output-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "requires SEW32 LMUL m1 data/mask/index config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask indexed scatter-store body "
        "requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask indexed scatter-store compare lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask indexed scatter-store compare rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask indexed scatter-store source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> index =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getIndex(),
          "pre-realized RVV computed-mask indexed scatter-store index operand",
          support::RuntimeABIParameterRole::IndexInputBuffer);
  if (!index)
    return index.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV computed-mask indexed scatter-store destination "
          "operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask indexed scatter-store runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskMemoryBody<
              tcrv::rvv::TypedComputedMaskIndexedScatterPreRealizedBodyOp>(
              variant, "computed-mask indexed scatter-store"))
    return error;
  return requireComputedMaskMemorySelectedVariantRequires(
      variant, "computed-mask indexed scatter-store");
}

bool isRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return true;
  default:
    return false;
  }
}

llvm::Error
verifyRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  if (isComputedMaskSegment2MemoryRouteOperation(analysis.description.operation))
    return llvm::Error::success();
  return verifyRVVSelectedBodyComputedMaskMemoryRouteFamilyProviderPlans(
      analysis, context);
}

llvm::Error verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyComputedMaskMemoryRouteStatementPlan
        &computedMaskMemoryStatementPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyOperationKind operation = analysis.description.operation;
  if (!isRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyConsumer(
          operation)) {
    if (computedMaskMemoryStatementPlan.plansComputedMaskMemoryRoute ||
        computedMaskMemoryStatementPlan.computedMaskMemoryPlan ||
        !computedMaskMemoryStatementPlan.preLoopSteps.empty() ||
        !computedMaskMemoryStatementPlan.loop.bodySteps.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " non computed-mask memory route construction must not carry a "
          "computed-mask memory statement plan before creating "
          "TCRVEmitCLowerableRoute");
    return llvm::Error::success();
  }

  switch (operation) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return verifyRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteProviderFacts(
        analysis, materializationFacts, memoryOperandBindingFacts,
        computedMaskMemoryStatementPlan, context);
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return verifyRVVSelectedBodyRegularComputedMaskMemoryRouteProviderFacts(
        analysis, materializationFacts, memoryOperandBindingFacts,
        computedMaskMemoryStatementPlan, context);
  default:
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask memory route construction reached an unsupported "
        "non-segment computed-mask memory consumer before creating "
        "TCRVEmitCLowerableRoute");
  }
}

llvm::ArrayRef<RVVSelectedBodyMemoryRouteFamilyOwner>
getRVVSelectedBodyComputedMaskMemoryRouteFamilyOwners() {
  static const RVVSelectedBodyMemoryRouteFamilyOwner owners[] = {
      {"computed-mask memory",
       isRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyConsumer,
       verifyRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyProviderPlans},
  };
  return owners;
}

} // namespace tianchenrv::plugin::rvv
