#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"

#include "TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

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
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
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

bool isPreRealizedRuntimeScalarComputedMaskIndexedGatherOpKind(
    llvm::StringRef opKind) {
  return opKind ==
         "runtime_scalar_cmp_masked_indexed_gather_load_unit_store";
}

bool isPreRealizedComputedMaskIndexedScatterOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_indexed_scatter_store_unit_load";
}

bool isPreRealizedRuntimeScalarComputedMaskIndexedScatterOpKind(
    llvm::StringRef opKind) {
  return opKind ==
         "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load";
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

bool isPreRealizedRuntimeScalarComputedMaskIndexedGatherPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isPreRealizedRuntimeScalarComputedMaskIndexedScatterPredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
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

constexpr llvm::StringLiteral kRVVMaskTailPolicyRouteFamilyPlanID(
    "rvv-mask-tail-policy-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskTailPolicyOwner(
    "computed-mask memory mask/tail policy");

bool isRVVSelectedBodyComputedMaskMemoryLoadMergeRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
         op ==
             RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedGatherLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
}

bool isRVVSelectedBodyComputedMaskMemoryStoreOnlyRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskSegment2StoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::
                   RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedScatterStoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
}

RVVSelectedBodyMemoryForm getComputedMaskMemoryRouteFamilyMemoryForm(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
  default:
    llvm_unreachable("unsupported computed-mask memory route-family op");
  }
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

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskIndexedGatherBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOp
        body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV runtime-scalar computed-mask indexed gather-load "
        "realization requires a pre-realized runtime-scalar computed-mask "
        "indexed gather body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body must be a direct child of the selected "
        "tcrv.exec.variant");

  if (!isPreRealizedRuntimeScalarComputedMaskIndexedGatherOpKind(
          body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body currently supports only op_kind "
        "'runtime_scalar_cmp_masked_indexed_gather_load_unit_store'");
  if (!isPreRealizedRuntimeScalarComputedMaskIndexedGatherPredicateKind(
          body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body currently supports only predicate_kind 'sle'");
  if (!isPreRealizedComputedMaskIndexedGatherMemoryForm(body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body currently supports only memory_form "
        "'computed-mask-indexed-gather-load-unit-store'");
  if (!isPreRealizedComputedMaskIndexedMemoryMovementIndexEEW(
          static_cast<std::int64_t>(body.getIndexEew())))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body currently supports only index_eew 32");
  if (!isPreRealizedComputedMaskIndexedMemoryMovementOffsetUnit(
          body.getOffsetUnit()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body currently supports only offset_unit 'element'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body currently supports only mask_memory_form "
        "'compare-produced-mask'");
  if (body.getInactiveLanePolicy() !=
      "preserve-passthrough-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body requires inactive_lane_policy "
        "'preserve-passthrough-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body requires SEW32 LMUL m1 data/mask/index config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "gather-load body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "gather-load lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "gather-load rhs scalar operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "gather-load source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> index =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getIndex(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "gather-load index operand",
          support::RuntimeABIParameterRole::IndexInputBuffer);
  if (!index)
    return index.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "gather-load destination/passthrough operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "gather-load runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  llvm::StringRef expectedElementCType =
      getRuntimeScalarComputedMaskElementCType(
          static_cast<std::int64_t>(body.getSew()));
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedElementCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedElementCType) + " *").str();
  if ((*lhs).getCType() != expectedConstPointer ||
      (*rhsScalar).getCType() != expectedElementCType ||
      (*source).getCType() != expectedConstPointer ||
      (*index).getCType() != "const uint32_t *" ||
      (*destination).getCType() != expectedMutablePointer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected runtime-scalar computed-mask "
                    "indexed gather-load body requires lhs/source ") +
        expectedConstPointer + ", rhs scalar " + expectedElementCType +
        ", index const uint32_t *, and destination " +
        expectedMutablePointer + " runtime ABI bindings");

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskMemoryBody<
              tcrv::rvv::
                  TypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOp>(
              variant, "runtime-scalar computed-mask indexed gather-load"))
    return error;
  return requireComputedMaskMemorySelectedVariantRequires(
      variant, "runtime-scalar computed-mask indexed gather-load");
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

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskIndexedScatterBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOp
        body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV runtime-scalar computed-mask indexed scatter-store "
        "realization requires a pre-realized runtime-scalar computed-mask "
        "indexed scatter body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body must be a direct child of the selected "
        "tcrv.exec.variant");

  if (!isPreRealizedRuntimeScalarComputedMaskIndexedScatterOpKind(
          body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body currently supports only op_kind "
        "'runtime_scalar_cmp_masked_indexed_scatter_store_unit_load'");
  if (!isPreRealizedRuntimeScalarComputedMaskIndexedScatterPredicateKind(
          body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body currently supports only predicate_kind 'sle'");
  if (!isPreRealizedComputedMaskIndexedScatterMemoryForm(body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body currently supports only memory_form "
        "'computed-mask-unit-load-indexed-scatter-store'");
  if (!isPreRealizedComputedMaskIndexedMemoryMovementIndexEEW(
          static_cast<std::int64_t>(body.getIndexEew())))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body currently supports only index_eew 32");
  if (!isPreRealizedComputedMaskIndexedMemoryMovementOffsetUnit(
          body.getOffsetUnit()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body currently supports only offset_unit 'element'");
  if (!isPreRealizedComputedMaskIndexedScatterIndexUniqueness(
          body.getIndexUniqueness()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body requires index_uniqueness 'unique' because "
        "duplicate-index masked scatter policy is unsupported");
  if (!isPreRealizedComputedMaskMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskSource(
          body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body currently supports only mask_memory_form "
        "'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body requires inactive_lane_policy "
        "'preserve-output-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body requires SEW32 LMUL m1 data/mask/index config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask indexed "
        "scatter-store body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "scatter-store lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "scatter-store rhs scalar operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getSource(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "scatter-store source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> index =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getIndex(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "scatter-store index operand",
          support::RuntimeABIParameterRole::IndexInputBuffer);
  if (!index)
    return index.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "scatter-store destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskMemoryRuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime-scalar computed-mask indexed "
          "scatter-store runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  llvm::StringRef expectedElementCType =
      getRuntimeScalarComputedMaskElementCType(
          static_cast<std::int64_t>(body.getSew()));
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedElementCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedElementCType) + " *").str();
  if ((*lhs).getCType() != expectedConstPointer ||
      (*rhsScalar).getCType() != expectedElementCType ||
      (*source).getCType() != expectedConstPointer ||
      (*index).getCType() != "const uint32_t *" ||
      (*destination).getCType() != expectedMutablePointer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected runtime-scalar computed-mask "
                    "indexed scatter-store body requires lhs/source ") +
        expectedConstPointer + ", rhs scalar " + expectedElementCType +
        ", index const uint32_t *, and destination " +
        expectedMutablePointer + " runtime ABI bindings");

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskMemoryBody<
              tcrv::rvv::
                  TypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOp>(
              variant, "runtime-scalar computed-mask indexed scatter-store"))
    return error;
  return requireComputedMaskMemorySelectedVariantRequires(
      variant, "runtime-scalar computed-mask indexed scatter-store");
}

bool isRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
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

static llvm::Error
verifyRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyComputedMaskMemoryRouteStatementPlan
        &computedMaskMemoryStatementPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  const bool isRuntimeScalarStore =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
  const bool isRuntimeScalarLoadStore =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
  const bool isRuntimeScalarIndexedGather =
      description.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
  const bool isRuntimeScalarIndexedScatter =
      description.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
  const bool isRuntimeScalarIndexedGatherMAccScatter =
      description.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  if (!isRuntimeScalarStore && !isRuntimeScalarLoadStore &&
      !isRuntimeScalarIndexedGather && !isRuntimeScalarIndexedScatter &&
      !isRuntimeScalarIndexedGatherMAccScatter)
    return llvm::Error::success();

  const bool isRuntimeScalarLoadMerge =
      isRuntimeScalarLoadStore || isRuntimeScalarIndexedGather;
  const bool isRuntimeScalarStoreOnly =
      isRuntimeScalarStore || isRuntimeScalarIndexedScatter;
  const bool expectsIndexedGather =
      isRuntimeScalarIndexedGather || isRuntimeScalarIndexedGatherMAccScatter;
  const bool expectsIndexedScatter =
      isRuntimeScalarIndexedScatter || isRuntimeScalarIndexedGatherMAccScatter;
  const RVVSelectedBodyMemoryForm expectedMemoryForm =
      isRuntimeScalarIndexedGatherMAccScatter
          ? RVVSelectedBodyMemoryForm::
                RuntimeScalarComputedMaskIndexedGatherMAccScatter
      : isRuntimeScalarIndexedGather
          ? RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore
      : isRuntimeScalarIndexedScatter
          ? RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore
          : (isRuntimeScalarLoadStore
                 ? RVVSelectedBodyMemoryForm::
                       RuntimeScalarComputedMaskLoadStore
                 : RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore);

  std::optional<RVVComputedMaskIndexedMemoryRouteFacts> indexedRouteFacts;
  if (expectsIndexedGather || expectsIndexedScatter) {
    indexedRouteFacts =
        getRVVComputedMaskIndexedMemoryRouteFacts(description.operation);
    if (!indexedRouteFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " runtime scalar computed-mask indexed route construction "
          "requires canonical provider-owned indexed route facts before "
          "creating TCRVEmitCLowerableRoute");
  }

  if (!analysis.computedMaskMemoryRouteFamilyPlan ||
      materializationFacts.computedMaskMemoryPlan !=
          &*analysis.computedMaskMemoryRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires "
        "exactly the verified computed-mask memory family plan before "
        "creating TCRVEmitCLowerableRoute");
  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan &plan =
      *materializationFacts.computedMaskMemoryPlan;

  if (plan.operation != description.operation ||
      plan.memoryForm != description.memoryForm ||
      plan.memoryForm != expectedMemoryForm || !plan.usesRuntimeScalarProducer ||
      plan.usesVectorCompareProducer ||
      plan.usesStoreOnly != isRuntimeScalarStoreOnly ||
      plan.usesLoadMerge != isRuntimeScalarLoadMerge ||
      plan.usesIndexedGather != expectsIndexedGather ||
      plan.usesIndexedScatter != expectsIndexedScatter ||
      plan.usesSegment2Load || plan.usesSegment2Store ||
      plan.usesSegment2Update)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires "
        "runtime-scalar producer facts and the matching store/load-store/"
        "indexed-gather/indexed-scatter "
        "memory form before creating TCRVEmitCLowerableRoute");

  if (indexedRouteFacts &&
      (plan.memoryForm != indexedRouteFacts->memoryForm ||
       plan.sew != indexedRouteFacts->sew ||
       plan.lmul != indexedRouteFacts->lmul ||
       plan.runtimeControlPlan.controlPlanID !=
           indexedRouteFacts->runtimeControlPlanID ||
       plan.runtimeABIOrder != indexedRouteFacts->runtimeABIOrder ||
       plan.targetLeafProfile != indexedRouteFacts->targetLeafProfile ||
       plan.providerSupportedMirror !=
           indexedRouteFacts->providerSupportedMirror ||
       plan.requiredHeaderDeclarations !=
           indexedRouteFacts->requiredHeaderDeclarations ||
       plan.cTypeMappingSummary != indexedRouteFacts->cTypeMappingSummary ||
       plan.familyPlanID !=
           indexedRouteFacts->computedMaskMemoryRouteFamilyPlanID ||
       plan.maskProducerSource !=
           indexedRouteFacts->computedMaskMemoryMaskProducerSource ||
       plan.vlCType != indexedRouteFacts->vlCType ||
       plan.vectorTypeName != indexedRouteFacts->vectorTypeName ||
       plan.vectorCType != indexedRouteFacts->vectorCType ||
       plan.indexVectorTypeName != indexedRouteFacts->indexVectorTypeName ||
       plan.indexVectorCType != indexedRouteFacts->indexVectorCType ||
       plan.maskTypeName != indexedRouteFacts->maskTypeName ||
       plan.maskCType != indexedRouteFacts->maskCType ||
       plan.setVLIntrinsic != indexedRouteFacts->setVLIntrinsic ||
       plan.vectorLoadIntrinsic != indexedRouteFacts->vectorLoadIntrinsic ||
       plan.indexLoadIntrinsic != indexedRouteFacts->indexLoadIntrinsic ||
       plan.indexScaleIntrinsic != indexedRouteFacts->indexScaleIntrinsic ||
       plan.rhsScalarSplatIntrinsic !=
           indexedRouteFacts->rhsScalarSplatIntrinsic ||
       plan.maskedLoadIntrinsic !=
           indexedRouteFacts->maskedIndexedLoadIntrinsic ||
       plan.indexedStoreIntrinsic !=
           indexedRouteFacts->maskedIndexedStoreIntrinsic ||
       plan.maskedStoreIntrinsic != indexedRouteFacts->maskedStoreIntrinsic ||
       plan.compareIntrinsic != indexedRouteFacts->compareIntrinsic ||
       plan.maskRole != indexedRouteFacts->maskRole ||
       plan.maskSource != indexedRouteFacts->maskSource ||
       plan.maskMemoryForm != indexedRouteFacts->maskMemoryForm ||
       plan.inactiveLaneContract !=
           indexedRouteFacts->inactiveLaneContract ||
       plan.maskedPassthroughLayout !=
           indexedRouteFacts->maskedPassthroughLayout ||
       plan.maskedMemoryLayout != indexedRouteFacts->indexedMemoryLayout ||
       plan.sourceMemoryForm != indexedRouteFacts->sourceMemoryForm ||
       plan.destinationMemoryForm !=
           indexedRouteFacts->destinationMemoryForm ||
       plan.indexEEW != indexedRouteFacts->indexEEW ||
       plan.offsetUnit != indexedRouteFacts->offsetUnit ||
       plan.indexSource != indexedRouteFacts->indexSource ||
       plan.indexUniqueness != indexedRouteFacts->indexUniqueness ||
       plan.indexedDataMemoryForm !=
           indexedRouteFacts->indexedDataMemoryForm ||
       plan.indexedDestinationMemoryForm !=
           indexedRouteFacts->indexedDestinationMemoryForm ||
       !support::runtimeABIParametersEqual(
           plan.runtimeABIParameters,
           indexedRouteFacts->runtimeABIParameters)))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask indexed route construction "
        "requires the computed-mask memory family plan to mirror canonical "
        "provider-owned indexed route facts before creating "
        "TCRVEmitCLowerableRoute");

  if (description.computedMaskMemoryRouteFamilyPlanID != plan.familyPlanID ||
      description.computedMaskMemoryMaskProducerSource !=
          plan.maskProducerSource ||
      description.maskTailPolicyRouteFamilyPlanID !=
          kRVVMaskTailPolicyRouteFamilyPlanID ||
      description.maskTailPolicyOwner !=
          kRVVComputedMaskMemoryMaskTailPolicyOwner ||
      description.maskRole != plan.maskRole ||
      description.maskSource != plan.maskSource ||
      description.maskMemoryForm != plan.maskMemoryForm ||
      description.inactiveLaneContract != plan.inactiveLaneContract ||
      description.maskedPassthroughLayout != plan.maskedPassthroughLayout ||
      description.providerSupportedMirror != plan.providerSupportedMirror ||
      description.runtimeABIOrder != plan.runtimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires "
        "provider-built family, mask/tail, inactive-lane, provider support, "
        "and ABI mirror facts before creating TCRVEmitCLowerableRoute");

  if ((expectsIndexedGather || expectsIndexedScatter) &&
      (description.sourceMemoryForm != plan.sourceMemoryForm ||
       description.destinationMemoryForm != plan.destinationMemoryForm ||
       description.indexEEW != plan.indexEEW ||
       description.offsetUnit != plan.offsetUnit ||
       description.indexSource != plan.indexSource ||
       description.indexUniqueness != plan.indexUniqueness ||
       description.indexedDataMemoryForm != plan.indexedDataMemoryForm ||
       description.indexedDestinationMemoryForm !=
           plan.indexedDestinationMemoryForm))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask indexed route construction "
        "requires provider-built source/index/destination memory-form mirror "
        "facts before creating TCRVEmitCLowerableRoute");

  if (indexedRouteFacts &&
      (description.routeOperandBindingPlanID !=
           indexedRouteFacts->routeOperandBindingPlanID ||
       description.routeOperandBindingSummary !=
           indexedRouteFacts->routeOperandBindingSummary ||
       analysis.routeOperandBindingPlan.planID !=
           indexedRouteFacts->routeOperandBindingPlanID))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask indexed route construction "
        "requires route operand binding plan and summary to mirror canonical "
        "provider-owned indexed route facts before creating "
        "TCRVEmitCLowerableRoute");

  if (!support::runtimeABIParametersEqual(description.runtimeABIParameters,
                                          plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires "
        "runtime ABI parameters from the verified family plan before creating "
        "TCRVEmitCLowerableRoute");

  if (llvm::Error error =
          verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context))
    return error;

  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      materializationFacts.typedConfigFacts;
  if (!typedFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires "
        "typed RVV body/config facts before creating "
        "TCRVEmitCLowerableRoute");
  if (typedFacts.sew != plan.sew || typedFacts.lmul != plan.lmul ||
      typedFacts.vlCType != plan.vlCType ||
      typedFacts.vectorTypeName != plan.vectorTypeName ||
      typedFacts.vectorCType != plan.vectorCType ||
      typedFacts.maskTypeName != plan.maskTypeName ||
      typedFacts.maskCType != plan.maskCType ||
      typedFacts.setVLIntrinsic != plan.setVLIntrinsic ||
      typedFacts.vectorLoadIntrinsic != plan.vectorLoadIntrinsic ||
      ((expectsIndexedGather || expectsIndexedScatter) &&
       (typedFacts.indexVectorTypeName != plan.indexVectorTypeName ||
        typedFacts.indexVectorCType != plan.indexVectorCType)))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires "
        "family-plan type/config facts to mirror the selected typed RVV body "
        "before creating TCRVEmitCLowerableRoute");

  if (materializationFacts.vlCType != plan.vlCType ||
      materializationFacts.resultVectorTypeName != plan.vectorTypeName ||
      materializationFacts.resultVectorCType != plan.vectorCType ||
      materializationFacts.maskTypeName != plan.maskTypeName ||
      materializationFacts.maskCType != plan.maskCType ||
      materializationFacts.setVLLeaf != plan.setVLIntrinsic ||
      materializationFacts.vectorLoadLeaf != plan.vectorLoadIntrinsic ||
      materializationFacts.rhsScalarBroadcastLeaf !=
          plan.rhsScalarSplatIntrinsic ||
      materializationFacts.compareLeaf != plan.compareIntrinsic ||
      materializationFacts.storeLeaf != plan.maskedStoreIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires "
        "materialization facts to come from the verified computed-mask memory "
        "family plan before creating TCRVEmitCLowerableRoute");

  auto requireABI = [&](const support::RuntimeABIParameter *parameter,
                        llvm::StringRef logicalName,
                        support::RuntimeABIParameterRole expectedRole)
      -> llvm::Error {
    if (!parameter)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " runtime scalar computed-mask memory route construction requires "
          "lhs/rhs_scalar/src/dst/n plus indexed route operand-binding facts "
          "before creating "
          "TCRVEmitCLowerableRoute");
    if (parameter->role != expectedRole)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " runtime scalar computed-mask memory route construction requires "
          "ABI role for " +
          logicalName + " to be '" +
          support::stringifyRuntimeABIParameterRole(expectedRole) +
          "' before creating TCRVEmitCLowerableRoute, but saw '" +
          support::stringifyRuntimeABIParameterRole(parameter->role) + "'");
    return llvm::Error::success();
  };

  if (memoryOperandBindingFacts.bindingPlan !=
          &analysis.routeOperandBindingPlan ||
      !memoryOperandBindingFacts.bindsMemoryCluster ||
      !memoryOperandBindingFacts.bindsComputedMaskMemory ||
      !memoryOperandBindingFacts.bindsRuntimeScalarComputedMaskMemory ||
      memoryOperandBindingFacts.bindsSegment2Memory)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires "
        "memory operand-binding facts from the same selected route analysis "
        "before creating TCRVEmitCLowerableRoute");

  if (llvm::Error error = requireABI(
          memoryOperandBindingFacts.compareLhsABI,
          isRuntimeScalarIndexedGatherMAccScatter ? "cmp_lhs" : "lhs",
          support::RuntimeABIParameterRole::LHSInputBuffer))
    return error;
  if (llvm::Error error = requireABI(
          memoryOperandBindingFacts.rhsScalarABI, "rhs_scalar",
          support::RuntimeABIParameterRole::RHSScalarValue))
    return error;
  if (llvm::Error error = requireABI(
          memoryOperandBindingFacts.sourceABI,
          isRuntimeScalarIndexedGatherMAccScatter ? "gather_src" : "src",
          support::RuntimeABIParameterRole::SourceInputBuffer))
    return error;
  if (isRuntimeScalarIndexedGatherMAccScatter) {
    if (llvm::Error error = requireABI(
            memoryOperandBindingFacts.dotRHSABI, "payload",
            support::RuntimeABIParameterRole::DotRHSInputBuffer))
      return error;
    if (llvm::Error error = requireABI(
            memoryOperandBindingFacts.accumulatorABI, "acc",
            support::RuntimeABIParameterRole::AccumulatorInputBuffer))
      return error;
  }
  if (llvm::Error error = requireABI(
          memoryOperandBindingFacts.destinationABI, "dst",
          support::RuntimeABIParameterRole::OutputBuffer))
    return error;
  if (llvm::Error error = requireABI(
          memoryOperandBindingFacts.runtimeElementCountABI, "n",
          support::RuntimeABIParameterRole::RuntimeElementCount))
    return error;
  if (expectsIndexedGather || expectsIndexedScatter)
    if (llvm::Error error = requireABI(
            memoryOperandBindingFacts.indexABI, "index",
            support::RuntimeABIParameterRole::IndexInputBuffer))
      return error;

  if (isRuntimeScalarLoadMerge || isRuntimeScalarIndexedGatherMAccScatter) {
    if (!memoryOperandBindingFacts.passthroughABI)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " runtime scalar computed-mask load-merge/composite route "
          "construction requires old-destination passthrough operand-binding "
          "facts before creating TCRVEmitCLowerableRoute");
    if (memoryOperandBindingFacts.passthroughABI->role !=
            support::RuntimeABIParameterRole::OutputBuffer ||
        memoryOperandBindingFacts.passthroughABI->cName !=
            memoryOperandBindingFacts.destinationABI->cName)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " runtime scalar computed-mask load-merge/composite route "
          "construction requires old-destination passthrough and destination "
          "store ABI facts to refer to the same output buffer before creating "
          "TCRVEmitCLowerableRoute");
  }

  if (!computedMaskMemoryStatementPlan.plansComputedMaskMemoryRoute ||
      computedMaskMemoryStatementPlan.computedMaskMemoryPlan != &plan ||
      computedMaskMemoryStatementPlan.preLoopSteps.empty() ||
      computedMaskMemoryStatementPlan.loop.bodySteps.empty() ||
      computedMaskMemoryStatementPlan.plansRuntimeScalarComputedMaskStore !=
          isRuntimeScalarStore ||
      computedMaskMemoryStatementPlan.plansRuntimeScalarComputedMaskLoadStore !=
          isRuntimeScalarLoadStore ||
      computedMaskMemoryStatementPlan
              .plansRuntimeScalarComputedMaskIndexedGatherLoadUnitStore !=
          isRuntimeScalarIndexedGather ||
      computedMaskMemoryStatementPlan
              .plansRuntimeScalarComputedMaskIndexedGatherMAccScatter !=
          isRuntimeScalarIndexedGatherMAccScatter ||
      computedMaskMemoryStatementPlan
              .plansRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad !=
          isRuntimeScalarIndexedScatter ||
      computedMaskMemoryStatementPlan
              .plansComputedMaskIndexedScatterStoreUnitLoad !=
          expectsIndexedScatter ||
      computedMaskMemoryStatementPlan.plansComputedMaskUnitLoadStore ||
      computedMaskMemoryStatementPlan.plansComputedMaskStridedStore ||
      computedMaskMemoryStatementPlan.plansComputedMaskStridedLoadUnitStore ||
      computedMaskMemoryStatementPlan
          .plansComputedMaskIndexedGatherLoadUnitStore)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires the "
        "matching runtime-scalar store/load-store/indexed-scatter statement "
        "plan, or the gather-MAcc-scatter statement plan, before creating "
        "TCRVEmitCLowerableRoute");

  auto statementHasCallee = [&](llvm::StringRef callee) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         computedMaskMemoryStatementPlan.loop.bodySteps)
      if (step.callee == callee)
        return true;
    return false;
  };
  if (!statementHasCallee(plan.rhsScalarSplatIntrinsic) ||
      !statementHasCallee(plan.compareIntrinsic) ||
      (!isRuntimeScalarIndexedScatter &&
       !isRuntimeScalarIndexedGatherMAccScatter &&
       !statementHasCallee(plan.maskedStoreIntrinsic)) ||
      ((isRuntimeScalarLoadMerge || isRuntimeScalarIndexedGatherMAccScatter) &&
       !statementHasCallee(plan.maskedLoadIntrinsic)) ||
      ((expectsIndexedGather || expectsIndexedScatter) &&
       !statementHasCallee(plan.indexLoadIntrinsic)) ||
      ((expectsIndexedGather || expectsIndexedScatter) &&
       !statementHasCallee(plan.indexScaleIntrinsic)) ||
      (expectsIndexedScatter &&
       !statementHasCallee(plan.indexedStoreIntrinsic)) ||
      (isRuntimeScalarIndexedGatherMAccScatter &&
       !statementHasCallee(plan.arithmeticIntrinsic)))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires "
        "statement/leaf facts for scalar splat, compare, masked memory, "
        "indexed route, MAcc, and store calls from the RVV statement-plan "
        "owner before creating TCRVEmitCLowerableRoute");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsComputedMaskMemory ||
      routeControlPlan->runtimeControlPlan != &plan.runtimeControlPlan ||
      routeControlPlan->typedConfigFacts != &analysis.typedConfigFacts ||
      routeControlPlan->selectedTargetCapabilityFacts !=
          &analysis.selectedTargetCapabilityFacts ||
      routeControlPlan->controlPlanIDMirror !=
          plan.runtimeControlPlan.controlPlanID ||
      routeControlPlan->configContractIDMirror !=
          plan.runtimeControlPlan.configContractID ||
      routeControlPlan->runtimeVLContractIDMirror !=
          plan.runtimeControlPlan.runtimeVLContractID ||
      routeControlPlan->runtimeAVLASourceMirror !=
          plan.runtimeControlPlan.runtimeAVLASource ||
      routeControlPlan->runtimeABIOrderMirror != plan.runtimeABIOrder ||
      routeControlPlan->tailPolicyMirror != plan.runtimeControlPlan.tailPolicy ||
      routeControlPlan->maskPolicyMirror != plan.runtimeControlPlan.maskPolicy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires the "
        "RVV-owned route-control provider plan and runtime n/AVL/VL facts "
        "before creating TCRVEmitCLowerableRoute");

  const RVVSelectedBodyMaskTailPolicyProviderPlan &maskTailPlan =
      computedMaskMemoryStatementPlan.maskTailPolicyPlan;
  if (!maskTailPlan.plansMaskTailPolicy ||
      !maskTailPlan.controlsComputedMaskMemory ||
      maskTailPlan.controlsComputedMaskSelect ||
      maskTailPlan.computedMaskMemoryPlan != &plan ||
      maskTailPlan.typedConfigFacts != &analysis.typedConfigFacts ||
      maskTailPlan.selectedTargetCapabilityFacts !=
          &analysis.selectedTargetCapabilityFacts ||
      maskTailPlan.bindingPlan != &analysis.routeOperandBindingPlan ||
      maskTailPlan.familyPlanIDMirror != kRVVMaskTailPolicyRouteFamilyPlanID ||
      maskTailPlan.ownerNameMirror !=
          kRVVComputedMaskMemoryMaskTailPolicyOwner ||
      maskTailPlan.maskProducerSourceMirror != plan.maskProducerSource ||
      maskTailPlan.maskRoleMirror != plan.maskRole ||
      maskTailPlan.maskSourceMirror != plan.maskSource ||
      maskTailPlan.maskMemoryFormMirror != plan.maskMemoryForm ||
      maskTailPlan.tailPolicyMirror != plan.runtimeControlPlan.tailPolicy ||
      maskTailPlan.maskPolicyMirror != plan.runtimeControlPlan.maskPolicy ||
      maskTailPlan.inactiveLaneContractMirror !=
          plan.inactiveLaneContract ||
      maskTailPlan.maskedPassthroughLayoutMirror !=
          plan.maskedPassthroughLayout ||
      maskTailPlan.runtimeABIOrderMirror != plan.runtimeABIOrder ||
      maskTailPlan.routeOperandBindingPlanIDMirror !=
          analysis.routeOperandBindingPlan.planID ||
      maskTailPlan.providerSupportedMirror != plan.providerSupportedMirror ||
      maskTailPlan.selectedProviderMirror !=
          routeControlPlan->selectedProviderMirror ||
      maskTailPlan.selectedLegalityMirror !=
          routeControlPlan->selectedLegalityMirror)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime scalar computed-mask memory route construction requires the "
        "RVV-owned mask/tail policy provider plan from the same computed-mask "
        "memory statement plan before creating TCRVEmitCLowerableRoute");

  return llvm::Error::success();
}

static llvm::Error
verifyRVVSelectedBodyRegularComputedMaskMemoryRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyComputedMaskMemoryRouteStatementPlan
        &computedMaskMemoryStatementPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  const RVVSelectedBodyOperationKind operation = description.operation;
  const bool isUnitLoadStore =
      operation == RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore;
  const bool isStridedStore =
      operation == RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
  const bool isStridedLoad =
      operation ==
      RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore;
  const bool isIndexedGather =
      operation ==
      RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
  const bool isIndexedScatter =
      operation ==
      RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
  const bool isRegularComputedMaskMemory =
      isUnitLoadStore || isStridedStore || isStridedLoad || isIndexedGather ||
      isIndexedScatter;
  if (!isRegularComputedMaskMemory)
    return llvm::Error::success();

  const bool isLoadMerge =
      isRVVSelectedBodyComputedMaskMemoryLoadMergeRoute(operation);
  const bool isStoreOnly =
      isRVVSelectedBodyComputedMaskMemoryStoreOnlyRoute(operation);
  const bool isIndexed = isIndexedGather || isIndexedScatter;
  const bool isStrided = isStridedStore || isStridedLoad;
  std::optional<RVVComputedMaskStridedMemoryRouteFacts> stridedRouteFacts;
  if (isStrided) {
    stridedRouteFacts = getRVVComputedMaskStridedMemoryRouteFacts(operation);
    if (!stridedRouteFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " regular computed-mask strided memory route construction requires "
          "canonical provider-owned strided route facts before creating "
          "TCRVEmitCLowerableRoute");
  }
  std::optional<RVVComputedMaskIndexedMemoryRouteFacts> indexedRouteFacts;
  if (isIndexed) {
    indexedRouteFacts = getRVVComputedMaskIndexedMemoryRouteFacts(operation);
    if (!indexedRouteFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " regular computed-mask indexed memory route construction requires "
          "canonical provider-owned indexed route facts before creating "
          "TCRVEmitCLowerableRoute");
  }

  if (!analysis.computedMaskMemoryRouteFamilyPlan ||
      materializationFacts.computedMaskMemoryPlan !=
          &*analysis.computedMaskMemoryRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires exactly "
        "the verified computed-mask memory family plan before creating "
        "TCRVEmitCLowerableRoute");
  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan &plan =
      *materializationFacts.computedMaskMemoryPlan;

  if (plan.operation != operation || plan.memoryForm != description.memoryForm ||
      plan.memoryForm != getComputedMaskMemoryRouteFamilyMemoryForm(operation) ||
      plan.usesRuntimeScalarProducer || !plan.usesVectorCompareProducer ||
      plan.usesStoreOnly != isStoreOnly ||
      plan.usesLoadMerge != isLoadMerge ||
      plan.usesIndexedGather != isIndexedGather ||
      plan.usesIndexedScatter != isIndexedScatter || plan.usesSegment2Load ||
      plan.usesSegment2Store || plan.usesSegment2Update)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires vector "
        "compare-mask producer facts and the matching regular memory form "
        "before creating TCRVEmitCLowerableRoute");

  if (stridedRouteFacts &&
      (plan.memoryForm != stridedRouteFacts->memoryForm ||
       plan.sew != stridedRouteFacts->sew ||
       plan.lmul != stridedRouteFacts->lmul ||
       plan.runtimeControlPlan.controlPlanID !=
           stridedRouteFacts->runtimeControlPlanID ||
       plan.runtimeABIOrder != stridedRouteFacts->runtimeABIOrder ||
       plan.targetLeafProfile != stridedRouteFacts->targetLeafProfile ||
       plan.providerSupportedMirror !=
           stridedRouteFacts->providerSupportedMirror ||
       plan.requiredHeaderDeclarations !=
           stridedRouteFacts->requiredHeaderDeclarations ||
       plan.cTypeMappingSummary != stridedRouteFacts->cTypeMappingSummary ||
       plan.familyPlanID !=
           stridedRouteFacts->computedMaskMemoryRouteFamilyPlanID ||
       plan.maskProducerSource !=
           stridedRouteFacts->computedMaskMemoryMaskProducerSource ||
       plan.vlCType != stridedRouteFacts->vlCType ||
       plan.vectorTypeName != stridedRouteFacts->vectorTypeName ||
       plan.vectorCType != stridedRouteFacts->vectorCType ||
       plan.maskTypeName != stridedRouteFacts->maskTypeName ||
       plan.maskCType != stridedRouteFacts->maskCType ||
       plan.setVLIntrinsic != stridedRouteFacts->setVLIntrinsic ||
       plan.vectorLoadIntrinsic != stridedRouteFacts->vectorLoadIntrinsic ||
       plan.maskedLoadIntrinsic != stridedRouteFacts->maskedLoadIntrinsic ||
       plan.maskedStoreIntrinsic != stridedRouteFacts->storeIntrinsic ||
       plan.stridedStoreIntrinsic !=
           stridedRouteFacts->stridedStoreIntrinsic ||
       plan.compareIntrinsic != stridedRouteFacts->compareIntrinsic ||
       plan.maskRole != stridedRouteFacts->maskRole ||
       plan.maskSource != stridedRouteFacts->maskSource ||
       plan.maskMemoryForm != stridedRouteFacts->maskMemoryForm ||
       plan.inactiveLaneContract !=
           stridedRouteFacts->inactiveLaneContract ||
       plan.maskedPassthroughLayout !=
           stridedRouteFacts->maskedPassthroughLayout ||
       plan.maskedMemoryLayout != stridedRouteFacts->maskedMemoryLayout ||
       plan.stridedMemoryLayout != stridedRouteFacts->stridedMemoryLayout ||
       plan.sourceMemoryForm != stridedRouteFacts->sourceMemoryForm ||
       plan.destinationMemoryForm !=
           stridedRouteFacts->destinationMemoryForm ||
       plan.sourceStrideSource != stridedRouteFacts->sourceStrideSource ||
       plan.destinationStrideSource !=
           stridedRouteFacts->destinationStrideSource ||
       !support::runtimeABIParametersEqual(
           plan.runtimeABIParameters,
           stridedRouteFacts->runtimeABIParameters)))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask strided memory route construction requires "
        "the computed-mask memory family plan to mirror canonical "
        "provider-owned strided route facts before creating "
        "TCRVEmitCLowerableRoute");

  if (indexedRouteFacts &&
      (plan.memoryForm != indexedRouteFacts->memoryForm ||
       plan.sew != indexedRouteFacts->sew ||
       plan.lmul != indexedRouteFacts->lmul ||
       plan.runtimeControlPlan.controlPlanID !=
           indexedRouteFacts->runtimeControlPlanID ||
       plan.runtimeABIOrder != indexedRouteFacts->runtimeABIOrder ||
       plan.targetLeafProfile != indexedRouteFacts->targetLeafProfile ||
       plan.providerSupportedMirror !=
           indexedRouteFacts->providerSupportedMirror ||
       plan.requiredHeaderDeclarations !=
           indexedRouteFacts->requiredHeaderDeclarations ||
       plan.cTypeMappingSummary != indexedRouteFacts->cTypeMappingSummary ||
       plan.familyPlanID !=
           indexedRouteFacts->computedMaskMemoryRouteFamilyPlanID ||
       plan.maskProducerSource !=
           indexedRouteFacts->computedMaskMemoryMaskProducerSource ||
       plan.vlCType != indexedRouteFacts->vlCType ||
       plan.vectorTypeName != indexedRouteFacts->vectorTypeName ||
       plan.vectorCType != indexedRouteFacts->vectorCType ||
       plan.indexVectorTypeName != indexedRouteFacts->indexVectorTypeName ||
       plan.indexVectorCType != indexedRouteFacts->indexVectorCType ||
       plan.maskTypeName != indexedRouteFacts->maskTypeName ||
       plan.maskCType != indexedRouteFacts->maskCType ||
       plan.setVLIntrinsic != indexedRouteFacts->setVLIntrinsic ||
       plan.vectorLoadIntrinsic != indexedRouteFacts->vectorLoadIntrinsic ||
       plan.indexLoadIntrinsic != indexedRouteFacts->indexLoadIntrinsic ||
       plan.indexScaleIntrinsic != indexedRouteFacts->indexScaleIntrinsic ||
       plan.maskedLoadIntrinsic !=
           indexedRouteFacts->maskedIndexedLoadIntrinsic ||
       plan.indexedStoreIntrinsic !=
           indexedRouteFacts->maskedIndexedStoreIntrinsic ||
       plan.maskedStoreIntrinsic != indexedRouteFacts->maskedStoreIntrinsic ||
       plan.compareIntrinsic != indexedRouteFacts->compareIntrinsic ||
       plan.maskRole != indexedRouteFacts->maskRole ||
       plan.maskSource != indexedRouteFacts->maskSource ||
       plan.maskMemoryForm != indexedRouteFacts->maskMemoryForm ||
       plan.inactiveLaneContract !=
           indexedRouteFacts->inactiveLaneContract ||
       plan.maskedPassthroughLayout !=
           indexedRouteFacts->maskedPassthroughLayout ||
       plan.maskedMemoryLayout != indexedRouteFacts->indexedMemoryLayout ||
       plan.sourceMemoryForm != indexedRouteFacts->sourceMemoryForm ||
       plan.destinationMemoryForm !=
           indexedRouteFacts->destinationMemoryForm ||
       plan.indexEEW != indexedRouteFacts->indexEEW ||
       plan.offsetUnit != indexedRouteFacts->offsetUnit ||
       plan.indexSource != indexedRouteFacts->indexSource ||
       plan.indexUniqueness != indexedRouteFacts->indexUniqueness ||
       plan.indexedDataMemoryForm !=
           indexedRouteFacts->indexedDataMemoryForm ||
       plan.indexedDestinationMemoryForm !=
           indexedRouteFacts->indexedDestinationMemoryForm))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask indexed memory route construction requires "
        "the computed-mask memory family plan to mirror canonical "
        "provider-owned indexed route facts before creating "
        "TCRVEmitCLowerableRoute");

  if (description.computedMaskMemoryRouteFamilyPlanID != plan.familyPlanID ||
      description.computedMaskMemoryMaskProducerSource !=
          plan.maskProducerSource ||
      description.maskTailPolicyRouteFamilyPlanID !=
          kRVVMaskTailPolicyRouteFamilyPlanID ||
      description.maskTailPolicyOwner !=
          kRVVComputedMaskMemoryMaskTailPolicyOwner ||
      description.maskRole != plan.maskRole ||
      description.maskSource != plan.maskSource ||
      description.maskMemoryForm != plan.maskMemoryForm ||
      description.inactiveLaneContract != plan.inactiveLaneContract ||
      description.maskedPassthroughLayout != plan.maskedPassthroughLayout ||
      description.providerSupportedMirror != plan.providerSupportedMirror ||
      description.runtimeABIOrder != plan.runtimeABIOrder ||
      description.sourceMemoryForm != plan.sourceMemoryForm ||
      description.destinationMemoryForm != plan.destinationMemoryForm ||
      description.sourceStrideSource != plan.sourceStrideSource ||
      description.outStrideSource != plan.destinationStrideSource ||
      description.indexEEW != plan.indexEEW ||
      description.offsetUnit != plan.offsetUnit ||
      description.indexSource != plan.indexSource ||
      description.indexUniqueness != plan.indexUniqueness ||
      description.indexedDataMemoryForm != plan.indexedDataMemoryForm ||
      description.indexedDestinationMemoryForm !=
          plan.indexedDestinationMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires "
        "provider-built family, memory-form, stride/index, mask/tail, "
        "inactive-lane, provider support, and ABI mirror facts before "
        "creating TCRVEmitCLowerableRoute");

  if (!support::runtimeABIParametersEqual(description.runtimeABIParameters,
                                          plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires runtime "
        "ABI parameters from the verified family plan before creating "
        "TCRVEmitCLowerableRoute");
  if (stridedRouteFacts &&
      (description.routeOperandBindingPlanID !=
           stridedRouteFacts->routeOperandBindingPlanID ||
       description.routeOperandBindingSummary !=
           stridedRouteFacts->routeOperandBindingSummary ||
       analysis.routeOperandBindingPlan.planID !=
           stridedRouteFacts->routeOperandBindingPlanID))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask strided memory route construction requires "
        "route operand binding plan and summary to mirror canonical "
        "provider-owned strided route facts before creating "
        "TCRVEmitCLowerableRoute");

  if (llvm::Error error =
          verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context))
    return error;

  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      materializationFacts.typedConfigFacts;
  if (!typedFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires typed RVV "
        "body/config facts before creating TCRVEmitCLowerableRoute");
  if (typedFacts.sew != plan.sew || typedFacts.lmul != plan.lmul ||
      typedFacts.vlCType != plan.vlCType ||
      typedFacts.vectorTypeName != plan.vectorTypeName ||
      typedFacts.vectorCType != plan.vectorCType ||
      typedFacts.maskTypeName != plan.maskTypeName ||
      typedFacts.maskCType != plan.maskCType ||
      typedFacts.setVLIntrinsic != plan.setVLIntrinsic ||
      typedFacts.vectorLoadIntrinsic != plan.vectorLoadIntrinsic ||
      (isIndexed &&
       (typedFacts.indexVectorTypeName != plan.indexVectorTypeName ||
        typedFacts.indexVectorCType != plan.indexVectorCType)))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires "
        "family-plan type/config facts to mirror the selected typed RVV body "
        "before creating TCRVEmitCLowerableRoute");

  if (materializationFacts.vlCType != plan.vlCType ||
      materializationFacts.resultVectorTypeName != plan.vectorTypeName ||
      materializationFacts.resultVectorCType != plan.vectorCType ||
      materializationFacts.maskTypeName != plan.maskTypeName ||
      materializationFacts.maskCType != plan.maskCType ||
      materializationFacts.setVLLeaf != plan.setVLIntrinsic ||
      materializationFacts.vectorLoadLeaf != plan.vectorLoadIntrinsic ||
      materializationFacts.compareLeaf != plan.compareIntrinsic ||
      materializationFacts.storeLeaf != plan.maskedStoreIntrinsic ||
      materializationFacts.stridedStoreLeaf != plan.stridedStoreIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires "
        "materialization facts to come from the verified computed-mask memory "
        "family plan before creating TCRVEmitCLowerableRoute");

  auto requireABI = [&](const support::RuntimeABIParameter *parameter,
                        llvm::StringRef logicalName,
                        support::RuntimeABIParameterRole expectedRole)
      -> llvm::Error {
    if (!parameter)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " regular computed-mask memory route construction requires "
          "cmp_lhs/cmp_rhs/src/dst/n plus stride/index operand-binding facts "
          "before creating TCRVEmitCLowerableRoute");
    if (parameter->role != expectedRole)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " regular computed-mask memory route construction requires ABI role "
          "for " +
          logicalName + " to be '" +
          support::stringifyRuntimeABIParameterRole(expectedRole) +
          "' before creating TCRVEmitCLowerableRoute, but saw '" +
          support::stringifyRuntimeABIParameterRole(parameter->role) + "'");
    return llvm::Error::success();
  };

  if (memoryOperandBindingFacts.bindingPlan !=
          &analysis.routeOperandBindingPlan ||
      !memoryOperandBindingFacts.bindsMemoryCluster ||
      !memoryOperandBindingFacts.bindsComputedMaskMemory ||
      memoryOperandBindingFacts.bindsRuntimeScalarComputedMaskMemory ||
      memoryOperandBindingFacts.bindsSegment2Memory)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires memory "
        "operand-binding facts from the same selected route analysis before "
        "creating TCRVEmitCLowerableRoute");

  if (llvm::Error error = requireABI(
          memoryOperandBindingFacts.compareLhsABI, "cmp_lhs",
          support::RuntimeABIParameterRole::LHSInputBuffer))
    return error;
  if (llvm::Error error = requireABI(
          memoryOperandBindingFacts.compareRhsABI, "cmp_rhs",
          support::RuntimeABIParameterRole::RHSInputBuffer))
    return error;
  if (llvm::Error error = requireABI(
          memoryOperandBindingFacts.sourceABI, "src",
          support::RuntimeABIParameterRole::SourceInputBuffer))
    return error;
  if (llvm::Error error = requireABI(
          memoryOperandBindingFacts.destinationABI, "dst/out",
          support::RuntimeABIParameterRole::OutputBuffer))
    return error;
  if (llvm::Error error = requireABI(
          memoryOperandBindingFacts.runtimeElementCountABI, "n",
          support::RuntimeABIParameterRole::RuntimeElementCount))
    return error;

  if (isLoadMerge) {
    if (!memoryOperandBindingFacts.passthroughABI)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " regular computed-mask load-merge route construction requires "
          "old-destination passthrough operand-binding facts before creating "
          "TCRVEmitCLowerableRoute");
    if (memoryOperandBindingFacts.passthroughABI->role !=
            support::RuntimeABIParameterRole::OutputBuffer ||
        memoryOperandBindingFacts.passthroughABI->cName !=
            memoryOperandBindingFacts.destinationABI->cName)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " regular computed-mask load-merge route construction requires "
          "old-destination passthrough and destination store ABI facts to "
          "refer to the same output buffer before creating "
          "TCRVEmitCLowerableRoute");
  }
  if (isStridedLoad)
    if (llvm::Error error = requireABI(
            memoryOperandBindingFacts.sourceStrideABI, "src_stride_bytes",
            support::RuntimeABIParameterRole::SourceByteStride))
      return error;
  if (isStridedStore)
    if (llvm::Error error = requireABI(
            memoryOperandBindingFacts.destinationStrideABI,
            "dst_stride_bytes",
            support::RuntimeABIParameterRole::DestinationByteStride))
      return error;
  if (isIndexed)
    if (llvm::Error error = requireABI(
            memoryOperandBindingFacts.indexABI, "index",
            support::RuntimeABIParameterRole::IndexInputBuffer))
      return error;

  if (!computedMaskMemoryStatementPlan.plansComputedMaskMemoryRoute ||
      computedMaskMemoryStatementPlan.computedMaskMemoryPlan != &plan ||
      computedMaskMemoryStatementPlan.preLoopSteps.empty() ||
      computedMaskMemoryStatementPlan.loop.bodySteps.empty() ||
      computedMaskMemoryStatementPlan.plansRuntimeScalarComputedMaskStore ||
      computedMaskMemoryStatementPlan.plansRuntimeScalarComputedMaskLoadStore ||
      computedMaskMemoryStatementPlan
          .plansRuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
      computedMaskMemoryStatementPlan.plansComputedMaskUnitLoadStore !=
          isUnitLoadStore ||
      computedMaskMemoryStatementPlan.plansComputedMaskStridedStore !=
          isStridedStore ||
      computedMaskMemoryStatementPlan.plansComputedMaskStridedLoadUnitStore !=
          isStridedLoad ||
      computedMaskMemoryStatementPlan.plansComputedMaskIndexedGatherLoadUnitStore !=
          isIndexedGather ||
      computedMaskMemoryStatementPlan
              .plansComputedMaskIndexedScatterStoreUnitLoad !=
          isIndexedScatter)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires the "
        "matching regular computed-mask memory statement plan before creating "
        "TCRVEmitCLowerableRoute");

  auto preLoopHasCallee = [&](llvm::StringRef callee) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         computedMaskMemoryStatementPlan.preLoopSteps)
      if (step.callee == callee)
        return true;
    return false;
  };
  auto statementHasCallee = [&](llvm::StringRef callee) {
    for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
         computedMaskMemoryStatementPlan.loop.bodySteps)
      if (step.callee == callee)
        return true;
    return false;
  };
  if (!preLoopHasCallee(plan.setVLIntrinsic) ||
      !statementHasCallee(plan.setVLIntrinsic) ||
      !statementHasCallee(plan.vectorLoadIntrinsic) ||
      !statementHasCallee(plan.compareIntrinsic) ||
      (isLoadMerge && !statementHasCallee(plan.maskedLoadIntrinsic)) ||
      (isLoadMerge && !statementHasCallee(plan.maskedStoreIntrinsic)) ||
      (isStridedStore && !statementHasCallee(plan.stridedStoreIntrinsic)) ||
      (isIndexed && !statementHasCallee(plan.indexLoadIntrinsic)) ||
      (isIndexed && !statementHasCallee(plan.indexScaleIntrinsic)) ||
      (isIndexedScatter &&
       !statementHasCallee(plan.indexedStoreIntrinsic)))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires statement/"
        "leaf facts for setvl, vector compare, masked load/store, stride, and "
        "index calls from the RVV statement-plan owner before creating "
        "TCRVEmitCLowerableRoute");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsComputedMaskMemory ||
      routeControlPlan->runtimeControlPlan != &plan.runtimeControlPlan ||
      routeControlPlan->typedConfigFacts != &analysis.typedConfigFacts ||
      routeControlPlan->selectedTargetCapabilityFacts !=
          &analysis.selectedTargetCapabilityFacts ||
      routeControlPlan->controlPlanIDMirror !=
          plan.runtimeControlPlan.controlPlanID ||
      routeControlPlan->configContractIDMirror !=
          plan.runtimeControlPlan.configContractID ||
      routeControlPlan->runtimeVLContractIDMirror !=
          plan.runtimeControlPlan.runtimeVLContractID ||
      routeControlPlan->runtimeAVLASourceMirror !=
          plan.runtimeControlPlan.runtimeAVLASource ||
      routeControlPlan->runtimeABIOrderMirror != plan.runtimeABIOrder ||
      routeControlPlan->tailPolicyMirror != plan.runtimeControlPlan.tailPolicy ||
      routeControlPlan->maskPolicyMirror != plan.runtimeControlPlan.maskPolicy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires the "
        "RVV-owned route-control provider plan and runtime n/AVL/VL facts "
        "before creating TCRVEmitCLowerableRoute");

  const RVVSelectedBodyMaskTailPolicyProviderPlan &maskTailPlan =
      computedMaskMemoryStatementPlan.maskTailPolicyPlan;
  if (!maskTailPlan.plansMaskTailPolicy ||
      !maskTailPlan.controlsComputedMaskMemory ||
      maskTailPlan.controlsComputedMaskSelect ||
      maskTailPlan.computedMaskMemoryPlan != &plan ||
      maskTailPlan.typedConfigFacts != &analysis.typedConfigFacts ||
      maskTailPlan.selectedTargetCapabilityFacts !=
          &analysis.selectedTargetCapabilityFacts ||
      maskTailPlan.bindingPlan != &analysis.routeOperandBindingPlan ||
      maskTailPlan.familyPlanIDMirror != kRVVMaskTailPolicyRouteFamilyPlanID ||
      maskTailPlan.ownerNameMirror !=
          kRVVComputedMaskMemoryMaskTailPolicyOwner ||
      maskTailPlan.maskProducerSourceMirror != plan.maskProducerSource ||
      maskTailPlan.maskRoleMirror != plan.maskRole ||
      maskTailPlan.maskSourceMirror != plan.maskSource ||
      maskTailPlan.maskMemoryFormMirror != plan.maskMemoryForm ||
      maskTailPlan.tailPolicyMirror != plan.runtimeControlPlan.tailPolicy ||
      maskTailPlan.maskPolicyMirror != plan.runtimeControlPlan.maskPolicy ||
      maskTailPlan.inactiveLaneContractMirror !=
          plan.inactiveLaneContract ||
      maskTailPlan.maskedPassthroughLayoutMirror !=
          plan.maskedPassthroughLayout ||
      maskTailPlan.runtimeABIOrderMirror != plan.runtimeABIOrder ||
      maskTailPlan.routeOperandBindingPlanIDMirror !=
          analysis.routeOperandBindingPlan.planID ||
      maskTailPlan.providerSupportedMirror != plan.providerSupportedMirror ||
      maskTailPlan.selectedProviderMirror !=
          routeControlPlan->selectedProviderMirror ||
      maskTailPlan.selectedLegalityMirror !=
          routeControlPlan->selectedLegalityMirror)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " regular computed-mask memory route construction requires the "
        "RVV-owned mask/tail policy provider plan from the same computed-mask "
        "memory statement plan before creating TCRVEmitCLowerableRoute");

  return llvm::Error::success();
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
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
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
