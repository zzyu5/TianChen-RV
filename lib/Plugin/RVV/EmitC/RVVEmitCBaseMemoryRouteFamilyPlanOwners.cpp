#include "TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h"

#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVStridedLoadUnitStoreOperandBindingPlanID(
    "rvv-route-operand-binding:strided_load_unit_store.v1");
constexpr llvm::StringLiteral kRVVUnitLoadStridedStoreOperandBindingPlanID(
    "rvv-route-operand-binding:unit_load_strided_store.v1");
constexpr llvm::StringLiteral kRVVIndexedGatherOperandBindingPlanID(
    "rvv-route-operand-binding:indexed_gather_unit_store.v1");
constexpr llvm::StringLiteral kRVVIndexedScatterOperandBindingPlanID(
    "rvv-route-operand-binding:indexed_scatter_unit_load.v1");
constexpr llvm::StringLiteral kRVVMaskedUnitLoadStoreOperandBindingPlanID(
    "rvv-route-operand-binding:masked_unit_load_store.v1");
constexpr llvm::StringLiteral kRVVMaskedUnitStoreOperandBindingPlanID(
    "rvv-route-operand-binding:masked_unit_store.v1");

constexpr llvm::StringLiteral kRVVStridedLoadUnitStoreRuntimeABIOrder(
    "src,out,n,stride_bytes");
constexpr llvm::StringLiteral kRVVUnitLoadStridedStoreRuntimeABIOrder(
    "src,dst,n,dst_stride_bytes");
constexpr llvm::StringLiteral kRVVIndexedGatherRuntimeABIOrder(
    "data,index,out,n");
constexpr llvm::StringLiteral kRVVIndexedScatterRuntimeABIOrder(
    "src,index,dst,n");
constexpr llvm::StringLiteral kRVVMaskedMemoryRuntimeABIOrder(
    "src,mask,dst,n");
constexpr llvm::StringLiteral kRVVMaskedStoreRuntimeABIOrder(
    "src,mask,dst,n");

constexpr llvm::StringLiteral kRVVBaseMemoryMovementRouteFamilyPlanID(
    "rvv-base-memory-movement-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVStridedLoadUnitStoreTargetLeafProfile(
    "rvv-v1-e32m1-strided-load-unit-store-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVUnitLoadStridedStoreTargetLeafProfile(
    "rvv-v1-e32m1-unit-load-strided-store-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVIndexedGatherUnitStoreTargetLeafProfile(
    "rvv-v1-e32m1-indexed-gather-unit-store-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVIndexedScatterUnitLoadTargetLeafProfile(
    "rvv-v1-e32m1-indexed-scatter-unit-load-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVMaskedUnitLoadStoreTargetLeafProfile(
    "rvv-v1-e32m1-masked-unit-load-store-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVMaskedUnitStoreTargetLeafProfile(
    "rvv-v1-e32m1-masked-unit-store-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVStridedLoadUnitStoreProviderSupportedMirror(
    "provider_supported_mirror:rvv-strided-load-unit-store-plan-validated");
constexpr llvm::StringLiteral kRVVUnitLoadStridedStoreProviderSupportedMirror(
    "provider_supported_mirror:rvv-unit-load-strided-store-plan-validated");
constexpr llvm::StringLiteral kRVVIndexedGatherUnitStoreProviderSupportedMirror(
    "provider_supported_mirror:rvv-indexed-gather-unit-store-plan-validated");
constexpr llvm::StringLiteral
    kRVVIndexedScatterUnitLoadProviderSupportedMirror(
        "provider_supported_mirror:rvv-indexed-scatter-unit-load-plan-validated");
constexpr llvm::StringLiteral kRVVMaskedUnitLoadStoreProviderSupportedMirror(
    "provider_supported_mirror:rvv-masked-unit-load-store-plan-validated");
constexpr llvm::StringLiteral kRVVMaskedUnitStoreProviderSupportedMirror(
    "provider_supported_mirror:rvv-masked-unit-store-plan-validated");
constexpr llvm::StringLiteral
    kRVVBaseMemoryMovementRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVBaseMemoryVLType("size_t");
constexpr llvm::StringLiteral kRVVBaseMemoryVectorTypeName(
    "!tcrv_rvv.vector<i32, \"m1\">");
constexpr llvm::StringLiteral kRVVBaseMemoryVectorCType("vint32m1_t");
constexpr llvm::StringLiteral kRVVBaseMemorySetVLIntrinsic(
    "__riscv_vsetvl_e32m1");
constexpr llvm::StringLiteral kRVVBaseMemoryVectorLoadIntrinsic(
    "__riscv_vle32_v_i32m1");
constexpr llvm::StringLiteral kRVVBaseMemoryStridedLoadIntrinsic(
    "__riscv_vlse32_v_i32m1");
constexpr llvm::StringLiteral kRVVBaseMemoryStoreIntrinsic(
    "__riscv_vse32_v_i32m1");
constexpr llvm::StringLiteral kRVVBaseMemoryStridedStoreIntrinsic(
    "__riscv_vsse32_v_i32m1");
constexpr llvm::StringLiteral kRVVBaseMemoryStrideCType("size_t");
constexpr llvm::StringLiteral kRVVBaseMemoryStrideUnit("byte");
constexpr llvm::StringLiteral kRVVStridedLoadUnitStoreCTypeMappingSummary(
    "vl:size_t,source:byte-strided-e32m1,result:signed-e32m1");
constexpr llvm::StringLiteral kRVVUnitLoadStridedStoreCTypeMappingSummary(
    "vl:size_t,source:signed-e32m1,destination:byte-strided-e32m1");
constexpr llvm::StringLiteral kRVVIndexedGatherUnitStoreCTypeMappingSummary(
    "vl:size_t,data:signed-e32m1,index:u32m1,result:signed-e32m1");
constexpr llvm::StringLiteral kRVVIndexedScatterUnitLoadCTypeMappingSummary(
    "vl:size_t,source:signed-e32m1,index:u32m1,destination:indexed-e32m1");
constexpr llvm::StringLiteral kRVVMaskedUnitLoadStoreCTypeMappingSummary(
    "vl:size_t,source/passthrough:signed-e32m1,mask:b32,result:masked-load-store");
constexpr llvm::StringLiteral kRVVMaskedUnitStoreCTypeMappingSummary(
    "vl:size_t,source:signed-e32m1,mask:b32,destination:masked-store");

constexpr llvm::StringLiteral kRVVStridedLoadUnitStoreMemoryLayout(
    "byte-strided-source-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral kRVVUnitLoadStridedStoreMemoryLayout(
    "unit-stride-source-byte-strided-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVIndexedGatherMemoryLayout(
    "element-indexed-data-index-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral kRVVIndexedScatterMemoryLayout(
    "unit-stride-source-indexed-destination-index-runtime-abi");
constexpr llvm::StringLiteral kRVVMaskedMemoryLayout(
    "unit-stride-source-mask-old-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVMaskedStoreMemoryLayout(
    "unit-stride-source-mask-destination-masked-store-runtime-abi");
constexpr llvm::StringLiteral kRVVSourceStrideSource(
    "runtime_abi:stride_bytes");
constexpr llvm::StringLiteral kRVVDestinationByteStrideSource(
    "runtime_abi:dst_stride_bytes");
constexpr llvm::StringLiteral kRVVUnitStrideSourceMemoryForm(
    "unit-stride-load");
constexpr llvm::StringLiteral kRVVSourceMemoryForm("strided-load");
constexpr llvm::StringLiteral kRVVDestinationMemoryForm("unit-stride-store");
constexpr llvm::StringLiteral kRVVIndexedDataMemoryForm("indexed-load");
constexpr llvm::StringLiteral kRVVIndexedDestinationMemoryForm(
    "indexed-store");
constexpr llvm::StringLiteral kRVVIndexSource("runtime_abi:index");
constexpr llvm::StringLiteral kRVVIndexedGatherOffsetUnit("element");
constexpr llvm::StringLiteral kRVVIndexedScatterIndexUniqueness("unique");
constexpr llvm::StringLiteral kRVVMaskRole("predicate-mask-input-buffer");
constexpr llvm::StringLiteral kRVVMaskSource("runtime_abi:mask");
constexpr llvm::StringLiteral kRVVMaskMemoryForm("unit-stride-mask-load");
constexpr llvm::StringLiteral kRVVMaskedMemoryInactiveLaneContract(
    "masked-off-lanes-preserve-old-destination");
constexpr llvm::StringLiteral kRVVMaskedMemoryPassthroughLayout(
    "old-destination-vector-preserves-inactive-lanes");
constexpr llvm::StringLiteral kRVVMaskedStoreInactiveLaneContract(
    "masked-store-false-lanes-preserve-output-buffer");
constexpr llvm::StringLiteral kRVVMaskedStorePassthroughLayout(
    "masked-store-has-no-passthrough-load");
constexpr llvm::StringLiteral kRVVMaskedStoreDestinationMemoryForm(
    "masked-unit-store");

bool isPreRealizedStridedMemoryMovementOpKind(llvm::StringRef opKind) {
  return opKind == "strided_load_unit_store";
}

bool isPreRealizedStridedMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "strided-load-unit-store";
}

bool isPreRealizedStridedLoadUnitStoreStrideUnit(llvm::StringRef strideUnit) {
  return strideUnit == "byte";
}

bool isPreRealizedStridedStoreMemoryMovementOpKind(llvm::StringRef opKind) {
  return opKind == "unit_load_strided_store";
}

bool isPreRealizedStridedStoreMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-load-strided-store";
}

bool isPreRealizedStridedStoreMemoryMovementStrideUnit(
    llvm::StringRef strideUnit) {
  return strideUnit == "byte";
}

bool isPreRealizedIndexedGatherMemoryMovementOpKind(
    llvm::StringRef opKind) {
  return opKind == "indexed_gather_unit_store";
}

bool isPreRealizedIndexedScatterMemoryMovementOpKind(
    llvm::StringRef opKind) {
  return opKind == "indexed_scatter_unit_load";
}

bool isPreRealizedIndexedGatherMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "indexed-load-unit-store";
}

bool isPreRealizedIndexedScatterMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-load-indexed-store";
}

bool isPreRealizedIndexedGatherMemoryMovementIndexEEW(
    std::int64_t indexEEW) {
  return indexEEW == 32;
}

bool isPreRealizedIndexedGatherMemoryMovementOffsetUnit(
    llvm::StringRef offsetUnit) {
  return offsetUnit == "element";
}

bool isPreRealizedIndexedScatterIndexUniqueness(
    llvm::StringRef indexUniqueness) {
  return indexUniqueness == "unique";
}

bool isPreRealizedMaskedMemoryMovementOpKind(llvm::StringRef opKind) {
  return opKind == "masked_unit_load_store" || opKind == "masked_unit_store";
}

bool isPreRealizedMaskedMemoryMovementMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "masked-unit-load-store" ||
         memoryForm == "masked-unit-store";
}

bool isPreRealizedMaskedMemoryMovementMaskRole(llvm::StringRef role) {
  return role == "predicate-mask-input-buffer";
}

bool isPreRealizedMaskedMemoryMovementMaskMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-mask-load";
}

bool isPreRealizedMaskedMemoryMovementInactiveLanePolicy(
    llvm::StringRef policy) {
  return policy == "preserve-old-destination" ||
         policy == "preserve-output-on-false-lanes";
}

llvm::Expected<tcrv::rvv::RuntimeABIValueOp>
requirePreRealizedBaseMemoryRuntimeABIValue(
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
llvm::Error rejectMixedPreRealizedBaseMemoryBody(
    tcrv::exec::VariantOp variant, mlir::Operation *bodyOp,
    llvm::StringRef bodyDescription) {
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

llvm::Error requireBaseMemorySelectedVariantRequires(
    tcrv::exec::VariantOp variant, llvm::StringRef context) {
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requires || requires.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected ") + context +
        " realization requires non-empty selected variant requires metadata");
  return llvm::Error::success();
}

void applyBaseMemoryRuntimeAVLVLControlPlanToDescription(
    const RVVRuntimeAVLVLControlPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  description.sew = plan.sew;
  description.lmul = plan.lmul;
  description.tailPolicy = plan.tailPolicy;
  description.maskPolicy = plan.maskPolicy;
  description.runtimeControlPlanID = plan.controlPlanID;
  description.configContractID = plan.configContractID;
  description.runtimeVLContractID = plan.runtimeVLContractID;
  description.runtimeAVLASource = plan.runtimeAVLASource;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.vlDefOpName = plan.vlDefOpName;
  description.vlScopeOpName = plan.vlScopeOpName;
  description.vlUses = plan.vlUses;
  description.emitCLoopKind = plan.emitCLoopKind;
  description.emitCLoopInductionName = plan.emitCLoopInductionName;
  description.emitCFullChunkVLName = plan.emitCFullChunkVLName;
  description.emitCLoopVLName = plan.emitCLoopVLName;
  description.remainingAVLMetadata = plan.remainingAVLMetadata;
  description.pointerAdvanceMetadata = plan.pointerAdvanceMetadata;
  description.boundedSlice = plan.boundedSlice;
  description.multiVL = plan.multiVL;
}

void addBaseMemoryRouteOperandBinding(
    RVVRouteOperandBindingPlan &plan, llvm::StringRef logicalOperand,
    const support::RuntimeABIParameter &parameter,
    llvm::ArrayRef<llvm::StringRef> materializedUses) {
  RVVRouteOperandBinding binding;
  binding.logicalOperand = logicalOperand.str();
  binding.parameter = parameter;
  for (llvm::StringRef use : materializedUses)
    binding.materializedUses.push_back(use.str());
  plan.bindings.push_back(std::move(binding));
}

llvm::Error requireBaseMemoryPlanField(
    const RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("base memory movement route-family plan validation for "
                  "operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireBaseMemoryPlanDerivedField(
    const RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, bool required) {
  if (required) {
    if (!actual.trim().empty())
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("base memory movement route-family plan validation for "
                    "operation '") +
        stringifyRVVSelectedBodyOperationKind(plan.operation) +
        "' requires provider-derived " + field +
        " for the active typed memory route");
  }
  if (actual.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("base memory movement route-family plan validation for "
                  "operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) +
      "' requires inactive " + field + " to be empty but found '" + actual +
      "'");
}

llvm::Error requireBaseMemoryRouteDescriptionField(
    llvm::StringRef context, llvm::StringRef field, llvm::StringRef actual,
    llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " " + field +
      " must mirror selected base memory movement owner fact '" + expected +
      "' but was '" + actual + "'");
}

llvm::Error requireBaseMemoryRouteDescriptionDerivedField(
    llvm::StringRef context, llvm::StringRef field, llvm::StringRef actual,
    bool required) {
  if (required) {
    if (!actual.trim().empty())
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + field +
        " must mirror a provider-derived selected base memory movement owner "
        "fact for the active typed memory route");
  }
  if (actual.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " " + field +
      " must be empty for inactive base memory movement route facts but was '" +
      actual + "'");
}

llvm::Error requireBaseMemoryTypedConfigLeaf(
    const RVVSelectedBodyTypedConfigFacts &typedFacts, llvm::StringRef field,
    llvm::StringRef value, RVVSelectedBodyOperationKind operation) {
  if (!value.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("base memory movement route-family plan requires "
                  "typed config-derived ") +
      field + " before deriving memory route facts for operation '" +
      stringifyRVVSelectedBodyOperationKind(operation) +
      "' from typed config facts '" + typedFacts.factsID + "'");
}

llvm::Error requireBaseMemoryTypedConfigLeaves(
    const RVVSelectedBodyTypedConfigFacts &typedFacts,
    RVVSelectedBodyOperationKind operation, bool isIndexed,
    bool isIndexedGather, bool isIndexedScatter, bool isStridedLoad,
    bool isStridedStore, bool isStaticMaskLoad, bool isStaticMaskStore) {
  if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
          typedFacts, "VL C type", typedFacts.vlCType, operation))
    return error;
  if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
          typedFacts, "vector type", typedFacts.vectorTypeName, operation))
    return error;
  if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
          typedFacts, "vector C type", typedFacts.vectorCType, operation))
    return error;
  if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
          typedFacts, "setvl leaf", typedFacts.setVLIntrinsic, operation))
    return error;
  if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
          typedFacts, "vector-load leaf", typedFacts.vectorLoadIntrinsic,
          operation))
    return error;
  if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
          typedFacts, "store leaf", typedFacts.storeIntrinsic, operation))
    return error;
  if (isIndexed) {
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "index vector type", typedFacts.indexVectorTypeName,
            operation))
      return error;
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "index vector C type", typedFacts.indexVectorCType,
            operation))
      return error;
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "index-load leaf", typedFacts.indexLoadIntrinsic,
            operation))
      return error;
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "index-scale leaf", typedFacts.indexScaleIntrinsic,
            operation))
      return error;
  }
  if (isIndexedGather) {
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "indexed-load leaf", typedFacts.indexedLoadIntrinsic,
            operation))
      return error;
  }
  if (isIndexedScatter) {
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "indexed-store leaf", typedFacts.indexedStoreIntrinsic,
            operation))
      return error;
  }
  if (isStridedLoad) {
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "strided-load leaf", typedFacts.stridedLoadIntrinsic,
            operation))
      return error;
  }
  if (isStridedStore) {
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "strided-store leaf", typedFacts.stridedStoreIntrinsic,
            operation))
      return error;
  }
  if (isStaticMaskLoad) {
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "mask type", typedFacts.maskTypeName, operation))
      return error;
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "mask C type", typedFacts.maskCType, operation))
      return error;
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "masked-load leaf", typedFacts.maskedLoadIntrinsic,
            operation))
      return error;
  }
  if (isStaticMaskStore) {
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "mask type", typedFacts.maskTypeName, operation))
      return error;
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "mask C type", typedFacts.maskCType, operation))
      return error;
    if (llvm::Error error = requireBaseMemoryTypedConfigLeaf(
            typedFacts, "masked-store leaf", typedFacts.maskedStoreIntrinsic,
            operation))
      return error;
  }
  return llvm::Error::success();
}

RVVSelectedBodyMemoryForm
getBaseMemoryMovementRouteFamilyMemoryForm(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return RVVSelectedBodyMemoryForm::UnitLoadStridedStore;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return RVVSelectedBodyMemoryForm::MaskedUnitStore;
  default:
    llvm_unreachable("unsupported base memory movement route-family op");
  }
}

llvm::StringRef
getBaseMemoryMovementRuntimeABIOrder(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return kRVVStridedLoadUnitStoreRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return kRVVUnitLoadStridedStoreRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return kRVVIndexedGatherRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return kRVVIndexedScatterRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return kRVVMaskedMemoryRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return kRVVMaskedStoreRuntimeABIOrder;
  default:
    return {};
  }
}

llvm::StringRef
getBaseMemoryMovementTargetLeafProfile(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return kRVVStridedLoadUnitStoreTargetLeafProfile;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return kRVVUnitLoadStridedStoreTargetLeafProfile;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return kRVVIndexedGatherUnitStoreTargetLeafProfile;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return kRVVIndexedScatterUnitLoadTargetLeafProfile;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return kRVVMaskedUnitLoadStoreTargetLeafProfile;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return kRVVMaskedUnitStoreTargetLeafProfile;
  default:
    return {};
  }
}

llvm::StringRef
getBaseMemoryMovementProviderSupportedMirror(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return kRVVStridedLoadUnitStoreProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return kRVVUnitLoadStridedStoreProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return kRVVIndexedGatherUnitStoreProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return kRVVIndexedScatterUnitLoadProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return kRVVMaskedUnitLoadStoreProviderSupportedMirror;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return kRVVMaskedUnitStoreProviderSupportedMirror;
  default:
    return {};
  }
}

llvm::StringRef
getBaseMemoryMovementCTypeMappingSummary(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return kRVVStridedLoadUnitStoreCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return kRVVUnitLoadStridedStoreCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return kRVVIndexedGatherUnitStoreCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return kRVVIndexedScatterUnitLoadCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return kRVVMaskedUnitLoadStoreCTypeMappingSummary;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return kRVVMaskedUnitStoreCTypeMappingSummary;
  default:
    return {};
  }
}

llvm::StringRef
getBaseMemoryMovementStridedLayout(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return kRVVStridedLoadUnitStoreMemoryLayout;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return kRVVUnitLoadStridedStoreMemoryLayout;
  default:
    return {};
  }
}

llvm::StringRef
getBaseMemoryMovementIndexedLayout(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return kRVVIndexedGatherMemoryLayout;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return kRVVIndexedScatterMemoryLayout;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return kRVVMaskedMemoryLayout;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return kRVVMaskedStoreMemoryLayout;
  default:
    return {};
  }
}

llvm::StringRef
getBaseMemoryMovementSourceMemoryForm(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return kRVVSourceMemoryForm;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return kRVVUnitStrideSourceMemoryForm;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return {};
  default:
    return {};
  }
}

llvm::StringRef
getBaseMemoryMovementDestinationMemoryForm(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return kRVVDestinationMemoryForm;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return "strided-store";
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return kRVVIndexedDestinationMemoryForm;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return kRVVMaskedStoreDestinationMemoryForm;
  default:
    return {};
  }
}

llvm::StringRef getBaseMemoryMovementResultName(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return "masked_loaded_vec";
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return "payload_vec";
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return "loaded_vec";
  default:
    return {};
  }
}

llvm::StringRef getBaseMemoryMovementMaskName(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return "memory_mask";
  default:
    return {};
  }
}

llvm::SmallVector<support::RuntimeABIParameter, 8>
getBaseMemoryMovementRuntimeABIParameters(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return tcrv::rvv::getRVVSelectedBodyStridedLoadUnitStoreRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return tcrv::rvv::getRVVSelectedBodyUnitLoadStridedStoreRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return tcrv::rvv::getRVVSelectedBodyIndexedGatherRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return tcrv::rvv::getRVVSelectedBodyIndexedScatterRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return tcrv::rvv::getRVVSelectedBodyMaskedMemoryRuntimeABIParameters();
  default:
    return {};
  }
}

RVVRouteOperandBindingPlan buildBaseMemoryRouteOperandBindingPlanFromFacts(
    const RVVBaseMemoryMovementRouteFacts &facts) {
  RVVRouteOperandBindingPlan plan;
  plan.planID = facts.routeOperandBindingPlanID.str();
  auto parameter = [&](std::size_t index) -> const support::RuntimeABIParameter & {
    return facts.runtimeABIParameters[index];
  };

  switch (facts.operation) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    addBaseMemoryRouteOperandBinding(
        plan, "src", parameter(0),
        {"runtime-abi-mirror", "materialized-strided-load-base",
         "move-source"});
    addBaseMemoryRouteOperandBinding(
        plan, "out", parameter(1),
        {"runtime-abi-mirror", "materialized-store-base", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", parameter(2),
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "stride_bytes", parameter(3),
        {"runtime-abi-mirror", "materialized-strided-load-stride",
         "materialized-byte-address", "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    addBaseMemoryRouteOperandBinding(
        plan, "src", parameter(0),
        {"runtime-abi-mirror", "materialized-load-base", "move-source"});
    addBaseMemoryRouteOperandBinding(
        plan, "dst", parameter(1),
        {"runtime-abi-mirror", "materialized-strided-store-base",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", parameter(2),
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "dst_stride_bytes", parameter(3),
        {"runtime-abi-mirror", "materialized-strided-store-stride",
         "materialized-byte-address", "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    addBaseMemoryRouteOperandBinding(
        plan, "data", parameter(0),
        {"runtime-abi-mirror", "materialized-indexed-data-base",
         "indexed-load-base", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "index", parameter(1),
        {"runtime-abi-mirror", "materialized-index-load-base",
         "index-offset-scale", "index-source-mirror", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "out", parameter(2),
        {"runtime-abi-mirror", "materialized-store-base", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", parameter(3),
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    addBaseMemoryRouteOperandBinding(
        plan, "src", parameter(0),
        {"runtime-abi-mirror", "materialized-load-base", "move-source",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "index", parameter(1),
        {"runtime-abi-mirror", "materialized-index-load-base",
         "index-offset-scale", "index-source-mirror", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "dst", parameter(2),
        {"runtime-abi-mirror", "materialized-indexed-store-base",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", parameter(3),
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    addBaseMemoryRouteOperandBinding(
        plan, "src", parameter(0),
        {"runtime-abi-mirror", "materialized-masked-load-base",
         "masked-load-source-call", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "mask", parameter(1),
        {"runtime-abi-mirror", "materialized-mask-load-base",
         "masked-load-mask-call"});
    addBaseMemoryRouteOperandBinding(
        plan, "dst", parameter(2),
        {"runtime-abi-mirror", "materialized-old-destination-load-base",
         "masked-load-passthrough-call", "materialized-store-base",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", parameter(3),
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    addBaseMemoryRouteOperandBinding(
        plan, "src", parameter(0),
        {"runtime-abi-mirror", "materialized-load-base",
         "masked-store-source-call"});
    addBaseMemoryRouteOperandBinding(
        plan, "mask", parameter(1),
        {"runtime-abi-mirror", "materialized-mask-load-base",
         "masked-store-mask-call"});
    addBaseMemoryRouteOperandBinding(
        plan, "dst", parameter(2),
        {"runtime-abi-mirror", "materialized-masked-store-base",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", parameter(3),
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    break;
  default:
    break;
  }
  return plan;
}

llvm::Expected<RVVRouteOperandBindingPlan>
deriveBaseMemoryRouteOperandBindingPlanImpl(
    const RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyRouteSlice &slice = analysis.slice;
  RVVRouteOperandBindingPlan plan;
  llvm::StringRef expectedRuntimeABIOrder;
  llvm::StringRef context;

  switch (slice.arithmeticKind) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    plan.planID = kRVVStridedLoadUnitStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVStridedLoadUnitStoreRuntimeABIOrder;
    context = "strided_load_unit_store route";
    addBaseMemoryRouteOperandBinding(
        plan, "src", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-strided-load-base",
         "move-source"});
    addBaseMemoryRouteOperandBinding(
        plan, "out", slice.outABI,
        {"runtime-abi-mirror", "materialized-store-base", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "stride_bytes", slice.lhsStrideABI,
        {"runtime-abi-mirror", "materialized-strided-load-stride",
         "materialized-byte-address", "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    plan.planID = kRVVUnitLoadStridedStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVUnitLoadStridedStoreRuntimeABIOrder;
    context = "unit_load_strided_store route";
    addBaseMemoryRouteOperandBinding(
        plan, "src", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base", "move-source"});
    addBaseMemoryRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"runtime-abi-mirror", "materialized-strided-store-base",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "dst_stride_bytes", slice.outStrideABI,
        {"runtime-abi-mirror", "materialized-strided-store-stride",
         "materialized-byte-address", "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    plan.planID = kRVVIndexedGatherOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVIndexedGatherRuntimeABIOrder;
    context = "indexed_gather_unit_store route";
    addBaseMemoryRouteOperandBinding(
        plan, "data", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-indexed-data-base",
         "indexed-load-base", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "index", slice.indexABI,
        {"runtime-abi-mirror", "materialized-index-load-base",
         "index-offset-scale", "index-source-mirror", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "out", slice.outABI,
        {"runtime-abi-mirror", "materialized-store-base", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    plan.planID = kRVVIndexedScatterOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVIndexedScatterRuntimeABIOrder;
    context = "indexed_scatter_unit_load route";
    addBaseMemoryRouteOperandBinding(
        plan, "src", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base", "move-source",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "index", slice.indexABI,
        {"runtime-abi-mirror", "materialized-index-load-base",
         "index-offset-scale", "index-source-mirror", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"runtime-abi-mirror", "materialized-indexed-store-base",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    plan.planID = kRVVMaskedUnitLoadStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVMaskedMemoryRuntimeABIOrder;
    context = "masked_unit_load_store route";
    addBaseMemoryRouteOperandBinding(
        plan, "src", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-masked-load-base",
         "masked-load-source-call", "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "mask", slice.maskABI,
        {"runtime-abi-mirror", "materialized-mask-load-base",
         "masked-load-mask-call"});
    addBaseMemoryRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"runtime-abi-mirror", "materialized-old-destination-load-base",
         "masked-load-passthrough-call", "materialized-store-base",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    plan.planID = kRVVMaskedUnitStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVMaskedStoreRuntimeABIOrder;
    context = "masked_unit_store route";
    addBaseMemoryRouteOperandBinding(
        plan, "src", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base",
         "masked-store-source-call"});
    addBaseMemoryRouteOperandBinding(
        plan, "mask", slice.maskABI,
        {"runtime-abi-mirror", "materialized-mask-load-base",
         "masked-store-mask-call"});
    addBaseMemoryRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"runtime-abi-mirror", "materialized-masked-store-base",
         "header-mirror"});
    addBaseMemoryRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    break;
  default:
    return plan;
  }

  if (llvm::Error error = verifyRVVRouteOperandBindingPlan(
          plan, plan.planID, expectedRuntimeABIOrder, context))
    return std::move(error);
  if (expectedRuntimeABIOrder != analysis.description.runtimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context + " requires description runtime ABI order '" +
        expectedRuntimeABIOrder + "' but found '" +
        analysis.description.runtimeABIOrder + "'");
  llvm::SmallVector<support::RuntimeABIParameter, 8> planParameters;
  for (const RVVRouteOperandBinding &binding : plan.bindings)
    planParameters.push_back(binding.parameter);
  if (!support::runtimeABIParametersEqual(
          planParameters, analysis.description.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context +
        " requires runtime ABI parameter mirrors to match the binding plan");
  return plan;
}

} // namespace

std::optional<RVVBaseMemoryMovementRouteFacts>
getRVVBaseMemoryMovementRouteFacts(RVVSelectedBodyOperationKind operation) {
  if (!isRVVSelectedBodyBaseMemoryMovementRouteOperation(operation))
    return std::nullopt;

  const bool isIndexedGather =
      operation == RVVSelectedBodyOperationKind::IndexedGatherUnitStore;
  const bool isIndexedScatter =
      operation == RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
  const bool isStaticMaskLoad =
      operation == RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  const bool isStaticMaskStore =
      operation == RVVSelectedBodyOperationKind::MaskedUnitStore;
  const bool isIndexed = isIndexedGather || isIndexedScatter;
  const bool isMasked = isStaticMaskLoad || isStaticMaskStore;
  const bool isStridedLoad =
      operation == RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  const bool isStridedStore =
      operation == RVVSelectedBodyOperationKind::UnitLoadStridedStore;

  RVVBaseMemoryMovementRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = getBaseMemoryMovementRouteFamilyMemoryForm(operation);
  facts.sew = tcrv::rvv::getRVVFirstSliceSEWBits();
  facts.lmul = tcrv::rvv::getRVVLMULM1();
  facts.tailPolicy =
      isStaticMaskStore ? llvm::StringRef("undisturbed")
                        : llvm::StringRef("agnostic");
  facts.maskPolicy =
      isStaticMaskStore ? llvm::StringRef("undisturbed")
                        : llvm::StringRef("agnostic");
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = getBaseMemoryMovementRuntimeABIOrder(operation);
  facts.targetLeafProfile = getBaseMemoryMovementTargetLeafProfile(operation);
  facts.providerSupportedMirror =
      getBaseMemoryMovementProviderSupportedMirror(operation);
  facts.requiredHeaderDeclarations =
      kRVVBaseMemoryMovementRequiredHeaderDeclarations;
  facts.cTypeMappingSummary = getBaseMemoryMovementCTypeMappingSummary(operation);
  facts.routeOperandBindingPlanID =
      *getExpectedRVVSelectedBodyBaseMemoryRouteOperandBindingPlanID(operation);
  facts.routeFamilyPlanID = kRVVBaseMemoryMovementRouteFamilyPlanID;
  facts.typedComputeOpName = "tcrv_rvv.move";
  facts.vlCType = kRVVBaseMemoryVLType;
  facts.vectorTypeName = kRVVBaseMemoryVectorTypeName;
  facts.vectorCType = kRVVBaseMemoryVectorCType;
  facts.setVLIntrinsic = kRVVBaseMemorySetVLIntrinsic;
  facts.vectorLoadIntrinsic = kRVVBaseMemoryVectorLoadIntrinsic;
  facts.stridedLoadIntrinsic =
      isStridedLoad ? llvm::StringRef(kRVVBaseMemoryStridedLoadIntrinsic)
                    : llvm::StringRef();
  facts.storeIntrinsic =
      !isStaticMaskStore ? llvm::StringRef(kRVVBaseMemoryStoreIntrinsic)
                         : llvm::StringRef();
  facts.stridedStoreIntrinsic =
      isStridedStore ? llvm::StringRef(kRVVBaseMemoryStridedStoreIntrinsic)
                     : llvm::StringRef();
  facts.stridedMemoryLayout = getBaseMemoryMovementStridedLayout(operation);
  facts.indexedMemoryLayout = getBaseMemoryMovementIndexedLayout(operation);
  facts.sourceMemoryForm = getBaseMemoryMovementSourceMemoryForm(operation);
  facts.destinationMemoryForm =
      getBaseMemoryMovementDestinationMemoryForm(operation);
  facts.sourceStrideSource =
      isStridedLoad ? llvm::StringRef(kRVVSourceStrideSource)
                    : llvm::StringRef();
  facts.destinationStrideSource =
      isStridedStore ? llvm::StringRef(kRVVDestinationByteStrideSource)
                     : llvm::StringRef();
  facts.sourceStrideCType =
      isStridedLoad ? llvm::StringRef(kRVVBaseMemoryStrideCType)
                    : llvm::StringRef();
  facts.destinationStrideCType =
      isStridedStore ? llvm::StringRef(kRVVBaseMemoryStrideCType)
                     : llvm::StringRef();
  facts.sourceStrideUnit =
      isStridedLoad ? llvm::StringRef(kRVVBaseMemoryStrideUnit)
                    : llvm::StringRef();
  facts.destinationStrideUnit =
      isStridedStore ? llvm::StringRef(kRVVBaseMemoryStrideUnit)
                     : llvm::StringRef();
  facts.indexEEW = isIndexed ? 32 : 0;
  facts.offsetUnit = isIndexed ? llvm::StringRef(kRVVIndexedGatherOffsetUnit)
                               : llvm::StringRef();
  facts.indexSource =
      isIndexed ? llvm::StringRef(kRVVIndexSource) : llvm::StringRef();
  facts.indexUniqueness =
      isIndexedScatter ? llvm::StringRef(kRVVIndexedScatterIndexUniqueness)
                       : llvm::StringRef();
  facts.indexedDataMemoryForm =
      isIndexedGather ? llvm::StringRef(kRVVIndexedDataMemoryForm)
                      : llvm::StringRef();
  facts.indexedDestinationMemoryForm =
      isIndexedScatter ? llvm::StringRef(kRVVIndexedDestinationMemoryForm)
                       : llvm::StringRef();
  facts.maskRole = isMasked ? llvm::StringRef(kRVVMaskRole) : llvm::StringRef();
  facts.maskSource =
      isMasked ? llvm::StringRef(kRVVMaskSource) : llvm::StringRef();
  facts.maskMemoryForm =
      isMasked ? llvm::StringRef(kRVVMaskMemoryForm) : llvm::StringRef();
  facts.inactiveLaneContract =
      isStaticMaskStore ? llvm::StringRef(kRVVMaskedStoreInactiveLaneContract)
      : isStaticMaskLoad ? llvm::StringRef(kRVVMaskedMemoryInactiveLaneContract)
                         : llvm::StringRef();
  facts.maskedPassthroughLayout =
      isStaticMaskStore ? llvm::StringRef(kRVVMaskedStorePassthroughLayout)
      : isStaticMaskLoad ? llvm::StringRef(kRVVMaskedMemoryPassthroughLayout)
                         : llvm::StringRef();
  facts.runtimeABIParameters =
      getBaseMemoryMovementRuntimeABIParameters(operation);

  RVVRouteOperandBindingPlan plan =
      buildBaseMemoryRouteOperandBindingPlanFromFacts(facts);
  facts.routeOperandBindingSummary =
      stringifyRVVRouteOperandBindingPlan(plan);
  for (const RVVRouteOperandBinding &binding : plan.bindings)
    facts.logicalOperands.push_back(binding.logicalOperand);

  return facts;
}

bool isRVVSelectedBodyBaseMemoryMovementRouteOperation(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return true;
  default:
    return false;
  }
}

bool isRVVSelectedBodyBaseMemoryMovementRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return isRVVSelectedBodyBaseMemoryMovementRouteOperation(operation);
}

llvm::Error validateRVVSelectedBodyBaseMemoryMovementRouteFamilyPlan(
    const RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "base memory movement route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyBaseMemoryMovementRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan supports only active "
        "strided, indexed, and static masked unit load/store routes");

  const bool isStridedLoad =
      plan.operation == RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  const bool isStridedStore =
      plan.operation == RVVSelectedBodyOperationKind::UnitLoadStridedStore;
  const bool isIndexedGather =
      plan.operation == RVVSelectedBodyOperationKind::IndexedGatherUnitStore;
  const bool isIndexedScatter =
      plan.operation == RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
  const bool isStaticMaskLoad =
      plan.operation == RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  const bool isStaticMaskStore =
      plan.operation == RVVSelectedBodyOperationKind::MaskedUnitStore;
  const bool isIndexed = isIndexedGather || isIndexedScatter;
  const bool isMasked = isStaticMaskLoad || isStaticMaskStore;

  if (plan.usesStridedLoad != isStridedLoad ||
      plan.usesStridedStore != isStridedStore ||
      plan.usesIndexedGather != isIndexedGather ||
      plan.usesIndexedScatter != isIndexedScatter ||
      plan.usesStaticMaskLoad != isStaticMaskLoad ||
      plan.usesStaticMaskStore != isStaticMaskStore)
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan has stale route consumer "
        "classification markers");
  if (plan.memoryForm !=
      getBaseMemoryMovementRouteFamilyMemoryForm(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan requires matching typed body "
        "memory form");
  if (llvm::Error error =
          requireBaseMemoryPlanField(plan, "runtime control plan",
                                     plan.runtimeControlPlan.controlPlanID,
                                     getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "family plan id", plan.familyPlanID,
          kRVVBaseMemoryMovementRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "runtime ABI order", plan.runtimeABIOrder,
          getBaseMemoryMovementRuntimeABIOrder(plan.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "target leaf profile", plan.targetLeafProfile,
          getBaseMemoryMovementTargetLeafProfile(plan.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "provider_supported_mirror", plan.providerSupportedMirror,
          getBaseMemoryMovementProviderSupportedMirror(plan.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "header declarations", plan.requiredHeaderDeclarations,
          kRVVBaseMemoryMovementRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "C type mapping summary", plan.cTypeMappingSummary,
          getBaseMemoryMovementCTypeMappingSummary(plan.operation)))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan requires provider-owned "
        "header declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (plan.typedConfigFactsID.empty() || plan.elementTypeName.empty() ||
      plan.elementBitWidth == 0 || plan.sew == 0 || plan.lmul.empty() ||
      plan.tailPolicy.empty() || plan.maskPolicy.empty() ||
      plan.configContractID.empty())
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan requires provider-derived "
        "typed config facts for element type, SEW, LMUL, policy, and config "
        "contract");
  if (plan.elementBitWidth != plan.sew)
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan requires element bit width "
        "to mirror provider-derived SEW");
  if (plan.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      plan.lmul != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan currently supports only "
        "typed SEW32 LMUL m1 data config");
  if (plan.sew != plan.runtimeControlPlan.sew ||
      plan.lmul != plan.runtimeControlPlan.lmul ||
      plan.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      plan.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      plan.configContractID != plan.runtimeControlPlan.configContractID)
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan requires typed config "
        "SEW/LMUL/policy/contract facts to mirror runtime AVL/VL control "
        "facts");
  if (llvm::Error error =
          requireBaseMemoryPlanField(plan, "VL C type", plan.vlCType, "size_t"))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "vector type", plan.vectorTypeName, true))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "vector C type", plan.vectorCType, true))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "index vector type", plan.indexVectorTypeName, isIndexed))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "index vector C type", plan.indexVectorCType, isIndexed))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "mask type", plan.maskTypeName, isMasked))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "mask C type", plan.maskCType, isMasked))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "setvl leaf", plan.setVLIntrinsic, true))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "vector-load leaf", plan.vectorLoadIntrinsic, true))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "index-load leaf", plan.indexLoadIntrinsic, isIndexed))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "index-scale leaf", plan.indexScaleIntrinsic, isIndexed))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "indexed-load leaf", plan.indexedLoadIntrinsic,
          isIndexedGather))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "indexed-store leaf", plan.indexedStoreIntrinsic,
          isIndexedScatter))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "strided-load leaf", plan.stridedLoadIntrinsic, isStridedLoad))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "masked-load leaf", plan.maskedLoadIntrinsic,
          isStaticMaskLoad))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "store leaf", plan.storeIntrinsic, true))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanDerivedField(
          plan, "strided-store leaf", plan.stridedStoreIntrinsic,
          isStridedStore))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "result name", plan.resultName,
          getBaseMemoryMovementResultName(plan.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "mask name", plan.maskName,
          getBaseMemoryMovementMaskName(plan.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "mask role", plan.maskRole,
          isMasked ? llvm::StringRef(kRVVMaskRole) : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "mask source", plan.maskSource,
          isMasked ? llvm::StringRef(kRVVMaskSource) : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "mask memory form", plan.maskMemoryForm,
          isMasked ? llvm::StringRef(kRVVMaskMemoryForm)
                   : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "inactive-lane contract", plan.inactiveLaneContract,
          isStaticMaskStore
              ? llvm::StringRef(kRVVMaskedStoreInactiveLaneContract)
          : isStaticMaskLoad
              ? llvm::StringRef(kRVVMaskedMemoryInactiveLaneContract)
              : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "masked passthrough layout", plan.maskedPassthroughLayout,
          isStaticMaskStore
              ? llvm::StringRef(kRVVMaskedStorePassthroughLayout)
          : isStaticMaskLoad
              ? llvm::StringRef(kRVVMaskedMemoryPassthroughLayout)
              : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "strided memory layout", plan.stridedMemoryLayout,
          getBaseMemoryMovementStridedLayout(plan.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "indexed memory layout", plan.indexedMemoryLayout,
          getBaseMemoryMovementIndexedLayout(plan.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "source memory form", plan.sourceMemoryForm,
          getBaseMemoryMovementSourceMemoryForm(plan.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "destination memory form", plan.destinationMemoryForm,
          getBaseMemoryMovementDestinationMemoryForm(plan.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "source stride source", plan.sourceStrideSource,
          isStridedLoad ? llvm::StringRef(kRVVSourceStrideSource)
                        : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "destination stride source", plan.destinationStrideSource,
          isStridedStore ? llvm::StringRef(kRVVDestinationByteStrideSource)
                         : llvm::StringRef()))
    return error;
  if (plan.indexEEW != (isIndexed ? 32 : 0))
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan requires index EEW to mirror "
        "the selected indexed memory consumer");
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "offset unit", plan.offsetUnit,
          isIndexed ? llvm::StringRef(kRVVIndexedGatherOffsetUnit)
                    : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "index source", plan.indexSource,
          isIndexed ? llvm::StringRef(kRVVIndexSource) : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "index uniqueness", plan.indexUniqueness,
          isIndexedScatter ? llvm::StringRef(kRVVIndexedScatterIndexUniqueness)
                           : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "indexed data memory form", plan.indexedDataMemoryForm,
          isIndexedGather ? llvm::StringRef(kRVVIndexedDataMemoryForm)
                          : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryPlanField(
          plan, "indexed destination memory form",
          plan.indexedDestinationMemoryForm,
          isIndexedScatter ? llvm::StringRef(kRVVIndexedDestinationMemoryForm)
                           : llvm::StringRef()))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan>
deriveRVVSelectedBodyBaseMemoryMovementRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyOperationKind operation = analysis.slice.arithmeticKind;
  if (!isRVVSelectedBodyBaseMemoryMovementRouteOperation(operation))
    return makeRVVEmitCRouteProviderError(
        "requested base memory movement route-family plan for non-base-memory "
        "RVV operation");
  if (analysis.slice.memoryForm !=
      getBaseMemoryMovementRouteFamilyMemoryForm(operation))
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan requires matching typed body "
        "memory form");
  if (!analysis.typedConfigFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan requires typed RVV config "
        "facts before deriving memory route facts");

  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      analysis.typedConfigFacts;
  const bool isStridedLoad =
      operation == RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  const bool isStridedStore =
      operation == RVVSelectedBodyOperationKind::UnitLoadStridedStore;
  const bool isIndexedGather =
      operation == RVVSelectedBodyOperationKind::IndexedGatherUnitStore;
  const bool isIndexedScatter =
      operation == RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
  const bool isStaticMaskLoad =
      operation == RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  const bool isStaticMaskStore =
      operation == RVVSelectedBodyOperationKind::MaskedUnitStore;
  const bool isIndexed = isIndexedGather || isIndexedScatter;
  const bool isMasked = isStaticMaskLoad || isStaticMaskStore;

  if (typedFacts.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      typedFacts.lmul != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan currently supports only "
        "typed SEW32 LMUL m1 vector configuration facts");
  if (llvm::Error error = requireBaseMemoryTypedConfigLeaves(
          typedFacts, operation, isIndexed, isIndexedGather, isIndexedScatter,
          isStridedLoad, isStridedStore, isStaticMaskLoad,
          isStaticMaskStore))
    return std::move(error);
  if (isStridedLoad &&
      (!analysis.slice.lhsStridedLoad || !analysis.slice.genericStore ||
       !analysis.slice.moveOp ||
       analysis.slice.lhsStrideABI.role !=
           support::RuntimeABIParameterRole::SourceByteStride ||
       analysis.slice.lhsABI.role !=
           support::RuntimeABIParameterRole::SourceInputBuffer ||
       analysis.slice.outABI.role !=
           support::RuntimeABIParameterRole::OutputBuffer))
    return makeRVVEmitCRouteProviderError(
        "base memory movement strided-load route-family plan requires a "
        "strided source load, move, unit store, source buffer, output buffer, "
        "and source stride ABI roles");
  if (isStridedStore &&
      (!analysis.slice.lhsGenericLoad || !analysis.slice.stridedStore ||
       !analysis.slice.moveOp ||
       analysis.slice.lhsABI.role !=
           support::RuntimeABIParameterRole::LHSInputBuffer ||
       analysis.slice.outABI.role !=
           support::RuntimeABIParameterRole::OutputBuffer ||
       analysis.slice.outStrideABI.role !=
           support::RuntimeABIParameterRole::DestinationByteStride))
    return makeRVVEmitCRouteProviderError(
        "base memory movement strided-store route-family plan requires a unit "
        "source load, move, strided store, source buffer, output buffer, and "
        "destination stride ABI roles");
  if (isIndexedGather &&
      (!analysis.slice.indexLoadOperation || !analysis.slice.indexedLoad ||
       !analysis.slice.genericStore ||
       analysis.slice.lhsABI.role !=
           support::RuntimeABIParameterRole::LHSInputBuffer ||
       analysis.slice.indexABI.role !=
           support::RuntimeABIParameterRole::IndexInputBuffer ||
       analysis.slice.outABI.role !=
           support::RuntimeABIParameterRole::OutputBuffer))
    return makeRVVEmitCRouteProviderError(
        "base memory movement indexed-gather route-family plan requires an "
        "index load, indexed data load, unit store, data/index/output ABI "
        "roles, and runtime AVL");
  if (isIndexedScatter &&
      (!analysis.slice.lhsGenericLoad || !analysis.slice.indexLoadOperation ||
       !analysis.slice.indexedStore ||
       analysis.slice.lhsABI.role !=
           support::RuntimeABIParameterRole::LHSInputBuffer ||
       analysis.slice.indexABI.role !=
           support::RuntimeABIParameterRole::IndexInputBuffer ||
       analysis.slice.outABI.role !=
           support::RuntimeABIParameterRole::OutputBuffer))
    return makeRVVEmitCRouteProviderError(
        "base memory movement indexed-scatter route-family plan requires a "
        "unit source load, index load, indexed store, source/index/output ABI "
        "roles, and runtime AVL");
  if (isStaticMaskLoad &&
      (!analysis.slice.maskLoadOperation || !analysis.slice.maskedLoadOp ||
       !analysis.slice.accumulatorLoadOperation ||
       !analysis.slice.genericStore ||
       analysis.slice.lhsABI.role !=
           support::RuntimeABIParameterRole::LHSInputBuffer ||
       analysis.slice.maskABI.role !=
           support::RuntimeABIParameterRole::MaskInputBuffer ||
       analysis.slice.outABI.role !=
           support::RuntimeABIParameterRole::OutputBuffer))
    return makeRVVEmitCRouteProviderError(
        "base memory movement masked load-store route-family plan requires "
        "unit source/old-destination/mask loads, masked load, unit store, and "
        "source/mask/output ABI roles");
  if (isStaticMaskStore &&
      (!analysis.slice.lhsGenericLoad || !analysis.slice.maskLoadOperation ||
       !analysis.slice.maskedStore ||
       analysis.slice.lhsABI.role !=
           support::RuntimeABIParameterRole::LHSInputBuffer ||
       analysis.slice.maskABI.role !=
           support::RuntimeABIParameterRole::MaskInputBuffer ||
       analysis.slice.outABI.role !=
           support::RuntimeABIParameterRole::OutputBuffer))
    return makeRVVEmitCRouteProviderError(
        "base memory movement masked-store route-family plan requires a unit "
        "source load, mask load, masked store, and source/mask/output ABI "
        "roles");
  if (analysis.slice.runtimeElementCountABI.role !=
      support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "base memory movement route-family plan requires runtime element-count "
        "ABI role for AVL/VL control");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          getBaseMemoryMovementRuntimeABIOrder(operation),
          "base memory movement route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesStridedLoad = isStridedLoad;
  plan.usesStridedStore = isStridedStore;
  plan.usesIndexedGather = isIndexedGather;
  plan.usesIndexedScatter = isIndexedScatter;
  plan.usesStaticMaskLoad = isStaticMaskLoad;
  plan.usesStaticMaskStore = isStaticMaskStore;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.familyPlanID = kRVVBaseMemoryMovementRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = getBaseMemoryMovementTargetLeafProfile(operation);
  plan.providerSupportedMirror =
      getBaseMemoryMovementProviderSupportedMirror(operation);
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations =
      kRVVBaseMemoryMovementRequiredHeaderDeclarations;
  plan.cTypeMappingSummary = getBaseMemoryMovementCTypeMappingSummary(operation);
  plan.typedConfigFactsID = typedFacts.factsID;
  plan.elementTypeName = typedFacts.elementTypeName;
  plan.elementBitWidth = typedFacts.elementBitWidth;
  plan.sew = typedFacts.sew;
  plan.lmul = typedFacts.lmul;
  plan.tailPolicy = typedFacts.tailPolicy;
  plan.maskPolicy = typedFacts.maskPolicy;
  plan.configContractID = typedFacts.configContractID;
  plan.vlCType = typedFacts.vlCType;
  plan.vectorTypeName = typedFacts.vectorTypeName;
  plan.vectorCType = typedFacts.vectorCType;
  plan.indexVectorTypeName = isIndexed ? typedFacts.indexVectorTypeName : "";
  plan.indexVectorCType = isIndexed ? typedFacts.indexVectorCType : "";
  plan.maskTypeName = isMasked ? typedFacts.maskTypeName : "";
  plan.maskCType = isMasked ? typedFacts.maskCType : "";
  plan.setVLIntrinsic = typedFacts.setVLIntrinsic;
  plan.vectorLoadIntrinsic = typedFacts.vectorLoadIntrinsic;
  plan.indexLoadIntrinsic = isIndexed ? typedFacts.indexLoadIntrinsic : "";
  plan.indexScaleIntrinsic = isIndexed ? typedFacts.indexScaleIntrinsic : "";
  plan.indexedLoadIntrinsic =
      isIndexedGather ? typedFacts.indexedLoadIntrinsic : "";
  plan.indexedStoreIntrinsic =
      isIndexedScatter ? typedFacts.indexedStoreIntrinsic : "";
  plan.stridedLoadIntrinsic =
      isStridedLoad ? typedFacts.stridedLoadIntrinsic : "";
  plan.maskedLoadIntrinsic =
      isStaticMaskLoad ? typedFacts.maskedLoadIntrinsic : "";
  plan.storeIntrinsic =
      isStaticMaskStore ? typedFacts.maskedStoreIntrinsic
                        : typedFacts.storeIntrinsic;
  plan.stridedStoreIntrinsic =
      isStridedStore ? typedFacts.stridedStoreIntrinsic : llvm::StringRef();
  plan.resultName = getBaseMemoryMovementResultName(operation);
  plan.maskName = getBaseMemoryMovementMaskName(operation);
  plan.maskRole = isMasked ? kRVVMaskRole : "";
  plan.maskSource = isMasked ? kRVVMaskSource : "";
  plan.maskMemoryForm = isMasked ? kRVVMaskMemoryForm : "";
  plan.inactiveLaneContract =
      isStaticMaskStore ? llvm::StringRef(kRVVMaskedStoreInactiveLaneContract)
      : isStaticMaskLoad ? llvm::StringRef(kRVVMaskedMemoryInactiveLaneContract)
                         : llvm::StringRef();
  plan.maskedPassthroughLayout =
      isStaticMaskStore ? llvm::StringRef(kRVVMaskedStorePassthroughLayout)
      : isStaticMaskLoad ? llvm::StringRef(kRVVMaskedMemoryPassthroughLayout)
                         : llvm::StringRef();
  plan.stridedMemoryLayout = getBaseMemoryMovementStridedLayout(operation);
  plan.indexedMemoryLayout = getBaseMemoryMovementIndexedLayout(operation);
  plan.sourceMemoryForm = getBaseMemoryMovementSourceMemoryForm(operation);
  plan.destinationMemoryForm = getBaseMemoryMovementDestinationMemoryForm(operation);
  plan.sourceStrideSource =
      isStridedLoad ? llvm::StringRef(kRVVSourceStrideSource)
                    : llvm::StringRef();
  plan.destinationStrideSource =
      isStridedStore ? llvm::StringRef(kRVVDestinationByteStrideSource)
                     : llvm::StringRef();
  plan.indexEEW =
      isIndexed
          ? static_cast<std::int64_t>(analysis.slice.indexLoad.getIndexEew())
          : 0;
  plan.offsetUnit =
      isIndexedGather ? analysis.slice.indexedLoad.getOffsetUnit()
      : isIndexedScatter ? analysis.slice.indexedStore.getOffsetUnit()
                         : llvm::StringRef();
  plan.indexSource = isIndexed ? llvm::StringRef(kRVVIndexSource)
                               : llvm::StringRef();
  plan.indexUniqueness =
      isIndexedScatter ? analysis.slice.indexedStore.getIndexUniqueness()
                       : llvm::StringRef();
  plan.indexedDataMemoryForm =
      isIndexedGather ? llvm::StringRef(kRVVIndexedDataMemoryForm)
                      : llvm::StringRef();
  plan.indexedDestinationMemoryForm =
      isIndexedScatter ? llvm::StringRef(kRVVIndexedDestinationMemoryForm)
                       : llvm::StringRef();
  if (isStridedLoad) {
    plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    plan.runtimeABIParameters.push_back(analysis.slice.outABI);
    plan.runtimeABIParameters.push_back(
        plan.runtimeControlPlan.runtimeAVLParameter);
    plan.runtimeABIParameters.push_back(analysis.slice.lhsStrideABI);
  } else if (isStridedStore) {
    plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    plan.runtimeABIParameters.push_back(analysis.slice.outABI);
    plan.runtimeABIParameters.push_back(
        plan.runtimeControlPlan.runtimeAVLParameter);
    plan.runtimeABIParameters.push_back(analysis.slice.outStrideABI);
  } else if (isIndexedGather || isIndexedScatter) {
    plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    plan.runtimeABIParameters.push_back(analysis.slice.indexABI);
    plan.runtimeABIParameters.push_back(analysis.slice.outABI);
    plan.runtimeABIParameters.push_back(
        plan.runtimeControlPlan.runtimeAVLParameter);
  } else {
    plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    plan.runtimeABIParameters.push_back(analysis.slice.maskABI);
    plan.runtimeABIParameters.push_back(analysis.slice.outABI);
    plan.runtimeABIParameters.push_back(
        plan.runtimeControlPlan.runtimeAVLParameter);
  }

  if (llvm::Error error =
          validateRVVSelectedBodyBaseMemoryMovementRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyBaseMemoryMovementRouteFamilyPlan(
    const RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyBaseMemoryRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                                      description);
  description.elementTypeName = plan.elementTypeName;
  description.sew = plan.sew;
  description.lmul = plan.lmul;
  description.tailPolicy = plan.tailPolicy;
  description.maskPolicy = plan.maskPolicy;
  description.configContractID = plan.configContractID;
  description.baseMemoryMovementRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.indexVectorTypeName = plan.indexVectorTypeName;
  description.indexVectorCType = plan.indexVectorCType;
  description.maskTypeName = plan.maskTypeName;
  description.maskCType = plan.maskCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.vectorLoadIntrinsic = plan.vectorLoadIntrinsic;
  description.indexLoadIntrinsic = plan.indexLoadIntrinsic;
  description.indexScaleIntrinsic = plan.indexScaleIntrinsic;
  description.indexedLoadIntrinsic = plan.indexedLoadIntrinsic;
  description.indexedStoreIntrinsic = plan.indexedStoreIntrinsic;
  description.stridedLoadIntrinsic = plan.stridedLoadIntrinsic;
  description.maskedLoadIntrinsic = plan.maskedLoadIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.stridedStoreIntrinsic = plan.stridedStoreIntrinsic;
  description.intrinsic = plan.maskedLoadIntrinsic;
  description.resultName = plan.resultName;
  description.maskName = plan.maskName;
  description.maskRole = plan.maskRole;
  description.maskSource = plan.maskSource;
  description.maskMemoryForm = plan.maskMemoryForm;
  description.inactiveLaneContract = plan.inactiveLaneContract;
  description.maskedPassthroughLayout = plan.maskedPassthroughLayout;
  description.stridedMemoryLayout = plan.stridedMemoryLayout;
  description.indexedMemoryLayout = plan.indexedMemoryLayout;
  description.sourceMemoryForm = plan.sourceMemoryForm;
  description.destinationMemoryForm = plan.destinationMemoryForm;
  description.sourceStrideSource = plan.sourceStrideSource;
  description.outStrideSource = plan.destinationStrideSource;
  description.indexEEW = plan.indexEEW;
  description.offsetUnit = plan.offsetUnit;
  description.indexSource = plan.indexSource;
  description.indexUniqueness = plan.indexUniqueness;
  description.indexedDataMemoryForm = plan.indexedDataMemoryForm;
  description.indexedDestinationMemoryForm =
      plan.indexedDestinationMemoryForm;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

llvm::Error verifyRVVSelectedBodyBaseMemoryMovementRouteDescriptionMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  const bool isConsumer =
      isRVVSelectedBodyBaseMemoryMovementRouteFamilyConsumer(
          description.operation);
  if (!isConsumer)
    return requireBaseMemoryRouteDescriptionField(
        context, "base memory movement route family plan",
        description.baseMemoryMovementRouteFamilyPlanID, "");

  const bool isStridedLoad =
      description.operation == RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  const bool isStridedStore =
      description.operation == RVVSelectedBodyOperationKind::UnitLoadStridedStore;
  const bool isIndexedGather =
      description.operation == RVVSelectedBodyOperationKind::IndexedGatherUnitStore;
  const bool isIndexedScatter =
      description.operation ==
      RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
  const bool isStaticMaskLoad =
      description.operation == RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  const bool isStaticMaskStore =
      description.operation == RVVSelectedBodyOperationKind::MaskedUnitStore;
  const bool isIndexed = isIndexedGather || isIndexedScatter;
  const bool isMasked = isStaticMaskLoad || isStaticMaskStore;

  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "base memory movement route family plan",
          description.baseMemoryMovementRouteFamilyPlanID,
          kRVVBaseMemoryMovementRouteFamilyPlanID))
    return error;
  if (description.memoryForm !=
      getBaseMemoryMovementRouteFamilyMemoryForm(description.operation))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " memory form must mirror selected base memory movement owner facts "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "runtime ABI order", description.runtimeABIOrder,
          getBaseMemoryMovementRuntimeABIOrder(description.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "target leaf profile", description.targetLeafProfile,
          getBaseMemoryMovementTargetLeafProfile(description.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "provider_supported_mirror",
          description.providerSupportedMirror,
          getBaseMemoryMovementProviderSupportedMirror(description.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "required header declarations",
          description.requiredHeaderDeclarations,
          kRVVBaseMemoryMovementRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "C type mapping summary", description.cTypeMappingSummary,
          getBaseMemoryMovementCTypeMappingSummary(description.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "element type", description.elementTypeName, true))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "vector type", description.vectorTypeName, true))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "vector C type", description.vectorCType, true))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "index vector type", description.indexVectorTypeName,
          isIndexed))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "index vector C type", description.indexVectorCType,
          isIndexed))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "mask type", description.maskTypeName, isMasked))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "mask C type", description.maskCType, isMasked))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "setvl leaf", description.setVLIntrinsic, true))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "vector-load leaf", description.vectorLoadIntrinsic, true))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "index-load leaf", description.indexLoadIntrinsic,
          isIndexed))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "index-scale leaf", description.indexScaleIntrinsic,
          isIndexed))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "indexed-load leaf", description.indexedLoadIntrinsic,
          isIndexedGather))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "indexed-store leaf", description.indexedStoreIntrinsic,
          isIndexedScatter))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "strided-load leaf", description.stridedLoadIntrinsic,
          isStridedLoad))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "masked-load leaf", description.maskedLoadIntrinsic,
          isStaticMaskLoad))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "store leaf", description.storeIntrinsic, true))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionDerivedField(
          context, "strided-store leaf", description.stridedStoreIntrinsic,
          isStridedStore))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "result name", description.resultName,
          getBaseMemoryMovementResultName(description.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "mask name", description.maskName,
          getBaseMemoryMovementMaskName(description.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "mask role", description.maskRole,
          isMasked ? llvm::StringRef(kRVVMaskRole) : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "mask source", description.maskSource,
          isMasked ? llvm::StringRef(kRVVMaskSource) : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "mask memory form", description.maskMemoryForm,
          isMasked ? llvm::StringRef(kRVVMaskMemoryForm)
                   : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "inactive-lane contract", description.inactiveLaneContract,
          isStaticMaskStore
              ? llvm::StringRef(kRVVMaskedStoreInactiveLaneContract)
          : isStaticMaskLoad
              ? llvm::StringRef(kRVVMaskedMemoryInactiveLaneContract)
              : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "masked passthrough layout",
          description.maskedPassthroughLayout,
          isStaticMaskStore
              ? llvm::StringRef(kRVVMaskedStorePassthroughLayout)
          : isStaticMaskLoad
              ? llvm::StringRef(kRVVMaskedMemoryPassthroughLayout)
              : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "strided memory layout", description.stridedMemoryLayout,
          getBaseMemoryMovementStridedLayout(description.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "indexed memory layout", description.indexedMemoryLayout,
          getBaseMemoryMovementIndexedLayout(description.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "lhs stride source", description.lhsStrideSource, ""))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "rhs stride source", description.rhsStrideSource, ""))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "out stride source", description.outStrideSource,
          isStridedStore ? llvm::StringRef(kRVVDestinationByteStrideSource)
                         : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "source stride source", description.sourceStrideSource,
          isStridedLoad ? llvm::StringRef(kRVVSourceStrideSource)
                        : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "source memory form", description.sourceMemoryForm,
          getBaseMemoryMovementSourceMemoryForm(description.operation)))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "destination memory form",
          description.destinationMemoryForm,
          getBaseMemoryMovementDestinationMemoryForm(description.operation)))
    return error;
  if (description.indexEEW != (isIndexed ? 32 : 0))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " index EEW must mirror base memory movement indexed route facts");
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "offset unit", description.offsetUnit,
          isIndexed ? llvm::StringRef(kRVVIndexedGatherOffsetUnit)
                    : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "index source", description.indexSource,
          isIndexed ? llvm::StringRef(kRVVIndexSource) : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "index uniqueness", description.indexUniqueness,
          isIndexedScatter ? llvm::StringRef(kRVVIndexedScatterIndexUniqueness)
                           : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireBaseMemoryRouteDescriptionField(
          context, "indexed data memory form",
          description.indexedDataMemoryForm,
          isIndexedGather ? llvm::StringRef(kRVVIndexedDataMemoryForm)
                          : llvm::StringRef()))
    return error;
  return requireBaseMemoryRouteDescriptionField(
      context, "indexed destination memory form",
      description.indexedDestinationMemoryForm,
      isIndexedScatter ? llvm::StringRef(kRVVIndexedDestinationMemoryForm)
                       : llvm::StringRef());
}

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyBaseMemoryRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return kRVVStridedLoadUnitStoreOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return kRVVUnitLoadStridedStoreOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return kRVVIndexedGatherOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return kRVVIndexedScatterOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return kRVVMaskedUnitLoadStoreOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return kRVVMaskedUnitStoreOperandBindingPlanID;
  default:
    return std::nullopt;
  }
}

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodyBaseMemoryRouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand) {
  using support::RuntimeABIParameterRole;
  if (planID == kRVVStridedLoadUnitStoreOperandBindingPlanID) {
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (logicalOperand == "stride_bytes")
      return RuntimeABIParameterRole::SourceByteStride;
  }
  if (planID == kRVVUnitLoadStridedStoreOperandBindingPlanID) {
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (logicalOperand == "dst_stride_bytes")
      return RuntimeABIParameterRole::DestinationByteStride;
  }
  if (planID == kRVVIndexedGatherOperandBindingPlanID) {
    if (logicalOperand == "data")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "index")
      return RuntimeABIParameterRole::IndexInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVIndexedScatterOperandBindingPlanID) {
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "index")
      return RuntimeABIParameterRole::IndexInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVMaskedUnitLoadStoreOperandBindingPlanID ||
      planID == kRVVMaskedUnitStoreOperandBindingPlanID) {
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "mask")
      return RuntimeABIParameterRole::MaskInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  return std::nullopt;
}

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyBaseMemoryRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis) {
  return deriveBaseMemoryRouteOperandBindingPlanImpl(analysis);
}

llvm::Error verifyRVVSelectedBodyBaseMemoryMovementRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  const RVVSelectedBodyOperationKind operation = analysis.description.operation;
  const bool isConsumer =
      isRVVSelectedBodyBaseMemoryMovementRouteFamilyConsumer(operation);
  if (isConsumer && !analysis.baseMemoryMovementRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires the base memory movement route-family plan before provider "
        "materialization for operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!isConsumer && analysis.baseMemoryMovementRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must not carry a base memory movement route-family plan for "
        "non-base-memory operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!analysis.baseMemoryMovementRouteFamilyPlan)
    return llvm::Error::success();

  const RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan &plan =
      *analysis.baseMemoryMovementRouteFamilyPlan;
  if (llvm::Error error =
          validateRVVSelectedBodyBaseMemoryMovementRouteFamilyPlan(plan))
    return error;
  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      analysis.typedConfigFacts;
  if (!typedFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement provider requires typed config facts before "
        "provider materialization");
  const bool isStridedLoad =
      operation == RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  const bool isStridedStore =
      operation == RVVSelectedBodyOperationKind::UnitLoadStridedStore;
  const bool isIndexedGather =
      operation == RVVSelectedBodyOperationKind::IndexedGatherUnitStore;
  const bool isIndexedScatter =
      operation == RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
  const bool isStaticMaskLoad =
      operation == RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  const bool isStaticMaskStore =
      operation == RVVSelectedBodyOperationKind::MaskedUnitStore;
  const bool isIndexed = isIndexedGather || isIndexedScatter;
  const bool isMasked = isStaticMaskLoad || isStaticMaskStore;
  std::optional<RVVBaseMemoryMovementRouteFacts> routeFacts =
      getRVVBaseMemoryMovementRouteFacts(operation);
  if (!routeFacts)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement provider requires canonical route facts for "
        "the selected operation before provider materialization");
  if (plan.memoryForm != routeFacts->memoryForm ||
      plan.sew != routeFacts->sew || plan.lmul != routeFacts->lmul ||
      plan.tailPolicy != routeFacts->tailPolicy ||
      plan.maskPolicy != routeFacts->maskPolicy ||
      plan.familyPlanID != routeFacts->routeFamilyPlanID ||
      plan.runtimeControlPlan.controlPlanID !=
          routeFacts->runtimeControlPlanID ||
      plan.runtimeABIOrder != routeFacts->runtimeABIOrder ||
      plan.targetLeafProfile != routeFacts->targetLeafProfile ||
      plan.providerSupportedMirror != routeFacts->providerSupportedMirror ||
      plan.requiredHeaderDeclarations !=
          routeFacts->requiredHeaderDeclarations ||
      plan.cTypeMappingSummary != routeFacts->cTypeMappingSummary ||
      plan.vlCType != routeFacts->vlCType ||
      plan.vectorTypeName != routeFacts->vectorTypeName ||
      plan.vectorCType != routeFacts->vectorCType ||
      plan.setVLIntrinsic != routeFacts->setVLIntrinsic ||
      plan.vectorLoadIntrinsic != routeFacts->vectorLoadIntrinsic ||
      plan.stridedLoadIntrinsic != routeFacts->stridedLoadIntrinsic ||
      (!isStaticMaskStore &&
       plan.storeIntrinsic != routeFacts->storeIntrinsic) ||
      plan.stridedStoreIntrinsic != routeFacts->stridedStoreIntrinsic ||
      plan.stridedMemoryLayout != routeFacts->stridedMemoryLayout ||
      plan.indexedMemoryLayout != routeFacts->indexedMemoryLayout ||
      plan.sourceMemoryForm != routeFacts->sourceMemoryForm ||
      plan.destinationMemoryForm != routeFacts->destinationMemoryForm ||
      plan.sourceStrideSource != routeFacts->sourceStrideSource ||
      plan.destinationStrideSource != routeFacts->destinationStrideSource ||
      plan.indexEEW != routeFacts->indexEEW ||
      plan.offsetUnit != routeFacts->offsetUnit ||
      plan.indexSource != routeFacts->indexSource ||
      plan.indexUniqueness != routeFacts->indexUniqueness ||
      plan.indexedDataMemoryForm != routeFacts->indexedDataMemoryForm ||
      plan.indexedDestinationMemoryForm !=
          routeFacts->indexedDestinationMemoryForm ||
      plan.maskRole != routeFacts->maskRole ||
      plan.maskSource != routeFacts->maskSource ||
      plan.maskMemoryForm != routeFacts->maskMemoryForm ||
      plan.inactiveLaneContract != routeFacts->inactiveLaneContract ||
      plan.maskedPassthroughLayout != routeFacts->maskedPassthroughLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement route-family plan must mirror canonical "
        "provider-owned memory/index/mask facts before provider "
        "materialization");
  const llvm::StringRef expectedIndexVectorType =
      isIndexed ? typedFacts.indexVectorTypeName : llvm::StringRef();
  const llvm::StringRef expectedIndexVectorCType =
      isIndexed ? typedFacts.indexVectorCType : llvm::StringRef();
  const llvm::StringRef expectedMaskType =
      isMasked ? typedFacts.maskTypeName : llvm::StringRef();
  const llvm::StringRef expectedMaskCType =
      isMasked ? typedFacts.maskCType : llvm::StringRef();
  const llvm::StringRef expectedIndexLoadIntrinsic =
      isIndexed ? typedFacts.indexLoadIntrinsic : llvm::StringRef();
  const llvm::StringRef expectedIndexScaleIntrinsic =
      isIndexed ? typedFacts.indexScaleIntrinsic : llvm::StringRef();
  const llvm::StringRef expectedIndexedLoadIntrinsic =
      isIndexedGather ? typedFacts.indexedLoadIntrinsic : llvm::StringRef();
  const llvm::StringRef expectedIndexedStoreIntrinsic =
      isIndexedScatter ? typedFacts.indexedStoreIntrinsic : llvm::StringRef();
  const llvm::StringRef expectedStridedLoadIntrinsic =
      isStridedLoad ? typedFacts.stridedLoadIntrinsic : llvm::StringRef();
  const llvm::StringRef expectedMaskedLoadIntrinsic =
      isStaticMaskLoad ? typedFacts.maskedLoadIntrinsic : llvm::StringRef();
  const llvm::StringRef expectedStoreIntrinsic =
      isStaticMaskStore ? typedFacts.maskedStoreIntrinsic
                        : typedFacts.storeIntrinsic;
  const llvm::StringRef expectedStridedStoreIntrinsic =
      isStridedStore ? typedFacts.stridedStoreIntrinsic : llvm::StringRef();
  if (plan.typedConfigFactsID != typedFacts.factsID ||
      plan.elementTypeName != typedFacts.elementTypeName ||
      plan.elementBitWidth != typedFacts.elementBitWidth ||
      plan.sew != typedFacts.sew || plan.lmul != typedFacts.lmul ||
      plan.tailPolicy != typedFacts.tailPolicy ||
      plan.maskPolicy != typedFacts.maskPolicy ||
      plan.configContractID != typedFacts.configContractID ||
      plan.vlCType != typedFacts.vlCType ||
      plan.vectorTypeName != typedFacts.vectorTypeName ||
      plan.vectorCType != typedFacts.vectorCType ||
      plan.indexVectorTypeName != expectedIndexVectorType ||
      plan.indexVectorCType != expectedIndexVectorCType ||
      plan.maskTypeName != expectedMaskType ||
      plan.maskCType != expectedMaskCType ||
      plan.setVLIntrinsic != typedFacts.setVLIntrinsic ||
      plan.vectorLoadIntrinsic != typedFacts.vectorLoadIntrinsic ||
      plan.indexLoadIntrinsic != expectedIndexLoadIntrinsic ||
      plan.indexScaleIntrinsic != expectedIndexScaleIntrinsic ||
      plan.indexedLoadIntrinsic != expectedIndexedLoadIntrinsic ||
      plan.indexedStoreIntrinsic != expectedIndexedStoreIntrinsic ||
      plan.stridedLoadIntrinsic != expectedStridedLoadIntrinsic ||
      plan.maskedLoadIntrinsic != expectedMaskedLoadIntrinsic ||
      plan.storeIntrinsic != expectedStoreIntrinsic ||
      plan.stridedStoreIntrinsic != expectedStridedStoreIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement route-family typed config snapshot must "
        "mirror selected typed RVV body/config facts before provider "
        "materialization");
  if (plan.operation != operation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement route-family plan operation must match the "
        "selected route description");
  if (analysis.description.baseMemoryMovementRouteFamilyPlanID !=
      plan.familyPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement route-family plan mirror must match the "
        "validated family plan");
  if (analysis.description.memoryForm != plan.memoryForm ||
      analysis.description.elementTypeName != plan.elementTypeName ||
      analysis.description.sew != plan.sew ||
      analysis.description.lmul != plan.lmul ||
      analysis.description.tailPolicy != plan.tailPolicy ||
      analysis.description.maskPolicy != plan.maskPolicy ||
      analysis.description.configContractID != plan.configContractID ||
      analysis.description.sew != plan.runtimeControlPlan.sew ||
      analysis.description.lmul != plan.runtimeControlPlan.lmul ||
      analysis.description.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      analysis.description.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      analysis.description.runtimeControlPlanID !=
          plan.runtimeControlPlan.controlPlanID ||
      analysis.description.configContractID !=
          plan.runtimeControlPlan.configContractID ||
      analysis.description.runtimeVLContractID !=
          plan.runtimeControlPlan.runtimeVLContractID ||
      analysis.description.runtimeAVLASource !=
          plan.runtimeControlPlan.runtimeAVLASource ||
      analysis.description.runtimeABIOrder != plan.runtimeABIOrder ||
      analysis.description.vlDefOpName != plan.runtimeControlPlan.vlDefOpName ||
      analysis.description.vlScopeOpName !=
          plan.runtimeControlPlan.vlScopeOpName ||
      analysis.description.vlUses != plan.runtimeControlPlan.vlUses ||
      analysis.description.emitCLoopKind !=
          plan.runtimeControlPlan.emitCLoopKind ||
      analysis.description.emitCLoopInductionName !=
          plan.runtimeControlPlan.emitCLoopInductionName ||
      analysis.description.emitCFullChunkVLName !=
          plan.runtimeControlPlan.emitCFullChunkVLName ||
      analysis.description.emitCLoopVLName !=
          plan.runtimeControlPlan.emitCLoopVLName ||
      analysis.description.remainingAVLMetadata !=
          plan.runtimeControlPlan.remainingAVLMetadata ||
      analysis.description.pointerAdvanceMetadata !=
          plan.runtimeControlPlan.pointerAdvanceMetadata ||
      analysis.description.boundedSlice != plan.runtimeControlPlan.boundedSlice ||
      analysis.description.multiVL != plan.runtimeControlPlan.multiVL ||
      analysis.description.targetLeafProfile != plan.targetLeafProfile ||
      analysis.description.providerSupportedMirror !=
          plan.providerSupportedMirror ||
      analysis.description.requiredHeaderDeclarations !=
          plan.requiredHeaderDeclarations ||
      analysis.description.cTypeMappingSummary != plan.cTypeMappingSummary ||
      analysis.description.vlCType != plan.vlCType ||
      analysis.description.vectorTypeName != plan.vectorTypeName ||
      analysis.description.vectorCType != plan.vectorCType ||
      analysis.description.indexVectorTypeName != plan.indexVectorTypeName ||
      analysis.description.indexVectorCType != plan.indexVectorCType ||
      analysis.description.maskTypeName != plan.maskTypeName ||
      analysis.description.maskCType != plan.maskCType ||
      analysis.description.setVLIntrinsic != plan.setVLIntrinsic ||
      analysis.description.vectorLoadIntrinsic != plan.vectorLoadIntrinsic ||
      analysis.description.indexLoadIntrinsic != plan.indexLoadIntrinsic ||
      analysis.description.indexScaleIntrinsic != plan.indexScaleIntrinsic ||
      analysis.description.indexedLoadIntrinsic != plan.indexedLoadIntrinsic ||
      analysis.description.indexedStoreIntrinsic !=
          plan.indexedStoreIntrinsic ||
      analysis.description.stridedLoadIntrinsic != plan.stridedLoadIntrinsic ||
      analysis.description.maskedLoadIntrinsic != plan.maskedLoadIntrinsic ||
      analysis.description.storeIntrinsic != plan.storeIntrinsic ||
      analysis.description.stridedStoreIntrinsic !=
          plan.stridedStoreIntrinsic ||
      analysis.description.resultName != plan.resultName ||
      analysis.description.maskName != plan.maskName ||
      analysis.description.maskRole != plan.maskRole ||
      analysis.description.maskSource != plan.maskSource ||
      analysis.description.maskMemoryForm != plan.maskMemoryForm ||
      analysis.description.inactiveLaneContract !=
          plan.inactiveLaneContract ||
      analysis.description.maskedPassthroughLayout !=
          plan.maskedPassthroughLayout ||
      analysis.description.stridedMemoryLayout != plan.stridedMemoryLayout ||
      analysis.description.indexedMemoryLayout != plan.indexedMemoryLayout ||
      analysis.description.sourceMemoryForm != plan.sourceMemoryForm ||
      analysis.description.destinationMemoryForm !=
          plan.destinationMemoryForm ||
      analysis.description.sourceStrideSource != plan.sourceStrideSource ||
      analysis.description.outStrideSource != plan.destinationStrideSource ||
      analysis.description.indexEEW != plan.indexEEW ||
      analysis.description.offsetUnit != plan.offsetUnit ||
      analysis.description.indexSource != plan.indexSource ||
      analysis.description.indexUniqueness != plan.indexUniqueness ||
      analysis.description.indexedDataMemoryForm !=
          plan.indexedDataMemoryForm ||
      analysis.description.indexedDestinationMemoryForm !=
          plan.indexedDestinationMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement route-family route, runtime, type, "
        "memory-form, intrinsic, and layout mirrors must be populated from "
        "the validated family plan before provider materialization");
  if (!support::runtimeABIParametersEqual(
          analysis.description.runtimeABIParameters, plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement route-family runtime ABI parameters must match "
        "the validated family plan");
  std::optional<llvm::StringRef> expectedBindingPlanID =
      getExpectedRVVSelectedBodyBaseMemoryRouteOperandBindingPlanID(operation);
  if (!expectedBindingPlanID ||
      analysis.routeOperandBindingPlan.planID != *expectedBindingPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement provider requires the route operand binding "
        "plan for the selected operation");
  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, analysis.description, context))
    return error;
  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedStridedMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV strided memory realization requires a pre-realized "
        "strided memory body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided memory realization requires a "
        "selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided memory body must be a direct child "
        "of the selected tcrv.exec.variant");

  if (!isPreRealizedStridedMemoryMovementOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided memory body currently supports "
        "only op_kind 'strided_load_unit_store'");
  if (!isPreRealizedStridedMemoryMovementMemoryForm(body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided memory body currently supports "
        "only memory_form 'strided-load-unit-store'");
  if (!isPreRealizedStridedLoadUnitStoreStrideUnit(body.getStrideUnit()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided memory body currently supports "
        "only stride_unit 'byte'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided memory body requires SEW32 LMUL "
        "m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided memory body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getSource(), "pre-realized RVV strided source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getOut(), "pre-realized RVV strided output operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getN(), "pre-realized RVV strided runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> sourceStride =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getSourceStride(),
          "pre-realized RVV source byte stride operand",
          support::RuntimeABIParameterRole::SourceByteStride);
  if (!sourceStride)
    return sourceStride.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedBaseMemoryBody<
              tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp>(
              variant, body.getOperation(), "strided memory"))
    return error;
  return requireBaseMemorySelectedVariantRequires(variant, "strided memory");
}

llvm::Error validatePreRealizedRVVSelectedStridedStoreMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV strided-store realization requires a pre-realized "
        "strided-store body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-store realization requires a "
        "selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-store body must be a direct child "
        "of the selected tcrv.exec.variant");

  if (!isPreRealizedStridedStoreMemoryMovementOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-store body currently supports "
        "only op_kind 'unit_load_strided_store'");
  if (!isPreRealizedStridedStoreMemoryMovementMemoryForm(body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-store body currently supports "
        "only memory_form 'unit-load-strided-store'");
  if (!isPreRealizedStridedStoreMemoryMovementStrideUnit(
          body.getStrideUnit()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-store body currently supports "
        "only stride_unit 'byte'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-store body requires SEW32 LMUL "
        "m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-store body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getSource(), "pre-realized RVV strided-store source operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> dst =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getDst(), "pre-realized RVV strided-store destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!dst)
    return dst.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getN(), "pre-realized RVV strided-store runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destinationStride =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getDestinationStride(),
          "pre-realized RVV destination byte stride operand",
          support::RuntimeABIParameterRole::DestinationByteStride);
  if (!destinationStride)
    return destinationStride.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedBaseMemoryBody<
              tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp>(
              variant, body.getOperation(), "strided-store"))
    return error;
  return requireBaseMemorySelectedVariantRequires(variant, "strided-store");
}

llvm::Error validatePreRealizedRVVSelectedIndexedGatherMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV indexed gather realization requires a pre-realized "
        "indexed gather body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed gather realization requires a "
        "selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed gather body must be a direct child "
        "of the selected tcrv.exec.variant");

  if (!isPreRealizedIndexedGatherMemoryMovementOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed gather body currently supports "
        "only op_kind 'indexed_gather_unit_store'");
  if (!isPreRealizedIndexedGatherMemoryMovementMemoryForm(
          body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed gather body currently supports "
        "only memory_form 'indexed-load-unit-store'");
  if (!isPreRealizedIndexedGatherMemoryMovementIndexEEW(
          static_cast<std::int64_t>(body.getIndexEew())))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed gather body currently supports "
        "only index_eew 32");
  if (!isPreRealizedIndexedGatherMemoryMovementOffsetUnit(
          body.getOffsetUnit()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed gather body currently supports "
        "only offset_unit 'element'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed gather body requires SEW32 LMUL "
        "m1 data config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed gather body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> data =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getData(), "pre-realized RVV indexed gather data operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!data)
    return data.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> index =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getIndex(), "pre-realized RVV indexed gather index operand",
          support::RuntimeABIParameterRole::IndexInputBuffer);
  if (!index)
    return index.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getOut(), "pre-realized RVV indexed gather output operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getN(), "pre-realized RVV indexed gather runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedBaseMemoryBody<
              tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp>(
              variant, body.getOperation(), "indexed gather"))
    return error;
  return requireBaseMemorySelectedVariantRequires(variant,
                                                  "indexed gather");
}

llvm::Error validatePreRealizedRVVSelectedIndexedScatterMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV indexed scatter realization requires a pre-realized "
        "indexed scatter body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed scatter realization requires a "
        "selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed scatter body must be a direct "
        "child of the selected tcrv.exec.variant");

  if (!isPreRealizedIndexedScatterMemoryMovementOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed scatter body currently supports "
        "only op_kind 'indexed_scatter_unit_load'");
  if (!isPreRealizedIndexedScatterMemoryMovementMemoryForm(
          body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed scatter body currently supports "
        "only memory_form 'unit-load-indexed-store'");
  if (!isPreRealizedIndexedGatherMemoryMovementIndexEEW(
          static_cast<std::int64_t>(body.getIndexEew())))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed scatter body currently supports "
        "only index_eew 32");
  if (!isPreRealizedIndexedGatherMemoryMovementOffsetUnit(
          body.getOffsetUnit()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed scatter body currently supports "
        "only offset_unit 'element'");
  if (!isPreRealizedIndexedScatterIndexUniqueness(
          body.getIndexUniqueness()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed scatter body requires "
        "index_uniqueness unique because duplicate-index scatter policy is "
        "unsupported");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed scatter body requires SEW32 LMUL "
        "m1 data config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected indexed scatter body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getSource(), "pre-realized RVV indexed scatter source operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> index =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getIndex(), "pre-realized RVV indexed scatter index operand",
          support::RuntimeABIParameterRole::IndexInputBuffer);
  if (!index)
    return index.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV indexed scatter destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getN(), "pre-realized RVV indexed scatter runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedBaseMemoryBody<
              tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp>(
              variant, body.getOperation(), "indexed scatter"))
    return error;
  return requireBaseMemorySelectedVariantRequires(variant,
                                                  "indexed scatter");
}

llvm::Error validatePreRealizedRVVSelectedMaskedMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV masked memory realization requires a pre-realized "
        "masked memory body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory realization requires a "
        "selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body must be a direct child "
        "of the selected tcrv.exec.variant");

  const bool isMaskedUnitLoadStore =
      body.getOpKind() == "masked_unit_load_store";
  const bool isMaskedUnitStore = body.getOpKind() == "masked_unit_store";
  if (!isPreRealizedMaskedMemoryMovementOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body currently supports only "
        "op_kind 'masked_unit_load_store' or 'masked_unit_store'");
  if (!isPreRealizedMaskedMemoryMovementMemoryForm(body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body currently supports only "
        "memory_form 'masked-unit-load-store' or 'masked-unit-store'");
  if ((isMaskedUnitLoadStore &&
       body.getMemoryForm() != "masked-unit-load-store") ||
      (isMaskedUnitStore && body.getMemoryForm() != "masked-unit-store"))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body requires op_kind and "
        "memory_form to agree");
  if (!isPreRealizedMaskedMemoryMovementMaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body currently supports only "
        "mask_role 'predicate-mask-input-buffer'");
  if (!isPreRealizedMaskedMemoryMovementMaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body currently supports only "
        "mask_memory_form 'unit-stride-mask-load'");
  if (!isPreRealizedMaskedMemoryMovementInactiveLanePolicy(
          body.getInactiveLanePolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body requires "
        "inactive_lane_policy 'preserve-old-destination' or "
        "'preserve-output-on-false-lanes'");
  if (isMaskedUnitLoadStore &&
      body.getInactiveLanePolicy() != "preserve-old-destination")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body requires "
        "inactive_lane_policy 'preserve-old-destination' for "
        "masked_unit_load_store");
  if (isMaskedUnitStore &&
      body.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body requires "
        "inactive_lane_policy 'preserve-output-on-false-lanes' for "
        "masked_unit_store");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body requires SEW32 LMUL m1 "
        "data/mask config");
  if (isMaskedUnitLoadStore &&
      !tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body requires tail "
        "agnostic, mask agnostic policy for masked_unit_load_store");
  if (isMaskedUnitStore &&
      !tcrv::rvv::isRVVUndisturbedPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected masked memory body requires tail "
        "undisturbed, mask undisturbed policy for masked_unit_store");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getSource(), "pre-realized RVV masked memory source operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> mask =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getMask(), "pre-realized RVV masked memory mask operand",
          support::RuntimeABIParameterRole::MaskInputBuffer);
  if (!mask)
    return mask.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getDestination(),
          "pre-realized RVV masked memory destination operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedBaseMemoryRuntimeABIValue(
          body.getN(), "pre-realized RVV masked memory runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedBaseMemoryBody<
              tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp>(
              variant, body.getOperation(), "masked memory"))
    return error;
  return requireBaseMemorySelectedVariantRequires(variant, "masked memory");
}

llvm::Expected<RVVSelectedBodyBaseMemoryMovementRouteProviderPlan>
getRVVSelectedBodyBaseMemoryMovementRouteProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyBaseMemoryMovementRouteProviderPlan providerPlan;
  if (!isRVVSelectedBodyBaseMemoryMovementStatementPlanConsumer(description))
    return providerPlan;

  if (llvm::Error error =
          verifyRVVSelectedBodyBaseMemoryMovementRouteFamilyProviderPlans(
              analysis, context))
    return std::move(error);

  const RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan *basePlan =
      materializationFacts.baseMemoryMovementPlan;
  if (!basePlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement provider plan requires the verified base "
        "memory movement route-family plan before provider route construction "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.baseMemoryMovementRouteFamilyPlan ||
      basePlan != &*analysis.baseMemoryMovementRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement provider plan requires route materialization "
        "facts from the same selected route analysis before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!memoryOperandBindingFacts.bindsBaseMemoryMovement ||
      !memoryOperandBindingFacts.bindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement provider plan requires RVV-owned memory "
        "operand-binding facts before provider route construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (memoryOperandBindingFacts.bindingPlan !=
      &analysis.routeOperandBindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement provider plan requires memory "
        "operand-binding facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsBaseMemoryMovement ||
      routeControlPlan->runtimeControlPlan != &basePlan->runtimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " base memory movement provider plan requires the RVV-owned "
        "route-control provider plan before provider route construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  providerPlan.baseMemoryMovementPlan = basePlan;
  providerPlan.bindingPlan = memoryOperandBindingFacts.bindingPlan;
  providerPlan.plansBaseMemoryMovementRoute = true;
  providerPlan.familyPlanIDMirror = basePlan->familyPlanID;
  providerPlan.providerSupportedMirror = basePlan->providerSupportedMirror;
  providerPlan.targetLeafProfileMirror = basePlan->targetLeafProfile;
  providerPlan.runtimeABIOrderMirror = basePlan->runtimeABIOrder;
  providerPlan.routeOperandBindingPlanIDMirror =
      analysis.routeOperandBindingPlan.planID;
  providerPlan.routeOperandBindingSummaryMirror =
      description.routeOperandBindingSummary;
  providerPlan.requiredHeaderDeclarationsMirror =
      basePlan->requiredHeaderDeclarations;
  providerPlan.cTypeMappingSummaryMirror = basePlan->cTypeMappingSummary;
  providerPlan.typedComputeOpNameMirror = description.typedComputeOpName;
  providerPlan.sourceMemoryFormMirror = basePlan->sourceMemoryForm;
  providerPlan.destinationMemoryFormMirror = basePlan->destinationMemoryForm;
  providerPlan.indexedMemoryLayoutMirror = basePlan->indexedMemoryLayout;
  providerPlan.indexEEWMirror = basePlan->indexEEW;
  providerPlan.offsetUnitMirror = basePlan->offsetUnit;
  providerPlan.indexSourceMirror = basePlan->indexSource;
  providerPlan.indexUniquenessMirror = basePlan->indexUniqueness;
  providerPlan.indexedDataMemoryFormMirror = basePlan->indexedDataMemoryForm;
  providerPlan.indexedDestinationMemoryFormMirror =
      basePlan->indexedDestinationMemoryForm;
  return providerPlan;
}

} // namespace tianchenrv::plugin::rvv
