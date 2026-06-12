#include "TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"

#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVMAccOperandBindingPlanID(
    "rvv-route-operand-binding:macc_add.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccOperandBindingPlanID(
    "rvv-route-operand-binding:scalar_broadcast_macc_add.v1");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccOperandBindingPlanID(
    "rvv-route-operand-binding:computed_masked_macc_add.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskedMAccOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1");

constexpr llvm::StringLiteral kRVVMAccRuntimeABIOrder("lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccRuntimeABIOrder(
    "lhs,rhs_scalar,acc,out,n");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral kRVVRuntimeScalarComputedMaskedMAccRuntimeABIOrder(
    "cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n");

constexpr llvm::StringLiteral kRVVPlainMAccRouteFamilyPlanID(
    "rvv-plain-macc-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccRouteFamilyPlanID(
    "rvv-scalar-broadcast-macc-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVComputedMaskAccumulationRouteFamilyPlanID(
    "rvv-computed-mask-accumulation-route-family-plan.v1");

constexpr llvm::StringLiteral kRVVMAccAccumulatorLayout(
    "separate-i32-vector-accumulator-input");
constexpr llvm::StringLiteral
    kRVVMAccResultLayout("store-multiply-accumulate-result-to-output-buffer");

constexpr llvm::StringLiteral kRVVPlainMAccTargetLeafProfile(
    "rvv-v1-typed-plain-macc-add-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccTargetLeafProfile(
    "rvv-v1-typed-scalar-broadcast-macc-add-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccTargetLeafProfile(
    "rvv-v1-typed-computed-mask-macc-add-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskedMAccTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-masked-macc-add-leaf-profile.v1");

constexpr llvm::StringLiteral kRVVPlainMAccProviderSupportedMirror(
    "provider_supported_mirror:rvv-plain-macc-add-plan-validated");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccProviderSupportedMirror(
    "provider_supported_mirror:rvv-scalar-broadcast-macc-add-composition-plan-validated");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccProviderSupportedMirror(
    "provider_supported_mirror:rvv-computed-mask-macc-add-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskedMAccProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-macc-add-plan-validated");

constexpr llvm::StringLiteral kRVVPlainMAccRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskedMAccRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");

constexpr llvm::StringLiteral kRVVPlainMAccCTypeMappingSummary(
    "vl:size_t,lhs/rhs/acc:typed-vector,result:typed-vector");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccCTypeMappingSummary(
    "vl:size_t,lhs/acc:typed-vector,rhs_scalar:typed-scalar,"
    "result:typed-vector");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccCTypeMappingSummary(
    "vl:size_t,cmp_lhs/cmp_rhs/lhs/rhs/acc:typed-vector,mask:typed-mask,result:typed-vector");
constexpr llvm::StringLiteral kRVVRuntimeScalarComputedMaskedMAccCTypeMappingSummary(
    "vl:size_t,cmp_lhs/lhs/rhs/acc:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:typed-vector");

constexpr llvm::StringLiteral kRVVComputedMaskAccumulationVectorCompareProducerSource(
    "vector-compare-rhs-load");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationRuntimeScalarProducerSource(
        "runtime-scalar-splat-compare-rhs");
constexpr llvm::StringLiteral kRVVComputedMaskAccumulationVectorMAccSuffix(
    "vector-masked-macc-add");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationStandaloneReductionSuffix(
        "scalar-horizontal-masked-standalone-reduction");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationMAccAccumulatorContract(
        "vector-accumulator-input-preserves-inactive-lanes");
constexpr llvm::StringLiteral kRVVComputedMaskAccumulationMAccResultContract(
    "vector-macc-result-stored-to-output-buffer");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationReductionAccumulatorContract(
        "scalar-seed-input-feeds-masked-horizontal-reduction");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationReductionResultContract(
        "scalar-horizontal-reduction-lane0-stored-to-output");
constexpr llvm::StringLiteral
    kRVVComputedMaskAccumulationReductionScalarCarryContract(
        "scalar-result-carries-across-runtime-vl-chunks");

constexpr llvm::StringLiteral kRVVStandaloneReductionTargetLeafProfile(
    "rvv-v1-typed-standalone-reduction-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionTargetLeafProfile(
        "rvv-v1-typed-computed-mask-standalone-reduction-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionTargetLeafProfile(
        "rvv-v1-typed-runtime-scalar-cmp-masked-standalone-reduction-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVStandaloneReductionProviderSupportedMirror(
    "provider_supported_mirror:rvv-standalone-reduction-plan-validated");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionProviderSupportedMirror(
        "provider_supported_mirror:rvv-computed-mask-standalone-reduction-plan-validated");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionProviderSupportedMirror(
        "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated");
constexpr llvm::StringLiteral kRVVStandaloneReductionRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVStandaloneReductionCTypeMappingSummary(
    "vl:size_t,input:typed-source-vector,seed:typed-scalar,result:typed-scalar-reduction-vector");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionCTypeMappingSummary(
        "vl:size_t,compare/source:typed-source-vector,mask:typed-mask,seed:typed-scalar,result:typed-scalar-reduction-vector");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionCTypeMappingSummary(
        "vl:size_t,cmp_lhs/source:typed-source-vector,rhs_scalar:typed-scalar,mask:typed-mask,seed:typed-scalar,result:typed-scalar-reduction-vector");
constexpr llvm::StringLiteral
    kRVVStandaloneReductionMaskedInactiveLaneZeroingRequirement(
        "masked-standalone-reduction-zero-inactive-lanes-before-reduction");
constexpr llvm::StringLiteral
    kRVVStandaloneReductionMaskedInactiveLaneNeutralRequirement(
        "masked-standalone-reduction-neutral-inactive-lanes-before-reduction");

constexpr llvm::StringLiteral kRVVMaskedPredicateMaskRole(
    "predicate-mask-produced-by-compare");
constexpr llvm::StringLiteral kRVVMaskedCompareMaskSource(
    "compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskMemoryForm(
    "compare-produced-mask");
constexpr llvm::StringLiteral kRVVUnitStrideSourceMemoryForm(
    "unit-stride-load");
constexpr llvm::StringLiteral kRVVDestinationMemoryForm("unit-stride-store");
constexpr llvm::StringLiteral kRVVComputedMaskedMAccMemoryLayout(
    "unit-stride-compare-lhs-rhs-accumulator-masked-macc-output-runtime-abi");
constexpr llvm::StringLiteral kRVVRuntimeScalarComputedMaskedMAccMemoryLayout(
    "unit-stride-compare-lhs-runtime-scalar-threshold-lhs-rhs-accumulator-masked-macc-output-runtime-abi");

void addMAccRouteOperandBinding(
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

RVVRouteOperandBindingPlan buildUnitStrideMAccRouteOperandBindingPlanFromFacts(
    const RVVUnitStrideMAccRouteFacts &facts) {
  RVVRouteOperandBindingPlan plan;
  plan.planID = facts.routeOperandBindingPlanID.str();
  if (facts.runtimeABIParameters.size() != 5)
    return plan;

  addMAccRouteOperandBinding(plan, "lhs", facts.runtimeABIParameters[0],
                             {"abi", "lhs-load", "macc-lhs", "hdr"});
  if (facts.usesScalarBroadcastRHS)
    addMAccRouteOperandBinding(plan, "rhs_scalar",
                               facts.runtimeABIParameters[1],
                               {"abi", "splat", "macc-rhs", "hdr"});
  else
    addMAccRouteOperandBinding(plan, "rhs", facts.runtimeABIParameters[1],
                               {"abi", "rhs-load", "macc-rhs", "hdr"});
  addMAccRouteOperandBinding(plan, "acc", facts.runtimeABIParameters[2],
                             {"abi", "acc-load", "macc-acc", "macc-pass",
                              "hdr"});
  addMAccRouteOperandBinding(plan, "out", facts.runtimeABIParameters[3],
                             {"abi", "store", "hdr"});
  addMAccRouteOperandBinding(plan, "n", facts.runtimeABIParameters[4],
                             {"abi", "setvl-avl", "loop", "hdr"});
  return plan;
}

RVVRouteOperandBindingPlan
buildComputedMaskMAccRouteOperandBindingPlanFromFacts(
    const RVVComputedMaskMAccRouteFacts &facts) {
  RVVRouteOperandBindingPlan plan;
  plan.planID = facts.routeOperandBindingPlanID.str();
  if (facts.runtimeABIParameters.size() != 7)
    return plan;

  addMAccRouteOperandBinding(plan, "cmp_lhs", facts.runtimeABIParameters[0],
                             {"abi", "cmp-lhs", "cmp-call", "hdr"});
  addMAccRouteOperandBinding(plan, "cmp_rhs", facts.runtimeABIParameters[1],
                             {"abi", "cmp-rhs", "cmp-call", "hdr"});
  addMAccRouteOperandBinding(plan, "lhs", facts.runtimeABIParameters[2],
                             {"abi", "lhs-load", "macc-lhs", "hdr"});
  addMAccRouteOperandBinding(plan, "rhs", facts.runtimeABIParameters[3],
                             {"abi", "rhs-load", "macc-rhs", "hdr"});
  addMAccRouteOperandBinding(plan, "acc", facts.runtimeABIParameters[4],
                             {"abi", "acc-load", "macc-acc", "macc-pass",
                              "hdr"});
  addMAccRouteOperandBinding(plan, "out", facts.runtimeABIParameters[5],
                             {"abi", "store", "hdr"});
  addMAccRouteOperandBinding(plan, "n", facts.runtimeABIParameters[6],
                             {"abi", "setvl-avl", "loop", "hdr"});
  return plan;
}

RVVRouteOperandBindingPlan
buildRuntimeScalarComputedMaskMAccRouteOperandBindingPlanFromFacts(
    const RVVRuntimeScalarComputedMaskMAccRouteFacts &facts) {
  RVVRouteOperandBindingPlan plan;
  plan.planID = facts.routeOperandBindingPlanID.str();
  if (facts.runtimeABIParameters.size() != 7)
    return plan;

  addMAccRouteOperandBinding(plan, "cmp_lhs", facts.runtimeABIParameters[0],
                             {"abi", "cmp-lhs", "cmp-call", "hdr"});
  addMAccRouteOperandBinding(plan, "rhs_scalar",
                             facts.runtimeABIParameters[1],
                             {"abi", "splat", "cmp-rhs", "hdr"});
  addMAccRouteOperandBinding(plan, "lhs", facts.runtimeABIParameters[2],
                             {"abi", "lhs-load", "macc-lhs", "hdr"});
  addMAccRouteOperandBinding(plan, "rhs", facts.runtimeABIParameters[3],
                             {"abi", "rhs-load", "macc-rhs", "hdr"});
  addMAccRouteOperandBinding(plan, "acc", facts.runtimeABIParameters[4],
                             {"abi", "acc-load", "macc-acc", "macc-pass",
                              "hdr"});
  addMAccRouteOperandBinding(plan, "out", facts.runtimeABIParameters[5],
                             {"abi", "store", "hdr"});
  addMAccRouteOperandBinding(plan, "n", facts.runtimeABIParameters[6],
                             {"abi", "setvl-avl", "loop", "hdr"});
  return plan;
}

void applyMAccRuntimeAVLVLControlPlanToDescription(
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

llvm::Error requireMAccPlanField(llvm::StringRef planKind,
                                 RVVSelectedBodyOperationKind operation,
                                 llvm::StringRef field,
                                 llvm::StringRef actual,
                                 llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(planKind) + " route-family plan validation for operation '" +
      stringifyRVVSelectedBodyOperationKind(operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireMAccRouteDescriptionField(llvm::StringRef context,
                                             llvm::StringRef field,
                                             llvm::StringRef actual,
                                             llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " " + field +
      " must mirror selected MAcc owner fact '" + expected + "' but was '" +
      actual + "'");
}

llvm::Error requireMAccPlanDerivedField(llvm::StringRef planKind,
                                        RVVSelectedBodyOperationKind operation,
                                        llvm::StringRef field,
                                        llvm::StringRef actual) {
  if (!actual.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(planKind) + " route-family plan validation for operation '" +
      stringifyRVVSelectedBodyOperationKind(operation) +
      "' requires provider-derived " + field +
      " from selected typed RVV body/config facts");
}

llvm::Error requireMAccTypedConfigLeaf(
    const RVVSelectedBodyTypedConfigFacts &typedFacts, llvm::StringRef field,
    llvm::StringRef leaf, RVVSelectedBodyOperationKind operation) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("MAcc route-family typed config derivation for operation '") +
      stringifyRVVSelectedBodyOperationKind(operation) +
      "' requires provider-derived " + field + " from typed config facts '" +
      typedFacts.factsID + "'");
}

llvm::StringRef internMAccDerivedLeaf(std::string leaf) {
  static llvm::StringSet<> leafPool;
  return leafPool.insert(std::move(leaf)).first->getKey();
}

template <typename PlanT>
void applyMAccTypedConfigSnapshot(
    PlanT &plan, const RVVSelectedBodyTypedConfigFacts &typedFacts) {
  plan.typedConfigFactsID = typedFacts.factsID;
  plan.elementTypeName = typedFacts.elementTypeName;
  plan.elementBitWidth = typedFacts.elementBitWidth;
  plan.sew = typedFacts.sew;
  plan.lmul = typedFacts.lmul;
  plan.tailPolicy = typedFacts.tailPolicy;
  plan.maskPolicy = typedFacts.maskPolicy;
  plan.configContractID = typedFacts.configContractID;
}

template <typename PlanT>
llvm::Error requireMAccPlanTypedConfigSnapshot(llvm::StringRef planKind,
                                               const PlanT &plan) {
  if (plan.typedConfigFactsID.empty() || plan.elementTypeName.empty() ||
      plan.elementBitWidth == 0 || plan.sew == 0 || plan.lmul.empty() ||
      plan.tailPolicy.empty() || plan.maskPolicy.empty() ||
      plan.configContractID.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(planKind) +
        " route-family plan requires provider-derived typed config facts for "
        "element type, SEW, LMUL, policy, and config contract");
  if (plan.elementBitWidth != plan.sew)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(planKind) +
        " route-family plan requires element bit width to mirror "
        "provider-derived SEW");
  if (plan.sew != plan.runtimeControlPlan.sew ||
      plan.lmul != plan.runtimeControlPlan.lmul ||
      plan.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      plan.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      plan.configContractID != plan.runtimeControlPlan.configContractID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(planKind) +
        " route-family plan typed config snapshot must mirror the RVV-owned "
        "runtime AVL/VL control plan");
  return llvm::Error::success();
}

template <typename PlanT>
llvm::Error verifyMAccPlanTypedConfigSnapshot(
    const RVVSelectedBodyTypedConfigFacts &typedFacts, const PlanT &plan,
    llvm::StringRef context, llvm::StringRef planKind,
    bool storeMustMatchTypedFacts) {
  if (!typedFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + planKind +
        " provider requires typed config facts before provider "
        "materialization");

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
      plan.setVLIntrinsic != typedFacts.setVLIntrinsic ||
      plan.vectorLoadIntrinsic != typedFacts.vectorLoadIntrinsic ||
      (storeMustMatchTypedFacts &&
       plan.storeIntrinsic != typedFacts.storeIntrinsic))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + planKind +
        " route-family typed config snapshot must mirror selected typed RVV "
        "body/config facts before provider materialization");

  return llvm::Error::success();
}

llvm::Error verifyMAccPlanMaskTypedConfigSnapshot(
    const RVVSelectedBodyTypedConfigFacts &typedFacts,
    const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan &plan,
    llvm::StringRef context, bool maskMustMatchTypedFacts) {
  if (!maskMustMatchTypedFacts)
    return llvm::Error::success();
  if (plan.maskTypeName != typedFacts.maskTypeName ||
      plan.maskCType != typedFacts.maskCType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation route-family typed mask snapshot must "
        "mirror selected typed RVV body/config facts before provider "
        "materialization");
  return llvm::Error::success();
}

llvm::Expected<tcrv::rvv::RuntimeABIValueOp> requirePreRealizedRuntimeABIValue(
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

} // namespace

llvm::Error validatePreRealizedRVVSelectedMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedMAccPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV macc realization requires a pre-realized macc body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected macc body must be a direct child of the "
        "selected tcrv.exec.variant");

  if (!(body.getOpKind() == "macc_add" ||
        body.getOpKind() == "scalar_broadcast_macc_add"))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected macc body currently supports only "
        "op_kind 'macc_add' or 'scalar_broadcast_macc_add'");
  if (!(body.getMemoryForm() == "vector-rhs-load" ||
        body.getMemoryForm() == "rhs-scalar-broadcast-macc"))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected macc body currently supports only "
        "memory_form 'vector-rhs-load' or 'rhs-scalar-broadcast-macc'");
  bool scalarBroadcastMAcc =
      body.getOpKind() == "scalar_broadcast_macc_add" ||
      body.getMemoryForm() == "rhs-scalar-broadcast-macc";
  if (scalarBroadcastMAcc &&
      (body.getOpKind() != "scalar_broadcast_macc_add" ||
       body.getMemoryForm() != "rhs-scalar-broadcast-macc"))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected scalar-broadcast macc body requires "
        "op_kind 'scalar_broadcast_macc_add' paired with memory_form "
        "'rhs-scalar-broadcast-macc'");
  if (!scalarBroadcastMAcc &&
      (body.getOpKind() != "macc_add" ||
       body.getMemoryForm() != "vector-rhs-load"))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected macc body requires op_kind 'macc_add' "
        "paired with memory_form 'vector-rhs-load'");
  if (body.getAccumulatorRole() != "accumulator-input-buffer")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected macc body currently supports only "
        "accumulator_role 'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() !=
      "separate-i32-vector-accumulator-input")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected macc body currently supports only "
        "accumulator_layout 'separate-i32-vector-accumulator-input'");
  if (body.getResultLayout() !=
      "store-multiply-accumulate-result-to-output-buffer")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected macc body currently supports only "
        "result_layout 'store-multiply-accumulate-result-to-output-buffer'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected macc body requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected macc body requires tail agnostic, mask "
        "agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV macc lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  support::RuntimeABIParameterRole rhsRole =
      scalarBroadcastMAcc ? support::RuntimeABIParameterRole::RHSScalarValue
                          : support::RuntimeABIParameterRole::RHSInputBuffer;
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV macc rhs operand", rhsRole);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(), "pre-realized RVV macc accumulator operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV macc out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV macc runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedMAccPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected macc body must not be mixed "
                    "with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected macc-body realization requires non-empty "
        "selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskMAccPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask macc realization requires a "
        "pre-realized computed-mask macc body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body must be a direct "
        "child of the selected tcrv.exec.variant");

  if (body.getOpKind() != "computed_masked_macc_add")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only op_kind 'computed_masked_macc_add'");
  if (body.getPredicateKind() != "slt")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only predicate_kind 'slt'");
  if (body.getMemoryForm() != "computed-mask-unit-stride-macc")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only memory_form 'computed-mask-unit-stride-macc'");
  if (body.getMaskRole() != "predicate-mask-produced-by-compare")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only mask_role 'predicate-mask-produced-by-compare'");
  if (body.getMaskSource() != "compare-produced-mask-same-vl-scope")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only mask_source 'compare-produced-mask-same-vl-scope'");
  if (body.getMaskMemoryForm() != "compare-produced-mask")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only mask_memory_form 'compare-produced-mask'");
  if (body.getAccumulatorRole() != "accumulator-input-buffer")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only accumulator_role 'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() !=
      "separate-i32-vector-accumulator-input")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only accumulator_layout 'separate-i32-vector-accumulator-input'");
  if (body.getResultLayout() !=
      "store-multiply-accumulate-result-to-output-buffer")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body currently supports "
        "only result_layout "
        "'store-multiply-accumulate-result-to-output-buffer'");
  if (!(static_cast<std::int64_t>(body.getSew()) ==
            tcrv::rvv::getRVVFirstSliceSEWBits() &&
        (body.getLmul() == tcrv::rvv::getRVVLMULM1() ||
         body.getLmul() == tcrv::rvv::getRVVLMULM2())))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body requires SEW32 "
        "LMUL m1 or SEW32 LMUL m2");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> cmpLHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask macc compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!cmpLHS)
    return cmpLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> cmpRHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask macc compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!cmpRHS)
    return cmpRHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV computed-mask macc lhs payload",
          support::RuntimeABIParameterRole::DotLHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV computed-mask macc rhs payload",
          support::RuntimeABIParameterRole::DotRHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(), "pre-realized RVV computed-mask macc accumulator",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV computed-mask macc out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV computed-mask macc runtime n/AVL "
                       "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedComputedMaskMAccPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected computed-mask macc body must "
                    "not be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask macc realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV runtime scalar computed-mask macc realization requires "
        "a pre-realized runtime scalar computed-mask macc body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "must be a direct child of the selected tcrv.exec.variant");

  if (body.getOpKind() != "runtime_scalar_cmp_masked_macc_add")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only op_kind "
        "'runtime_scalar_cmp_masked_macc_add'");
  if (body.getPredicateKind() != "sle")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only predicate_kind 'sle'");
  if (body.getMemoryForm() !=
      "runtime-scalar-computed-mask-unit-stride-macc")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only memory_form "
        "'runtime-scalar-computed-mask-unit-stride-macc'");
  if (body.getMaskRole() != "predicate-mask-produced-by-compare")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (body.getMaskSource() != "compare-produced-mask-same-vl-scope")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (body.getMaskMemoryForm() != "compare-produced-mask")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only mask_memory_form 'compare-produced-mask'");
  if (body.getAccumulatorRole() != "accumulator-input-buffer")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only accumulator_role 'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() !=
      "separate-i32-vector-accumulator-input")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only accumulator_layout "
        "'separate-i32-vector-accumulator-input'");
  if (body.getResultLayout() !=
      "store-multiply-accumulate-result-to-output-buffer")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "currently supports only result_layout "
        "'store-multiply-accumulate-result-to-output-buffer'");
  if (!(static_cast<std::int64_t>(body.getSew()) ==
            tcrv::rvv::getRVVFirstSliceSEWBits() &&
        (body.getLmul() == tcrv::rvv::getRVVLMULM1() ||
         body.getLmul() == tcrv::rvv::getRVVLMULM2())))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "requires SEW32 LMUL m1 or SEW32 LMUL m2");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc body "
        "requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> cmpLHS =
      requirePreRealizedRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV runtime scalar computed-mask macc compare lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!cmpLHS)
    return cmpLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedRuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime scalar computed-mask macc rhs scalar "
          "operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV runtime scalar computed-mask macc lhs payload",
          support::RuntimeABIParameterRole::DotLHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV runtime scalar computed-mask macc rhs payload",
          support::RuntimeABIParameterRole::DotRHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV runtime scalar computed-mask macc accumulator",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV runtime scalar computed-mask macc out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime scalar computed-mask macc runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<
            tcrv::rvv::RuntimeABIValueOp,
            tcrv::rvv::
                TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected runtime scalar computed-mask "
                    "macc body must not be mixed with already realized RVV "
                    "route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime scalar computed-mask macc "
        "realization requires non-empty selected variant requires metadata");

  return llvm::Error::success();
}

namespace {

bool isComputedMaskStandaloneReductionOperation(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isRuntimeScalarComputedMaskStandaloneReductionOperation(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isZeroInactiveStandaloneReductionOperation(
    RVVSelectedBodyOperationKind operation) {
  return operation ==
             RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd ||
         operation == RVVSelectedBodyOperationKind::
                          RuntimeScalarComputedMaskStandaloneReduceAdd;
}

llvm::StringRef getStandaloneReductionInactiveLaneRequirement(
    RVVSelectedBodyOperationKind operation) {
  return isZeroInactiveStandaloneReductionOperation(operation)
             ? llvm::StringRef(
                   kRVVStandaloneReductionMaskedInactiveLaneZeroingRequirement)
             : llvm::StringRef(
                   kRVVStandaloneReductionMaskedInactiveLaneNeutralRequirement);
}

bool isRuntimeScalarComputedMaskStandaloneReductionConfig(std::int64_t sew,
                                                         llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    return lmul == tcrv::rvv::getRVVLMULM1() ||
           lmul == tcrv::rvv::getRVVLMULM2();
  return sew == tcrv::rvv::getRVVSEW64Bits() &&
         lmul == tcrv::rvv::getRVVLMULM1();
}

bool isStandaloneReductionScalarChannelConfig(std::int64_t sew,
                                              llvm::StringRef lmul) {
  return sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         (lmul == tcrv::rvv::getRVVLMULM1() ||
          lmul == tcrv::rvv::getRVVLMULM2());
}

bool isComputedMaskMAccConfig(std::int64_t sew, llvm::StringRef lmul) {
  return sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         (lmul == tcrv::rvv::getRVVLMULM1() ||
          lmul == tcrv::rvv::getRVVLMULM2());
}

bool isSupportedMAccVectorSuffixConfig(std::int64_t sew,
                                       llvm::StringRef lmul) {
  return (sew == tcrv::rvv::getRVVFirstSliceSEWBits() ||
          sew == tcrv::rvv::getRVVSEW64Bits()) &&
         (lmul == tcrv::rvv::getRVVLMULM1() ||
          lmul == tcrv::rvv::getRVVLMULM2());
}

std::optional<std::string>
deriveMAccVectorIntrinsicSuffix(std::int64_t sew, llvm::StringRef lmul) {
  if (!isSupportedMAccVectorSuffixConfig(sew, lmul))
    return std::nullopt;
  return (llvm::Twine("i") + llvm::Twine(sew) + lmul).str();
}

std::optional<std::string>
deriveMAccMaskIntrinsicSuffix(std::int64_t sew, llvm::StringRef lmul) {
  std::optional<std::string> vectorSuffix =
      deriveMAccVectorIntrinsicSuffix(sew, lmul);
  if (!vectorSuffix)
    return std::nullopt;

  std::int64_t maskBits = 0;
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      lmul == tcrv::rvv::getRVVLMULM1())
    maskBits = 32;
  else if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
           lmul == tcrv::rvv::getRVVLMULM2())
    maskBits = 16;
  else if (sew == tcrv::rvv::getRVVSEW64Bits() &&
           lmul == tcrv::rvv::getRVVLMULM1())
    maskBits = 64;
  else if (sew == tcrv::rvv::getRVVSEW64Bits() &&
           lmul == tcrv::rvv::getRVVLMULM2())
    maskBits = 32;
  if (maskBits == 0)
    return std::nullopt;
  return (llvm::Twine(*vectorSuffix) + "_b" + llvm::Twine(maskBits)).str();
}

std::optional<std::string> deriveMAccIntrinsic(std::int64_t sew,
                                               llvm::StringRef lmul) {
  if (sew != tcrv::rvv::getRVVFirstSliceSEWBits())
    return std::nullopt;
  std::optional<std::string> suffix =
      deriveMAccVectorIntrinsicSuffix(sew, lmul);
  if (!suffix)
    return std::nullopt;
  return (llvm::Twine("__riscv_vmacc_vv_") + *suffix).str();
}

std::optional<std::string> deriveMAccSetVLIntrinsic(std::int64_t sew,
                                                    llvm::StringRef lmul) {
  if (!isSupportedMAccVectorSuffixConfig(sew, lmul))
    return std::nullopt;
  return (llvm::Twine("__riscv_vsetvl_e") + llvm::Twine(sew) + lmul).str();
}

std::optional<std::string> deriveMAccScalarSplatIntrinsic(std::int64_t sew,
                                                          llvm::StringRef lmul) {
  std::optional<std::string> suffix =
      deriveMAccVectorIntrinsicSuffix(sew, lmul);
  if (!suffix)
    return std::nullopt;
  return (llvm::Twine("__riscv_vmv_v_x_") + *suffix).str();
}

std::optional<std::string> deriveMAccCompareIntrinsic(
    llvm::StringRef predicateKind, std::int64_t sew, llvm::StringRef lmul) {
  llvm::StringRef stem;
  if (predicateKind == "slt")
    stem = "vmslt";
  else if (predicateKind == "sle")
    stem = "vmsle";
  else
    return std::nullopt;
  std::optional<std::string> suffix = deriveMAccMaskIntrinsicSuffix(sew, lmul);
  if (!suffix)
    return std::nullopt;
  return (llvm::Twine("__riscv_") + stem + "_vv_" + *suffix).str();
}

std::optional<std::string> deriveMAccMaskedMergeIntrinsic(std::int64_t sew,
                                                          llvm::StringRef lmul) {
  std::optional<std::string> suffix =
      deriveMAccVectorIntrinsicSuffix(sew, lmul);
  if (!suffix)
    return std::nullopt;
  return (llvm::Twine("__riscv_vmerge_vvm_") + *suffix).str();
}

std::optional<std::string> deriveMAccVectorStoreIntrinsic(std::int64_t sew,
                                                          llvm::StringRef lmul) {
  std::optional<std::string> suffix =
      deriveMAccVectorIntrinsicSuffix(sew, lmul);
  if (!suffix)
    return std::nullopt;
  return (llvm::Twine("__riscv_vse") + llvm::Twine(sew) + "_v_" + *suffix)
      .str();
}

std::optional<std::string>
deriveStandaloneReductionScalarResultStoreIntrinsic(std::int64_t sew,
                                                    llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      (lmul == tcrv::rvv::getRVVLMULM1() ||
       lmul == tcrv::rvv::getRVVLMULM2()))
    return (llvm::Twine("__riscv_vse") + llvm::Twine(sew) + "_v_i" +
            llvm::Twine(sew) + tcrv::rvv::getRVVLMULM1())
        .str();
  if (sew == tcrv::rvv::getRVVSEW64Bits() &&
      lmul == tcrv::rvv::getRVVLMULM1())
    return (llvm::Twine("__riscv_vse") + llvm::Twine(sew) + "_v_i" +
            llvm::Twine(sew) + lmul)
        .str();
  return std::nullopt;
}

} // namespace

bool isRVVSelectedBodyScalarBroadcastMAccRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
}

bool isRVVSelectedBodyPlainMAccRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::MAccAdd;
}

bool isRVVSelectedBodyComputedMaskMAccAccumulationRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVSelectedBodyComputedMaskAccumulationRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isRVVSelectedBodyComputedMaskAccumulationRouteOperation(
    RVVSelectedBodyOperationKind operation) {
  return isRVVSelectedBodyComputedMaskAccumulationRouteFamilyConsumer(
      operation);
}

llvm::StringRef getRVVSelectedBodyMAccAccumulatorLayout() {
  return kRVVMAccAccumulatorLayout;
}

llvm::StringRef getRVVSelectedBodyMAccResultLayout() {
  return kRVVMAccResultLayout;
}

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyMAccRuntimeABIOrder(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::MAccAdd:
    return kRVVMAccRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
    return kRVVScalarBroadcastMAccRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd: {
    std::optional<RVVComputedMaskMAccRouteFacts> routeFacts =
        getRVVComputedMaskMAccRouteFacts(operation);
    return routeFacts ? routeFacts->runtimeABIOrder : llvm::StringRef();
  }
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd: {
    std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts> routeFacts =
        getRVVRuntimeScalarComputedMaskMAccRouteFacts(operation);
    return routeFacts ? routeFacts->runtimeABIOrder : llvm::StringRef();
  }
  default:
    return std::nullopt;
  }
}

std::optional<RVVUnitStrideMAccRouteFacts>
getRVVUnitStrideMAccRouteFacts(RVVSelectedBodyOperationKind operation) {
  const bool isPlain = operation == RVVSelectedBodyOperationKind::MAccAdd;
  const bool isScalarBroadcast =
      operation == RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
  if (!isPlain && !isScalarBroadcast)
    return std::nullopt;

  RVVUnitStrideMAccRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = isPlain ? RVVSelectedBodyMemoryForm::VectorRHSLoad
                             : RVVSelectedBodyMemoryForm::RHSScalarBroadcastMAcc;
  facts.sew = tcrv::rvv::getRVVFirstSliceSEWBits();
  facts.lmul = tcrv::rvv::getRVVLMULM1();
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder =
      isPlain ? kRVVMAccRuntimeABIOrder
              : kRVVScalarBroadcastMAccRuntimeABIOrder;
  facts.targetLeafProfile =
      isPlain ? kRVVPlainMAccTargetLeafProfile
              : kRVVScalarBroadcastMAccTargetLeafProfile;
  facts.providerSupportedMirror =
      isPlain ? kRVVPlainMAccProviderSupportedMirror
              : kRVVScalarBroadcastMAccProviderSupportedMirror;
  facts.requiredHeaderDeclarations =
      isPlain ? kRVVPlainMAccRequiredHeaderDeclarations
              : kRVVScalarBroadcastMAccRequiredHeaderDeclarations;
  facts.cTypeMappingSummary =
      isPlain ? kRVVPlainMAccCTypeMappingSummary
              : kRVVScalarBroadcastMAccCTypeMappingSummary;
  facts.routeOperandBindingPlanID =
      isPlain ? kRVVMAccOperandBindingPlanID
              : kRVVScalarBroadcastMAccOperandBindingPlanID;
  facts.typedComputeOpName = "tcrv_rvv.macc";
  facts.routeFamilyPlanID =
      isPlain ? kRVVPlainMAccRouteFamilyPlanID
              : kRVVScalarBroadcastMAccRouteFamilyPlanID;
  facts.arithmeticKind = "add";
  facts.lhsRole = "lhs-input-buffer";
  facts.rhsRole = isPlain ? llvm::StringRef("rhs-input-buffer")
                          : llvm::StringRef("rhs-scalar-value");
  facts.accumulatorRole = "accumulator-input-buffer";
  facts.outputRole = "output-buffer";
  facts.runtimeCountRole = "runtime-element-count";
  facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.rhsMemoryForm =
      isPlain ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
              : llvm::StringRef("rhs-scalar-broadcast");
  facts.accumulatorMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  facts.usesVectorRHSLoad = isPlain;
  facts.usesScalarBroadcastRHS = isScalarBroadcast;
  facts.maccAccumulatorLayout = kRVVMAccAccumulatorLayout;
  facts.maccResultLayout = kRVVMAccResultLayout;
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "lhs", "const int32_t *", support::RuntimeABIParameterRole::LHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  if (isPlain)
    facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
        "rhs", "const int32_t *",
        support::RuntimeABIParameterRole::RHSInputBuffer,
        support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  else
    facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
        "rhs_scalar", "int32_t",
        support::RuntimeABIParameterRole::RHSScalarValue,
        support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));

  RVVRouteOperandBindingPlan plan =
      buildUnitStrideMAccRouteOperandBindingPlanFromFacts(facts);
  facts.routeOperandBindingSummary =
      stringifyRVVRouteOperandBindingPlan(plan);
  return facts;
}

std::optional<RVVComputedMaskMAccRouteFacts>
getRVVComputedMaskMAccRouteFacts(RVVSelectedBodyOperationKind operation) {
  return getRVVComputedMaskMAccRouteFacts(
      operation, tcrv::rvv::getRVVFirstSliceSEWBits(),
      tcrv::rvv::getRVVLMULM1());
}

std::optional<RVVComputedMaskMAccRouteFacts>
getRVVComputedMaskMAccRouteFacts(RVVSelectedBodyOperationKind operation,
                                 std::int64_t sew, llvm::StringRef lmul) {
  if (operation != RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd)
    return std::nullopt;
  if (!isComputedMaskMAccConfig(sew, lmul))
    return std::nullopt;

  RVVComputedMaskMAccRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideMAcc;
  facts.sew = sew;
  facts.lmul = lmul;
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = kRVVComputedMaskedMAccRuntimeABIOrder;
  facts.targetLeafProfile = kRVVComputedMaskedMAccTargetLeafProfile;
  facts.providerSupportedMirror = kRVVComputedMaskedMAccProviderSupportedMirror;
  facts.requiredHeaderDeclarations =
      kRVVComputedMaskedMAccRequiredHeaderDeclarations;
  facts.cTypeMappingSummary = kRVVComputedMaskedMAccCTypeMappingSummary;
  facts.routeOperandBindingPlanID =
      kRVVComputedMaskedMAccOperandBindingPlanID;
  facts.typedComputeOpName = "tcrv_rvv.masked_macc";
  facts.arithmeticKind = "add";
  facts.comparePredicateKind = "slt";
  facts.compareLhsRole = "lhs-input-buffer";
  facts.compareRhsRole = "rhs-input-buffer";
  facts.lhsRole = "dot-lhs-input-buffer";
  facts.rhsRole = "dot-rhs-input-buffer";
  facts.accumulatorRole = "accumulator-input-buffer";
  facts.outputRole = "output-buffer";
  facts.runtimeCountRole = "runtime-element-count";
  facts.usesVectorCompareRHSLoad = true;
  facts.usesRuntimeScalarCompareThreshold = false;
  facts.accumulationRouteFamilyPlanID =
      kRVVComputedMaskAccumulationRouteFamilyPlanID;
  facts.accumulationComputeSuffix =
      kRVVComputedMaskAccumulationVectorMAccSuffix;
  facts.accumulationMaskProducerSource =
      kRVVComputedMaskAccumulationVectorCompareProducerSource;
  facts.accumulationAccumulatorContract =
      kRVVComputedMaskAccumulationMAccAccumulatorContract;
  facts.accumulationResultContract =
      kRVVComputedMaskAccumulationMAccResultContract;
  facts.maskRole = kRVVMaskedPredicateMaskRole;
  facts.maskSource = kRVVMaskedCompareMaskSource;
  facts.maskMemoryForm = kRVVComputedMaskMemoryMaskMemoryForm;
  facts.inactiveLaneContract =
      "masked-macc-false-lanes-preserve-accumulator";
  facts.maskedPassthroughLayout =
      "accumulator-vector-preserves-inactive-lanes";
  facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  facts.indexedMemoryLayout = kRVVComputedMaskedMAccMemoryLayout;
  facts.maccAccumulatorLayout = kRVVMAccAccumulatorLayout;
  facts.maccResultLayout = kRVVMAccResultLayout;
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "cmp_rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::DotLHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "rhs", "const int32_t *",
      support::RuntimeABIParameterRole::DotRHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));

  RVVRouteOperandBindingPlan plan =
      buildComputedMaskMAccRouteOperandBindingPlanFromFacts(facts);
  facts.routeOperandBindingSummary =
      stringifyRVVRouteOperandBindingPlan(plan);
  return facts;
}

std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts>
getRVVRuntimeScalarComputedMaskMAccRouteFacts(
    RVVSelectedBodyOperationKind operation) {
  return getRVVRuntimeScalarComputedMaskMAccRouteFacts(
      operation, tcrv::rvv::getRVVFirstSliceSEWBits(),
      tcrv::rvv::getRVVLMULM1());
}

std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts>
getRVVRuntimeScalarComputedMaskMAccRouteFacts(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul) {
  if (operation != RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd)
    return std::nullopt;
  if (!isComputedMaskMAccConfig(sew, lmul))
    return std::nullopt;

  RVVRuntimeScalarComputedMaskMAccRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm =
      RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskUnitStrideMAcc;
  facts.sew = sew;
  facts.lmul = lmul;
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = kRVVRuntimeScalarComputedMaskedMAccRuntimeABIOrder;
  facts.targetLeafProfile = kRVVRuntimeScalarComputedMaskedMAccTargetLeafProfile;
  facts.providerSupportedMirror =
      kRVVRuntimeScalarComputedMaskedMAccProviderSupportedMirror;
  facts.requiredHeaderDeclarations =
      kRVVRuntimeScalarComputedMaskedMAccRequiredHeaderDeclarations;
  facts.cTypeMappingSummary =
      kRVVRuntimeScalarComputedMaskedMAccCTypeMappingSummary;
  facts.routeOperandBindingPlanID =
      kRVVRuntimeScalarComputedMaskedMAccOperandBindingPlanID;
  facts.typedComputeOpName = "tcrv_rvv.masked_macc";
  facts.arithmeticKind = "add";
  facts.comparePredicateKind = "sle";
  facts.compareLhsRole = "lhs-input-buffer";
  facts.compareRhsRole = "rhs-scalar-value";
  facts.lhsRole = "dot-lhs-input-buffer";
  facts.rhsRole = "dot-rhs-input-buffer";
  facts.accumulatorRole = "accumulator-input-buffer";
  facts.outputRole = "output-buffer";
  facts.runtimeCountRole = "runtime-element-count";
  facts.usesVectorCompareRHSLoad = false;
  facts.usesRuntimeScalarCompareThreshold = true;
  facts.accumulationRouteFamilyPlanID =
      kRVVComputedMaskAccumulationRouteFamilyPlanID;
  facts.accumulationComputeSuffix =
      kRVVComputedMaskAccumulationVectorMAccSuffix;
  facts.accumulationMaskProducerSource =
      kRVVComputedMaskAccumulationRuntimeScalarProducerSource;
  facts.accumulationAccumulatorContract =
      kRVVComputedMaskAccumulationMAccAccumulatorContract;
  facts.accumulationResultContract =
      kRVVComputedMaskAccumulationMAccResultContract;
  facts.maskRole = kRVVMaskedPredicateMaskRole;
  facts.maskSource = kRVVMaskedCompareMaskSource;
  facts.maskMemoryForm = kRVVComputedMaskMemoryMaskMemoryForm;
  facts.inactiveLaneContract =
      "masked-macc-false-lanes-preserve-accumulator";
  facts.maskedPassthroughLayout =
      "accumulator-vector-preserves-inactive-lanes";
  facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  facts.indexedMemoryLayout = kRVVRuntimeScalarComputedMaskedMAccMemoryLayout;
  facts.maccAccumulatorLayout = kRVVMAccAccumulatorLayout;
  facts.maccResultLayout = kRVVMAccResultLayout;
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "rhs_scalar", "int32_t",
      support::RuntimeABIParameterRole::RHSScalarValue,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::DotLHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "rhs", "const int32_t *",
      support::RuntimeABIParameterRole::DotRHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));

  RVVRouteOperandBindingPlan plan =
      buildRuntimeScalarComputedMaskMAccRouteOperandBindingPlanFromFacts(
          facts);
  facts.routeOperandBindingSummary =
      stringifyRVVRouteOperandBindingPlan(plan);
  return facts;
}














llvm::Error validateRVVSelectedBodyPlainMAccRouteFamilyPlan(
    const RVVSelectedBodyPlainMAccRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "plain MAcc route-family runtime AVL/VL control"))
    return error;
  if (plan.operation != RVVSelectedBodyOperationKind::MAccAdd)
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan currently supports only macc_add");
  if (plan.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan requires vector-rhs-load memory form");
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "runtime control plan",
          plan.runtimeControlPlan.controlPlanID,
          getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error =
          requireMAccPlanField("plain MAcc", plan.operation, "family plan",
                               plan.familyPlanID,
                               kRVVPlainMAccRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "runtime ABI order",
          plan.runtimeABIOrder, kRVVMAccRuntimeABIOrder))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "target leaf profile",
          plan.targetLeafProfile, kRVVPlainMAccTargetLeafProfile))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "provider_supported_mirror",
          plan.providerSupportedMirror, kRVVPlainMAccProviderSupportedMirror))
    return error;
  if (llvm::Error error =
          requireMAccPlanField("plain MAcc", plan.operation,
                               "multiply-add arithmetic kind",
                               plan.maccArithmeticKind, "add"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "header declarations",
          plan.requiredHeaderDeclarations, kRVVPlainMAccRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "C type mapping summary",
          plan.cTypeMappingSummary, kRVVPlainMAccCTypeMappingSummary))
    return error;
  if (llvm::Error error =
          requireMAccPlanTypedConfigSnapshot("plain MAcc", plan))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan requires provider-owned header "
        "declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "VL C type", plan.vlCType, "size_t"))
    return error;
  if (llvm::Error error = requireMAccPlanDerivedField(
          "plain MAcc", plan.operation, "vector type", plan.vectorTypeName))
    return error;
  if (llvm::Error error = requireMAccPlanDerivedField(
          "plain MAcc", plan.operation, "vector C type", plan.vectorCType))
    return error;
  if (llvm::Error error = requireMAccPlanDerivedField(
          "plain MAcc", plan.operation, "setvl leaf", plan.setVLIntrinsic))
    return error;
  if (llvm::Error error =
          requireMAccPlanDerivedField("plain MAcc", plan.operation,
                                      "vector-load leaf",
                                      plan.vectorLoadIntrinsic))
    return error;
  std::optional<std::string> expectedMAccIntrinsic =
      deriveMAccIntrinsic(plan.sew, plan.lmul);
  if (!expectedMAccIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan cannot derive MAcc compute leaf from "
        "typed operation/SEW/LMUL facts");
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "MAcc compute leaf",
          plan.maccIntrinsic, *expectedMAccIntrinsic))
    return error;
  if (llvm::Error error = requireMAccPlanDerivedField(
          "plain MAcc", plan.operation, "store leaf", plan.storeIntrinsic))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "result name", plan.resultName,
          "macc_sum_vec"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "accumulator layout",
          plan.accumulatorLayout, kRVVMAccAccumulatorLayout))
    return error;
  if (llvm::Error error =
          requireMAccPlanField("plain MAcc", plan.operation, "result layout",
                               plan.resultLayout, kRVVMAccResultLayout))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyPlainMAccRouteFamilyPlan>
deriveRVVSelectedBodyPlainMAccRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis) {
  if (analysis.slice.arithmeticKind != RVVSelectedBodyOperationKind::MAccAdd)
    return makeRVVEmitCRouteProviderError(
        "requested plain MAcc route-family plan for non-macc_add RVV "
        "operation");
  if (analysis.slice.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan requires vector-rhs-load typed body "
        "structure");
  if (!analysis.typedConfigFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan requires typed RVV config facts");
  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      analysis.typedConfigFacts;
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.rhsGenericLoad ||
      !analysis.slice.accumulatorLoadOperation || !analysis.slice.genericStore ||
      !analysis.slice.maccOp || !analysis.slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan requires explicit lhs load, rhs load, "
        "accumulator load, macc compute, and store body structure");
  if (typedFacts.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      typedFacts.lmul != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan currently requires SEW32 LMUL m1 "
        "typed config");
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      analysis.slice.rhsABI.role !=
          support::RuntimeABIParameterRole::RHSInputBuffer ||
      analysis.slice.accumulatorABI.role !=
          support::RuntimeABIParameterRole::AccumulatorInputBuffer ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan requires lhs buffer, rhs buffer, "
        "accumulator buffer, output buffer, and runtime element-count ABI "
        "roles");
  if (analysis.slice.accumulatorBuffer == analysis.slice.outBuffer)
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan requires separate accumulator input and "
        "output destination ABI values");
  if (*analysis.slice.maccOp.getAccumulatorLayout() !=
          kRVVMAccAccumulatorLayout ||
      *analysis.slice.maccOp.getResultLayout() != kRVVMAccResultLayout)
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan requires explicit MAcc accumulator and "
        "result layout contracts");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL, kRVVMAccRuntimeABIOrder,
          "plain MAcc route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  for (auto [field, leaf] : {std::pair<llvm::StringRef, llvm::StringRef>(
                                 "VL C type", typedFacts.vlCType),
                             {"vector type", typedFacts.vectorTypeName},
                             {"vector C type", typedFacts.vectorCType},
                             {"setvl leaf", typedFacts.setVLIntrinsic},
                             {"vector-load leaf",
                              typedFacts.vectorLoadIntrinsic},
                             {"store leaf", typedFacts.storeIntrinsic}}) {
    if (llvm::Error error = requireMAccTypedConfigLeaf(
            typedFacts, field, leaf, analysis.slice.arithmeticKind))
      return std::move(error);
  }
  std::optional<std::string> maccIntrinsic =
      deriveMAccIntrinsic(typedFacts.sew, typedFacts.lmul);
  if (!maccIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan cannot derive MAcc compute leaf from "
        "typed operation/SEW/LMUL facts");

  RVVSelectedBodyPlainMAccRouteFamilyPlan plan;
  plan.operation = analysis.slice.arithmeticKind;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  applyMAccTypedConfigSnapshot(plan, typedFacts);
  plan.familyPlanID = kRVVPlainMAccRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = kRVVPlainMAccTargetLeafProfile;
  plan.providerSupportedMirror = kRVVPlainMAccProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = kRVVPlainMAccRequiredHeaderDeclarations;
  plan.cTypeMappingSummary = kRVVPlainMAccCTypeMappingSummary;
  plan.vlCType = typedFacts.vlCType;
  plan.vectorTypeName = typedFacts.vectorTypeName;
  plan.vectorCType = typedFacts.vectorCType;
  plan.setVLIntrinsic = typedFacts.setVLIntrinsic;
  plan.vectorLoadIntrinsic = typedFacts.vectorLoadIntrinsic;
  plan.maccIntrinsic = internMAccDerivedLeaf(std::move(*maccIntrinsic));
  plan.storeIntrinsic = typedFacts.storeIntrinsic;
  plan.resultName = "macc_sum_vec";
  plan.accumulatorLayout = *analysis.slice.maccOp.getAccumulatorLayout();
  plan.resultLayout = *analysis.slice.maccOp.getResultLayout();
  plan.maccArithmeticKind = "add";
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.accumulatorABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error = validateRVVSelectedBodyPlainMAccRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyPlainMAccRouteFamilyPlan(
    const RVVSelectedBodyPlainMAccRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyMAccRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                                description);
  description.plainMAccRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.vectorLoadIntrinsic = plan.vectorLoadIntrinsic;
  description.intrinsic = plan.maccIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.resultName = plan.resultName;
  description.maccAccumulatorLayout = plan.accumulatorLayout;
  description.maccResultLayout = plan.resultLayout;
  description.maccArithmeticKind = plan.maccArithmeticKind;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

llvm::Error validateRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan(
    const RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "scalar-broadcast MAcc route-family runtime AVL/VL control"))
    return error;
  if (plan.operation != RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan currently supports only "
        "scalar_broadcast_macc_add");
  if (plan.memoryForm != RVVSelectedBodyMemoryForm::RHSScalarBroadcastMAcc)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan requires "
        "rhs-scalar-broadcast-macc memory form");
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "runtime control plan",
          plan.runtimeControlPlan.controlPlanID,
          getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "family plan",
          plan.familyPlanID, kRVVScalarBroadcastMAccRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "runtime ABI order",
          plan.runtimeABIOrder, kRVVScalarBroadcastMAccRuntimeABIOrder))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "target leaf profile",
          plan.targetLeafProfile, kRVVScalarBroadcastMAccTargetLeafProfile))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "provider_supported_mirror",
          plan.providerSupportedMirror,
          kRVVScalarBroadcastMAccProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation,
          "multiply-add arithmetic kind", plan.maccArithmeticKind, "add"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "header declarations",
          plan.requiredHeaderDeclarations,
          kRVVScalarBroadcastMAccRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "C type mapping summary",
          plan.cTypeMappingSummary, kRVVScalarBroadcastMAccCTypeMappingSummary))
    return error;
  if (llvm::Error error =
          requireMAccPlanTypedConfigSnapshot("scalar-broadcast MAcc", plan))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan requires provider-owned "
        "header declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "VL C type", plan.vlCType,
          "size_t"))
    return error;
  if (llvm::Error error =
          requireMAccPlanDerivedField("scalar-broadcast MAcc", plan.operation,
                                      "vector type", plan.vectorTypeName))
    return error;
  if (llvm::Error error =
          requireMAccPlanDerivedField("scalar-broadcast MAcc", plan.operation,
                                      "vector C type", plan.vectorCType))
    return error;
  if (llvm::Error error =
          requireMAccPlanDerivedField("scalar-broadcast MAcc", plan.operation,
                                      "setvl leaf", plan.setVLIntrinsic))
    return error;
  if (llvm::Error error =
          requireMAccPlanDerivedField("scalar-broadcast MAcc", plan.operation,
                                      "vector-load leaf",
                                      plan.vectorLoadIntrinsic))
    return error;
  std::optional<std::string> expectedSplatIntrinsic =
      deriveMAccScalarSplatIntrinsic(plan.sew, plan.lmul);
  if (!expectedSplatIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan cannot derive RHS scalar "
        "splat leaf from typed SEW/LMUL facts");
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "RHS scalar splat leaf",
          plan.rhsScalarSplatIntrinsic, *expectedSplatIntrinsic))
    return error;
  std::optional<std::string> expectedMAccIntrinsic =
      deriveMAccIntrinsic(plan.sew, plan.lmul);
  if (!expectedMAccIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan cannot derive MAcc compute "
        "leaf from typed operation/SEW/LMUL facts");
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "MAcc compute leaf",
          plan.maccIntrinsic, *expectedMAccIntrinsic))
    return error;
  if (llvm::Error error =
          requireMAccPlanDerivedField("scalar-broadcast MAcc", plan.operation,
                                      "store leaf", plan.storeIntrinsic))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "result name",
          plan.resultName, "macc_sum_vec"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "accumulator layout",
          plan.accumulatorLayout, kRVVMAccAccumulatorLayout))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "result layout",
          plan.resultLayout, kRVVMAccResultLayout))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan>
deriveRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis) {
  if (analysis.slice.arithmeticKind !=
      RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd)
    return makeRVVEmitCRouteProviderError(
        "requested scalar-broadcast MAcc route-family plan for "
        "non-scalar-broadcast-MAcc RVV operation");
  if (analysis.slice.memoryForm !=
      RVVSelectedBodyMemoryForm::RHSScalarBroadcastMAcc)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan requires "
        "rhs-scalar-broadcast-macc typed body structure");
  if (!analysis.typedConfigFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan requires typed RVV config "
        "facts");
  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      analysis.typedConfigFacts;
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.rhsScalarSplat ||
      !analysis.slice.accumulatorLoadOperation || !analysis.slice.genericStore ||
      !analysis.slice.maccOp || !analysis.slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan requires explicit load, "
        "scalar splat, accumulator load, macc compute, and store body "
        "structure");
  if (typedFacts.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      typedFacts.lmul != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan currently requires SEW32 "
        "LMUL m1 typed config");
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      analysis.slice.rhsABI.role !=
          support::RuntimeABIParameterRole::RHSScalarValue ||
      analysis.slice.accumulatorABI.role !=
          support::RuntimeABIParameterRole::AccumulatorInputBuffer ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan requires lhs buffer, RHS "
        "scalar, accumulator buffer, output buffer, and runtime "
        "element-count ABI roles");
  if (*analysis.slice.maccOp.getAccumulatorLayout() !=
          kRVVMAccAccumulatorLayout ||
      *analysis.slice.maccOp.getResultLayout() != kRVVMAccResultLayout)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan requires explicit MAcc "
        "accumulator and result layout contracts");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          kRVVScalarBroadcastMAccRuntimeABIOrder,
          "scalar-broadcast MAcc route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  for (auto [field, leaf] : {std::pair<llvm::StringRef, llvm::StringRef>(
                                 "VL C type", typedFacts.vlCType),
                             {"vector type", typedFacts.vectorTypeName},
                             {"vector C type", typedFacts.vectorCType},
                             {"setvl leaf", typedFacts.setVLIntrinsic},
                             {"vector-load leaf",
                              typedFacts.vectorLoadIntrinsic},
                             {"scalar-splat leaf",
                              typedFacts.scalarSplatIntrinsic},
                             {"store leaf", typedFacts.storeIntrinsic}}) {
    if (llvm::Error error = requireMAccTypedConfigLeaf(
            typedFacts, field, leaf, analysis.slice.arithmeticKind))
      return std::move(error);
  }
  std::optional<std::string> maccIntrinsic =
      deriveMAccIntrinsic(typedFacts.sew, typedFacts.lmul);
  if (!maccIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan cannot derive MAcc compute "
        "leaf from typed operation/SEW/LMUL facts");

  RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan plan;
  plan.operation = analysis.slice.arithmeticKind;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  applyMAccTypedConfigSnapshot(plan, typedFacts);
  plan.familyPlanID = kRVVScalarBroadcastMAccRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = kRVVScalarBroadcastMAccTargetLeafProfile;
  plan.providerSupportedMirror = kRVVScalarBroadcastMAccProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations =
      kRVVScalarBroadcastMAccRequiredHeaderDeclarations;
  plan.cTypeMappingSummary = kRVVScalarBroadcastMAccCTypeMappingSummary;
  plan.vlCType = typedFacts.vlCType;
  plan.vectorTypeName = typedFacts.vectorTypeName;
  plan.vectorCType = typedFacts.vectorCType;
  plan.setVLIntrinsic = typedFacts.setVLIntrinsic;
  plan.vectorLoadIntrinsic = typedFacts.vectorLoadIntrinsic;
  plan.rhsScalarSplatIntrinsic = typedFacts.scalarSplatIntrinsic;
  plan.maccIntrinsic = internMAccDerivedLeaf(std::move(*maccIntrinsic));
  plan.storeIntrinsic = typedFacts.storeIntrinsic;
  plan.resultName = "macc_sum_vec";
  plan.accumulatorLayout = *analysis.slice.maccOp.getAccumulatorLayout();
  plan.resultLayout = *analysis.slice.maccOp.getResultLayout();
  plan.maccArithmeticKind = "add";
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.accumulatorABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error =
          validateRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan(
    const RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyMAccRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                                description);
  description.scalarBroadcastMAccRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.vectorLoadIntrinsic = plan.vectorLoadIntrinsic;
  description.rhsBroadcastIntrinsic = plan.rhsScalarSplatIntrinsic;
  description.intrinsic = plan.maccIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.resultName = plan.resultName;
  description.maccAccumulatorLayout = plan.accumulatorLayout;
  description.maccResultLayout = plan.resultLayout;
  description.maccArithmeticKind = plan.maccArithmeticKind;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

llvm::Error validateRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(
    const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "computed-mask accumulation route-family runtime AVL/VL control"))
    return error;
  const bool isMAcc =
      plan.operation == RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
      plan.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
  const bool isStandaloneReduction =
      isComputedMaskStandaloneReductionOperation(plan.operation) ||
      isRuntimeScalarComputedMaskStandaloneReductionOperation(plan.operation);
  const bool isRuntimeScalarProducer =
      plan.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd ||
      isRuntimeScalarComputedMaskStandaloneReductionOperation(plan.operation);
  const bool isVectorCompareProducer = !isRuntimeScalarProducer;
  const std::optional<RVVComputedMaskMAccRouteFacts> computedMaskMAccFacts =
      getRVVComputedMaskMAccRouteFacts(plan.operation);
  const std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts>
      runtimeScalarMAccFacts =
          getRVVRuntimeScalarComputedMaskMAccRouteFacts(plan.operation);
  if (!isMAcc && !isStandaloneReduction)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan supports only masked "
        "macc or masked standalone reduction consumers");
  if (plan.operation == RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd &&
      !computedMaskMAccFacts)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires "
        "provider-owned computed-mask MAcc route facts for "
        "computed_masked_macc_add");
  if (plan.usesVectorMAccSuffix != isMAcc)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires "
        "usesVectorMAccSuffix to mirror the selected operation");
  if (plan.usesScalarHorizontalReductionSuffix != isStandaloneReduction)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires "
        "usesScalarHorizontalReductionSuffix to mirror the selected operation");
  if (plan.usesVectorCompareProducer != isVectorCompareProducer)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires "
        "usesVectorCompareProducer to mirror the selected mask producer");
  if (plan.usesRuntimeScalarProducer != isRuntimeScalarProducer)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires "
        "usesRuntimeScalarProducer to mirror the selected mask producer");

  const RVVSelectedBodyMemoryForm expectedMemoryForm =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->memoryForm
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->memoryForm
          : isMAcc
                ? (isRuntimeScalarProducer
                       ? RVVSelectedBodyMemoryForm::
                             RuntimeScalarComputedMaskUnitStrideMAcc
                       : RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideMAcc)
                : (isRuntimeScalarProducer
                       ? RVVSelectedBodyMemoryForm::
                             RuntimeScalarComputedMaskUnitStrideStandaloneReduction
                       : RVVSelectedBodyMemoryForm::
                             ComputedMaskUnitStrideStandaloneReduction);
  llvm::StringRef expectedSuffix =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->accumulationComputeSuffix
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->accumulationComputeSuffix
          : isMAcc ? kRVVComputedMaskAccumulationVectorMAccSuffix
                   : kRVVComputedMaskAccumulationStandaloneReductionSuffix;
  llvm::StringRef expectedProducerSource =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->accumulationMaskProducerSource
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->accumulationMaskProducerSource
      : isRuntimeScalarProducer
          ? kRVVComputedMaskAccumulationRuntimeScalarProducerSource
          : kRVVComputedMaskAccumulationVectorCompareProducerSource;
  llvm::StringRef expectedRuntimeABIOrder =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->runtimeABIOrder
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->runtimeABIOrder
      : isMAcc
          ? (isRuntimeScalarProducer
                 ? llvm::StringRef(
                       kRVVRuntimeScalarComputedMaskedMAccRuntimeABIOrder)
                 : llvm::StringRef(kRVVComputedMaskedMAccRuntimeABIOrder))
          : (isRuntimeScalarProducer
                 ? llvm::StringRef("cmp_lhs,rhs_scalar,src,acc,out,n")
                 : llvm::StringRef("cmp_lhs,cmp_rhs,src,acc,out,n"));
  llvm::StringRef expectedTargetLeafProfile =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->targetLeafProfile
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->targetLeafProfile
      : isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccTargetLeafProfile
                 : kRVVComputedMaskedMAccTargetLeafProfile)
          : (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskStandaloneReductionTargetLeafProfile
                 : kRVVComputedMaskStandaloneReductionTargetLeafProfile);
  llvm::StringRef expectedProviderSupportedMirror =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->providerSupportedMirror
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->providerSupportedMirror
      : isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccProviderSupportedMirror
                 : kRVVComputedMaskedMAccProviderSupportedMirror)
          : (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskStandaloneReductionProviderSupportedMirror
                 : kRVVComputedMaskStandaloneReductionProviderSupportedMirror);
  llvm::StringRef expectedHeaderDeclarations =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->requiredHeaderDeclarations
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->requiredHeaderDeclarations
      : isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccRequiredHeaderDeclarations
                 : kRVVComputedMaskedMAccRequiredHeaderDeclarations)
          : kRVVStandaloneReductionRequiredHeaderDeclarations;
  llvm::StringRef expectedCTypeMappingSummary =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->cTypeMappingSummary
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->cTypeMappingSummary
      : isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccCTypeMappingSummary
                 : kRVVComputedMaskedMAccCTypeMappingSummary)
          : (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskStandaloneReductionCTypeMappingSummary
                 : kRVVComputedMaskStandaloneReductionCTypeMappingSummary);
  llvm::StringRef expectedAccumulatorContract =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->accumulationAccumulatorContract
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->accumulationAccumulatorContract
      : isMAcc ? kRVVComputedMaskAccumulationMAccAccumulatorContract
               : kRVVComputedMaskAccumulationReductionAccumulatorContract;
  llvm::StringRef expectedResultContract =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->accumulationResultContract
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->accumulationResultContract
      : isMAcc ? kRVVComputedMaskAccumulationMAccResultContract
               : kRVVComputedMaskAccumulationReductionResultContract;
  llvm::StringRef expectedInactiveLaneContract =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->inactiveLaneContract
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->inactiveLaneContract
      : isMAcc
          ? llvm::StringRef("masked-macc-false-lanes-preserve-accumulator")
          : getStandaloneReductionInactiveLaneRequirement(plan.operation);
  llvm::StringRef expectedPassthroughLayout =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->maskedPassthroughLayout
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->maskedPassthroughLayout
          : isMAcc ? llvm::StringRef("accumulator-vector-preserves-inactive-lanes")
                   : llvm::StringRef();
  llvm::StringRef expectedMaskRole =
      computedMaskMAccFacts ? computedMaskMAccFacts->maskRole
      : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->maskRole
                             : llvm::StringRef(kRVVMaskedPredicateMaskRole);
  llvm::StringRef expectedMaskSource =
      computedMaskMAccFacts ? computedMaskMAccFacts->maskSource
      : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->maskSource
                             : llvm::StringRef(kRVVMaskedCompareMaskSource);
  llvm::StringRef expectedMaskMemoryForm =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->maskMemoryForm
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->maskMemoryForm
          : llvm::StringRef(kRVVComputedMaskMemoryMaskMemoryForm);
  llvm::StringRef expectedSourceMemoryForm =
      computedMaskMAccFacts ? computedMaskMAccFacts->sourceMemoryForm
      : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->sourceMemoryForm
                             : llvm::StringRef(kRVVUnitStrideSourceMemoryForm);
  llvm::StringRef expectedDestinationMemoryForm =
      computedMaskMAccFacts ? computedMaskMAccFacts->destinationMemoryForm
      : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->destinationMemoryForm
                             : llvm::StringRef(kRVVDestinationMemoryForm);
  llvm::StringRef expectedScalarCarry =
      isStandaloneReduction
          ? llvm::StringRef(
                kRVVComputedMaskAccumulationReductionScalarCarryContract)
          : llvm::StringRef();
  llvm::StringRef expectedPredicate =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->comparePredicateKind
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->comparePredicateKind
          : (isMAcc && !isRuntimeScalarProducer ? llvm::StringRef("slt")
                                                : llvm::StringRef("sle"));
  std::optional<std::string> expectedSetVLLeaf =
      deriveMAccSetVLIntrinsic(plan.runtimeControlPlan.sew,
                               plan.runtimeControlPlan.lmul);
  if (!expectedSetVLLeaf)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires a "
        "provider-derived setvl leaf for the selected typed config");
  std::optional<std::string> expectedRHSSplatLeaf =
      isRuntimeScalarProducer
          ? deriveMAccScalarSplatIntrinsic(plan.runtimeControlPlan.sew,
                                           plan.runtimeControlPlan.lmul)
          : std::optional<std::string>(std::string());
  if (isRuntimeScalarProducer && !expectedRHSSplatLeaf)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires a "
        "provider-derived RHS scalar splat leaf for the selected typed "
        "config");
  std::optional<std::string> expectedCompareLeaf =
      deriveMAccCompareIntrinsic(expectedPredicate, plan.runtimeControlPlan.sew,
                                 plan.runtimeControlPlan.lmul);
  if (!expectedCompareLeaf)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires a "
        "provider-derived compare leaf for the selected typed config");
  std::optional<std::string> expectedMAccLeaf =
      isMAcc ? deriveMAccIntrinsic(plan.runtimeControlPlan.sew,
                                   plan.runtimeControlPlan.lmul)
             : std::optional<std::string>(std::string());
  if (isMAcc && !expectedMAccLeaf)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires a "
        "provider-derived MAcc leaf for the selected typed config");
  std::optional<std::string> expectedMaskedMergeLeaf =
      isMAcc ? deriveMAccMaskedMergeIntrinsic(plan.runtimeControlPlan.sew,
                                              plan.runtimeControlPlan.lmul)
             : std::optional<std::string>(std::string());
  if (isMAcc && !expectedMaskedMergeLeaf)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires a "
        "provider-derived masked merge leaf for the selected typed config");
  std::optional<std::string> expectedStoreLeaf =
      isStandaloneReduction
          ? deriveStandaloneReductionScalarResultStoreIntrinsic(
                plan.runtimeControlPlan.sew, plan.runtimeControlPlan.lmul)
          : deriveMAccVectorStoreIntrinsic(plan.runtimeControlPlan.sew,
                                           plan.runtimeControlPlan.lmul);
  if (!expectedStoreLeaf)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires a "
        "provider-derived store leaf for the selected scalar result channel");

  if (plan.memoryForm != expectedMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires "
        "the operation-specific accumulation memory form");
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "runtime control plan",
          plan.runtimeControlPlan.controlPlanID,
          getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "family plan",
          plan.familyPlanID, kRVVComputedMaskAccumulationRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "compute suffix",
          plan.computeSuffix, expectedSuffix))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "mask producer source",
          plan.maskProducerSource, expectedProducerSource))
    return error;
  if (isMAcc)
    if (llvm::Error error = requireMAccPlanField(
            "computed-mask accumulation", plan.operation,
            "multiply-add arithmetic kind", plan.maccArithmeticKind, "add"))
      return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "runtime ABI order",
          plan.runtimeABIOrder, expectedRuntimeABIOrder))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "target leaf profile",
          plan.targetLeafProfile, expectedTargetLeafProfile))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation,
          "provider_supported_mirror", plan.providerSupportedMirror,
          expectedProviderSupportedMirror))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires "
        "provider-owned header declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "header declarations",
          plan.requiredHeaderDeclarations, expectedHeaderDeclarations))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation,
          "C type mapping summary", plan.cTypeMappingSummary,
          expectedCTypeMappingSummary))
    return error;
  if (llvm::Error error = requireMAccPlanTypedConfigSnapshot(
          "computed-mask accumulation", plan))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "VL C type",
          plan.vlCType, "size_t"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "setvl leaf",
          plan.setVLIntrinsic, *expectedSetVLLeaf))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "RHS scalar splat leaf",
          plan.rhsScalarSplatIntrinsic, *expectedRHSSplatLeaf))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "MAcc leaf",
          plan.maccIntrinsic, *expectedMAccLeaf))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "compare leaf",
          plan.compareIntrinsic, *expectedCompareLeaf))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "masked merge leaf",
          plan.maskedMergeIntrinsic, *expectedMaskedMergeLeaf))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "store leaf",
          plan.storeIntrinsic, *expectedStoreLeaf))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "mask role",
          plan.maskRole, expectedMaskRole))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "mask source",
          plan.maskSource, expectedMaskSource))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "mask memory form",
          plan.maskMemoryForm, expectedMaskMemoryForm))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "source memory form",
          plan.sourceMemoryForm, expectedSourceMemoryForm))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation,
          "destination memory form", plan.destinationMemoryForm,
          expectedDestinationMemoryForm))
    return error;
  if (isMAcc) {
    llvm::StringRef expectedMemoryLayout =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->indexedMemoryLayout
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->indexedMemoryLayout
        : isRuntimeScalarProducer
            ? llvm::StringRef(kRVVRuntimeScalarComputedMaskedMAccMemoryLayout)
            : llvm::StringRef(kRVVComputedMaskedMAccMemoryLayout);
    if (llvm::Error error = requireMAccPlanField(
            "computed-mask accumulation", plan.operation,
            "indexed memory layout", plan.indexedMemoryLayout,
            expectedMemoryLayout))
      return error;
    if (llvm::Error error = requireMAccPlanField(
            "computed-mask accumulation", plan.operation,
            "accumulator layout", plan.accumulatorLayout,
            computedMaskMAccFacts
                ? computedMaskMAccFacts->maccAccumulatorLayout
            : runtimeScalarMAccFacts
                ? runtimeScalarMAccFacts->maccAccumulatorLayout
                : llvm::StringRef(kRVVMAccAccumulatorLayout)))
      return error;
    if (llvm::Error error = requireMAccPlanField(
            "computed-mask accumulation", plan.operation, "result layout",
            plan.resultLayout,
            computedMaskMAccFacts
                ? computedMaskMAccFacts->maccResultLayout
            : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->maccResultLayout
                                     : llvm::StringRef(kRVVMAccResultLayout)))
      return error;
  } else {
    if (llvm::Error error = requireMAccPlanField(
            "computed-mask accumulation", plan.operation,
            "indexed memory layout", plan.indexedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireMAccPlanField(
            "computed-mask accumulation", plan.operation,
            "accumulator layout", plan.accumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireMAccPlanField(
            "computed-mask accumulation", plan.operation, "result layout",
            plan.resultLayout, ""))
      return error;
  }
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "accumulator contract",
          plan.accumulatorContract, expectedAccumulatorContract))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "result contract",
          plan.resultContract, expectedResultContract))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation,
          "inactive-lane contract", plan.inactiveLaneContract,
          expectedInactiveLaneContract))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation,
          "masked passthrough layout", plan.maskedPassthroughLayout,
          expectedPassthroughLayout))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation,
          "scalar carry contract", plan.scalarCarryContract,
          expectedScalarCarry))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan>
deriveRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyOperationKind operation = analysis.slice.arithmeticKind;
  if (!isRVVSelectedBodyComputedMaskAccumulationRouteOperation(operation))
    return makeRVVEmitCRouteProviderError(
        "requested computed-mask accumulation route-family "
        "plan for non-accumulation RVV operation");
  if (!analysis.typedConfigFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires typed RVV "
        "config facts");
  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      analysis.typedConfigFacts;
  const bool isMAcc =
      operation == RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
      operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
  const bool isStandaloneReduction = !isMAcc;
  const bool isRuntimeScalarProducer =
      operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd ||
      isRuntimeScalarComputedMaskStandaloneReductionOperation(operation);
  const std::optional<RVVComputedMaskMAccRouteFacts> computedMaskMAccFacts =
      getRVVComputedMaskMAccRouteFacts(operation, typedFacts.sew,
                                       typedFacts.lmul);
  const std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts>
      runtimeScalarMAccFacts =
          getRVVRuntimeScalarComputedMaskMAccRouteFacts(operation,
                                                        typedFacts.sew,
                                                        typedFacts.lmul);
  if (operation == RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd &&
      !computedMaskMAccFacts)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires "
        "provider-owned computed-mask MAcc route facts for "
        "computed_masked_macc_add");
  if (operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd &&
      !runtimeScalarMAccFacts)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires "
        "provider-owned runtime-scalar computed-mask MAcc route facts for "
        "runtime_scalar_cmp_masked_macc_add");
  const RVVSelectedBodyMemoryForm expectedMemoryForm =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->memoryForm
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->memoryForm
      : isMAcc
          ? (isRuntimeScalarProducer
                 ? RVVSelectedBodyMemoryForm::
                       RuntimeScalarComputedMaskUnitStrideMAcc
                 : RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideMAcc)
          : (isRuntimeScalarProducer
                 ? RVVSelectedBodyMemoryForm::
                       RuntimeScalarComputedMaskUnitStrideStandaloneReduction
                 : RVVSelectedBodyMemoryForm::
                       ComputedMaskUnitStrideStandaloneReduction);
  if (analysis.slice.memoryForm != expectedMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires "
        "the selected typed body to carry the operation-specific accumulation "
        "memory form");
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.compareOp ||
      !analysis.slice.genericStore ||
      (isRuntimeScalarProducer && !analysis.slice.rhsScalarSplat) ||
      (!isRuntimeScalarProducer && !analysis.slice.rhsGenericLoad) ||
      (isMAcc &&
       (!analysis.slice.dotLHSGenericLoad || !analysis.slice.dotRHSGenericLoad ||
        !analysis.slice.accumulatorLoadOperation ||
        !analysis.slice.maskedMAccOp)) ||
      (!isMAcc &&
       (!analysis.slice.sourceGenericLoad ||
        !analysis.slice.maskedStandaloneReduceOp)))
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires typed compare "
        "loads or runtime scalar splat producer, compare-produced mask, "
        "accumulator/result body structure, and route-specific payload suffix");
  if (isRuntimeScalarProducer && isStandaloneReduction) {
    if (!isRuntimeScalarComputedMaskStandaloneReductionConfig(
            typedFacts.sew, typedFacts.lmul))
      return makeRVVEmitCRouteProviderError(
          "computed-mask accumulation route-family plan requires runtime-"
          "scalar standalone reduction typed config to be SEW32 LMUL m1, "
          "SEW32 LMUL m2, or SEW64 LMUL m1 with a separate LMUL m1 scalar "
          "reduction accumulator/result channel");
  } else if (isStandaloneReduction) {
    if (!isStandaloneReductionScalarChannelConfig(
            typedFacts.sew, typedFacts.lmul))
      return makeRVVEmitCRouteProviderError(
          "computed-mask accumulation route-family plan requires "
          "non-runtime-scalar standalone reduction typed config to be SEW32 "
          "LMUL m1 or SEW32 LMUL m2 with a separate LMUL m1 scalar reduction "
          "accumulator/result channel");
  } else if (isMAcc && !isComputedMaskMAccConfig(typedFacts.sew,
                                                typedFacts.lmul)) {
    return makeRVVEmitCRouteProviderError(
        isRuntimeScalarProducer
            ? "computed-mask accumulation route-family plan requires runtime "
              "scalar masked macc config to be SEW32 LMUL m1 or SEW32 LMUL m2"
            : "computed-mask accumulation route-family plan requires vector "
              "masked macc config to be SEW32 LMUL m1 or SEW32 LMUL m2");
  }
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      analysis.slice.rhsABI.role !=
          (isRuntimeScalarProducer
               ? support::RuntimeABIParameterRole::RHSScalarValue
               : support::RuntimeABIParameterRole::RHSInputBuffer) ||
      (isMAcc &&
       (analysis.slice.dotLHSABI.role !=
            support::RuntimeABIParameterRole::DotLHSInputBuffer ||
        analysis.slice.dotRHSABI.role !=
            support::RuntimeABIParameterRole::DotRHSInputBuffer)) ||
      (!isMAcc &&
       analysis.slice.sourceABI.role !=
           support::RuntimeABIParameterRole::SourceInputBuffer) ||
      analysis.slice.accumulatorABI.role !=
          support::RuntimeABIParameterRole::AccumulatorInputBuffer ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires compare lhs, "
        "producer RHS, payload, accumulator, output, and runtime element-count "
        "ABI roles");

  llvm::StringRef runtimeABIOrder =
      runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->runtimeABIOrder
      : isMAcc
          ? (isRuntimeScalarProducer
                 ? llvm::StringRef(
                       kRVVRuntimeScalarComputedMaskedMAccRuntimeABIOrder)
                 : llvm::StringRef(kRVVComputedMaskedMAccRuntimeABIOrder))
          : (isRuntimeScalarProducer
                 ? llvm::StringRef("cmp_lhs,rhs_scalar,src,acc,out,n")
                 : llvm::StringRef("cmp_lhs,cmp_rhs,src,acc,out,n"));
  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL, runtimeABIOrder,
          "computed-mask accumulation route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  for (auto [field, leaf] : {std::pair<llvm::StringRef, llvm::StringRef>(
                                 "VL C type", typedFacts.vlCType),
                             {"vector type", typedFacts.vectorTypeName},
                             {"vector C type", typedFacts.vectorCType},
                             {"mask type", typedFacts.maskTypeName},
                             {"mask C type", typedFacts.maskCType},
                             {"setvl leaf", typedFacts.setVLIntrinsic},
                             {"vector-load leaf",
                              typedFacts.vectorLoadIntrinsic},
                             {"store leaf", typedFacts.storeIntrinsic}}) {
    if (llvm::Error error =
            requireMAccTypedConfigLeaf(typedFacts, field, leaf, operation))
      return std::move(error);
  }
  if (isRuntimeScalarProducer)
    if (llvm::Error error = requireMAccTypedConfigLeaf(
            typedFacts, "scalar-splat leaf", typedFacts.scalarSplatIntrinsic,
            operation))
      return std::move(error);

  std::optional<std::string> maccIntrinsic =
      isMAcc ? deriveMAccIntrinsic(typedFacts.sew, typedFacts.lmul)
             : std::optional<std::string>(std::string());
  if (isMAcc && !maccIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan cannot derive MAcc "
        "leaf from typed operation/SEW/LMUL facts");
  std::optional<std::string> compareIntrinsic = deriveMAccCompareIntrinsic(
      computedMaskMAccFacts
          ? computedMaskMAccFacts->comparePredicateKind
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->comparePredicateKind
          : (isMAcc && !isRuntimeScalarProducer ? llvm::StringRef("slt")
                                                : llvm::StringRef("sle")),
      typedFacts.sew, typedFacts.lmul);
  if (!compareIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan cannot derive compare "
        "leaf from typed predicate/SEW/LMUL facts");
  std::optional<std::string> maskedMergeIntrinsic =
      isMAcc ? deriveMAccMaskedMergeIntrinsic(typedFacts.sew, typedFacts.lmul)
             : std::optional<std::string>(std::string());
  if (isMAcc && !maskedMergeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan cannot derive masked "
        "merge leaf from typed SEW/LMUL facts");
  std::optional<std::string> standaloneStoreIntrinsic =
      isStandaloneReduction
          ? deriveStandaloneReductionScalarResultStoreIntrinsic(typedFacts.sew,
                                                                typedFacts.lmul)
          : std::optional<std::string>();
  if (isStandaloneReduction && !standaloneStoreIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan cannot derive "
        "standalone reduction scalar-result store leaf from typed SEW/LMUL "
        "facts");

  RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesVectorMAccSuffix = isMAcc;
  plan.usesScalarHorizontalReductionSuffix = !isMAcc;
  plan.usesRuntimeScalarProducer = isRuntimeScalarProducer;
  plan.usesVectorCompareProducer = !isRuntimeScalarProducer;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  applyMAccTypedConfigSnapshot(plan, typedFacts);
  plan.familyPlanID = kRVVComputedMaskAccumulationRouteFamilyPlanID;
  plan.computeSuffix =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->accumulationComputeSuffix
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->accumulationComputeSuffix
          : isMAcc ? kRVVComputedMaskAccumulationVectorMAccSuffix
                   : kRVVComputedMaskAccumulationStandaloneReductionSuffix;
  plan.maskProducerSource =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->accumulationMaskProducerSource
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->accumulationMaskProducerSource
      : isRuntimeScalarProducer
          ? kRVVComputedMaskAccumulationRuntimeScalarProducerSource
          : kRVVComputedMaskAccumulationVectorCompareProducerSource;
  plan.maccArithmeticKind =
      computedMaskMAccFacts ? computedMaskMAccFacts->arithmeticKind
      : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->arithmeticKind
                               : isMAcc ? llvm::StringRef("add")
                                        : llvm::StringRef();
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->targetLeafProfile
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->targetLeafProfile
      : isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccTargetLeafProfile
                 : kRVVComputedMaskedMAccTargetLeafProfile)
          : (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskStandaloneReductionTargetLeafProfile
                 : kRVVComputedMaskStandaloneReductionTargetLeafProfile);
  plan.providerSupportedMirror =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->providerSupportedMirror
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->providerSupportedMirror
      : isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccProviderSupportedMirror
                 : kRVVComputedMaskedMAccProviderSupportedMirror)
          : (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskStandaloneReductionProviderSupportedMirror
                 : kRVVComputedMaskStandaloneReductionProviderSupportedMirror);
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->requiredHeaderDeclarations
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->requiredHeaderDeclarations
      : isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccRequiredHeaderDeclarations
                 : kRVVComputedMaskedMAccRequiredHeaderDeclarations)
          : kRVVStandaloneReductionRequiredHeaderDeclarations;
  plan.cTypeMappingSummary =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->cTypeMappingSummary
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->cTypeMappingSummary
      : isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccCTypeMappingSummary
                 : kRVVComputedMaskedMAccCTypeMappingSummary)
          : (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskStandaloneReductionCTypeMappingSummary
                 : kRVVComputedMaskStandaloneReductionCTypeMappingSummary);
  plan.vlCType = typedFacts.vlCType;
  plan.vectorTypeName = typedFacts.vectorTypeName;
  plan.vectorCType = typedFacts.vectorCType;
  plan.maskTypeName = typedFacts.maskTypeName;
  plan.maskCType = typedFacts.maskCType;
  plan.setVLIntrinsic = typedFacts.setVLIntrinsic;
  plan.vectorLoadIntrinsic = typedFacts.vectorLoadIntrinsic;
  plan.rhsScalarSplatIntrinsic =
      isRuntimeScalarProducer ? typedFacts.scalarSplatIntrinsic : "";
  plan.maccIntrinsic = isMAcc ? internMAccDerivedLeaf(std::move(*maccIntrinsic))
                              : llvm::StringRef();
  plan.compareIntrinsic = internMAccDerivedLeaf(std::move(*compareIntrinsic));
  plan.maskedMergeIntrinsic =
      isMAcc ? internMAccDerivedLeaf(std::move(*maskedMergeIntrinsic))
             : llvm::StringRef();
  plan.storeIntrinsic =
      isStandaloneReduction ? internMAccDerivedLeaf(
                                  std::move(*standaloneStoreIntrinsic))
                            : typedFacts.storeIntrinsic;
  plan.maskRole = computedMaskMAccFacts ? computedMaskMAccFacts->maskRole
                  : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->maskRole
                                         : kRVVMaskedPredicateMaskRole;
  plan.maskSource = computedMaskMAccFacts ? computedMaskMAccFacts->maskSource
                    : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->maskSource
                                           : kRVVMaskedCompareMaskSource;
  plan.maskMemoryForm = computedMaskMAccFacts
                            ? computedMaskMAccFacts->maskMemoryForm
                        : runtimeScalarMAccFacts
                            ? runtimeScalarMAccFacts->maskMemoryForm
                            : kRVVComputedMaskMemoryMaskMemoryForm;
  plan.sourceMemoryForm =
      computedMaskMAccFacts ? computedMaskMAccFacts->sourceMemoryForm
      : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->sourceMemoryForm
                             : llvm::StringRef(kRVVUnitStrideSourceMemoryForm);
  plan.destinationMemoryForm =
      computedMaskMAccFacts ? computedMaskMAccFacts->destinationMemoryForm
      : runtimeScalarMAccFacts ? runtimeScalarMAccFacts->destinationMemoryForm
                             : llvm::StringRef(kRVVDestinationMemoryForm);
  if (isMAcc) {
    plan.indexedMemoryLayout =
        computedMaskMAccFacts
            ? computedMaskMAccFacts->indexedMemoryLayout
        : runtimeScalarMAccFacts
            ? runtimeScalarMAccFacts->indexedMemoryLayout
            : isRuntimeScalarProducer
                  ? llvm::StringRef(kRVVRuntimeScalarComputedMaskedMAccMemoryLayout)
                  : llvm::StringRef(kRVVComputedMaskedMAccMemoryLayout);
    plan.accumulatorLayout = analysis.slice.maskedMAccOp.getAccumulatorLayout();
    plan.resultLayout = analysis.slice.maskedMAccOp.getResultLayout();
  }
  plan.accumulatorContract =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->accumulationAccumulatorContract
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->accumulationAccumulatorContract
      : isMAcc ? kRVVComputedMaskAccumulationMAccAccumulatorContract
               : kRVVComputedMaskAccumulationReductionAccumulatorContract;
  plan.resultContract =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->accumulationResultContract
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->accumulationResultContract
      : isMAcc ? kRVVComputedMaskAccumulationMAccResultContract
               : kRVVComputedMaskAccumulationReductionResultContract;
  plan.inactiveLaneContract =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->inactiveLaneContract
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->inactiveLaneContract
      : isMAcc
          ? llvm::StringRef("masked-macc-false-lanes-preserve-accumulator")
          : getStandaloneReductionInactiveLaneRequirement(operation);
  plan.maskedPassthroughLayout =
      computedMaskMAccFacts
          ? computedMaskMAccFacts->maskedPassthroughLayout
      : runtimeScalarMAccFacts
          ? runtimeScalarMAccFacts->maskedPassthroughLayout
          : isMAcc ? llvm::StringRef("accumulator-vector-preserves-inactive-lanes")
                   : llvm::StringRef();
  plan.scalarCarryContract =
      isMAcc
          ? llvm::StringRef()
          : llvm::StringRef(
                kRVVComputedMaskAccumulationReductionScalarCarryContract);
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  if (isMAcc) {
    plan.runtimeABIParameters.push_back(analysis.slice.dotLHSABI);
    plan.runtimeABIParameters.push_back(analysis.slice.dotRHSABI);
  } else {
    plan.runtimeABIParameters.push_back(analysis.slice.sourceABI);
  }
  plan.runtimeABIParameters.push_back(analysis.slice.accumulatorABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error =
          validateRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(
    const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyMAccRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                                description);
  description.accumulationRouteFamilyPlanID = plan.familyPlanID;
  description.accumulationComputeSuffix = plan.computeSuffix;
  description.accumulationMaskProducerSource = plan.maskProducerSource;
  description.accumulationAccumulatorContract = plan.accumulatorContract;
  description.accumulationResultContract = plan.resultContract;
  description.accumulationScalarCarryContract = plan.scalarCarryContract;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.maskTypeName = plan.maskTypeName;
  description.maskCType = plan.maskCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.vectorLoadIntrinsic = plan.vectorLoadIntrinsic;
  description.rhsBroadcastIntrinsic = plan.rhsScalarSplatIntrinsic;
  if (plan.usesVectorMAccSuffix) {
    description.intrinsic = plan.maccIntrinsic;
    description.maskedMergeIntrinsic = plan.maskedMergeIntrinsic;
    description.maccArithmeticKind = plan.maccArithmeticKind;
  }
  description.compareIntrinsic = plan.compareIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.maskRole = plan.maskRole;
  description.maskSource = plan.maskSource;
  description.maskMemoryForm = plan.maskMemoryForm;
  if (plan.usesVectorMAccSuffix) {
    description.sourceMemoryForm = plan.sourceMemoryForm;
    description.destinationMemoryForm = plan.destinationMemoryForm;
    description.indexedMemoryLayout = plan.indexedMemoryLayout;
    description.inactiveLaneContract = plan.inactiveLaneContract;
    description.maskedPassthroughLayout = plan.maskedPassthroughLayout;
    description.maccAccumulatorLayout = plan.accumulatorLayout;
    description.maccResultLayout = plan.resultLayout;
  }
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyMAccRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::MAccAdd:
    return kRVVMAccOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
    return kRVVScalarBroadcastMAccOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd: {
    std::optional<RVVComputedMaskMAccRouteFacts> routeFacts =
        getRVVComputedMaskMAccRouteFacts(operation);
    return routeFacts ? routeFacts->routeOperandBindingPlanID
                      : llvm::StringRef();
  }
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd: {
    std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts> routeFacts =
        getRVVRuntimeScalarComputedMaskMAccRouteFacts(operation);
    return routeFacts ? routeFacts->routeOperandBindingPlanID
                      : llvm::StringRef();
  }
  default:
    return std::nullopt;
  }
}

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodyMAccRouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand) {
  using support::RuntimeABIParameterRole;
  if (planID == kRVVMAccOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVScalarBroadcastMAccOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVComputedMaskedMAccOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::DotLHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::DotRHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVRuntimeScalarComputedMaskedMAccOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::DotLHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::DotRHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  return std::nullopt;
}

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyMAccRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyRouteSlice &slice = analysis.slice;
  RVVRouteOperandBindingPlan plan;
  llvm::StringRef expectedRuntimeABIOrder;
  llvm::StringRef context;

  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd) {
    plan.planID = kRVVMAccOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVMAccRuntimeABIOrder;
    context = "macc_add route";
    addMAccRouteOperandBinding(
        plan, "lhs", slice.lhsABI, {"abi", "lhs-load", "macc-lhs", "hdr"});
    addMAccRouteOperandBinding(
        plan, "rhs", slice.rhsABI, {"abi", "rhs-load", "macc-rhs", "hdr"});
    addMAccRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"abi", "acc-load", "macc-acc", "macc-pass", "hdr"});
    addMAccRouteOperandBinding(plan, "out", slice.outABI,
                               {"abi", "store", "hdr"});
    addMAccRouteOperandBinding(plan, "n", slice.runtimeElementCountABI,
                               {"abi", "setvl-avl", "loop", "hdr"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd) {
    plan.planID = kRVVScalarBroadcastMAccOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVScalarBroadcastMAccRuntimeABIOrder;
    context = "scalar_broadcast_macc_add route";
    addMAccRouteOperandBinding(
        plan, "lhs", slice.lhsABI, {"abi", "lhs-load", "macc-lhs", "hdr"});
    addMAccRouteOperandBinding(
        plan, "rhs_scalar", slice.rhsABI,
        {"abi", "splat", "macc-rhs", "hdr"});
    addMAccRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"abi", "acc-load", "macc-acc", "macc-pass", "hdr"});
    addMAccRouteOperandBinding(
        plan, "out", slice.outABI, {"abi", "store", "hdr"});
    addMAccRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd) {
    std::optional<RVVComputedMaskMAccRouteFacts> routeFacts =
        getRVVComputedMaskMAccRouteFacts(
            slice.arithmeticKind, analysis.typedConfigFacts.sew,
            analysis.typedConfigFacts.lmul);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          "computed_masked_macc_add route requires provider-owned "
          "computed-mask MAcc route facts before route operand binding");
    plan.planID = routeFacts->routeOperandBindingPlanID.str();
    expectedRuntimeABIOrder = routeFacts->runtimeABIOrder;
    context = "computed_masked_macc_add route";
    addMAccRouteOperandBinding(plan, "cmp_lhs", slice.lhsABI,
                               {"abi", "cmp-lhs", "cmp-call", "hdr"});
    addMAccRouteOperandBinding(plan, "cmp_rhs", slice.rhsABI,
                               {"abi", "cmp-rhs", "cmp-call", "hdr"});
    addMAccRouteOperandBinding(plan, "lhs", slice.dotLHSABI,
                               {"abi", "lhs-load", "macc-lhs", "hdr"});
    addMAccRouteOperandBinding(plan, "rhs", slice.dotRHSABI,
                               {"abi", "rhs-load", "macc-rhs", "hdr"});
    addMAccRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"abi", "acc-load", "macc-acc", "macc-pass", "hdr"});
    addMAccRouteOperandBinding(plan, "out", slice.outABI,
                               {"abi", "store", "hdr"});
    addMAccRouteOperandBinding(plan, "n", slice.runtimeElementCountABI,
                               {"abi", "setvl-avl", "loop", "hdr"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskedMAccAdd) {
    std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts> routeFacts =
        getRVVRuntimeScalarComputedMaskMAccRouteFacts(
            slice.arithmeticKind, analysis.typedConfigFacts.sew,
            analysis.typedConfigFacts.lmul);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          "runtime_scalar_cmp_masked_macc_add route requires provider-owned "
          "runtime-scalar computed-mask MAcc route facts before route operand "
          "binding");
    plan.planID = routeFacts->routeOperandBindingPlanID.str();
    expectedRuntimeABIOrder = routeFacts->runtimeABIOrder;
    context = "runtime_scalar_cmp_masked_macc_add route";
    addMAccRouteOperandBinding(plan, "cmp_lhs", slice.lhsABI,
                               {"abi", "cmp-lhs", "cmp-call", "hdr"});
    addMAccRouteOperandBinding(plan, "rhs_scalar", slice.rhsABI,
                               {"abi", "splat", "cmp-rhs", "hdr"});
    addMAccRouteOperandBinding(plan, "lhs", slice.dotLHSABI,
                               {"abi", "lhs-load", "macc-lhs", "hdr"});
    addMAccRouteOperandBinding(plan, "rhs", slice.dotRHSABI,
                               {"abi", "rhs-load", "macc-rhs", "hdr"});
    addMAccRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"abi", "acc-load", "macc-acc", "macc-pass", "hdr"});
    addMAccRouteOperandBinding(plan, "out", slice.outABI,
                               {"abi", "store", "hdr"});
    addMAccRouteOperandBinding(plan, "n", slice.runtimeElementCountABI,
                               {"abi", "setvl-avl", "loop", "hdr"});
  } else {
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

llvm::Error verifyRVVSelectedBodyPlainMAccRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  const RVVSelectedBodyOperationKind operation = analysis.description.operation;
  const bool isConsumer =
      isRVVSelectedBodyPlainMAccRouteFamilyConsumer(operation);
  if (isConsumer && !analysis.plainMAccRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires the plain MAcc route-family plan before provider "
        "materialization for operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!isConsumer && analysis.plainMAccRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must not carry a plain MAcc route-family plan for non-plain-MAcc "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!analysis.plainMAccRouteFamilyPlan)
    return llvm::Error::success();

  const RVVSelectedBodyPlainMAccRouteFamilyPlan &plan =
      *analysis.plainMAccRouteFamilyPlan;
  if (llvm::Error error = validateRVVSelectedBodyPlainMAccRouteFamilyPlan(plan))
    return error;
  if (llvm::Error error = verifyMAccPlanTypedConfigSnapshot(
          analysis.typedConfigFacts, plan, context, "plain MAcc",
          /*storeMustMatchTypedFacts=*/true))
    return error;
  if (plan.operation != operation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " plain MAcc route-family plan operation must match the selected "
        "route description");
  if (analysis.description.plainMAccRouteFamilyPlanID != plan.familyPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " plain MAcc route-family plan mirror must match the validated "
        "family plan");
  if (analysis.description.memoryForm != plan.memoryForm ||
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
      analysis.description.vlDefOpName !=
          plan.runtimeControlPlan.vlDefOpName ||
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
      analysis.description.setVLIntrinsic != plan.setVLIntrinsic ||
      analysis.description.vectorLoadIntrinsic != plan.vectorLoadIntrinsic ||
      analysis.description.intrinsic != plan.maccIntrinsic ||
      analysis.description.storeIntrinsic != plan.storeIntrinsic ||
      analysis.description.resultName != plan.resultName ||
      analysis.description.maccAccumulatorLayout != plan.accumulatorLayout ||
      analysis.description.maccResultLayout != plan.resultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " plain MAcc route-family route, runtime, type, intrinsic, layout, "
        "and result mirrors must be populated from the validated family plan "
        "before provider materialization");
  if (!support::runtimeABIParametersEqual(
          analysis.description.runtimeABIParameters, plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " plain MAcc route-family runtime ABI parameters must match the "
        "validated family plan");
  std::optional<llvm::StringRef> expectedBindingPlanID =
      getExpectedRVVSelectedBodyMAccRouteOperandBindingPlanID(operation);
  if (!expectedBindingPlanID ||
      analysis.routeOperandBindingPlan.planID != *expectedBindingPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " plain MAcc provider requires the route operand binding plan for "
        "the selected operation");
  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, analysis.description, context))
    return error;
  return llvm::Error::success();
}

llvm::Error
verifyRVVSelectedBodyScalarBroadcastMAccRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  const RVVSelectedBodyOperationKind operation = analysis.description.operation;
  const bool isConsumer =
      isRVVSelectedBodyScalarBroadcastMAccRouteFamilyConsumer(operation);
  if (isConsumer && !analysis.scalarBroadcastMAccRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires the scalar-broadcast MAcc route-family plan before "
        "provider materialization for operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!isConsumer && analysis.scalarBroadcastMAccRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must not carry a scalar-broadcast MAcc route-family plan for "
        "non-scalar-broadcast-MAcc operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!analysis.scalarBroadcastMAccRouteFamilyPlan)
    return llvm::Error::success();

  const RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan &plan =
      *analysis.scalarBroadcastMAccRouteFamilyPlan;
  if (llvm::Error error =
          validateRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan(plan))
    return error;
  if (llvm::Error error = verifyMAccPlanTypedConfigSnapshot(
          analysis.typedConfigFacts, plan, context, "scalar-broadcast MAcc",
          /*storeMustMatchTypedFacts=*/true))
    return error;
  if (plan.rhsScalarSplatIntrinsic !=
      analysis.typedConfigFacts.scalarSplatIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast MAcc route-family scalar splat leaf must mirror "
        "selected typed RVV body/config facts before provider "
        "materialization");
  if (plan.operation != operation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast MAcc route-family plan operation must match the "
        "selected route description");
  if (analysis.description.scalarBroadcastMAccRouteFamilyPlanID !=
      plan.familyPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast MAcc route-family plan mirror must match the "
        "validated family plan");
  if (analysis.description.memoryForm != plan.memoryForm ||
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
      analysis.description.vlDefOpName !=
          plan.runtimeControlPlan.vlDefOpName ||
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
      analysis.description.setVLIntrinsic != plan.setVLIntrinsic ||
      analysis.description.vectorLoadIntrinsic != plan.vectorLoadIntrinsic ||
      analysis.description.rhsBroadcastIntrinsic !=
          plan.rhsScalarSplatIntrinsic ||
      analysis.description.intrinsic != plan.maccIntrinsic ||
      analysis.description.storeIntrinsic != plan.storeIntrinsic ||
      analysis.description.resultName != plan.resultName ||
      analysis.description.maccAccumulatorLayout != plan.accumulatorLayout ||
      analysis.description.maccResultLayout != plan.resultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast MAcc route-family route, runtime, type, "
        "intrinsic, layout, and result mirrors must be populated from the "
        "validated family plan before provider materialization");
  if (!support::runtimeABIParametersEqual(
          analysis.description.runtimeABIParameters, plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast MAcc route-family runtime ABI parameters must "
        "match the validated family plan");
  std::optional<llvm::StringRef> expectedBindingPlanID =
      getExpectedRVVSelectedBodyMAccRouteOperandBindingPlanID(operation);
  if (!expectedBindingPlanID ||
      analysis.routeOperandBindingPlan.planID != *expectedBindingPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast MAcc provider requires the route operand binding "
        "plan for the selected operation");
  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, analysis.description, context))
    return error;
  return llvm::Error::success();
}

llvm::Error
verifyRVVSelectedBodyComputedMaskAccumulationRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  const RVVSelectedBodyOperationKind operation = analysis.description.operation;
  const bool isConsumer =
      isRVVSelectedBodyComputedMaskAccumulationRouteFamilyConsumer(operation);
  const bool isMAccConsumer =
      isRVVSelectedBodyComputedMaskMAccAccumulationRouteFamilyConsumer(
          operation);
  if (isConsumer && !analysis.computedMaskAccumulationRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires the computed-mask accumulation route-family plan before "
        "provider materialization for operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!isConsumer && analysis.computedMaskAccumulationRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must not carry a computed-mask accumulation route-family plan for "
        "non-computed-mask-accumulation operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!analysis.computedMaskAccumulationRouteFamilyPlan)
    return llvm::Error::success();

  const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan &plan =
      *analysis.computedMaskAccumulationRouteFamilyPlan;
  if (llvm::Error error =
          validateRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(plan))
    return error;
  if (llvm::Error error = verifyMAccPlanTypedConfigSnapshot(
          analysis.typedConfigFacts, plan, context,
          "computed-mask accumulation",
          /*storeMustMatchTypedFacts=*/isMAccConsumer))
    return error;
  if (llvm::Error error = verifyMAccPlanMaskTypedConfigSnapshot(
          analysis.typedConfigFacts, plan, context,
          /*maskMustMatchTypedFacts=*/isConsumer))
    return error;
  if (plan.usesRuntimeScalarProducer &&
      plan.rhsScalarSplatIntrinsic !=
          analysis.typedConfigFacts.scalarSplatIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation route-family scalar splat leaf must "
        "mirror selected typed RVV body/config facts before provider "
        "materialization");
  if (plan.operation != operation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation route-family plan operation must match "
        "the selected route description");
  if (analysis.description.accumulationRouteFamilyPlanID != plan.familyPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation route-family plan mirror must match "
        "the validated family plan");
  if (analysis.description.accumulationComputeSuffix != plan.computeSuffix ||
      analysis.description.accumulationMaskProducerSource !=
          plan.maskProducerSource ||
      analysis.description.accumulationAccumulatorContract !=
          plan.accumulatorContract ||
      analysis.description.accumulationResultContract != plan.resultContract ||
      analysis.description.accumulationScalarCarryContract !=
          plan.scalarCarryContract)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation route-family mirrors must be populated "
        "from the validated family plan before provider materialization");
  if (analysis.description.memoryForm != plan.memoryForm ||
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
      analysis.description.vlDefOpName !=
          plan.runtimeControlPlan.vlDefOpName ||
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
      analysis.description.maskTypeName != plan.maskTypeName ||
      analysis.description.maskCType != plan.maskCType ||
      analysis.description.setVLIntrinsic != plan.setVLIntrinsic ||
      analysis.description.vectorLoadIntrinsic != plan.vectorLoadIntrinsic ||
      analysis.description.rhsBroadcastIntrinsic !=
          plan.rhsScalarSplatIntrinsic ||
      analysis.description.compareIntrinsic != plan.compareIntrinsic ||
      analysis.description.storeIntrinsic != plan.storeIntrinsic ||
      analysis.description.maskRole != plan.maskRole ||
      analysis.description.maskSource != plan.maskSource ||
      analysis.description.maskMemoryForm != plan.maskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation route-family route, runtime, type, "
        "intrinsic, and mask mirrors must be populated from the validated "
        "family plan before provider materialization");
  if (isMAccConsumer &&
      (analysis.description.intrinsic != plan.maccIntrinsic ||
       analysis.description.maskedMergeIntrinsic != plan.maskedMergeIntrinsic))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask MAcc route-family MAcc and merge leaves must be "
        "populated from the validated family plan before provider "
        "materialization");
  if (!support::runtimeABIParametersEqual(
          analysis.description.runtimeABIParameters, plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation route-family runtime ABI parameters "
        "must match the validated family plan");
  if (isMAccConsumer &&
      (analysis.description.sourceMemoryForm != plan.sourceMemoryForm ||
       analysis.description.destinationMemoryForm != plan.destinationMemoryForm ||
       analysis.description.indexedMemoryLayout != plan.indexedMemoryLayout ||
       analysis.description.inactiveLaneContract !=
           plan.inactiveLaneContract ||
       analysis.description.maskedPassthroughLayout !=
           plan.maskedPassthroughLayout ||
       analysis.description.maccAccumulatorLayout != plan.accumulatorLayout ||
       analysis.description.maccResultLayout != plan.resultLayout))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask MAcc route-family memory, layout, and inactive-lane "
        "mirrors "
        "must be populated from the validated family plan before provider "
        "materialization");
  std::optional<llvm::StringRef> expectedBindingPlanID =
      getExpectedRVVSelectedBodyMAccRouteOperandBindingPlanID(operation);
  if (isMAccConsumer && !expectedBindingPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask MAcc provider requires an owner-defined route "
        "operand binding plan for the selected operation");
  llvm::StringRef expectedPlanID =
      isMAccConsumer ? *expectedBindingPlanID
                     : getExpectedRVVRouteOperandBindingPlanID(operation);
  if (expectedPlanID.empty() ||
      analysis.routeOperandBindingPlan.planID != expectedPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask accumulation provider requires the route operand "
        "binding plan for the selected operation");
  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, analysis.description, context))
    return error;
  if (isMAccConsumer) {
    if (!plan.usesVectorMAccSuffix ||
        plan.usesScalarHorizontalReductionSuffix)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed-mask MAcc provider requires a vector MAcc suffix family "
          "plan, not the standalone-reduction suffix");
    if (plan.accumulatorContract !=
            "vector-accumulator-input-preserves-inactive-lanes" ||
        plan.resultContract !=
            "vector-macc-result-stored-to-output-buffer" ||
        plan.inactiveLaneContract !=
            "masked-macc-false-lanes-preserve-accumulator" ||
        plan.maskedPassthroughLayout !=
            "accumulator-vector-preserves-inactive-lanes")
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed-mask MAcc provider requires family-plan accumulator, "
          "result, inactive-lane, and passthrough contracts");
  } else if (plan.usesVectorMAccSuffix ||
             !plan.usesScalarHorizontalReductionSuffix) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask standalone reduction provider requires the shared "
        "accumulation plan to carry only the standalone-reduction suffix");
  }

  if (operation == RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd) {
    if (!plan.usesVectorCompareProducer || plan.usesRuntimeScalarProducer ||
        plan.maskProducerSource != "vector-compare-rhs-load")
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed_masked_macc_add provider requires a vector-compare "
          "computed-mask accumulation producer plan");
  } else if (operation ==
             RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd) {
    if (!plan.usesRuntimeScalarProducer || plan.usesVectorCompareProducer ||
        plan.maskProducerSource != "runtime-scalar-splat-compare-rhs")
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " runtime_scalar_cmp_masked_macc_add provider requires a "
          "runtime-scalar computed-mask accumulation producer plan");
  }

  return llvm::Error::success();
}

llvm::Error verifyRVVSelectedBodyMAccRouteDescriptionMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  const RVVSelectedBodyOperationKind operation = description.operation;
  const bool isPlainMAcc =
      isRVVSelectedBodyPlainMAccRouteFamilyConsumer(operation);
  const bool isScalarBroadcastMAcc =
      isRVVSelectedBodyScalarBroadcastMAccRouteFamilyConsumer(operation);
  const bool isComputedMaskMAcc =
      isRVVSelectedBodyComputedMaskMAccAccumulationRouteFamilyConsumer(
          operation);
  const bool isRuntimeScalarComputedMaskMAcc =
      operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
  const bool isMAcc = isPlainMAcc || isScalarBroadcastMAcc ||
                      isComputedMaskMAcc;

  if (isPlainMAcc) {
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            kRVVPlainMAccTargetLeafProfile))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            kRVVPlainMAccProviderSupportedMirror))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            kRVVPlainMAccRequiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            kRVVPlainMAccCTypeMappingSummary))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "plain MAcc route family plan",
            description.plainMAccRouteFamilyPlanID,
            kRVVPlainMAccRouteFamilyPlanID))
      return error;
  } else {
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "plain MAcc route family plan",
            description.plainMAccRouteFamilyPlanID, ""))
      return error;
  }

  if (isScalarBroadcastMAcc) {
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            kRVVScalarBroadcastMAccTargetLeafProfile))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            kRVVScalarBroadcastMAccProviderSupportedMirror))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            kRVVScalarBroadcastMAccRequiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            kRVVScalarBroadcastMAccCTypeMappingSummary))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "scalar-broadcast MAcc route family plan",
            description.scalarBroadcastMAccRouteFamilyPlanID,
            kRVVScalarBroadcastMAccRouteFamilyPlanID))
      return error;
  } else {
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "scalar-broadcast MAcc route family plan",
            description.scalarBroadcastMAccRouteFamilyPlanID, ""))
      return error;
  }

  if (isComputedMaskMAcc) {
    const std::optional<RVVComputedMaskMAccRouteFacts> computedMaskMAccFacts =
        getRVVComputedMaskMAccRouteFacts(operation, description.sew,
                                         description.lmul);
    const std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts>
        runtimeScalarMAccFacts =
            getRVVRuntimeScalarComputedMaskMAccRouteFacts(
                operation, description.sew, description.lmul);
    if (!isRuntimeScalarComputedMaskMAcc && !computedMaskMAccFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed_masked_macc_add description requires provider-owned "
          "computed-mask MAcc route facts for the selected SEW/LMUL");
    if (isRuntimeScalarComputedMaskMAcc && !runtimeScalarMAccFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " runtime_scalar_cmp_masked_macc_add description requires "
          "provider-owned runtime-scalar computed-mask MAcc route facts for "
          "the selected SEW/LMUL");
    llvm::StringRef expectedTargetLeafProfile =
        isRuntimeScalarComputedMaskMAcc
            ? runtimeScalarMAccFacts->targetLeafProfile
            : computedMaskMAccFacts->targetLeafProfile;
    llvm::StringRef expectedProviderSupportedMirror =
        isRuntimeScalarComputedMaskMAcc
            ? runtimeScalarMAccFacts->providerSupportedMirror
            : computedMaskMAccFacts->providerSupportedMirror;
    llvm::StringRef expectedHeaderDeclarations =
        isRuntimeScalarComputedMaskMAcc
            ? runtimeScalarMAccFacts->requiredHeaderDeclarations
            : computedMaskMAccFacts->requiredHeaderDeclarations;
    llvm::StringRef expectedCTypeMappingSummary =
        isRuntimeScalarComputedMaskMAcc
            ? runtimeScalarMAccFacts->cTypeMappingSummary
            : computedMaskMAccFacts->cTypeMappingSummary;
    llvm::StringRef expectedMemoryLayout =
        isRuntimeScalarComputedMaskMAcc
            ? runtimeScalarMAccFacts->indexedMemoryLayout
            : computedMaskMAccFacts->indexedMemoryLayout;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            expectedTargetLeafProfile))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror, expectedProviderSupportedMirror))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations, expectedHeaderDeclarations))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            expectedCTypeMappingSummary))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            expectedMemoryLayout))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            isRuntimeScalarComputedMaskMAcc
                ? runtimeScalarMAccFacts->sourceMemoryForm
                : computedMaskMAccFacts->sourceMemoryForm))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm,
            isRuntimeScalarComputedMaskMAcc
                ? runtimeScalarMAccFacts->destinationMemoryForm
                : computedMaskMAccFacts->destinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for computed-mask MAcc routes");
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "inactive-lane contract",
            description.inactiveLaneContract,
            isRuntimeScalarComputedMaskMAcc
                ? runtimeScalarMAccFacts->inactiveLaneContract
                : computedMaskMAccFacts->inactiveLaneContract))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout,
            isRuntimeScalarComputedMaskMAcc
                ? runtimeScalarMAccFacts->maskedPassthroughLayout
                : computedMaskMAccFacts->maskedPassthroughLayout))
      return error;
  }

  if (isMAcc) {
    const std::optional<RVVComputedMaskMAccRouteFacts> computedMaskMAccFacts =
        getRVVComputedMaskMAccRouteFacts(operation, description.sew,
                                         description.lmul);
    const std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts>
        runtimeScalarMAccFacts =
            getRVVRuntimeScalarComputedMaskMAccRouteFacts(
                operation, description.sew, description.lmul);
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "multiply-accumulate accumulator layout",
            description.maccAccumulatorLayout,
            runtimeScalarMAccFacts
                ? runtimeScalarMAccFacts->maccAccumulatorLayout
            : computedMaskMAccFacts
                ? computedMaskMAccFacts->maccAccumulatorLayout
                : llvm::StringRef(kRVVMAccAccumulatorLayout)))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "multiply-accumulate result layout",
            description.maccResultLayout,
            runtimeScalarMAccFacts ? runtimeScalarMAccFacts->maccResultLayout
            : computedMaskMAccFacts ? computedMaskMAccFacts->maccResultLayout
                                    : llvm::StringRef(kRVVMAccResultLayout)))
      return error;
  } else {
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "multiply-accumulate accumulator layout",
            description.maccAccumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "multiply-accumulate result layout",
            description.maccResultLayout, ""))
      return error;
  }

  if (std::optional<llvm::StringRef> expectedRuntimeABIOrder =
          getExpectedRVVSelectedBodyMAccRuntimeABIOrder(operation))
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "runtime ABI order", description.runtimeABIOrder,
            *expectedRuntimeABIOrder))
      return error;

  return llvm::Error::success();
}

llvm::ArrayRef<RVVSelectedBodyMAccRouteFamilyOwner>
getRVVSelectedBodyMAccRouteFamilyOwners() {
  static const RVVSelectedBodyMAccRouteFamilyOwner owners[] = {
      {"plain MAcc", isRVVSelectedBodyPlainMAccRouteFamilyConsumer,
       verifyRVVSelectedBodyPlainMAccRouteFamilyProviderPlans},
      {"scalar-broadcast MAcc",
       isRVVSelectedBodyScalarBroadcastMAccRouteFamilyConsumer,
       verifyRVVSelectedBodyScalarBroadcastMAccRouteFamilyProviderPlans},
      {"computed-mask MAcc",
       isRVVSelectedBodyComputedMaskMAccAccumulationRouteFamilyConsumer,
       verifyRVVSelectedBodyComputedMaskAccumulationRouteFamilyProviderPlans},
  };
  return owners;
}

bool isRVVSelectedBodyMAccRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  for (const RVVSelectedBodyMAccRouteFamilyOwner &owner :
       getRVVSelectedBodyMAccRouteFamilyOwners())
    if (owner.isConsumer && owner.isConsumer(operation))
      return true;
  return false;
}

llvm::Error verifyRVVSelectedBodyMAccRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  llvm::SmallVector<const RVVSelectedBodyMAccRouteFamilyOwner *, 2>
      selectedOwners;
  for (const RVVSelectedBodyMAccRouteFamilyOwner &owner :
       getRVVSelectedBodyMAccRouteFamilyOwners()) {
    if (!owner.isConsumer || !owner.verifyProviderPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " encountered an incomplete MAcc route-family owner registry entry");
    if (owner.isConsumer(analysis.description.operation))
      selectedOwners.push_back(&owner);
    if (llvm::Error error = owner.verifyProviderPlan(analysis, context))
      return error;
  }
  if (selectedOwners.size() > 1) {
    std::string owners;
    llvm::raw_string_ostream os(owners);
    for (const RVVSelectedBodyMAccRouteFamilyOwner *owner : selectedOwners) {
      if (!owners.empty())
        os << ", ";
      os << owner->familyName;
    }
    os.flush();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " matched multiple MAcc route-family owners for operation '" +
        stringifyRVVSelectedBodyOperationKind(
            analysis.description.operation) +
        "': " + owners);
  }
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
