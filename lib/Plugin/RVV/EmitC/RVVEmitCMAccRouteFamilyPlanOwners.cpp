#include "TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"

#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
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
    "rvv-v1-e32m1-plain-macc-add-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccTargetLeafProfile(
    "rvv-v1-e32m1-scalar-broadcast-macc-add-leaf-profile.v1");
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
    "vl:size_t,lhs/rhs/acc:signed-e32m1,result:signed-e32m1");
constexpr llvm::StringLiteral kRVVScalarBroadcastMAccCTypeMappingSummary(
    "vl:size_t,lhs/acc:signed-e32m1,rhs_scalar:i32,result:signed-e32m1");
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

llvm::StringRef getMAccIntrinsic(std::int64_t sew, llvm::StringRef lmul) {
  if (sew != tcrv::rvv::getRVVFirstSliceSEWBits())
    return {};
  if (lmul == tcrv::rvv::getRVVLMULM2())
    return "__riscv_vmacc_vv_i32m2";
  if (lmul == tcrv::rvv::getRVVLMULM1())
    return "__riscv_vmacc_vv_i32m1";
  return {};
}

llvm::StringRef getSetVLIntrinsic(std::int64_t sew, llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVSEW64Bits())
    return lmul == tcrv::rvv::getRVVLMULM2() ? "__riscv_vsetvl_e64m2"
                                             : "__riscv_vsetvl_e64m1";
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    return lmul == tcrv::rvv::getRVVLMULM2() ? "__riscv_vsetvl_e32m2"
                                             : "__riscv_vsetvl_e32m1";
  return {};
}

llvm::StringRef getScalarSplatIntrinsic(std::int64_t sew,
                                        llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVSEW64Bits() &&
      lmul == tcrv::rvv::getRVVLMULM1())
    return "__riscv_vmv_v_x_i64m1";
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits()) {
    if (lmul == tcrv::rvv::getRVVLMULM2())
      return "__riscv_vmv_v_x_i32m2";
    if (lmul == tcrv::rvv::getRVVLMULM1())
      return "__riscv_vmv_v_x_i32m1";
  }
  return {};
}

llvm::StringRef getSignedLessThanCompareIntrinsic(std::int64_t sew,
                                                  llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVSEW64Bits())
    return lmul == tcrv::rvv::getRVVLMULM2()
               ? "__riscv_vmslt_vv_i64m2_b32"
               : "__riscv_vmslt_vv_i64m1_b64";
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmslt_vv_i32m2_b16"
             : "__riscv_vmslt_vv_i32m1_b32";
}

llvm::StringRef getSignedLessEqualCompareIntrinsic(std::int64_t sew,
                                                   llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVSEW64Bits())
    return lmul == tcrv::rvv::getRVVLMULM2()
               ? "__riscv_vmsle_vv_i64m2_b32"
               : "__riscv_vmsle_vv_i64m1_b64";
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmsle_vv_i32m2_b16"
             : "__riscv_vmsle_vv_i32m1_b32";
}

llvm::StringRef getCompareIntrinsic(llvm::StringRef predicateKind,
                                    std::int64_t sew, llvm::StringRef lmul) {
  if (predicateKind == "slt")
    return getSignedLessThanCompareIntrinsic(sew, lmul);
  if (predicateKind == "sle")
    return getSignedLessEqualCompareIntrinsic(sew, lmul);
  return {};
}

llvm::StringRef getStandaloneReductionScalarResultStoreIntrinsic(
    std::int64_t sew, llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      (lmul == tcrv::rvv::getRVVLMULM1() ||
       lmul == tcrv::rvv::getRVVLMULM2()))
    return "__riscv_vse32_v_i32m1";
  if (sew == tcrv::rvv::getRVVSEW64Bits() &&
      lmul == tcrv::rvv::getRVVLMULM1())
    return "__riscv_vse64_v_i64m1";
  return {};
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
  case RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
    return kRVVComputedMaskedMAccRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd:
    return kRVVRuntimeScalarComputedMaskedMAccRuntimeABIOrder;
  default:
    return std::nullopt;
  }
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
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "header declarations",
          plan.requiredHeaderDeclarations, kRVVPlainMAccRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "C type mapping summary",
          plan.cTypeMappingSummary, kRVVPlainMAccCTypeMappingSummary))
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
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "vector type", plan.vectorTypeName,
          "!tcrv_rvv.vector<i32, \"m1\">"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "vector C type", plan.vectorCType,
          "vint32m1_t"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "setvl leaf", plan.setVLIntrinsic,
          "__riscv_vsetvl_e32m1"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "vector-load leaf",
          plan.vectorLoadIntrinsic, "__riscv_vle32_v_i32m1"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "MAcc compute leaf",
          plan.maccIntrinsic, "__riscv_vmacc_vv_i32m1"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "plain MAcc", plan.operation, "store leaf", plan.storeIntrinsic,
          "__riscv_vse32_v_i32m1"))
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
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.rhsGenericLoad ||
      !analysis.slice.accumulatorLoadOperation || !analysis.slice.genericStore ||
      !analysis.slice.maccOp || !analysis.slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "plain MAcc route-family plan requires explicit lhs load, rhs load, "
        "accumulator load, macc compute, and store body structure");
  if (analysis.typedConfigFacts.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      analysis.typedConfigFacts.lmul != tcrv::rvv::getRVVLMULM1())
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

  RVVSelectedBodyPlainMAccRouteFamilyPlan plan;
  plan.operation = analysis.slice.arithmeticKind;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.familyPlanID = kRVVPlainMAccRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = kRVVPlainMAccTargetLeafProfile;
  plan.providerSupportedMirror = kRVVPlainMAccProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = kRVVPlainMAccRequiredHeaderDeclarations;
  plan.cTypeMappingSummary = kRVVPlainMAccCTypeMappingSummary;
  plan.vlCType = analysis.typedConfigFacts.vlCType;
  plan.vectorTypeName = analysis.typedConfigFacts.vectorTypeName;
  plan.vectorCType = analysis.typedConfigFacts.vectorCType;
  plan.setVLIntrinsic = analysis.typedConfigFacts.setVLIntrinsic;
  plan.vectorLoadIntrinsic = analysis.typedConfigFacts.vectorLoadIntrinsic;
  plan.maccIntrinsic =
      getMAccIntrinsic(analysis.typedConfigFacts.sew,
                       analysis.typedConfigFacts.lmul);
  plan.storeIntrinsic = analysis.typedConfigFacts.storeIntrinsic;
  plan.resultName = "macc_sum_vec";
  plan.accumulatorLayout = *analysis.slice.maccOp.getAccumulatorLayout();
  plan.resultLayout = *analysis.slice.maccOp.getResultLayout();
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
          "scalar-broadcast MAcc", plan.operation, "header declarations",
          plan.requiredHeaderDeclarations,
          kRVVScalarBroadcastMAccRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "C type mapping summary",
          plan.cTypeMappingSummary, kRVVScalarBroadcastMAccCTypeMappingSummary))
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
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "vector type",
          plan.vectorTypeName, "!tcrv_rvv.vector<i32, \"m1\">"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "vector C type",
          plan.vectorCType, "vint32m1_t"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "setvl leaf",
          plan.setVLIntrinsic, "__riscv_vsetvl_e32m1"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "vector-load leaf",
          plan.vectorLoadIntrinsic, "__riscv_vle32_v_i32m1"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "RHS scalar splat leaf",
          plan.rhsScalarSplatIntrinsic, "__riscv_vmv_v_x_i32m1"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "MAcc compute leaf",
          plan.maccIntrinsic, "__riscv_vmacc_vv_i32m1"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "scalar-broadcast MAcc", plan.operation, "store leaf",
          plan.storeIntrinsic, "__riscv_vse32_v_i32m1"))
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
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.rhsScalarSplat ||
      !analysis.slice.accumulatorLoadOperation || !analysis.slice.genericStore ||
      !analysis.slice.maccOp || !analysis.slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast MAcc route-family plan requires explicit load, "
        "scalar splat, accumulator load, macc compute, and store body "
        "structure");
  if (analysis.typedConfigFacts.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      analysis.typedConfigFacts.lmul != tcrv::rvv::getRVVLMULM1())
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

  RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan plan;
  plan.operation = analysis.slice.arithmeticKind;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
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
  plan.vlCType = analysis.typedConfigFacts.vlCType;
  plan.vectorTypeName = analysis.typedConfigFacts.vectorTypeName;
  plan.vectorCType = analysis.typedConfigFacts.vectorCType;
  plan.setVLIntrinsic = analysis.typedConfigFacts.setVLIntrinsic;
  plan.vectorLoadIntrinsic = analysis.typedConfigFacts.vectorLoadIntrinsic;
  plan.rhsScalarSplatIntrinsic =
      getScalarSplatIntrinsic(analysis.typedConfigFacts.sew,
                              analysis.typedConfigFacts.lmul);
  plan.maccIntrinsic =
      getMAccIntrinsic(analysis.typedConfigFacts.sew,
                       analysis.typedConfigFacts.lmul);
  plan.storeIntrinsic = analysis.typedConfigFacts.storeIntrinsic;
  plan.resultName = "macc_sum_vec";
  plan.accumulatorLayout = *analysis.slice.maccOp.getAccumulatorLayout();
  plan.resultLayout = *analysis.slice.maccOp.getResultLayout();
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
  if (!isMAcc && !isStandaloneReduction)
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan supports only masked "
        "macc or masked standalone reduction consumers");
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
      isMAcc
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
      isMAcc ? kRVVComputedMaskAccumulationVectorMAccSuffix
             : kRVVComputedMaskAccumulationStandaloneReductionSuffix;
  llvm::StringRef expectedProducerSource =
      isRuntimeScalarProducer
          ? kRVVComputedMaskAccumulationRuntimeScalarProducerSource
          : kRVVComputedMaskAccumulationVectorCompareProducerSource;
  llvm::StringRef expectedRuntimeABIOrder =
      isMAcc
          ? (isRuntimeScalarProducer
                 ? llvm::StringRef(
                       kRVVRuntimeScalarComputedMaskedMAccRuntimeABIOrder)
                 : llvm::StringRef(kRVVComputedMaskedMAccRuntimeABIOrder))
          : (isRuntimeScalarProducer
                 ? llvm::StringRef(
                       "cmp_lhs,rhs_scalar,src,acc,out,n")
                 : llvm::StringRef("cmp_lhs,cmp_rhs,src,acc,out,n"));
  llvm::StringRef expectedTargetLeafProfile =
      isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccTargetLeafProfile
                 : kRVVComputedMaskedMAccTargetLeafProfile)
          : (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskStandaloneReductionTargetLeafProfile
                 : kRVVComputedMaskStandaloneReductionTargetLeafProfile);
  llvm::StringRef expectedProviderSupportedMirror =
      isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccProviderSupportedMirror
                 : kRVVComputedMaskedMAccProviderSupportedMirror)
          : (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskStandaloneReductionProviderSupportedMirror
                 : kRVVComputedMaskStandaloneReductionProviderSupportedMirror);
  llvm::StringRef expectedHeaderDeclarations =
      isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccRequiredHeaderDeclarations
                 : kRVVComputedMaskedMAccRequiredHeaderDeclarations)
          : kRVVStandaloneReductionRequiredHeaderDeclarations;
  llvm::StringRef expectedCTypeMappingSummary =
      isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccCTypeMappingSummary
                 : kRVVComputedMaskedMAccCTypeMappingSummary)
          : (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskStandaloneReductionCTypeMappingSummary
                 : kRVVComputedMaskStandaloneReductionCTypeMappingSummary);
  llvm::StringRef expectedAccumulatorContract =
      isMAcc ? kRVVComputedMaskAccumulationMAccAccumulatorContract
             : kRVVComputedMaskAccumulationReductionAccumulatorContract;
  llvm::StringRef expectedResultContract =
      isMAcc ? kRVVComputedMaskAccumulationMAccResultContract
             : kRVVComputedMaskAccumulationReductionResultContract;
  llvm::StringRef expectedInactiveLaneContract =
      isMAcc ? llvm::StringRef("masked-macc-false-lanes-preserve-accumulator")
             : getStandaloneReductionInactiveLaneRequirement(plan.operation);
  llvm::StringRef expectedPassthroughLayout =
      isMAcc ? llvm::StringRef("accumulator-vector-preserves-inactive-lanes")
             : llvm::StringRef();
  llvm::StringRef expectedScalarCarry =
      isStandaloneReduction
          ? llvm::StringRef(
                kRVVComputedMaskAccumulationReductionScalarCarryContract)
          : llvm::StringRef();
  llvm::StringRef expectedPredicate =
      isMAcc && !isRuntimeScalarProducer ? "slt" : "sle";
  llvm::StringRef expectedRHSSplatLeaf =
      isRuntimeScalarProducer
          ? getScalarSplatIntrinsic(plan.runtimeControlPlan.sew,
                                    plan.runtimeControlPlan.lmul)
          : llvm::StringRef();
  llvm::StringRef expectedCompareLeaf =
      getCompareIntrinsic(expectedPredicate, plan.runtimeControlPlan.sew,
                          plan.runtimeControlPlan.lmul);
  if (expectedCompareLeaf.empty())
    return makeRVVEmitCRouteProviderError(
        "computed-mask accumulation route-family plan requires a "
        "provider-derived compare leaf for the selected typed config");
  llvm::StringRef expectedStoreLeaf =
      isStandaloneReduction
          ? getStandaloneReductionScalarResultStoreIntrinsic(
                plan.runtimeControlPlan.sew, plan.runtimeControlPlan.lmul)
          : plan.runtimeControlPlan.sew ==
                    tcrv::rvv::getRVVFirstSliceSEWBits()
                ? (plan.runtimeControlPlan.lmul == tcrv::rvv::getRVVLMULM2()
                       ? llvm::StringRef("__riscv_vse32_v_i32m2")
                       : llvm::StringRef("__riscv_vse32_v_i32m1"))
                : llvm::StringRef("__riscv_vse64_v_i64m1");
  if (expectedStoreLeaf.empty())
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
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "VL C type",
          plan.vlCType, "size_t"))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "setvl leaf",
          plan.setVLIntrinsic,
          getSetVLIntrinsic(plan.runtimeControlPlan.sew,
                            plan.runtimeControlPlan.lmul)))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "RHS scalar splat leaf",
          plan.rhsScalarSplatIntrinsic, expectedRHSSplatLeaf))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "compare leaf",
          plan.compareIntrinsic, expectedCompareLeaf))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "store leaf",
          plan.storeIntrinsic, expectedStoreLeaf))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "mask role",
          plan.maskRole, kRVVMaskedPredicateMaskRole))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "mask source",
          plan.maskSource, kRVVMaskedCompareMaskSource))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "mask memory form",
          plan.maskMemoryForm, kRVVComputedMaskMemoryMaskMemoryForm))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation, "source memory form",
          plan.sourceMemoryForm, kRVVUnitStrideSourceMemoryForm))
    return error;
  if (llvm::Error error = requireMAccPlanField(
          "computed-mask accumulation", plan.operation,
          "destination memory form", plan.destinationMemoryForm,
          kRVVDestinationMemoryForm))
    return error;
  if (isMAcc) {
    llvm::StringRef expectedMemoryLayout =
        isRuntimeScalarProducer
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
            kRVVMAccAccumulatorLayout))
      return error;
    if (llvm::Error error = requireMAccPlanField(
            "computed-mask accumulation", plan.operation, "result layout",
            plan.resultLayout, kRVVMAccResultLayout))
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
  const bool isMAcc =
      operation == RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
      operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
  const bool isStandaloneReduction = !isMAcc;
  const bool isRuntimeScalarProducer =
      operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd ||
      isRuntimeScalarComputedMaskStandaloneReductionOperation(operation);
  const RVVSelectedBodyMemoryForm expectedMemoryForm =
      isMAcc
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
            analysis.typedConfigFacts.sew, analysis.typedConfigFacts.lmul))
      return makeRVVEmitCRouteProviderError(
          "computed-mask accumulation route-family plan requires runtime-"
          "scalar standalone reduction typed config to be SEW32 LMUL m1, "
          "SEW32 LMUL m2, or SEW64 LMUL m1 with a separate LMUL m1 scalar "
          "reduction accumulator/result channel");
  } else if (isStandaloneReduction) {
    if (!isStandaloneReductionScalarChannelConfig(
            analysis.typedConfigFacts.sew, analysis.typedConfigFacts.lmul))
      return makeRVVEmitCRouteProviderError(
          "computed-mask accumulation route-family plan requires "
          "non-runtime-scalar standalone reduction typed config to be SEW32 "
          "LMUL m1 or SEW32 LMUL m2 with a separate LMUL m1 scalar reduction "
          "accumulator/result channel");
  } else if (isMAcc && !isComputedMaskMAccConfig(analysis.typedConfigFacts.sew,
                                                analysis.typedConfigFacts.lmul)) {
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
      isMAcc
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

  RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesVectorMAccSuffix = isMAcc;
  plan.usesScalarHorizontalReductionSuffix = !isMAcc;
  plan.usesRuntimeScalarProducer = isRuntimeScalarProducer;
  plan.usesVectorCompareProducer = !isRuntimeScalarProducer;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.familyPlanID = kRVVComputedMaskAccumulationRouteFamilyPlanID;
  plan.computeSuffix =
      isMAcc ? kRVVComputedMaskAccumulationVectorMAccSuffix
             : kRVVComputedMaskAccumulationStandaloneReductionSuffix;
  plan.maskProducerSource =
      isRuntimeScalarProducer
          ? kRVVComputedMaskAccumulationRuntimeScalarProducerSource
          : kRVVComputedMaskAccumulationVectorCompareProducerSource;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile =
      isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccTargetLeafProfile
                 : kRVVComputedMaskedMAccTargetLeafProfile)
          : (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskStandaloneReductionTargetLeafProfile
                 : kRVVComputedMaskStandaloneReductionTargetLeafProfile);
  plan.providerSupportedMirror =
      isMAcc
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
      isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccRequiredHeaderDeclarations
                 : kRVVComputedMaskedMAccRequiredHeaderDeclarations)
          : kRVVStandaloneReductionRequiredHeaderDeclarations;
  plan.cTypeMappingSummary =
      isMAcc
          ? (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskedMAccCTypeMappingSummary
                 : kRVVComputedMaskedMAccCTypeMappingSummary)
          : (isRuntimeScalarProducer
                 ? kRVVRuntimeScalarComputedMaskStandaloneReductionCTypeMappingSummary
                 : kRVVComputedMaskStandaloneReductionCTypeMappingSummary);
  plan.vlCType = analysis.typedConfigFacts.vlCType;
  plan.vectorTypeName = analysis.typedConfigFacts.vectorTypeName;
  plan.vectorCType = analysis.typedConfigFacts.vectorCType;
  plan.maskTypeName = analysis.typedConfigFacts.maskTypeName;
  plan.maskCType = analysis.typedConfigFacts.maskCType;
  plan.setVLIntrinsic = analysis.typedConfigFacts.setVLIntrinsic;
  plan.vectorLoadIntrinsic = analysis.typedConfigFacts.vectorLoadIntrinsic;
  plan.rhsScalarSplatIntrinsic =
      isRuntimeScalarProducer
          ? getScalarSplatIntrinsic(analysis.typedConfigFacts.sew,
                                    analysis.typedConfigFacts.lmul)
          : "";
  plan.compareIntrinsic =
      getCompareIntrinsic(isMAcc && !isRuntimeScalarProducer ? "slt" : "sle",
                          analysis.typedConfigFacts.sew,
                          analysis.typedConfigFacts.lmul);
  plan.storeIntrinsic =
      isStandaloneReduction
          ? getStandaloneReductionScalarResultStoreIntrinsic(
                analysis.typedConfigFacts.sew, analysis.typedConfigFacts.lmul)
          : analysis.typedConfigFacts.storeIntrinsic;
  plan.maskRole = kRVVMaskedPredicateMaskRole;
  plan.maskSource = kRVVMaskedCompareMaskSource;
  plan.maskMemoryForm = kRVVComputedMaskMemoryMaskMemoryForm;
  plan.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  plan.destinationMemoryForm = kRVVDestinationMemoryForm;
  if (isMAcc) {
    plan.indexedMemoryLayout =
        isRuntimeScalarProducer ? kRVVRuntimeScalarComputedMaskedMAccMemoryLayout
                                : kRVVComputedMaskedMAccMemoryLayout;
    plan.accumulatorLayout = analysis.slice.maskedMAccOp.getAccumulatorLayout();
    plan.resultLayout = analysis.slice.maskedMAccOp.getResultLayout();
  }
  plan.accumulatorContract =
      isMAcc ? kRVVComputedMaskAccumulationMAccAccumulatorContract
             : kRVVComputedMaskAccumulationReductionAccumulatorContract;
  plan.resultContract =
      isMAcc ? kRVVComputedMaskAccumulationMAccResultContract
             : kRVVComputedMaskAccumulationReductionResultContract;
  plan.inactiveLaneContract =
      isMAcc ? llvm::StringRef("masked-macc-false-lanes-preserve-accumulator")
             : getStandaloneReductionInactiveLaneRequirement(operation);
  plan.maskedPassthroughLayout =
      isMAcc ? llvm::StringRef("accumulator-vector-preserves-inactive-lanes")
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
  case RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
    return kRVVComputedMaskedMAccOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd:
    return kRVVRuntimeScalarComputedMaskedMAccOperandBindingPlanID;
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
        plan, "lhs", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base", "macc-lhs-call"});
    addMAccRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"runtime-abi-mirror", "materialized-load-base", "macc-rhs-call"});
    addMAccRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"runtime-abi-mirror", "materialized-accumulator-load-base",
         "macc-accumulator-call"});
    addMAccRouteOperandBinding(
        plan, "out", slice.outABI,
        {"runtime-abi-mirror", "materialized-store-base", "header-mirror"});
    addMAccRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd) {
    plan.planID = kRVVScalarBroadcastMAccOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVScalarBroadcastMAccRuntimeABIOrder;
    context = "scalar_broadcast_macc_add route";
    addMAccRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base", "macc-lhs-call"});
    addMAccRouteOperandBinding(
        plan, "rhs_scalar", slice.rhsABI,
        {"runtime-abi-mirror", "scalar-broadcast-rhs-call",
         "macc-rhs-call"});
    addMAccRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"runtime-abi-mirror", "materialized-accumulator-load-base",
         "macc-accumulator-call"});
    addMAccRouteOperandBinding(
        plan, "out", slice.outABI,
        {"runtime-abi-mirror", "materialized-store-base", "header-mirror"});
    addMAccRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd) {
    plan.planID = kRVVComputedMaskedMAccOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVComputedMaskedMAccRuntimeABIOrder;
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
    plan.planID =
        kRVVRuntimeScalarComputedMaskedMAccOperandBindingPlanID.str();
    expectedRuntimeABIOrder =
        kRVVRuntimeScalarComputedMaskedMAccRuntimeABIOrder;
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
        " computed-mask MAcc route-family memory, layout, and inactive-lane mirrors "
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
    llvm::StringRef expectedTargetLeafProfile =
        isRuntimeScalarComputedMaskMAcc
            ? llvm::StringRef(kRVVRuntimeScalarComputedMaskedMAccTargetLeafProfile)
            : llvm::StringRef(kRVVComputedMaskedMAccTargetLeafProfile);
    llvm::StringRef expectedProviderSupportedMirror =
        isRuntimeScalarComputedMaskMAcc
            ? llvm::StringRef(
                  kRVVRuntimeScalarComputedMaskedMAccProviderSupportedMirror)
            : llvm::StringRef(kRVVComputedMaskedMAccProviderSupportedMirror);
    llvm::StringRef expectedHeaderDeclarations =
        isRuntimeScalarComputedMaskMAcc
            ? llvm::StringRef(
                  kRVVRuntimeScalarComputedMaskedMAccRequiredHeaderDeclarations)
            : llvm::StringRef(kRVVComputedMaskedMAccRequiredHeaderDeclarations);
    llvm::StringRef expectedCTypeMappingSummary =
        isRuntimeScalarComputedMaskMAcc
            ? llvm::StringRef(
                  kRVVRuntimeScalarComputedMaskedMAccCTypeMappingSummary)
            : llvm::StringRef(kRVVComputedMaskedMAccCTypeMappingSummary);
    llvm::StringRef expectedMemoryLayout =
        isRuntimeScalarComputedMaskMAcc
            ? llvm::StringRef(kRVVRuntimeScalarComputedMaskedMAccMemoryLayout)
            : llvm::StringRef(kRVVComputedMaskedMAccMemoryLayout);
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
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm, kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for computed-mask MAcc routes");
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "inactive-lane contract",
            description.inactiveLaneContract,
            "masked-macc-false-lanes-preserve-accumulator"))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout,
            "accumulator-vector-preserves-inactive-lanes"))
      return error;
  }

  if (isMAcc) {
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "multiply-accumulate accumulator layout",
            description.maccAccumulatorLayout, kRVVMAccAccumulatorLayout))
      return error;
    if (llvm::Error error = requireMAccRouteDescriptionField(
            context, "multiply-accumulate result layout",
            description.maccResultLayout, kRVVMAccResultLayout))
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
