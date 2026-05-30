#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {

namespace {

constexpr llvm::StringLiteral kRVVWideningMAccOperandBindingPlanID(
    "rvv-route-operand-binding:widening_macc_add.v1");
constexpr llvm::StringLiteral kRVVWideningDotReduceOperandBindingPlanID(
    "rvv-route-operand-binding:widening_dot_reduce.v1");
constexpr llvm::StringLiteral
    kRVVStridedInputWideningDotReduceOperandBindingPlanID(
        "rvv-route-operand-binding:strided_widening_dot_reduce.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskWideningDotReduceOperandBindingPlanID(
        "rvv-route-operand-binding:masked_widening_dot_reduce.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID(
        "rvv-route-operand-binding:masked_strided_wdot.v1");

constexpr llvm::StringLiteral kRVVWideningMAccAccumulatorLayout(
    "separate-i32-vector-accumulator-input");
constexpr llvm::StringLiteral kRVVWideningMAccResultLayout(
    "store-widening-multiply-accumulate-result-to-output-buffer");
constexpr llvm::StringLiteral kRVVWideningMAccRelation(
    "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1");
constexpr llvm::StringLiteral kRVVWideningDotProductAccumulatorLayout(
    "scalar-i32-seed-lane0-from-accumulator-input");
constexpr llvm::StringLiteral kRVVWideningDotProductResultLayout(
    "store-dot-reduction-lane0-to-output-scalar");
constexpr llvm::StringLiteral kRVVWideningDotProductRelation(
    "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32");
constexpr llvm::StringLiteral kRVVWideningDotProductStoreVL("1");
constexpr llvm::StringLiteral kRVVContractionRouteFamilyPlanID(
    "rvv-contraction-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVContractionTargetLeafProfile(
    "rvv-v1-i16mf2-i32m1-contraction-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVContractionProviderSupportedMirror(
    "provider_supported_mirror:rvv-contraction-family-plan-validated");
constexpr llvm::StringLiteral kRVVContractionRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVContractionCTypeMappingSummary(
    "vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32");
constexpr llvm::StringLiteral
    kRVVContractionMaskedInactiveLaneZeroingRequirement(
        "masked-widening-products-zero-inactive-lanes-before-reduction");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotMemoryLayout(
    "element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotMemoryLayout(
        "unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral kRVVLHSStrideSource("runtime_abi:lhs_stride");
constexpr llvm::StringLiteral kRVVRHSStrideSource("runtime_abi:rhs_stride");
constexpr llvm::StringLiteral kRVVStridedInputDotSourceMemoryForm(
    "strided-load");
constexpr llvm::StringLiteral kRVVDestinationMemoryForm("unit-stride-store");
constexpr llvm::StringLiteral
    kRVVMaskedCompareMaskSource("compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral
    kRVVMaskedPredicateMaskRole("predicate-mask-produced-by-compare");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskMemoryForm(
    "compare-produced-mask");

constexpr llvm::StringLiteral kRVVWideningMAccRuntimeABIOrder(
    "lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral kRVVWideningDotProductRuntimeABIOrder(
    "lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral
    kRVVStridedInputWideningDotProductRuntimeABIOrder(
        "lhs,rhs,acc,out,n,lhs_stride,rhs_stride");
constexpr llvm::StringLiteral kRVVComputedMaskWideningDotProductRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotProductRuntimeABIOrder(
        "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride");

bool isContractionDotReductionOperation(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return true;
  default:
    return false;
  }
}

llvm::StringRef getContractionRuntimeABIOrder(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    return kRVVWideningMAccRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    return kRVVWideningDotProductRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return kRVVStridedInputWideningDotProductRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return kRVVComputedMaskWideningDotProductRuntimeABIOrder;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return kRVVComputedMaskStridedInputWideningDotProductRuntimeABIOrder;
  default:
    return {};
  }
}

void addContractionRouteOperandBinding(
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

llvm::StringRef getContractionWideningMAccIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL, std::int64_t sew,
    llvm::StringRef lmul, llvm::StringRef relation) {
  if (sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
      sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
      sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      lmul == tcrv::rvv::getRVVLMULM1() &&
      relation == kRVVWideningMAccRelation)
    return "__riscv_vwmacc_vv_i32m1";
  return {};
}

llvm::StringRef getContractionWideningProductIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL, std::int64_t sew,
    llvm::StringRef lmul, llvm::StringRef relation) {
  if (sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
      sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
      sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      lmul == tcrv::rvv::getRVVLMULM1() &&
      relation == kRVVWideningDotProductRelation)
    return "__riscv_vwmul_vv_i32m1";
  return {};
}

llvm::StringRef getContractionMaskedWideningProductIntrinsic(
    std::int64_t sourceSEW, llvm::StringRef sourceLMUL, std::int64_t sew,
    llvm::StringRef lmul, llvm::StringRef relation) {
  if (sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
      sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
      sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      lmul == tcrv::rvv::getRVVLMULM1() &&
      relation == kRVVWideningDotProductRelation)
    return "__riscv_vwmul_vv_i32m1_m";
  return {};
}

llvm::StringRef getContractionReductionIntrinsic(llvm::StringRef lmul) {
  if (lmul == tcrv::rvv::getRVVLMULM1())
    return "__riscv_vredsum_vs_i32m1_i32m1";
  return {};
}

llvm::StringRef getContractionSignedLessThanCompareIntrinsic(
    llvm::StringRef lmul) {
  if (lmul == tcrv::rvv::getRVVLMULM1())
    return "__riscv_vmslt_vv_i32m1_b32";
  return {};
}

llvm::StringRef getContractionSelectIntrinsic(llvm::StringRef lmul) {
  if (lmul == tcrv::rvv::getRVVLMULM1())
    return "__riscv_vmerge_vvm_i32m1";
  return {};
}

llvm::StringRef getContractionScalarSeedSplatIntrinsic(llvm::StringRef lmul) {
  if (lmul == tcrv::rvv::getRVVLMULM1())
    return "__riscv_vmv_v_x_i32m1";
  return {};
}

llvm::Error verifyRVVSelectedBodyContractionRouteFamilyProviderPlanForOwner(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context,
    llvm::StringRef familyName) {
  const RVVSelectedBodyOperationKind operation = analysis.description.operation;
  if (!analysis.contractionRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " requires the " + familyName +
        " route-family plan before provider materialization for operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");

  const RVVSelectedBodyContractionRouteFamilyPlan &plan =
      *analysis.contractionRouteFamilyPlan;
  if (plan.operation != operation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + familyName +
        " route-family plan operation must match the selected route "
        "description");
  if (analysis.description.contractionRouteFamilyPlanID != plan.familyPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + familyName +
        " route-family plan mirror must match the validated family plan");
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
      analysis.description.boundedSlice !=
          plan.runtimeControlPlan.boundedSlice ||
      analysis.description.multiVL != plan.runtimeControlPlan.multiVL ||
      analysis.description.targetLeafProfile != plan.targetLeafProfile ||
      analysis.description.providerSupportedMirror !=
          plan.providerSupportedMirror ||
      analysis.description.requiredHeaderDeclarations !=
          plan.requiredHeaderDeclarations ||
      analysis.description.cTypeMappingSummary != plan.cTypeMappingSummary ||
      analysis.description.vlCType != plan.vlCType ||
      analysis.description.vectorTypeName != plan.resultVectorTypeName ||
      analysis.description.vectorCType != plan.resultVectorCType ||
      analysis.description.maskTypeName != plan.maskTypeName ||
      analysis.description.maskCType != plan.maskCType ||
      analysis.description.setVLIntrinsic != plan.setVLIntrinsic ||
      analysis.description.sourceSEW != plan.sourceSEW ||
      analysis.description.sourceLMUL != plan.sourceLMUL ||
      analysis.description.sourceVectorTypeName != plan.sourceVectorTypeName ||
      analysis.description.sourceVectorCType != plan.sourceVectorCType ||
      analysis.description.sourceVectorLoadIntrinsic !=
          plan.sourceVectorLoadIntrinsic ||
      analysis.description.stridedLoadIntrinsic != plan.stridedLoadIntrinsic ||
      analysis.description.storeIntrinsic != plan.storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + familyName +
        " route-family mirrors must be populated from the validated family "
        "plan before provider materialization");
  if (!support::runtimeABIParametersEqual(
          analysis.description.runtimeABIParameters, plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + familyName +
        " route-family runtime ABI parameters must match the validated "
        "family plan");

  std::optional<llvm::StringRef> expectedPlanID =
      getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(operation);
  if (!expectedPlanID || analysis.routeOperandBindingPlan.planID != *expectedPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " " + familyName +
        " provider requires the owner-defined route operand binding plan for "
        "the selected operation");
  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, analysis.description, context))
    return error;

  if (plan.usesWideningMAcc) {
    if (analysis.description.wideningMAccAccumulatorLayout !=
            plan.accumulatorLayout ||
        analysis.description.wideningMAccResultLayout != plan.resultLayout ||
        analysis.description.wideningMAccRelation != plan.relation ||
        analysis.description.intrinsic != plan.contractionComputeIntrinsic)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " widening MAcc contraction mirrors must come from the validated "
          "family plan");
  } else if (analysis.description.wideningDotProductAccumulatorLayout !=
                 plan.accumulatorLayout ||
             analysis.description.wideningDotProductResultLayout !=
                 plan.resultLayout ||
             analysis.description.wideningDotProductRelation !=
                 plan.relation ||
             analysis.description.intrinsic !=
                 plan.contractionComputeIntrinsic ||
             analysis.description.wideningProductIntrinsic !=
                 plan.wideningProductIntrinsic ||
             analysis.description.maskedWideningProductIntrinsic !=
                 plan.maskedWideningProductIntrinsic ||
             analysis.description.scalarSeedSplatIntrinsic !=
                 plan.scalarSeedSplatIntrinsic ||
             analysis.description.reductionStoreVL != plan.reductionStoreVL) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening dot-reduce contraction mirrors must come from the "
        "validated family plan");
  }

  if (plan.usesComputedMask &&
      (analysis.description.compareIntrinsic != plan.compareIntrinsic ||
       analysis.description.maskedMergeIntrinsic != plan.maskedMergeIntrinsic ||
       analysis.description.maskRole != plan.maskRole ||
       analysis.description.maskSource != plan.maskSource ||
       analysis.description.maskMemoryForm != plan.maskMemoryForm ||
       analysis.description.inactiveLaneZeroingRequirement !=
           plan.inactiveLaneZeroingRequirement))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask contraction mirrors must come from the validated "
        "family plan");
  if (!plan.usesComputedMask &&
      (!analysis.description.compareIntrinsic.empty() ||
       !analysis.description.maskedMergeIntrinsic.empty() ||
       !analysis.description.maskRole.empty() ||
       !analysis.description.maskSource.empty() ||
       !analysis.description.maskMemoryForm.empty() ||
       !analysis.description.inactiveLaneZeroingRequirement.empty()))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " non-masked contraction routes must not carry computed-mask "
        "contraction mirrors");
  if (plan.usesStridedInputs &&
      (analysis.description.stridedMemoryLayout != plan.stridedMemoryLayout ||
       analysis.description.lhsStrideSource != plan.lhsStrideSource ||
       analysis.description.rhsStrideSource != plan.rhsStrideSource ||
       analysis.description.sourceMemoryForm != plan.sourceMemoryForm ||
       analysis.description.destinationMemoryForm !=
           plan.destinationMemoryForm))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " strided-input contraction mirrors must come from the validated "
        "family plan");
  return llvm::Error::success();
}

llvm::Error verifyRVVSelectedBodyWideningMAccContractionRouteFamilyProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  return verifyRVVSelectedBodyContractionRouteFamilyProviderPlanForOwner(
      analysis, context, "widening MAcc contraction");
}

llvm::Error
verifyRVVSelectedBodyWideningDotReductionContractionRouteFamilyProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  return verifyRVVSelectedBodyContractionRouteFamilyProviderPlanForOwner(
      analysis, context, "widening dot-reduction contraction");
}

} // namespace

bool isRVVSelectedBodyContractionRouteOperation(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVSelectedBodyContractionDotReduction(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
         operation ==
             RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
         operation ==
             RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
         operation == RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyContractionComputedMask(
    RVVSelectedBodyOperationKind operation) {
  return operation ==
             RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
         operation == RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyContractionStridedInputs(
    RVVSelectedBodyOperationKind operation) {
  return operation ==
             RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
         operation == RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyWideningMAccContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
}

bool isRVVSelectedBodyWideningDotReductionContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return isContractionDotReductionOperation(operation);
}

llvm::ArrayRef<RVVSelectedBodyContractionRouteFamilyOwner>
getRVVSelectedBodyContractionRouteFamilyOwners() {
  static const RVVSelectedBodyContractionRouteFamilyOwner owners[] = {
      {"widening MAcc contraction",
       isRVVSelectedBodyWideningMAccContractionRouteFamilyConsumer,
       verifyRVVSelectedBodyWideningMAccContractionRouteFamilyProviderPlan},
      {"widening dot-reduction contraction",
       isRVVSelectedBodyWideningDotReductionContractionRouteFamilyConsumer,
       verifyRVVSelectedBodyWideningDotReductionContractionRouteFamilyProviderPlan},
  };
  return owners;
}

bool isRVVSelectedBodyContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  for (const RVVSelectedBodyContractionRouteFamilyOwner &owner :
       getRVVSelectedBodyContractionRouteFamilyOwners())
    if (owner.isConsumer && owner.isConsumer(operation))
      return true;
  return false;
}

llvm::StringRef getRVVSelectedBodyContractionRuntimeABIOrder(
    RVVSelectedBodyOperationKind operation) {
  return getContractionRuntimeABIOrder(operation);
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccAccumulatorLayout() {
  return kRVVWideningMAccAccumulatorLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccResultLayout() {
  return kRVVWideningMAccResultLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccRelation() {
  return kRVVWideningMAccRelation;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductAccumulatorLayout() {
  return kRVVWideningDotProductAccumulatorLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductResultLayout() {
  return kRVVWideningDotProductResultLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductRelation() {
  return kRVVWideningDotProductRelation;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedMaskedInactiveLaneZeroingRequirement() {
  return kRVVContractionMaskedInactiveLaneZeroingRequirement;
}

llvm::Error requireRVVSelectedBodyContractionPlanField(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("contraction route-family target-leaf/profile validation "
                  "for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error validateRVVSelectedBodyContractionRouteFamilyPlan(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "contraction route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyContractionRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "contraction route-family plan supports only active widening_macc_add "
        "and widening dot-reduce routes");
  const bool isWideningMAcc =
      plan.operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isDotReduction =
      isRVVSelectedBodyContractionDotReduction(plan.operation);
  const bool isComputedMask =
      isRVVSelectedBodyContractionComputedMask(plan.operation);
  const bool isStridedInput =
      isRVVSelectedBodyContractionStridedInputs(plan.operation);
  const RVVSelectedBodyMemoryForm expectedMemoryForm =
      isWideningMAcc
          ? RVVSelectedBodyMemoryForm::VectorRHSLoad
      : plan.operation == RVVSelectedBodyOperationKind::WideningDotReduceAdd
          ? RVVSelectedBodyMemoryForm::VectorRHSLoad
      : plan.operation ==
              RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd
          ? RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce
      : plan.operation ==
              RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd
          ? RVVSelectedBodyMemoryForm::
                ComputedMaskUnitStrideWideningDotReduce
          : RVVSelectedBodyMemoryForm::
                ComputedMaskStridedInputWideningDotReduce;
  if (plan.usesWideningMAcc != isWideningMAcc ||
      plan.usesDotReduction != isDotReduction ||
      plan.usesComputedMask != isComputedMask ||
      plan.usesStridedInputs != isStridedInput ||
      plan.usesScalarSeed != isDotReduction ||
      plan.usesVectorAccumulator != isWideningMAcc)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family plan stale operation classification "
        "markers");
  if (plan.memoryForm != expectedMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family plan requires the operation-specific memory "
        "form");
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "runtime control plan",
          plan.runtimeControlPlan.controlPlanID,
          getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "family plan", plan.familyPlanID,
          kRVVContractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "runtime ABI order", plan.runtimeABIOrder,
          getRVVSelectedBodyContractionRuntimeABIOrder(plan.operation)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "target leaf profile", plan.targetLeafProfile,
          kRVVContractionTargetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "provider_supported_mirror", plan.providerSupportedMirror,
          kRVVContractionProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "header declarations", plan.requiredHeaderDeclarations,
          kRVVContractionRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "C type mapping summary", plan.cTypeMappingSummary,
          kRVVContractionCTypeMappingSummary))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("contraction route-family target-leaf/profile "
                    "validation for operation '") +
        stringifyRVVSelectedBodyOperationKind(plan.operation) +
        "' requires provider-owned header declarations "
        "'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "VL C type", plan.vlCType, "size_t"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "result vector type", plan.resultVectorTypeName,
          "!tcrv_rvv.vector<i32, \"m1\">"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "result vector C type", plan.resultVectorCType,
          "vint32m1_t"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "setvl leaf", plan.setVLIntrinsic,
          "__riscv_vsetvl_e32m1"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "source vector-load leaf", plan.sourceVectorLoadIntrinsic,
          "__riscv_vle16_v_i16mf2"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "store leaf", plan.storeIntrinsic,
          "__riscv_vse32_v_i32m1"))
    return error;

  if (plan.usesStridedInputs) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "strided source-load leaf", plan.stridedLoadIntrinsic,
            "__riscv_vlse16_v_i16mf2"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "strided memory layout", plan.stridedMemoryLayout,
            plan.usesComputedMask
                ? kRVVComputedMaskStridedInputWideningDotMemoryLayout
                : kRVVStridedInputWideningDotMemoryLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "lhs stride source", plan.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "rhs stride source", plan.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "source memory form", plan.sourceMemoryForm,
            kRVVStridedInputDotSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "destination memory form", plan.destinationMemoryForm,
            kRVVDestinationMemoryForm))
      return error;
  } else if (llvm::Error error =
                 requireRVVSelectedBodyContractionPlanField(
                     plan, "strided source-load leaf",
                     plan.stridedLoadIntrinsic, ""))
    return error;
  if (!plan.usesStridedInputs) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "strided memory layout", plan.stridedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "lhs stride source", plan.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "rhs stride source", plan.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "source memory form", plan.sourceMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "destination memory form", plan.destinationMemoryForm, ""))
      return error;
  }

  if (plan.usesWideningMAcc) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "accumulator layout", plan.accumulatorLayout,
            kRVVWideningMAccAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "result layout", plan.resultLayout,
            kRVVWideningMAccResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening macc relation", plan.relation,
            kRVVWideningMAccRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening macc leaf", plan.contractionComputeIntrinsic,
            "__riscv_vwmacc_vv_i32m1"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product leaf", plan.wideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction store VL", plan.reductionStoreVL, ""))
      return error;
  } else {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "accumulator layout", plan.accumulatorLayout,
            kRVVWideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "result layout", plan.resultLayout,
            kRVVWideningDotProductResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening dot relation", plan.relation,
            kRVVWideningDotProductRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product leaf", plan.wideningProductIntrinsic,
            "__riscv_vwmul_vv_i32m1"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction leaf", plan.contractionComputeIntrinsic,
            "__riscv_vredsum_vs_i32m1_i32m1"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic,
            "__riscv_vmv_v_x_i32m1"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction store VL", plan.reductionStoreVL,
            kRVVWideningDotProductStoreVL))
      return error;
  }

  if (plan.usesComputedMask) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask type", plan.maskTypeName,
            "!tcrv_rvv.mask<i32, \"m1\">"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask C type", plan.maskCType, "vbool32_t"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "compare leaf", plan.compareIntrinsic,
            "__riscv_vmslt_vv_i32m1_b32"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked merge leaf", plan.maskedMergeIntrinsic,
            "__riscv_vmerge_vvm_i32m1"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic,
            "__riscv_vwmul_vv_i32m1_m"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "inactive-lane zeroing requirement",
            plan.inactiveLaneZeroingRequirement,
            kRVVContractionMaskedInactiveLaneZeroingRequirement))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask role", plan.maskRole,
            kRVVMaskedPredicateMaskRole))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask source", plan.maskSource,
            kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask memory form", plan.maskMemoryForm,
            kRVVComputedMaskMemoryMaskMemoryForm))
      return error;
  } else {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask type", plan.maskTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask C type", plan.maskCType, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "compare leaf", plan.compareIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked merge leaf", plan.maskedMergeIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "inactive-lane zeroing requirement",
            plan.inactiveLaneZeroingRequirement, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask role", plan.maskRole, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask source", plan.maskSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask memory form", plan.maskMemoryForm, ""))
      return error;
  }

  return llvm::Error::success();
}

void applyContractionRuntimeAVLVLControlPlanToDescription(
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

llvm::Expected<RVVSelectedBodyContractionRouteFamilyPlan>
deriveRVVSelectedBodyContractionRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyOperationKind operation = analysis.slice.arithmeticKind;
  if (!isRVVSelectedBodyContractionRouteOperation(operation))
    return makeRVVEmitCRouteProviderError(
        "requested contraction route-family plan for non-contraction RVV "
        "operation");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          getRVVSelectedBodyContractionRuntimeABIOrder(operation),
          "contraction route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  const RVVSelectedBodyTypedConfigFacts &typedConfig =
      analysis.typedConfigFacts;

  RVVSelectedBodyContractionRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesWideningMAcc =
      operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  plan.usesDotReduction =
      isRVVSelectedBodyContractionDotReduction(operation);
  plan.usesComputedMask =
      isRVVSelectedBodyContractionComputedMask(operation);
  plan.usesStridedInputs =
      isRVVSelectedBodyContractionStridedInputs(operation);
  plan.usesScalarSeed = plan.usesDotReduction;
  plan.usesVectorAccumulator = plan.usesWideningMAcc;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.familyPlanID = kRVVContractionRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = kRVVContractionTargetLeafProfile;
  plan.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  plan.cTypeMappingSummary = kRVVContractionCTypeMappingSummary;
  plan.vlCType = typedConfig.vlCType;
  plan.resultVectorTypeName = typedConfig.vectorTypeName;
  plan.resultVectorCType = typedConfig.vectorCType;
  plan.maskTypeName = plan.usesComputedMask ? typedConfig.maskTypeName : "";
  plan.maskCType = plan.usesComputedMask ? typedConfig.maskCType : "";
  plan.setVLIntrinsic = typedConfig.setVLIntrinsic;
  plan.sourceSEW = tcrv::rvv::getRVVSEW16Bits();
  plan.sourceLMUL = tcrv::rvv::getRVVLMULMF2();
  plan.sourceVectorTypeName = "!tcrv_rvv.vector<i16, \"mf2\">";
  plan.sourceVectorCType = "vint16mf2_t";
  plan.sourceVectorLoadIntrinsic = "__riscv_vle16_v_i16mf2";
  if (plan.usesStridedInputs)
    plan.stridedLoadIntrinsic = "__riscv_vlse16_v_i16mf2";
  plan.storeIntrinsic = typedConfig.storeIntrinsic;

  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  if (plan.usesComputedMask) {
    plan.runtimeABIParameters.push_back(analysis.slice.dotLHSABI);
    plan.runtimeABIParameters.push_back(analysis.slice.dotRHSABI);
  }
  plan.runtimeABIParameters.push_back(analysis.slice.accumulatorABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(
      plan.runtimeControlPlan.runtimeAVLParameter);
  if (plan.usesStridedInputs) {
    plan.runtimeABIParameters.push_back(analysis.slice.lhsStrideABI);
    plan.runtimeABIParameters.push_back(analysis.slice.rhsStrideABI);
  }

  if (plan.usesWideningMAcc) {
    plan.accumulatorLayout =
        analysis.slice.wideningMAccOp.getAccumulatorLayout();
    plan.resultLayout = analysis.slice.wideningMAccOp.getResultLayout();
    plan.relation = analysis.slice.wideningMAccOp.getMaccRelation();
    plan.contractionComputeIntrinsic =
        getContractionWideningMAccIntrinsic(
            plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
            typedConfig.lmul, plan.relation);
  } else {
    if (plan.usesComputedMask) {
      plan.accumulatorLayout =
          analysis.slice.maskedWideningDotReduceOp.getAccumulatorLayout();
      plan.resultLayout =
          analysis.slice.maskedWideningDotReduceOp.getResultLayout();
      plan.relation =
          analysis.slice.maskedWideningDotReduceOp.getDotProductRelation();
      plan.maskRole = analysis.slice.maskedWideningDotReduceOp.getMaskRole();
      plan.maskSource =
          analysis.slice.maskedWideningDotReduceOp.getMaskSource();
      plan.maskMemoryForm =
          analysis.slice.maskedWideningDotReduceOp.getMaskMemoryForm();
      plan.compareIntrinsic =
          getContractionSignedLessThanCompareIntrinsic(typedConfig.lmul);
      plan.maskedMergeIntrinsic =
          getContractionSelectIntrinsic(typedConfig.lmul);
      plan.maskedWideningProductIntrinsic =
          getContractionMaskedWideningProductIntrinsic(
              plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
              typedConfig.lmul, plan.relation);
      plan.inactiveLaneZeroingRequirement =
          kRVVContractionMaskedInactiveLaneZeroingRequirement;
    } else {
      plan.accumulatorLayout =
          analysis.slice.wideningDotReduceOp.getAccumulatorLayout();
      plan.resultLayout = analysis.slice.wideningDotReduceOp.getResultLayout();
      plan.relation =
          analysis.slice.wideningDotReduceOp.getDotProductRelation();
    }

    plan.contractionComputeIntrinsic =
        getContractionReductionIntrinsic(typedConfig.lmul);
    plan.wideningProductIntrinsic =
        getContractionWideningProductIntrinsic(
            plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
            typedConfig.lmul, plan.relation);
    plan.scalarSeedSplatIntrinsic =
        getContractionScalarSeedSplatIntrinsic(typedConfig.lmul);
    plan.reductionStoreVL = kRVVWideningDotProductStoreVL;
    if (plan.usesStridedInputs) {
      plan.stridedMemoryLayout =
          plan.usesComputedMask
              ? kRVVComputedMaskStridedInputWideningDotMemoryLayout
              : kRVVStridedInputWideningDotMemoryLayout;
      plan.lhsStrideSource = kRVVLHSStrideSource;
      plan.rhsStrideSource = kRVVRHSStrideSource;
      plan.sourceMemoryForm = kRVVStridedInputDotSourceMemoryForm;
      plan.destinationMemoryForm = kRVVDestinationMemoryForm;
    }
  }

  if (llvm::Error error =
          validateRVVSelectedBodyContractionRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyContractionRouteFamilyPlan(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyContractionRuntimeAVLVLControlPlanToDescription(
      plan.runtimeControlPlan, description);
  description.contractionRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.resultVectorTypeName;
  description.vectorCType = plan.resultVectorCType;
  description.maskTypeName = plan.maskTypeName;
  description.maskCType = plan.maskCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.sourceSEW = plan.sourceSEW;
  description.sourceLMUL = plan.sourceLMUL;
  description.sourceVectorTypeName = plan.sourceVectorTypeName;
  description.sourceVectorCType = plan.sourceVectorCType;
  description.sourceVectorLoadIntrinsic = plan.sourceVectorLoadIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());

  if (plan.usesWideningMAcc) {
    description.wideningMAccAccumulatorLayout = plan.accumulatorLayout;
    description.wideningMAccResultLayout = plan.resultLayout;
    description.wideningMAccRelation = plan.relation;
    description.intrinsic = plan.contractionComputeIntrinsic;
    return;
  }

  description.wideningDotProductAccumulatorLayout = plan.accumulatorLayout;
  description.wideningDotProductResultLayout = plan.resultLayout;
  description.wideningDotProductRelation = plan.relation;
  description.intrinsic = plan.contractionComputeIntrinsic;
  description.compareIntrinsic = plan.compareIntrinsic;
  description.maskedMergeIntrinsic = plan.maskedMergeIntrinsic;
  description.wideningProductIntrinsic = plan.wideningProductIntrinsic;
  description.maskedWideningProductIntrinsic =
      plan.maskedWideningProductIntrinsic;
  description.scalarSeedSplatIntrinsic = plan.scalarSeedSplatIntrinsic;
  description.reductionStoreVL = plan.reductionStoreVL;
  description.inactiveLaneZeroingRequirement =
      plan.inactiveLaneZeroingRequirement;
  if (plan.usesComputedMask) {
    description.maskRole = plan.maskRole;
    description.maskSource = plan.maskSource;
    description.maskMemoryForm = plan.maskMemoryForm;
  }
  if (plan.usesStridedInputs) {
    description.stridedLoadIntrinsic = plan.stridedLoadIntrinsic;
    description.stridedMemoryLayout = plan.stridedMemoryLayout;
    description.lhsStrideSource = plan.lhsStrideSource;
    description.rhsStrideSource = plan.rhsStrideSource;
    description.sourceMemoryForm = plan.sourceMemoryForm;
    description.destinationMemoryForm = plan.destinationMemoryForm;
  }
}

llvm::Error requireRVVSelectedBodyContractionDescriptionField(
    llvm::StringRef context, llvm::StringRef field, llvm::StringRef actual,
    llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " contraction route description requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error verifyRVVSelectedBodyContractionRouteDescriptionMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!isRVVSelectedBodyContractionRouteOperation(description.operation))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requested contraction route description verification for "
        "non-contraction operation");

  const bool usesWideningMAcc =
      description.operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool usesDotReduction =
      isRVVSelectedBodyContractionDotReduction(description.operation);
  const bool usesComputedMask =
      isRVVSelectedBodyContractionComputedMask(description.operation);
  const bool usesStridedInputs =
      isRVVSelectedBodyContractionStridedInputs(description.operation);

  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "family plan", description.contractionRouteFamilyPlanID,
          kRVVContractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "runtime ABI order", description.runtimeABIOrder,
          getRVVSelectedBodyContractionRuntimeABIOrder(description.operation)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "target leaf profile", description.targetLeafProfile,
          kRVVContractionTargetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "provider_supported_mirror",
          description.providerSupportedMirror,
          kRVVContractionProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "required header declarations",
          description.requiredHeaderDeclarations,
          kRVVContractionRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "C type mapping summary", description.cTypeMappingSummary,
          kRVVContractionCTypeMappingSummary))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector type", description.sourceVectorTypeName,
          "!tcrv_rvv.vector<i16, \"mf2\">"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector C type", description.sourceVectorCType,
          "vint16mf2_t"))
    return error;
  if (description.sourceSEW != tcrv::rvv::getRVVSEW16Bits() ||
      description.sourceLMUL != tcrv::rvv::getRVVLMULMF2())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " contraction route description requires source SEW16 LMUL mf2");
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector-load intrinsic",
          description.sourceVectorLoadIntrinsic, "__riscv_vle16_v_i16mf2"))
    return error;

  if (usesWideningMAcc) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate accumulator layout",
            description.wideningMAccAccumulatorLayout,
            kRVVWideningMAccAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate result layout",
            description.wideningMAccResultLayout,
            kRVVWideningMAccResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate relation",
            description.wideningMAccRelation, kRVVWideningMAccRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate intrinsic",
            description.intrinsic, "__riscv_vwmacc_vv_i32m1"))
      return error;
  }

  if (usesDotReduction) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product accumulator layout",
            description.wideningDotProductAccumulatorLayout,
            kRVVWideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product result layout",
            description.wideningDotProductResultLayout,
            kRVVWideningDotProductResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product relation",
            description.wideningDotProductRelation,
            kRVVWideningDotProductRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product intrinsic",
            description.wideningProductIntrinsic,
            "__riscv_vwmul_vv_i32m1"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "reduction intrinsic", description.intrinsic,
            "__riscv_vredsum_vs_i32m1_i32m1"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "scalar seed splat intrinsic",
            description.scalarSeedSplatIntrinsic, "__riscv_vmv_v_x_i32m1"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product reduction store VL",
            description.reductionStoreVL, kRVVWideningDotProductStoreVL))
      return error;
  }

  if (usesComputedMask) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic,
            "__riscv_vmslt_vv_i32m1_b32"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "masked merge intrinsic",
            description.maskedMergeIntrinsic, "__riscv_vmerge_vvm_i32m1"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "masked widening product intrinsic",
            description.maskedWideningProductIntrinsic,
            "__riscv_vwmul_vv_i32m1_m"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "inactive-lane zeroing requirement",
            description.inactiveLaneZeroingRequirement,
            kRVVContractionMaskedInactiveLaneZeroingRequirement))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "mask role", description.maskRole,
            kRVVMaskedPredicateMaskRole))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "mask source", description.maskSource,
            kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "mask memory form", description.maskMemoryForm,
            kRVVComputedMaskMemoryMaskMemoryForm))
      return error;
  }

  if (usesStridedInputs) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "strided-load intrinsic",
            description.stridedLoadIntrinsic, "__riscv_vlse16_v_i16mf2"))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout,
            usesComputedMask ? kRVVComputedMaskStridedInputWideningDotMemoryLayout
                             : kRVVStridedInputWideningDotMemoryLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "lhs stride source", description.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "rhs stride source", description.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVStridedInputDotSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm, kRVVDestinationMemoryForm))
      return error;
  }

  return llvm::Error::success();
}

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    return kRVVWideningMAccOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    return kRVVWideningDotReduceOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return kRVVStridedInputWideningDotReduceOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return kRVVComputedMaskWideningDotReduceOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID;
  default:
    return std::nullopt;
  }
}

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodyContractionRouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand) {
  using support::RuntimeABIParameterRole;
  if (planID == kRVVWideningMAccOperandBindingPlanID ||
      planID == kRVVWideningDotReduceOperandBindingPlanID ||
      planID == kRVVStridedInputWideningDotReduceOperandBindingPlanID) {
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
    if (planID == kRVVStridedInputWideningDotReduceOperandBindingPlanID) {
      if (logicalOperand == "lhs_stride")
        return RuntimeABIParameterRole::LHSInputStride;
      if (logicalOperand == "rhs_stride")
        return RuntimeABIParameterRole::RHSInputStride;
    }
  }
  if (planID == kRVVComputedMaskWideningDotReduceOperandBindingPlanID ||
      planID ==
          kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "dot_lhs")
      return RuntimeABIParameterRole::DotLHSInputBuffer;
    if (logicalOperand == "dot_rhs")
      return RuntimeABIParameterRole::DotRHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (planID ==
        kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID) {
      if (logicalOperand == "lhs_stride")
        return RuntimeABIParameterRole::LHSInputStride;
      if (logicalOperand == "rhs_stride")
        return RuntimeABIParameterRole::RHSInputStride;
    }
  }
  return std::nullopt;
}

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyContractionRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyRouteSlice &slice = analysis.slice;
  RVVRouteOperandBindingPlan plan;
  llvm::StringRef expectedRuntimeABIOrder =
      getContractionRuntimeABIOrder(slice.arithmeticKind);
  std::optional<llvm::StringRef> planID =
      getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(
          slice.arithmeticKind);
  if (!planID)
    return plan;

  if (!analysis.contractionRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
        " route requires the contraction route-family plan before deriving "
        "widening-contraction operand bindings");
  if (analysis.contractionRouteFamilyPlan->operation != slice.arithmeticKind)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
        " route requires same-operation contraction route-family plan facts");

  plan.planID = planID->str();
  llvm::StringRef context;
  switch (slice.arithmeticKind) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    context = "widening_macc_add route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wmacc-lhs", "src-i16mf2", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wmacc-rhs", "src-i16mf2", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"abi", "acc-load", "wmacc-acc", "acc-i32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "res-store", "res-i32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    context = "widening_dot_reduce_add route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI, {"abi", "ld", "dot-lhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI, {"abi", "ld", "dot-rhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    context = "strided_input_widening_dot_reduce_add route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI, {"abi", "sld", "dot-lhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI, {"abi", "sld", "dot-rhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    addContractionRouteOperandBinding(plan, "lhs_stride", slice.lhsStrideABI,
                                      {"abi", "str", "addr", "hdr"});
    addContractionRouteOperandBinding(plan, "rhs_stride", slice.rhsStrideABI,
                                      {"abi", "str", "addr", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    context = "computed_masked_widening_dot_reduce_add route";
    addContractionRouteOperandBinding(plan, "cmp_lhs", slice.lhsABI,
                                      {"abi", "cmp", "mask", "hdr"});
    addContractionRouteOperandBinding(plan, "cmp_rhs", slice.rhsABI,
                                      {"abi", "cmp", "mask", "hdr"});
    addContractionRouteOperandBinding(plan, "dot_lhs", slice.dotLHSABI,
                                      {"abi", "ld", "mlhs", "i16", "hdr"});
    addContractionRouteOperandBinding(plan, "dot_rhs", slice.dotRHSABI,
                                      {"abi", "ld", "mrhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    context = "computed_masked_strided_input_widening_dot_reduce_add route";
    addContractionRouteOperandBinding(plan, "cmp_lhs", slice.lhsABI,
                                      {"abi", "cmp", "mask"});
    addContractionRouteOperandBinding(plan, "cmp_rhs", slice.rhsABI,
                                      {"abi", "cmp", "mask"});
    addContractionRouteOperandBinding(plan, "dot_lhs", slice.dotLHSABI,
                                      {"abi", "sld", "mlhs", "i16"});
    addContractionRouteOperandBinding(plan, "dot_rhs", slice.dotRHSABI,
                                      {"abi", "sld", "mrhs", "i16"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    addContractionRouteOperandBinding(plan, "lhs_stride", slice.lhsStrideABI,
                                      {"abi", "str", "addr"});
    addContractionRouteOperandBinding(plan, "rhs_stride", slice.rhsStrideABI,
                                      {"abi", "str", "addr"});
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

llvm::Error verifyRVVSelectedBodyContractionRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  llvm::SmallVector<const RVVSelectedBodyContractionRouteFamilyOwner *, 2>
      selectedOwners;
  for (const RVVSelectedBodyContractionRouteFamilyOwner &owner :
       getRVVSelectedBodyContractionRouteFamilyOwners()) {
    if (!owner.isConsumer || !owner.verifyProviderPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " encountered an incomplete contraction route-family owner registry "
          "entry");
    if (owner.isConsumer(analysis.description.operation))
      selectedOwners.push_back(&owner);
  }
  if (selectedOwners.size() > 1) {
    std::string owners;
    llvm::raw_string_ostream os(owners);
    for (const RVVSelectedBodyContractionRouteFamilyOwner *owner :
         selectedOwners) {
      if (!owners.empty())
        os << ", ";
      os << owner->familyName;
    }
    os.flush();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " matched multiple contraction route-family owners for operation '" +
        stringifyRVVSelectedBodyOperationKind(
            analysis.description.operation) +
        "': " + owners);
  }
  if (selectedOwners.empty()) {
    if (analysis.contractionRouteFamilyPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " must not carry a contraction route-family plan for "
          "non-contraction operation '" +
          stringifyRVVSelectedBodyOperationKind(
              analysis.description.operation) +
          "'");
    return llvm::Error::success();
  }
  return selectedOwners.front()->verifyProviderPlan(analysis, context);
}

} // namespace tianchenrv::plugin::rvv
