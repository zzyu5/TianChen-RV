#include "TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {

llvm::Error validateRVVSelectedBodyPlainMAccRouteFamilyPlanForOwner(
    const RVVSelectedBodyPlainMAccRouteFamilyPlan &plan);
llvm::Error
validateRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlanForOwner(
    const RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan &plan);
llvm::Error
validateRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlanForOwner(
    const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan &plan);

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

bool isRVVSelectedBodyScalarBroadcastMAccRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
}

bool isRVVSelectedBodyPlainMAccRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::MAccAdd;
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
  if (llvm::Error error =
          validateRVVSelectedBodyPlainMAccRouteFamilyPlanForOwner(plan))
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
          validateRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlanForOwner(
              plan))
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
          validateRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlanForOwner(
              plan))
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
       analysis.description.inactiveLaneContract !=
           plan.inactiveLaneContract ||
       analysis.description.maskedPassthroughLayout !=
           plan.maskedPassthroughLayout))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " computed-mask MAcc route-family memory and inactive-lane mirrors "
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
