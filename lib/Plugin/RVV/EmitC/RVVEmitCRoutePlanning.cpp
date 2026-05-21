#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {

llvm::Error makeRVVEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine(
          "TianChen-RV RVV plugin-owned EmitC route provider failed: ") +
          message,
      llvm::errc::invalid_argument);
}

constexpr llvm::StringLiteral kRVVMAccOperandBindingPlanID(
    "rvv-route-operand-binding:macc_add.v1");
constexpr llvm::StringLiteral kRVVStridedLoadUnitStoreOperandBindingPlanID(
    "rvv-route-operand-binding:strided_load_unit_store.v1");
constexpr llvm::StringLiteral kRVVUnitLoadStridedStoreOperandBindingPlanID(
    "rvv-route-operand-binding:unit_load_strided_store.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastOperandBindingPlanID(
    "rvv-route-operand-binding:scalar_broadcast_add.v1");
constexpr llvm::StringLiteral kRVVStandaloneReductionOperandBindingPlanID(
    "rvv-route-operand-binding:standalone_reduce_add.v1");
constexpr llvm::StringLiteral kRVVMaskedUnitStoreOperandBindingPlanID(
    "rvv-route-operand-binding:masked_unit_store.v1");
constexpr llvm::StringLiteral kRVVComputedMaskStridedStoreOperandBindingPlanID(
    "rvv-route-operand-binding:computed_masked_strided_store.v1");
constexpr llvm::StringLiteral kRVVAddOperandBindingPlanID(
    "rvv-route-operand-binding:add.v1");
constexpr llvm::StringLiteral kRVVSubOperandBindingPlanID(
    "rvv-route-operand-binding:sub.v1");
constexpr llvm::StringLiteral kRVVMulOperandBindingPlanID(
    "rvv-route-operand-binding:mul.v1");
constexpr llvm::StringLiteral kRVVCmpSelectOperandBindingPlanID(
    "rvv-route-operand-binding:cmp_select.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSelectOperandBindingPlanID(
    "rvv-route-operand-binding:computed_mask_select.v1");
constexpr llvm::StringLiteral kRVVMaskedAddOperandBindingPlanID(
    "rvv-route-operand-binding:masked_add.v1");
constexpr llvm::StringLiteral kRVVReduceAddOperandBindingPlanID(
    "rvv-route-operand-binding:reduce_add.v1");
constexpr llvm::StringLiteral kRVVStridedAddOperandBindingPlanID(
    "rvv-route-operand-binding:strided_add.v1");

bool isRVVFourOperandPlanID(llvm::StringRef planID) {
  return planID == kRVVAddOperandBindingPlanID ||
         planID == kRVVSubOperandBindingPlanID ||
         planID == kRVVMulOperandBindingPlanID ||
         planID == kRVVCmpSelectOperandBindingPlanID ||
         planID == kRVVMaskedAddOperandBindingPlanID ||
         planID == kRVVReduceAddOperandBindingPlanID;
}

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVRouteOperandBindingRole(llvm::StringRef planID,
                                      llvm::StringRef logicalOperand) {
  using support::RuntimeABIParameterRole;
  if (isRVVFourOperandPlanID(planID)) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
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
  if (planID == kRVVScalarBroadcastOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVStandaloneReductionOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVMaskedUnitStoreOperandBindingPlanID) {
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "mask")
      return RuntimeABIParameterRole::MaskInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVComputedMaskStridedStoreOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (logicalOperand == "dst_stride_bytes")
      return RuntimeABIParameterRole::DestinationByteStride;
  }
  if (planID == kRVVComputedMaskSelectOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "true_value")
      return RuntimeABIParameterRole::TrueValueInputBuffer;
    if (logicalOperand == "false_value")
      return RuntimeABIParameterRole::FalseValueInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVStridedAddOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (logicalOperand == "lhs_stride")
      return RuntimeABIParameterRole::LHSInputStride;
    if (logicalOperand == "rhs_stride")
      return RuntimeABIParameterRole::RHSInputStride;
    if (logicalOperand == "out_stride")
      return RuntimeABIParameterRole::OutputStride;
  }
  return std::nullopt;
}

llvm::Expected<const support::RuntimeABIParameter *>
getRVVRouteOperandBindingParameter(
    const RVVRouteOperandBindingPlan &plan, llvm::StringRef logicalOperand,
    llvm::StringRef materializedUse, llvm::StringRef context) {
  if (plan.planID.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding lookup for ") + context +
        " requires a non-empty RouteOperandBindingPlan");

  const RVVRouteOperandBinding *matched = nullptr;
  unsigned count = 0;
  for (const RVVRouteOperandBinding &binding : plan.bindings) {
    if (llvm::StringRef(binding.logicalOperand) != logicalOperand)
      continue;
    matched = &binding;
    ++count;
  }

  if (count != 1)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding lookup for ") + context +
        " requires exactly one logical operand '" + logicalOperand +
        "' in plan '" + plan.planID + "'");

  const bool hasUse = llvm::any_of(
      matched->materializedUses, [&](const std::string &use) {
        return llvm::StringRef(use) == materializedUse;
      });
  if (!hasUse)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding lookup for ") + context +
        " requires logical operand '" + logicalOperand +
        "' to record materialized use '" + materializedUse + "'");

  if (llvm::StringRef(matched->parameter.cName).empty() ||
      llvm::StringRef(matched->parameter.cType).empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding lookup for ") + context +
        " found empty C ABI name/type for logical operand '" +
        logicalOperand + "'");

  return &matched->parameter;
}

std::string
stringifyRVVRouteOperandBindingPlan(const RVVRouteOperandBindingPlan &plan) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << plan.planID;
  for (const RVVRouteOperandBinding &binding : plan.bindings) {
    os << ";" << binding.logicalOperand << "="
       << support::stringifyRuntimeABIParameterRole(binding.parameter.role)
       << ":" << binding.parameter.cName << ":";
    for (auto indexedUse : llvm::enumerate(binding.materializedUses)) {
      if (indexedUse.index() != 0)
        os << "|";
      os << indexedUse.value();
    }
  }
  os.flush();
  return text;
}

llvm::Error verifyRVVRouteOperandBindingPlan(
    const RVVRouteOperandBindingPlan &plan, llvm::StringRef expectedPlanID,
    llvm::StringRef expectedRuntimeABIOrder, llvm::StringRef context) {
  if (llvm::StringRef(plan.planID) != expectedPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context + " requires plan id '" + expectedPlanID + "' but found '" +
        plan.planID + "'");
  if (plan.bindings.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context + " requires at least one binding");

  llvm::StringSet<> logicalOperands;
  llvm::StringSet<> runtimeRoles;
  std::string runtimeABIOrder;
  llvm::raw_string_ostream orderOS(runtimeABIOrder);
  bool firstBinding = true;
  for (const RVVRouteOperandBinding &binding : plan.bindings) {
    if (binding.logicalOperand.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " found empty logical operand");
    if (!logicalOperands.insert(binding.logicalOperand).second)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " found duplicate logical operand '" +
          binding.logicalOperand + "'");
    llvm::StringRef role =
        support::stringifyRuntimeABIParameterRole(binding.parameter.role);
    std::optional<support::RuntimeABIParameterRole> expectedRole =
        getExpectedRVVRouteOperandBindingRole(plan.planID,
                                              binding.logicalOperand);
    if (!expectedRole)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " found unsupported logical operand '" +
          binding.logicalOperand + "' in plan '" + plan.planID + "'");
    if (binding.parameter.role != *expectedRole)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " requires logical operand '" + binding.logicalOperand +
          "' to bind runtime ABI role '" +
          support::stringifyRuntimeABIParameterRole(*expectedRole) +
          "' but found '" + role + "'");
    if (!runtimeRoles.insert(role).second)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " found duplicate runtime ABI role '" + role + "'");
    if (binding.parameter.cName.empty() || binding.parameter.cType.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " requires non-empty C ABI name/type for logical operand '" +
          binding.logicalOperand + "'");
    if (binding.materializedUses.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " requires materialized uses for logical operand '" +
          binding.logicalOperand + "'");
    if (!firstBinding)
      orderOS << ",";
    firstBinding = false;
    orderOS << binding.parameter.cName;
  }
  orderOS.flush();
  if (runtimeABIOrder != expectedRuntimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context + " requires runtime ABI mirror order '" +
        expectedRuntimeABIOrder + "' but plan records '" + runtimeABIOrder +
        "'");

  return llvm::Error::success();
}

namespace {

constexpr llvm::StringLiteral kRVVRuntimeAVLVLControlPlanID(
    "rvv-runtime-avl-vl-control-plan.v1");

llvm::Error makeRVVRuntimeAVLVLControlPlanError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV runtime AVL/VL control plan invalid: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::StringRef stringifyRuntimeControlTailPolicy(
    tcrv::rvv::TailPolicy policy) {
  switch (policy) {
  case tcrv::rvv::TailPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::TailPolicy::Undisturbed:
    return "undisturbed";
  }
  return "unknown";
}

llvm::StringRef stringifyRuntimeControlMaskPolicy(
    tcrv::rvv::MaskPolicy policy) {
  switch (policy) {
  case tcrv::rvv::MaskPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  return "unknown";
}

llvm::Expected<support::RuntimeABIParameter>
getRuntimeAVLParameterBindingFromValue(mlir::Value value,
                                       llvm::StringRef context) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " AVL must be defined by explicit tcrv_rvv.runtime_abi_value");
  if (llvm::Error error =
          verifyRVVRuntimeABIValueRoleOpInterface(binding.getOperation()))
    return std::move(error);

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " AVL binding carries unsupported runtime ABI "
                              "role '" +
        binding.getRole() + "'");
  if (*role != support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " AVL must bind the single runtime-element-count ABI role; got '" +
        binding.getRole() + "'");

  std::optional<support::RuntimeABIParameterOwnership> ownership =
      support::symbolizeRuntimeABIParameterOwnership(binding.getOwnership());
  if (!ownership)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " AVL binding carries unsupported runtime ABI "
                              "ownership '" +
        binding.getOwnership() + "'");

  support::RuntimeABIParameter parameter(binding.getCName(),
                                         binding.getCType(), *role,
                                         *ownership);
  if (parameter.cName != tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName())
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " AVL runtime ABI parameter must be named '" +
        tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName() +
        "' but found '" + parameter.cName + "'");
  if (parameter.cType != "size_t")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " AVL runtime ABI parameter must have C type "
                              "'size_t' but found '" +
        parameter.cType + "'");
  if (parameter.ownership !=
      support::RuntimeABIParameterOwnership::TargetExportABIOwned)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " AVL runtime ABI parameter must be target-export ABI owned");

  return parameter;
}

llvm::Error validateSingleRuntimeAVLBinding(tcrv::exec::VariantOp variant,
                                            mlir::Value runtimeAVLValue,
                                            llvm::StringRef context) {
  if (!variant)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " requires a materialized selected tcrv.exec.variant");

  auto selectedBinding =
      runtimeAVLValue.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!selectedBinding)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " runtime AVL value must be a tcrv_rvv.runtime_abi_value");

  unsigned runtimeElementCountBindings = 0;
  bool selectedIsTheRuntimeCount = false;
  variant.getBody().walk([&](tcrv::rvv::RuntimeABIValueOp op) {
    std::optional<support::RuntimeABIParameterRole> role =
        support::symbolizeRuntimeABIParameterRole(op.getRole());
    if (!role || *role != support::RuntimeABIParameterRole::RuntimeElementCount)
      return;
    ++runtimeElementCountBindings;
    if (op.getOperation() == selectedBinding.getOperation())
      selectedIsTheRuntimeCount = true;
  });

  if (runtimeElementCountBindings != 1)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " requires exactly one runtime-element-count ABI binding in the "
        "selected RVV variant");
  if (!selectedIsTheRuntimeCount)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " setvl AVL must consume the selected variant's single runtime n ABI "
        "binding");
  return llvm::Error::success();
}

llvm::Expected<RVVRuntimeAVLVLControlPlan>
buildRuntimeAVLVLControlPlan(
    tcrv::exec::VariantOp variant, mlir::Value runtimeAVLValue,
    std::int64_t sew, llvm::StringRef lmul, tcrv::rvv::PolicyAttr policy,
    llvm::StringRef runtimeABIOrder, llvm::StringRef context) {
  llvm::Expected<support::RuntimeABIParameter> runtimeAVL =
      getRuntimeAVLParameterBindingFromValue(runtimeAVLValue, context);
  if (!runtimeAVL)
    return runtimeAVL.takeError();
  if (llvm::Error error =
          validateSingleRuntimeAVLBinding(variant, runtimeAVLValue, context))
    return std::move(error);

  const tcrv::rvv::RVVSelectedBodyConfigVLContract &configContract =
      tcrv::rvv::getRVVSelectedBodyConfigVLContract(sew, lmul, policy);

  RVVRuntimeAVLVLControlPlan plan;
  plan.sew = sew;
  plan.lmul = lmul;
  plan.policy = policy;
  if (policy) {
    plan.tailPolicy = stringifyRuntimeControlTailPolicy(policy.getTail());
    plan.maskPolicy = stringifyRuntimeControlMaskPolicy(policy.getMask());
  }
  plan.controlPlanID = kRVVRuntimeAVLVLControlPlanID;
  plan.configContractID = configContract.configContractID;
  plan.runtimeVLContractID = configContract.runtimeVLContractID;
  plan.runtimeAVLABIParameterName =
      configContract.runtimeAVLABIParameterName;
  plan.runtimeAVLASource = configContract.runtimeAVLASource;
  plan.runtimeABIOrder = runtimeABIOrder;
  plan.vlDefOpName = configContract.vlDefOpName;
  plan.vlScopeOpName = configContract.vlScopeOpName;
  plan.vlUses = configContract.vlUses;
  plan.emitCLoopKind = configContract.emitCLoopKind;
  plan.emitCLoopInductionName = configContract.emitCLoopInductionName;
  plan.emitCFullChunkVLName = configContract.emitCFullChunkVLName;
  plan.emitCLoopVLName = tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName();
  plan.remainingAVLMetadata = configContract.remainingAVLMetadata;
  plan.pointerAdvanceMetadata = configContract.pointerAdvanceMetadata;
  plan.boundedSlice = configContract.boundedSlice;
  plan.multiVL = configContract.multiVL;
  plan.runtimeAVLValue = runtimeAVLValue;
  plan.runtimeAVLParameter = std::move(*runtimeAVL);

  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(plan, context))
    return std::move(error);
  return plan;
}

void applyRVVRuntimeAVLVLControlPlanToDescription(
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

} // namespace

llvm::StringRef getRVVRuntimeAVLVLControlPlanID() {
  return kRVVRuntimeAVLVLControlPlanID;
}

llvm::Error verifyRVVRuntimeAVLVLControlPlan(
    const RVVRuntimeAVLVLControlPlan &plan, llvm::StringRef context) {
  if (context.trim().empty())
    return makeRVVRuntimeAVLVLControlPlanError(
        "verification requires a non-empty context");
  if (plan.controlPlanID != kRVVRuntimeAVLVLControlPlanID)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " must use runtime control plan '" +
        kRVVRuntimeAVLVLControlPlanID + "'");
  if (!plan.policy)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " requires explicit RVV policy");
  if (!tcrv::rvv::isRVVFirstSliceDataflowConfig(plan.sew, plan.lmul))
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " requires a supported typed RVV SEW/LMUL config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(plan.policy) &&
      !tcrv::rvv::isRVVUndisturbedPolicy(plan.policy))
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " requires an explicitly supported runtime VL policy");
  if (plan.runtimeAVLABIParameterName !=
      tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName())
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " must use runtime AVL ABI parameter '" +
        tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName() + "'");
  if (plan.runtimeAVLASource != "runtime_abi:n")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " must derive runtime AVL source from runtime_abi:n");
  if (plan.runtimeAVLParameter.role !=
      support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " must carry runtime-element-count as AVL parameter role");
  if (plan.runtimeAVLParameter.cName !=
      tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName())
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " must carry runtime AVL ABI parameter name '" +
        tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName() + "'");
  if (plan.runtimeAVLParameter.cType != "size_t")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " must carry runtime AVL ABI C type 'size_t'");
  if (plan.runtimeAVLParameter.ownership !=
      support::RuntimeABIParameterOwnership::TargetExportABIOwned)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " must carry target-export-owned runtime AVL ABI ownership");
  if (plan.vlDefOpName != "tcrv_rvv.setvl" ||
      plan.vlScopeOpName != "tcrv_rvv.with_vl")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " must route VL through tcrv_rvv.setvl and tcrv_rvv.with_vl");
  if (plan.remainingAVLMetadata != "n-offset" ||
      plan.pointerAdvanceMetadata != "offset")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " must carry remaining AVL 'n-offset' and pointer advance 'offset'");
  if (plan.multiVL != "supported")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " must keep multi-VL support validated");
  return llvm::Error::success();
}

llvm::Expected<RVVRuntimeAVLVLControlPlan>
deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
    tcrv::exec::VariantOp variant, mlir::Value runtimeAVLValue,
    std::int64_t sew, llvm::StringRef lmul, tcrv::rvv::PolicyAttr policy,
    llvm::StringRef runtimeABIOrder, llvm::StringRef context) {
  return buildRuntimeAVLVLControlPlan(variant, runtimeAVLValue, sew, lmul,
                                      policy, runtimeABIOrder, context);
}

llvm::Expected<RVVRuntimeAVLVLControlPlan>
deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
    tcrv::exec::VariantOp variant, tcrv::rvv::SetVLOp setvl,
    tcrv::rvv::WithVLOp withVL, llvm::StringRef runtimeABIOrder,
    llvm::StringRef context) {
  if (!setvl)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " requires a validated tcrv_rvv.setvl op");
  if (!withVL)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " requires a validated tcrv_rvv.with_vl op");
  if (setvl->getBlock() != withVL->getBlock() ||
      !setvl->isBeforeInBlock(withVL.getOperation()))
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " requires tcrv_rvv.setvl to dominate the tcrv_rvv.with_vl scope");

  tcrv::rvv::RVVConfigContractDiagnostic structure =
      tcrv::rvv::validateRVVSelectedBodyConfigVLStructure(setvl, withVL);
  if (!structure.ok)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " " + structure.message);

  tcrv::rvv::RVVCompileTimeConfig config =
      tcrv::rvv::getRVVSetVLCompileTimeConfig(setvl);
  return buildRuntimeAVLVLControlPlan(variant, setvl.getAvl(), config.sew,
                                      config.lmul, config.policy,
                                      runtimeABIOrder, context);
}

namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral
    kRVVSelectedBodyEmissionKind("materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral
    kRVVSelectedBodyLoweringBoundaryOpName("tcrv_rvv.with_vl");
constexpr llvm::StringLiteral
    kRVVSelectedBodyRuntimeABIKind("plugin-owned-runtime-abi");
constexpr llvm::StringLiteral
    kRVVSelectedBodyRuntimeGlueRole("emitc-cpp-rvv-intrinsic-runtime-glue");
constexpr llvm::StringLiteral
    kRVVReductionAccumulatorLayout("rhs-vector-seed-lane0-per-vl-chunk");
constexpr llvm::StringLiteral
    kRVVReductionResultLayout("store-reduction-lane0-to-output-chunk-base");
constexpr llvm::StringLiteral kRVVReductionStoreVL("1");
constexpr llvm::StringLiteral kRVVStandaloneReductionAccumulatorLayout(
    "scalar-i32-seed-lane0-from-accumulator-input");
constexpr llvm::StringLiteral kRVVStandaloneReductionResultLayout(
    "store-standalone-reduction-lane0-to-output-scalar");
constexpr llvm::StringLiteral kRVVStandaloneReductionStoreVL("1");
constexpr llvm::StringLiteral
    kRVVMaskedCompareMaskSource("compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral
    kRVVMaskedPredicateMaskRole("predicate-mask-produced-by-compare");
constexpr llvm::StringLiteral kRVVMaskedInactiveLaneContract(
    "masked-off-lanes-preserve-passthrough-vector");
constexpr llvm::StringLiteral kRVVMaskedPassthroughLayout(
    "passthrough-vector-preserves-inactive-lanes");
constexpr llvm::StringLiteral
    kRVVMAccAccumulatorLayout("separate-i32-vector-accumulator-input");
constexpr llvm::StringLiteral
    kRVVMAccResultLayout("store-multiply-accumulate-result-to-output-buffer");
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
constexpr llvm::StringLiteral kRVVGenericBinaryRuntimeABIOrder(
    "lhs,rhs,out,n");
constexpr llvm::StringLiteral kRVVStridedRuntimeABIOrder(
    "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride");
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
constexpr llvm::StringLiteral kRVVComputedMaskMemoryRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,dst,n");
constexpr llvm::StringLiteral kRVVComputedMaskSelectRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,true_value,false_value,out,n");
constexpr llvm::StringLiteral kRVVComputedMaskStridedStoreRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,dst,n,dst_stride_bytes");
constexpr llvm::StringLiteral kRVVSegment2RuntimeABIOrder(
    "src,out0,out1,n");
constexpr llvm::StringLiteral kRVVSegment2InterleaveRuntimeABIOrder(
    "src0,src1,dst,n");
constexpr llvm::StringLiteral kRVVScalarBroadcastRuntimeABIOrder(
    "lhs,rhs_scalar,out,n");
constexpr llvm::StringLiteral kRVVStandaloneReductionRuntimeABIOrder(
    "lhs,acc,out,n");
constexpr llvm::StringLiteral kRVVStandaloneReductionTargetLeafProfile(
    "rvv-v1-e32m1-standalone-reduction-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVStandaloneReductionProviderSupportedMirror(
    "provider_supported_mirror:rvv-standalone-reduction-plan-validated");
constexpr llvm::StringLiteral kRVVStandaloneReductionRequiredHeaderDeclarations(
    "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVStandaloneReductionCTypeMappingSummary(
    "vl:size_t,input:signed-e32m1,seed:i32,result:signed-e32m1");
constexpr llvm::StringLiteral kRVVScalarBroadcastElementwiseTargetLeafProfile(
    "rvv-v1-e32m1-scalar-broadcast-elementwise-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVScalarBroadcastElementwiseProviderSupportedMirror(
        "provider_supported_mirror:rvv-scalar-broadcast-elementwise-plan-validated");
constexpr llvm::StringLiteral
    kRVVScalarBroadcastElementwiseRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVScalarBroadcastElementwiseCTypeMappingSummary(
    "vl:size_t,lhs:signed-e32m1,rhs_scalar:i32,result:signed-e32m1");
constexpr llvm::StringLiteral kRVVWideningConversionRuntimeABIOrder(
    "lhs,out,n");
constexpr llvm::StringLiteral kRVVMAccRuntimeABIOrder("lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral kRVVWideningMAccRuntimeABIOrder(
    "lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral kRVVWideningDotProductRuntimeABIOrder(
    "lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral
    kRVVStridedInputWideningDotProductRuntimeABIOrder(
        "lhs,rhs,acc,out,n,lhs_stride,rhs_stride");
constexpr llvm::StringLiteral
    kRVVComputedMaskWideningDotProductRuntimeABIOrder(
        "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotProductRuntimeABIOrder(
        "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride");
constexpr llvm::StringLiteral kRVVStridedMemoryLayout(
    "element-strided-lhs-rhs-output-runtime-abi");
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
constexpr llvm::StringLiteral kRVVComputedMaskMemoryLayout(
    "unit-stride-compare-source-old-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskSelectMemoryLayout(
    "unit-stride-compare-true-false-select-output-runtime-abi");
constexpr llvm::StringLiteral kRVVComputedMaskStridedStoreMemoryLayout(
    "unit-stride-compare-source-byte-strided-old-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVStridedInputWideningDotMemoryLayout(
    "element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral
    kRVVComputedMaskStridedInputWideningDotMemoryLayout(
        "unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi");
constexpr llvm::StringLiteral kRVVSegment2MemoryLayout(
    "segment2-interleaved-source-dual-unit-stride-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVSegment2InterleaveMemoryLayout(
    "dual-unit-stride-source-segment2-interleaved-destination-runtime-abi");
constexpr llvm::StringLiteral kRVVLHSStrideSource("runtime_abi:lhs_stride");
constexpr llvm::StringLiteral kRVVRHSStrideSource("runtime_abi:rhs_stride");
constexpr llvm::StringLiteral kRVVOutStrideSource("runtime_abi:out_stride");
constexpr llvm::StringLiteral kRVVSourceStrideSource(
    "runtime_abi:stride_bytes");
constexpr llvm::StringLiteral kRVVDestinationByteStrideSource(
    "runtime_abi:dst_stride_bytes");
constexpr llvm::StringLiteral kRVVSourceMemoryForm("strided-load");
constexpr llvm::StringLiteral kRVVStridedInputDotSourceMemoryForm(
    "strided-load");
constexpr llvm::StringLiteral kRVVUnitStrideSourceMemoryForm(
    "unit-stride-load");
constexpr llvm::StringLiteral kRVVIndexedDataMemoryForm("indexed-load");
constexpr llvm::StringLiteral kRVVIndexedDestinationMemoryForm(
    "indexed-store");
constexpr llvm::StringLiteral kRVVIndexSource("runtime_abi:index");
constexpr llvm::StringLiteral kRVVMaskSource("runtime_abi:mask");
constexpr llvm::StringLiteral kRVVMaskRole("predicate-mask-input-buffer");
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
constexpr llvm::StringLiteral kRVVMaskedStoreIntrinsic(
    "__riscv_vse32_v_i32m1_m");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskMemoryForm(
    "compare-produced-mask");
constexpr llvm::StringLiteral kRVVSegment2SourceMemoryForm(
    "segment2-interleaved-unit-stride-load");
constexpr llvm::StringLiteral kRVVSegment2Field0Role(
    "segment-field0-output-buffer");
constexpr llvm::StringLiteral kRVVSegment2Field1Role(
    "segment-field1-output-buffer");
constexpr llvm::StringLiteral kRVVSegment2Field0InputRole(
    "segment-field0-input-buffer");
constexpr llvm::StringLiteral kRVVSegment2Field1InputRole(
    "segment-field1-input-buffer");
constexpr llvm::StringLiteral kRVVSegment2TupleCType("vint32m1x2_t");
constexpr llvm::StringLiteral kRVVSegment2LoadIntrinsic(
    "__riscv_vlseg2e32_v_i32m1x2");
constexpr llvm::StringLiteral kRVVSegment2StoreIntrinsic(
    "__riscv_vsseg2e32_v_i32m1x2");
constexpr llvm::StringLiteral kRVVSegment2FieldExtractIntrinsic(
    "__riscv_vget_v_i32m1x2_i32m1");
constexpr llvm::StringLiteral kRVVSegment2TupleCreateIntrinsic(
    "__riscv_vcreate_v_i32m1x2");
constexpr llvm::StringLiteral kRVVSegment2InterleavedDestinationMemoryForm(
    "segment2-interleaved-unit-stride-store");
constexpr llvm::StringLiteral kRVVIndexedGatherOffsetUnit("element");
constexpr llvm::StringLiteral kRVVIndexedScatterIndexUniqueness("unique");
constexpr llvm::StringLiteral kRVVDestinationMemoryForm("unit-stride-store");
constexpr llvm::StringLiteral kRVVWideningConversionRelation(
    "signed-i32m1-to-i64m2");
constexpr llvm::StringLiteral kRVVWidenI16ToI32ConversionRelation(
    "signed-i16mf2-to-i32m1");

struct RVVSelectedBodyOperationProfile {
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef operationMnemonic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  bool isCompareSelect;
  bool isReduction;
  bool isMaskedArithmetic;
  bool isMultiplyAccumulate;
  bool isStridedMemory;
  bool isMemoryMovement;
  bool isIndexedMemoryMovement;
  bool isMaskedMemoryMovement;
  bool isSegmentedMemoryMovement;
  bool isWideningConversion;
};

struct RVVSelectedBodyConfigProfile {
  std::int64_t sew;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  const tcrv::rvv::RVVSelectedBodyConfigVLContract *configContract = nullptr;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef indexVectorTypeName;
  llvm::StringRef maskTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef indexVectorCType;
  llvm::StringRef maskCType;
  llvm::StringRef scalarCType;
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
  llvm::StringRef elementByteSize;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef indexLoadIntrinsic;
  llvm::StringRef indexScaleIntrinsic;
  llvm::StringRef indexedLoadIntrinsic;
  llvm::StringRef indexedStoreIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef stridedStoreIntrinsic;
};

struct RVVSelectedBodyTargetLeafProfile {
  llvm::StringRef intrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
};

struct RVVSelectedBodyRouteProfile {
  RVVSelectedBodyOperationProfile operation;
  RVVSelectedBodyConfigProfile config;
  RVVSelectedBodyTargetLeafProfile targetLeaves;
};

constexpr RVVSelectedBodyOperationKind kRVVSelectedBodyOperationKinds[] = {
    RVVSelectedBodyOperationKind::Add, RVVSelectedBodyOperationKind::Sub,
    RVVSelectedBodyOperationKind::Mul,
    RVVSelectedBodyOperationKind::CmpSelect,
    RVVSelectedBodyOperationKind::ComputedMaskSelect,
    RVVSelectedBodyOperationKind::ReduceAdd,
    RVVSelectedBodyOperationKind::StandaloneReduceAdd,
    RVVSelectedBodyOperationKind::MaskedAdd,
    RVVSelectedBodyOperationKind::MAccAdd,
    RVVSelectedBodyOperationKind::StridedAdd,
    RVVSelectedBodyOperationKind::StridedLoadUnitStore,
    RVVSelectedBodyOperationKind::UnitLoadStridedStore,
    RVVSelectedBodyOperationKind::IndexedGatherUnitStore,
    RVVSelectedBodyOperationKind::IndexedScatterUnitLoad,
    RVVSelectedBodyOperationKind::MaskedUnitLoadStore,
    RVVSelectedBodyOperationKind::MaskedUnitStore,
    RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore,
    RVVSelectedBodyOperationKind::ComputedMaskStridedStore,
    RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore,
    RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad,
    RVVSelectedBodyOperationKind::ScalarBroadcastAdd,
    RVVSelectedBodyOperationKind::WidenI32ToI64,
    RVVSelectedBodyOperationKind::WidenI16ToI32,
    RVVSelectedBodyOperationKind::WideningMAccAdd,
    RVVSelectedBodyOperationKind::WideningDotReduceAdd,
    RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd,
    RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd,
    RVVSelectedBodyOperationKind::
        ComputedMaskStridedInputWideningDotReduceAdd};

const RVVSelectedBodyOperationProfile &
getRVVSelectedBodyOperationProfile(RVVSelectedBodyOperationKind op) {
  static const RVVSelectedBodyOperationProfile kAdd = {
      RVVSelectedBodyOperationKind::Add, "add", "sum_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kSub = {
      RVVSelectedBodyOperationKind::Sub, "sub", "difference_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMul = {
      RVVSelectedBodyOperationKind::Mul, "mul", "product_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kCmpSelect = {
      RVVSelectedBodyOperationKind::CmpSelect, "cmp_select", "selected_vec",
      "cmp_mask", /*isCompareSelect=*/true, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kComputedMaskSelect = {
      RVVSelectedBodyOperationKind::ComputedMaskSelect,
      "computed_mask_select", "computed_selected_vec",
      "computed_select_mask", /*isCompareSelect=*/true,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kReduceAdd = {
      RVVSelectedBodyOperationKind::ReduceAdd, "reduce_add", "reduced_vec",
      "", /*isCompareSelect=*/false, /*isReduction=*/true,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kStandaloneReduceAdd = {
      RVVSelectedBodyOperationKind::StandaloneReduceAdd,
      "standalone_reduce_add", "standalone_reduced_vec",
      "", /*isCompareSelect=*/false, /*isReduction=*/true,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMaskedAdd = {
      RVVSelectedBodyOperationKind::MaskedAdd, "masked_add",
      "masked_sum_vec", "add_mask", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/true,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMAccAdd = {
      RVVSelectedBodyOperationKind::MAccAdd, "macc_add", "macc_sum_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/true,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kStridedAdd = {
      RVVSelectedBodyOperationKind::StridedAdd, "strided_add",
      "strided_sum_vec", "", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/true,
      /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kStridedLoadUnitStore = {
      RVVSelectedBodyOperationKind::StridedLoadUnitStore,
      "strided_load_unit_store", "loaded_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/true, /*isMemoryMovement=*/true,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kUnitLoadStridedStore = {
      RVVSelectedBodyOperationKind::UnitLoadStridedStore,
      "unit_load_strided_store", "loaded_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/true, /*isMemoryMovement=*/true,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kIndexedGatherUnitStore = {
      RVVSelectedBodyOperationKind::IndexedGatherUnitStore,
      "indexed_gather_unit_store", "loaded_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/true, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kIndexedScatterUnitLoad = {
      RVVSelectedBodyOperationKind::IndexedScatterUnitLoad,
      "indexed_scatter_unit_load", "loaded_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/true, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMaskedUnitLoadStore = {
      RVVSelectedBodyOperationKind::MaskedUnitLoadStore,
      "masked_unit_load_store", "masked_loaded_vec", "memory_mask",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/true,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kMaskedUnitStore = {
      RVVSelectedBodyOperationKind::MaskedUnitStore,
      "masked_unit_store", "payload_vec", "memory_mask",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/true,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kComputedMaskUnitLoadStore = {
      RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore,
      "computed_masked_unit_load_store", "computed_masked_loaded_vec",
      "computed_memory_mask", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/true,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kComputedMaskStridedStore = {
      RVVSelectedBodyOperationKind::ComputedMaskStridedStore,
      "computed_masked_strided_store", "computed_masked_strided_vec",
      "computed_strided_memory_mask", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/false, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/true,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kSegment2DeinterleaveUnitStore = {
      RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore,
      "segment2_deinterleave_unit_store", "field0_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/true,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kSegment2InterleaveUnitLoad = {
      RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad,
      "segment2_interleave_unit_load", "segment2_tuple", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/true,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kScalarBroadcastAdd = {
      RVVSelectedBodyOperationKind::ScalarBroadcastAdd,
      "scalar_broadcast_add", "sum_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kWidenI32ToI64 = {
      RVVSelectedBodyOperationKind::WidenI32ToI64, "widen_i32_to_i64",
      "widened_vec", "", /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/true};
  static const RVVSelectedBodyOperationProfile kWidenI16ToI32 = {
      RVVSelectedBodyOperationKind::WidenI16ToI32, "widen_i16_to_i32",
      "widened_vec", "", /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/true};
  static const RVVSelectedBodyOperationProfile kWideningMAccAdd = {
      RVVSelectedBodyOperationKind::WideningMAccAdd, "widening_macc_add",
      "widening_macc_sum_vec", "", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/false,
      /*isMultiplyAccumulate=*/true, /*isStridedMemory=*/false,
      /*isMemoryMovement=*/false, /*isIndexedMemoryMovement=*/false,
      /*isMaskedMemoryMovement=*/false, /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile kWideningDotReduceAdd = {
      RVVSelectedBodyOperationKind::WideningDotReduceAdd,
      "widening_dot_reduce_add", "widening_dot_reduced_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
      /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
      /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
      /*isSegmentedMemoryMovement=*/false,
      /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kStridedInputWideningDotReduceAdd = {
          RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd,
          "strided_input_widening_dot_reduce_add",
          "strided_input_widening_dot_reduced_vec", "",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/true, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskWideningDotReduceAdd = {
          RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd,
          "computed_masked_widening_dot_reduce_add",
          "computed_masked_widening_dot_reduced_vec", "dot_reduce_mask",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/false, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};
  static const RVVSelectedBodyOperationProfile
      kComputedMaskStridedInputWideningDotReduceAdd = {
          RVVSelectedBodyOperationKind::
              ComputedMaskStridedInputWideningDotReduceAdd,
          "computed_masked_strided_input_widening_dot_reduce_add",
          "computed_masked_strided_input_widening_dot_reduced_vec",
          "dot_reduce_mask",
          /*isCompareSelect=*/false, /*isReduction=*/false,
          /*isMaskedArithmetic=*/false, /*isMultiplyAccumulate=*/false,
          /*isStridedMemory=*/true, /*isMemoryMovement=*/false,
          /*isIndexedMemoryMovement=*/false, /*isMaskedMemoryMovement=*/false,
          /*isSegmentedMemoryMovement=*/false,
          /*isWideningConversion=*/false};

  switch (op) {
  case RVVSelectedBodyOperationKind::Add:
    return kAdd;
  case RVVSelectedBodyOperationKind::Sub:
    return kSub;
  case RVVSelectedBodyOperationKind::Mul:
    return kMul;
  case RVVSelectedBodyOperationKind::CmpSelect:
    return kCmpSelect;
  case RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return kComputedMaskSelect;
  case RVVSelectedBodyOperationKind::ReduceAdd:
    return kReduceAdd;
  case RVVSelectedBodyOperationKind::StandaloneReduceAdd:
    return kStandaloneReduceAdd;
  case RVVSelectedBodyOperationKind::MaskedAdd:
    return kMaskedAdd;
  case RVVSelectedBodyOperationKind::MAccAdd:
    return kMAccAdd;
  case RVVSelectedBodyOperationKind::StridedAdd:
    return kStridedAdd;
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return kStridedLoadUnitStore;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return kUnitLoadStridedStore;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return kIndexedGatherUnitStore;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return kIndexedScatterUnitLoad;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return kMaskedUnitLoadStore;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return kMaskedUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return kComputedMaskUnitLoadStore;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kComputedMaskStridedStore;
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return kSegment2DeinterleaveUnitStore;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return kSegment2InterleaveUnitLoad;
  case RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
    return kScalarBroadcastAdd;
  case RVVSelectedBodyOperationKind::WidenI32ToI64:
    return kWidenI32ToI64;
  case RVVSelectedBodyOperationKind::WidenI16ToI32:
    return kWidenI16ToI32;
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    return kWideningMAccAdd;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    return kWideningDotReduceAdd;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return kStridedInputWideningDotReduceAdd;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return kComputedMaskWideningDotReduceAdd;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return kComputedMaskStridedInputWideningDotReduceAdd;
  }
  llvm_unreachable("unknown RVV selected-body operation");
}

llvm::Error makeUnsupportedRVVSelectedBodyRouteProfileError(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV selected-body route profile: operation=") +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      ", memory_form=" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) +
      ", SEW=" + llvm::Twine(description.sew) + ", LMUL=" + description.lmul +
      ", tail_policy=" + description.tailPolicy +
      ", mask_policy=" + description.maskPolicy);
}

llvm::Expected<RVVSelectedBodyConfigProfile>
deriveRVVSelectedBodyConfigProfile(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  const bool isMaskedStorePolicy =
      description.operation == RVVSelectedBodyOperationKind::MaskedUnitStore;
  if (isMaskedStorePolicy) {
    if (description.tailPolicy != "undisturbed" ||
        description.maskPolicy != "undisturbed")
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (description.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
        description.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
  } else if (description.tailPolicy != "agnostic" ||
             description.maskPolicy != "agnostic") {
    return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
  }

  if (description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1())
    return RVVSelectedBodyConfigProfile{
        32,
        "m1",
        isMaskedStorePolicy ? "undisturbed" : "agnostic",
        isMaskedStorePolicy ? "undisturbed" : "agnostic",
        isMaskedStorePolicy
            ? &tcrv::rvv::getRVVSelectedBodyM1UndisturbedConfigVLContract()
            : &tcrv::rvv::getRVVSelectedBodyConfigVLContract("m1"),
        "size_t",
        "!tcrv_rvv.vector<i32, \"m1\">",
        "!tcrv_rvv.index_vector<i32, \"m1\">",
        "!tcrv_rvv.mask<i32, \"m1\">",
        "vint32m1_t",
        "vuint32m1_t",
        "vbool32_t",
        "int32_t",
        "const int32_t *",
        "int32_t *",
        "4",
        "__riscv_vsetvl_e32m1",
        "__riscv_vle32_v_i32m1",
        "__riscv_vle32_v_u32m1",
        "__riscv_vmul_vx_u32m1",
        "__riscv_vloxei32_v_i32m1",
        "__riscv_vsoxei32_v_i32m1",
        "__riscv_vlse32_v_i32m1",
        "__riscv_vmv_v_x_i32m1",
        "__riscv_vse32_v_i32m1",
        "__riscv_vsse32_v_i32m1"};

  if (description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM2())
    return RVVSelectedBodyConfigProfile{
        32,
        "m2",
        "agnostic",
        "agnostic",
        &tcrv::rvv::getRVVSelectedBodyConfigVLContract("m2"),
        "size_t",
        "!tcrv_rvv.vector<i32, \"m2\">",
        "",
        "!tcrv_rvv.mask<i32, \"m2\">",
        "vint32m2_t",
        "",
        "vbool16_t",
        "int32_t",
        "const int32_t *",
        "int32_t *",
        "4",
        "__riscv_vsetvl_e32m2",
        "__riscv_vle32_v_i32m2",
        "",
        "",
        "",
        "",
        "__riscv_vlse32_v_i32m2",
        "__riscv_vmv_v_x_i32m2",
        "__riscv_vse32_v_i32m2",
        "__riscv_vsse32_v_i32m2"};

  if (description.sew == tcrv::rvv::getRVVSEW64Bits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1() &&
      description.tailPolicy == "agnostic" &&
      description.maskPolicy == "agnostic")
    return RVVSelectedBodyConfigProfile{
        64,
        "m1",
        "agnostic",
        "agnostic",
        &tcrv::rvv::getRVVSelectedBodyConfigVLContract(64, "m1"),
        "size_t",
        "!tcrv_rvv.vector<i64, \"m1\">",
        "",
        "!tcrv_rvv.mask<i64, \"m1\">",
        "vint64m1_t",
        "",
        "vbool64_t",
        "int64_t",
        "const int64_t *",
        "int64_t *",
        "8",
        "__riscv_vsetvl_e64m1",
        "__riscv_vle64_v_i64m1",
        "",
        "",
        "",
        "",
        "__riscv_vlse64_v_i64m1",
        "__riscv_vmv_v_x_i64m1",
        "__riscv_vse64_v_i64m1",
        "__riscv_vsse64_v_i64m1"};

  if (description.sew == tcrv::rvv::getRVVSEW64Bits() &&
      description.lmul == tcrv::rvv::getRVVLMULM2() &&
      description.tailPolicy == "agnostic" &&
      description.maskPolicy == "agnostic")
    return RVVSelectedBodyConfigProfile{
        64,
        "m2",
        "agnostic",
        "agnostic",
        &tcrv::rvv::getRVVSelectedBodyConfigVLContract(64, "m2"),
        "size_t",
        "!tcrv_rvv.vector<i64, \"m2\">",
        "",
        "!tcrv_rvv.mask<i64, \"m2\">",
        "vint64m2_t",
        "",
        "vbool32_t",
        "int64_t",
        "const int64_t *",
        "int64_t *",
        "8",
        "__riscv_vsetvl_e64m2",
        "__riscv_vle64_v_i64m2",
        "",
        "",
        "",
        "",
        "__riscv_vlse64_v_i64m2",
        "__riscv_vmv_v_x_i64m2",
        "__riscv_vse64_v_i64m2",
        "__riscv_vsse64_v_i64m2"};

  return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
}

llvm::StringRef getRVVSelectedBodyArithmeticIntrinsic(
    RVVSelectedBodyOperationKind operation,
    const RVVSelectedBodyConfigProfile &config) {
  if (config.sew == tcrv::rvv::getRVVSEW64Bits()) {
    if (operation == RVVSelectedBodyOperationKind::Add &&
        config.lmul == tcrv::rvv::getRVVLMULM1())
      return "__riscv_vadd_vv_i64m1";
    return {};
  }

  switch (operation) {
  case RVVSelectedBodyOperationKind::Add:
  case RVVSelectedBodyOperationKind::StridedAdd:
  case RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
    return config.lmul == tcrv::rvv::getRVVLMULM2()
               ? "__riscv_vadd_vv_i32m2"
               : "__riscv_vadd_vv_i32m1";
  case RVVSelectedBodyOperationKind::Sub:
    return config.lmul == tcrv::rvv::getRVVLMULM2()
               ? "__riscv_vsub_vv_i32m2"
               : "__riscv_vsub_vv_i32m1";
  case RVVSelectedBodyOperationKind::Mul:
    return config.lmul == tcrv::rvv::getRVVLMULM2()
               ? "__riscv_vmul_vv_i32m2"
               : "__riscv_vmul_vv_i32m1";
  case RVVSelectedBodyOperationKind::CmpSelect:
  case RVVSelectedBodyOperationKind::ComputedMaskSelect:
    llvm_unreachable("compare/select uses dedicated compare and merge leaves");
  case RVVSelectedBodyOperationKind::ReduceAdd:
  case RVVSelectedBodyOperationKind::StandaloneReduceAdd:
    llvm_unreachable("reduction uses dedicated reduction intrinsic leaf");
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    llvm_unreachable(
        "widening dot-product reduction uses dedicated widening product and "
        "reduction leaves");
  case RVVSelectedBodyOperationKind::MaskedAdd:
    llvm_unreachable("masked arithmetic uses dedicated masked intrinsic leaf");
  case RVVSelectedBodyOperationKind::MAccAdd:
    llvm_unreachable("multiply-accumulate uses dedicated macc intrinsic leaf");
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    llvm_unreachable("strided memory movement uses load/store leaves only");
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    llvm_unreachable("strided memory movement uses load/store leaves only");
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    llvm_unreachable("indexed memory movement uses load/store leaves only");
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    llvm_unreachable("indexed memory movement uses load/store leaves only");
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    llvm_unreachable("masked memory movement uses mask/load/store leaves only");
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    llvm_unreachable("masked store uses mask/load/masked-store leaves only");
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    llvm_unreachable(
        "computed-mask memory movement uses compare/mask/load/store leaves "
        "only");
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    llvm_unreachable(
        "segment2 memory movement uses segment load/extract/store leaves only");
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    llvm_unreachable(
        "segment2 memory movement uses load/tuple/store leaves only");
  case RVVSelectedBodyOperationKind::WidenI32ToI64:
  case RVVSelectedBodyOperationKind::WidenI16ToI32:
    llvm_unreachable("widening conversion uses dedicated conversion leaf");
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    llvm_unreachable("widening macc uses dedicated widening macc leaf");
  }
  llvm_unreachable("unknown RVV selected-body operation");
}

llvm::StringRef getRVVSelectedBodyWideningConversionIntrinsic(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.sourceSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.sourceLMUL == tcrv::rvv::getRVVLMULM1() &&
      description.sew == tcrv::rvv::getRVVSEW64Bits() &&
      description.lmul == tcrv::rvv::getRVVLMULM2() &&
      description.conversionRelation == kRVVWideningConversionRelation)
    return "__riscv_vwcvt_x_x_v_i64m2";
  if (description.sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
      description.sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
      description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1() &&
      description.conversionRelation == kRVVWidenI16ToI32ConversionRelation)
    return "__riscv_vwcvt_x_x_v_i32m1";
  return {};
}

llvm::StringRef
getRVVSelectedBodyMAccIntrinsic(llvm::StringRef lmul) {
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmacc_vv_i32m2"
             : "__riscv_vmacc_vv_i32m1";
}

llvm::StringRef getRVVSelectedBodyWideningMAccIntrinsic(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
      description.sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
      description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1() &&
      description.wideningMAccRelation == kRVVWideningMAccRelation)
    return "__riscv_vwmacc_vv_i32m1";
  return {};
}

llvm::StringRef getRVVSelectedBodyWideningProductIntrinsic(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
      description.sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
      description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1() &&
      description.wideningDotProductRelation ==
          kRVVWideningDotProductRelation)
    return "__riscv_vwmul_vv_i32m1";
  return {};
}

llvm::StringRef getRVVSelectedBodyMaskedWideningProductIntrinsic(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
      description.sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
      description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1() &&
      description.wideningDotProductRelation ==
          kRVVWideningDotProductRelation)
    return "__riscv_vwmul_vv_i32m1_m";
  return {};
}

llvm::StringRef
getRVVSelectedBodyReductionIntrinsic(llvm::StringRef lmul) {
  if (lmul == tcrv::rvv::getRVVLMULM1())
    return "__riscv_vredsum_vs_i32m1_i32m1";
  return {};
}

llvm::StringRef
getRVVSelectedBodyCompareIntrinsic(llvm::StringRef lmul) {
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmseq_vv_i32m2_b16"
             : "__riscv_vmseq_vv_i32m1_b32";
}

llvm::StringRef
getRVVSelectedBodySignedLessThanCompareIntrinsic(llvm::StringRef lmul) {
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmslt_vv_i32m2_b16"
             : "__riscv_vmslt_vv_i32m1_b32";
}

llvm::StringRef getRVVSelectedBodyMaskFromI32Intrinsic(llvm::StringRef lmul) {
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmsne_vx_i32m2_b16"
             : "__riscv_vmsne_vx_i32m1_b32";
}

llvm::StringRef
getRVVSelectedBodySelectIntrinsic(llvm::StringRef lmul) {
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmerge_vvm_i32m2"
             : "__riscv_vmerge_vvm_i32m1";
}

bool isRVVSelectedBodyContractionRouteOperation(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
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
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
         op == RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
         op == RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyContractionComputedMask(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyContractionStridedInputs(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskStridedInputWideningDotReduceAdd;
}

llvm::StringRef getRVVSelectedBodyContractionRuntimeABIOrder(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
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
  } else if (llvm::Error error =
                 requireRVVSelectedBodyContractionPlanField(
                     plan, "strided source-load leaf",
                     plan.stridedLoadIntrinsic, ""))
    return error;

  if (plan.usesWideningMAcc) {
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
  } else {
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
  }

  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyContractionRouteFamilyPlan>
deriveRVVSelectedBodyContractionRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile) {
  const RVVSelectedBodyOperationKind operation =
      analysis.slice.arithmeticKind;
  if (!isRVVSelectedBodyContractionRouteOperation(operation))
    return makeRVVEmitCRouteProviderError(
        "requested contraction route-family plan for non-contraction RVV "
        "operation");

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
  plan.runtimeABIOrder =
      getRVVSelectedBodyContractionRuntimeABIOrder(operation);
  plan.targetLeafProfile = kRVVContractionTargetLeafProfile;
  plan.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  plan.cTypeMappingSummary = kRVVContractionCTypeMappingSummary;
  plan.vlCType = configProfile.vlCType;
  plan.resultVectorTypeName = configProfile.vectorTypeName;
  plan.resultVectorCType = configProfile.vectorCType;
  plan.maskTypeName = plan.usesComputedMask ? configProfile.maskTypeName : "";
  plan.maskCType = plan.usesComputedMask ? configProfile.maskCType : "";
  plan.setVLIntrinsic = configProfile.setVLIntrinsic;
  plan.sourceSEW = tcrv::rvv::getRVVSEW16Bits();
  plan.sourceLMUL = tcrv::rvv::getRVVLMULMF2();
  plan.sourceVectorTypeName = "!tcrv_rvv.vector<i16, \"mf2\">";
  plan.sourceVectorCType = "vint16mf2_t";
  plan.sourceVectorLoadIntrinsic = "__riscv_vle16_v_i16mf2";
  if (plan.usesStridedInputs)
    plan.stridedLoadIntrinsic = "__riscv_vlse16_v_i16mf2";
  plan.storeIntrinsic = configProfile.storeIntrinsic;

  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  if (plan.usesComputedMask) {
    plan.runtimeABIParameters.push_back(analysis.slice.dotLHSABI);
    plan.runtimeABIParameters.push_back(analysis.slice.dotRHSABI);
  }
  plan.runtimeABIParameters.push_back(analysis.slice.accumulatorABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(analysis.slice.runtimeElementCountABI);
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
        getRVVSelectedBodyWideningMAccIntrinsic(analysis.description);
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
          getRVVSelectedBodySignedLessThanCompareIntrinsic(configProfile.lmul);
      plan.maskedMergeIntrinsic =
          getRVVSelectedBodySelectIntrinsic(configProfile.lmul);
      plan.maskedWideningProductIntrinsic =
          getRVVSelectedBodyMaskedWideningProductIntrinsic(
              analysis.description);
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
        getRVVSelectedBodyReductionIntrinsic(configProfile.lmul);
    plan.wideningProductIntrinsic =
        getRVVSelectedBodyWideningProductIntrinsic(analysis.description);
    plan.scalarSeedSplatIntrinsic = configProfile.rhsBroadcastIntrinsic;
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

bool isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ScalarBroadcastAdd;
}

llvm::Error requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
    const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("scalar-broadcast elementwise route-family plan validation "
                  "for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error validateRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
    const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "scalar-broadcast elementwise route-family runtime AVL/VL control"))
    return error;
  if (plan.operation != RVVSelectedBodyOperationKind::ScalarBroadcastAdd)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan currently supports "
        "only scalar_broadcast_add");
  if (plan.memoryForm != RVVSelectedBodyMemoryForm::RHSScalarBroadcast)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires "
        "rhs-scalar-broadcast memory form");
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "runtime control plan",
              plan.runtimeControlPlan.controlPlanID,
              getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "runtime ABI order", plan.runtimeABIOrder,
              kRVVScalarBroadcastRuntimeABIOrder))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "target leaf profile", plan.targetLeafProfile,
              kRVVScalarBroadcastElementwiseTargetLeafProfile))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "provider_supported_mirror",
              plan.providerSupportedMirror,
              kRVVScalarBroadcastElementwiseProviderSupportedMirror))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "header declarations", plan.requiredHeaderDeclarations,
              kRVVScalarBroadcastElementwiseRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "C type mapping summary", plan.cTypeMappingSummary,
              kRVVScalarBroadcastElementwiseCTypeMappingSummary))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires "
        "provider-owned header declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "VL C type", plan.vlCType, "size_t"))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "vector type", plan.vectorTypeName,
              "!tcrv_rvv.vector<i32, \"m1\">"))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "vector C type", plan.vectorCType, "vint32m1_t"))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "setvl leaf", plan.setVLIntrinsic,
              "__riscv_vsetvl_e32m1"))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "vector-load leaf", plan.vectorLoadIntrinsic,
              "__riscv_vle32_v_i32m1"))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "RHS scalar splat leaf",
              plan.rhsScalarSplatIntrinsic,
              "__riscv_vmv_v_x_i32m1"))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "elementwise compute leaf", plan.arithmeticIntrinsic,
              "__riscv_vadd_vv_i32m1"))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "store leaf", plan.storeIntrinsic,
              "__riscv_vse32_v_i32m1"))
    return error;
  if (llvm::Error error =
          requireRVVSelectedBodyScalarBroadcastElementwisePlanField(
              plan, "result name", plan.resultName, "sum_vec"))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan>
deriveRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves) {
  if (!isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
          analysis.slice.arithmeticKind))
    return makeRVVEmitCRouteProviderError(
        "requested scalar-broadcast elementwise route-family plan for "
        "non-scalar-broadcast RVV operation");
  if (analysis.slice.memoryForm != RVVSelectedBodyMemoryForm::RHSScalarBroadcast)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires "
        "rhs-scalar-broadcast typed body structure");
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.rhsScalarSplat ||
      !analysis.slice.genericStore || !analysis.slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires explicit "
        "load, scalar splat, binary compute, and store body structure");
  if (analysis.slice.arithmeticKind !=
      RVVSelectedBodyOperationKind::ScalarBroadcastAdd)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan currently requires "
        "add binary compute");
  if (configProfile.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      configProfile.lmul != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan currently requires "
        "SEW32 LMUL m1 typed config");
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      analysis.slice.rhsABI.role !=
          support::RuntimeABIParameterRole::RHSScalarValue ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires lhs buffer, "
        "RHS scalar, output buffer, and runtime element-count ABI roles");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          kRVVScalarBroadcastRuntimeABIOrder,
          "scalar-broadcast elementwise route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan plan;
  plan.operation = analysis.slice.arithmeticKind;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = kRVVScalarBroadcastElementwiseTargetLeafProfile;
  plan.providerSupportedMirror =
      kRVVScalarBroadcastElementwiseProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations =
      kRVVScalarBroadcastElementwiseRequiredHeaderDeclarations;
  plan.cTypeMappingSummary =
      kRVVScalarBroadcastElementwiseCTypeMappingSummary;
  plan.vlCType = configProfile.vlCType;
  plan.vectorTypeName = configProfile.vectorTypeName;
  plan.vectorCType = configProfile.vectorCType;
  plan.setVLIntrinsic = configProfile.setVLIntrinsic;
  plan.vectorLoadIntrinsic = configProfile.vectorLoadIntrinsic;
  plan.rhsScalarSplatIntrinsic = targetLeaves.rhsBroadcastIntrinsic;
  plan.arithmeticIntrinsic = targetLeaves.intrinsic;
  plan.storeIntrinsic = configProfile.storeIntrinsic;
  plan.resultName = "sum_vec";
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error =
          validateRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
              plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
    const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyRVVRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                               description);
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
  description.intrinsic = plan.arithmeticIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.resultName = plan.resultName;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

bool isRVVSelectedBodyStandaloneReductionRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::StandaloneReduceAdd;
}

llvm::Error requireRVVSelectedBodyStandaloneReductionPlanField(
    const RVVSelectedBodyStandaloneReductionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("standalone reduction route-family plan validation for "
                  "operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error validateRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
    const RVVSelectedBodyStandaloneReductionRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "standalone reduction route-family runtime AVL/VL control"))
    return error;
  if (plan.operation != RVVSelectedBodyOperationKind::StandaloneReduceAdd)
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan currently supports only "
        "standalone_reduce_add");
  if (plan.memoryForm != RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction)
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires "
        "unit-stride-standalone-reduction memory form");
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "runtime control plan", plan.runtimeControlPlan.controlPlanID,
          getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "runtime ABI order", plan.runtimeABIOrder,
          kRVVStandaloneReductionRuntimeABIOrder))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "target leaf profile", plan.targetLeafProfile,
          kRVVStandaloneReductionTargetLeafProfile))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "provider_supported_mirror", plan.providerSupportedMirror,
          kRVVStandaloneReductionProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "header declarations", plan.requiredHeaderDeclarations,
          kRVVStandaloneReductionRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "C type mapping summary", plan.cTypeMappingSummary,
          kRVVStandaloneReductionCTypeMappingSummary))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires provider-owned header "
        "declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "VL C type", plan.vlCType, "size_t"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "vector type", plan.vectorTypeName,
          "!tcrv_rvv.vector<i32, \"m1\">"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "vector C type", plan.vectorCType, "vint32m1_t"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "setvl leaf", plan.setVLIntrinsic, "__riscv_vsetvl_e32m1"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "vector-load leaf", plan.vectorLoadIntrinsic,
          "__riscv_vle32_v_i32m1"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic,
          "__riscv_vmv_v_x_i32m1"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "reduction leaf", plan.reductionIntrinsic,
          "__riscv_vredsum_vs_i32m1_i32m1"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "store leaf", plan.storeIntrinsic, "__riscv_vse32_v_i32m1"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "accumulator layout", plan.accumulatorLayout,
          kRVVStandaloneReductionAccumulatorLayout))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "result layout", plan.resultLayout,
          kRVVStandaloneReductionResultLayout))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "reduction store VL", plan.reductionStoreVL,
          kRVVStandaloneReductionStoreVL))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyStandaloneReductionPlanField(
          plan, "result name", plan.resultName, "standalone_reduced_vec"))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyStandaloneReductionRouteFamilyPlan>
deriveRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyConfigProfile &configProfile,
    const RVVSelectedBodyTargetLeafProfile &targetLeaves) {
  if (!isRVVSelectedBodyStandaloneReductionRouteOperation(
          analysis.slice.arithmeticKind))
    return makeRVVEmitCRouteProviderError(
        "requested standalone reduction route-family plan for non-standalone "
        "RVV operation");
  if (analysis.slice.memoryForm !=
      RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction)
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires "
        "unit-stride-standalone-reduction typed body structure");
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.standaloneReduceOp ||
      !analysis.slice.genericStore || !analysis.slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires explicit input load, "
        "standalone_reduce compute, and scalar-output store body structure");
  if (configProfile.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      configProfile.lmul != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan currently requires SEW32 LMUL "
        "m1 typed config");
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      analysis.slice.accumulatorABI.role !=
          support::RuntimeABIParameterRole::AccumulatorInputBuffer ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "standalone reduction route-family plan requires lhs buffer, "
        "accumulator seed buffer, output buffer, and runtime element-count ABI "
        "roles");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          kRVVStandaloneReductionRuntimeABIOrder,
          "standalone reduction route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  RVVSelectedBodyStandaloneReductionRouteFamilyPlan plan;
  plan.operation = analysis.slice.arithmeticKind;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = kRVVStandaloneReductionTargetLeafProfile;
  plan.providerSupportedMirror = kRVVStandaloneReductionProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations =
      kRVVStandaloneReductionRequiredHeaderDeclarations;
  plan.cTypeMappingSummary = kRVVStandaloneReductionCTypeMappingSummary;
  plan.vlCType = configProfile.vlCType;
  plan.vectorTypeName = configProfile.vectorTypeName;
  plan.vectorCType = configProfile.vectorCType;
  plan.setVLIntrinsic = configProfile.setVLIntrinsic;
  plan.vectorLoadIntrinsic = configProfile.vectorLoadIntrinsic;
  plan.scalarSeedSplatIntrinsic = configProfile.rhsBroadcastIntrinsic;
  plan.reductionIntrinsic = targetLeaves.intrinsic;
  plan.storeIntrinsic = configProfile.storeIntrinsic;
  plan.accumulatorLayout =
      analysis.slice.standaloneReduceOp.getAccumulatorLayout();
  plan.resultLayout = analysis.slice.standaloneReduceOp.getResultLayout();
  plan.reductionStoreVL = kRVVStandaloneReductionStoreVL;
  plan.resultName = "standalone_reduced_vec";
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.accumulatorABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error =
          validateRVVSelectedBodyStandaloneReductionRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
    const RVVSelectedBodyStandaloneReductionRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyRVVRuntimeAVLVLControlPlanToDescription(plan.runtimeControlPlan,
                                               description);
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
  description.scalarSeedSplatIntrinsic = plan.scalarSeedSplatIntrinsic;
  description.intrinsic = plan.reductionIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.reductionAccumulatorLayout = plan.accumulatorLayout;
  description.reductionResultLayout = plan.resultLayout;
  description.reductionStoreVL = plan.reductionStoreVL;
  description.resultName = plan.resultName;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

llvm::Expected<RVVSelectedBodyTargetLeafProfile>
deriveRVVSelectedBodyTargetLeafProfile(
    const RVVSelectedBodyEmitCRouteDescription &description,
    const RVVSelectedBodyOperationProfile &operationProfile,
    const RVVSelectedBodyConfigProfile &configProfile) {
  if (configProfile.sew == tcrv::rvv::getRVVSEW64Bits() &&
      description.operation != RVVSelectedBodyOperationKind::WidenI32ToI64) {
    if (description.operation != RVVSelectedBodyOperationKind::Add ||
        description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad ||
        configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
  }

  if (operationProfile.isWideningConversion) {
    if (description.memoryForm !=
        RVVSelectedBodyMemoryForm::UnitStrideConversion)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef intrinsic =
        getRVVSelectedBodyWideningConversionIntrinsic(description);
    if (intrinsic.empty())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{intrinsic, "", "", ""};
  }

  if (operationProfile.isCompareSelect) {
    const bool isComputedMaskSelect =
        description.operation == RVVSelectedBodyOperationKind::ComputedMaskSelect;
    if ((!isComputedMaskSelect &&
         description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad) ||
        (isComputedMaskSelect &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodySelectIntrinsic(configProfile.lmul),
        isComputedMaskSelect
            ? getRVVSelectedBodySignedLessThanCompareIntrinsic(
                  configProfile.lmul)
            : getRVVSelectedBodyCompareIntrinsic(configProfile.lmul),
        "", ""};
  }

  if (description.operation ==
          RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              ComputedMaskStridedInputWideningDotReduceAdd) {
    const bool isComputedMaskDotReduce =
        description.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
        description.operation ==
            RVVSelectedBodyOperationKind::
                ComputedMaskStridedInputWideningDotReduceAdd;
    const bool isStridedInputDotReduce =
        description.operation ==
            RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
        description.operation ==
            RVVSelectedBodyOperationKind::
                ComputedMaskStridedInputWideningDotReduceAdd;
    const bool isComputedMaskStridedInputDotReduce =
        description.operation ==
        RVVSelectedBodyOperationKind::
            ComputedMaskStridedInputWideningDotReduceAdd;
    if ((!isComputedMaskDotReduce && !isStridedInputDotReduce &&
         description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad) ||
        (isStridedInputDotReduce && !isComputedMaskStridedInputDotReduce &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce) ||
        (isComputedMaskStridedInputDotReduce &&
         description.memoryForm != RVVSelectedBodyMemoryForm::
                                       ComputedMaskStridedInputWideningDotReduce) ||
        (isComputedMaskDotReduce && !isStridedInputDotReduce &&
         description.memoryForm != RVVSelectedBodyMemoryForm::
                                       ComputedMaskUnitStrideWideningDotReduce))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef wideningProductIntrinsic =
        getRVVSelectedBodyWideningProductIntrinsic(description);
    llvm::StringRef reductionIntrinsic =
        getRVVSelectedBodyReductionIntrinsic(configProfile.lmul);
    llvm::StringRef maskedWideningProductIntrinsic =
        getRVVSelectedBodyMaskedWideningProductIntrinsic(description);
    if (wideningProductIntrinsic.empty() || reductionIntrinsic.empty() ||
        (isComputedMaskDotReduce && maskedWideningProductIntrinsic.empty()))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        reductionIntrinsic,
        isComputedMaskDotReduce
            ? getRVVSelectedBodySignedLessThanCompareIntrinsic(
                  configProfile.lmul)
            : llvm::StringRef(),
        isComputedMaskDotReduce ? getRVVSelectedBodySelectIntrinsic(
                                      configProfile.lmul)
                                : llvm::StringRef(),
        ""};
  }

  if (operationProfile.isReduction) {
    const bool isStandalone =
        description.operation == RVVSelectedBodyOperationKind::StandaloneReduceAdd;
    if ((!isStandalone &&
         description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad) ||
        (isStandalone &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef reductionIntrinsic =
        getRVVSelectedBodyReductionIntrinsic(configProfile.lmul);
    if (reductionIntrinsic.empty())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{reductionIntrinsic, "", "", ""};
  }

  if (operationProfile.isMaskedArithmetic) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyArithmeticIntrinsic(
            RVVSelectedBodyOperationKind::Add, configProfile),
        getRVVSelectedBodyCompareIntrinsic(configProfile.lmul),
        getRVVSelectedBodySelectIntrinsic(configProfile.lmul), ""};
  }

  if (operationProfile.isMultiplyAccumulate) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (description.operation == RVVSelectedBodyOperationKind::WideningMAccAdd) {
      llvm::StringRef intrinsic =
          getRVVSelectedBodyWideningMAccIntrinsic(description);
      if (intrinsic.empty())
        return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
      return RVVSelectedBodyTargetLeafProfile{intrinsic, "", "", ""};
    }
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyMAccIntrinsic(configProfile.lmul), "", "", ""};
  }

  if (operationProfile.isMemoryMovement) {
    const bool isStridedLoadUnitStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::StridedLoadUnitStore;
    const bool isUnitLoadStridedStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::UnitLoadStridedStore;
    if ((isStridedLoadUnitStore &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::StridedLoadUnitStore) ||
        (isUnitLoadStridedStore &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::UnitLoadStridedStore))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{"", "", "", ""};
  }

  if (operationProfile.isSegmentedMemoryMovement) {
    if (description.memoryForm !=
            RVVSelectedBodyMemoryForm::Segment2LoadUnitStore &&
        description.memoryForm !=
            RVVSelectedBodyMemoryForm::UnitLoadSegment2Store)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1() ||
        configProfile.sew != tcrv::rvv::getRVVFirstSliceSEWBits())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{"", "", "", ""};
  }

  if (operationProfile.isIndexedMemoryMovement) {
    const bool isGather =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedGatherUnitStore;
    const bool isScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
    if ((isGather &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::IndexedLoadUnitStore) ||
        (isScatter &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::UnitLoadIndexedStore))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1() ||
        configProfile.indexLoadIntrinsic.empty() ||
        configProfile.indexScaleIntrinsic.empty() ||
        (isGather && configProfile.indexedLoadIntrinsic.empty()) ||
        (isScatter && configProfile.indexedStoreIntrinsic.empty()))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{"", "", "", ""};
  }

  if (operationProfile.isMaskedMemoryMovement) {
    const bool isRuntimeMask =
        description.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
    const bool isMaskedStore =
        description.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore;
    const bool isComputedMask =
        description.memoryForm ==
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore;
    const bool isComputedMaskStridedStore =
        description.memoryForm ==
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
    if (!isRuntimeMask && !isMaskedStore && !isComputedMask &&
        !isComputedMaskStridedStore)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        "",
        (isComputedMask || isComputedMaskStridedStore)
            ? getRVVSelectedBodySignedLessThanCompareIntrinsic(
                  configProfile.lmul)
            : getRVVSelectedBodyMaskFromI32Intrinsic(configProfile.lmul),
        isMaskedStore ? llvm::StringRef()
                      : getRVVSelectedBodySelectIntrinsic(configProfile.lmul),
        ""};
  }

  if (operationProfile.isStridedMemory) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::StridedLoadStore)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyArithmeticIntrinsic(
            RVVSelectedBodyOperationKind::StridedAdd, configProfile),
        "", "", ""};
  }

  if (description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad ||
      description.memoryForm == RVVSelectedBodyMemoryForm::RHSScalarBroadcast)
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyArithmeticIntrinsic(description.operation,
                                             configProfile),
        "", "", configProfile.rhsBroadcastIntrinsic};

  if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
    return makeUnsupportedRVVSelectedBodyRouteProfileError(description);

  return RVVSelectedBodyTargetLeafProfile{
      getRVVSelectedBodyArithmeticIntrinsic(description.operation,
                                           configProfile),
      "", "", ""};
}

llvm::Expected<RVVSelectedBodyRouteProfile>
deriveRVVSelectedBodyRouteProfile(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  RVVSelectedBodyRouteProfile profile;
  profile.operation = getRVVSelectedBodyOperationProfile(description.operation);

  llvm::Expected<RVVSelectedBodyConfigProfile> config =
      deriveRVVSelectedBodyConfigProfile(description);
  if (!config)
    return config.takeError();
  profile.config = *config;

  llvm::Expected<RVVSelectedBodyTargetLeafProfile> targetLeaves =
      deriveRVVSelectedBodyTargetLeafProfile(description, profile.operation,
                                            profile.config);
  if (!targetLeaves)
    return targetLeaves.takeError();
  profile.targetLeaves = *targetLeaves;
  return profile;
}

llvm::Error requireRouteDescriptionText(llvm::StringRef context,
                                        llvm::StringRef field,
                                        llvm::StringRef value) {
  if (!value.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(llvm::Twine(context) + " " + field +
                                        " must be provider-derived and "
                                        "non-empty");
}

llvm::Error requireRouteDescriptionField(llvm::StringRef context,
                                         llvm::StringRef field,
                                         llvm::StringRef actual,
                                         llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " " + field +
      " must mirror selected-body route profile fact '" + expected +
      "' but was '" + actual + "'");
}

llvm::StringRef stringifyRVVTailPolicy(tcrv::rvv::TailPolicy policy) {
  switch (policy) {
  case tcrv::rvv::TailPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::TailPolicy::Undisturbed:
    return "undisturbed";
  }
  llvm_unreachable("unknown RVV tail policy");
}

llvm::StringRef stringifyRVVMaskPolicy(tcrv::rvv::MaskPolicy policy) {
  switch (policy) {
  case tcrv::rvv::MaskPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  llvm_unreachable("unknown RVV mask policy");
}

const RVVSelectedBodyConstructionRoute &
getRVVSelectedBodyConstructionRouteOrDie(RVVSelectedBodyOperationKind op) {
  const RVVSelectedBodyOperationProfile &profile =
      getRVVSelectedBodyOperationProfile(op);
  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          profile.operationMnemonic);
  if (!route) {
    std::string message = llvm::toString(route.takeError());
    llvm::report_fatal_error(llvm::StringRef(message));
  }
  return **route;
}

bool variantContainsExplicitTypedRVVBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found || op == variant.getOperation())
      return;
    if (op->getName().getDialectNamespace() == "tcrv_rvv")
      found = true;
  });
  return found;
}

llvm::Error requireRVVVariantLegality(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "requires a materialized tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>("origin");
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVEmitCRouteProviderError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  if (variantContainsExplicitTypedRVVBody(variant))
    return llvm::Error::success();

  return makeRVVEmitCRouteProviderError(
      "materialized RVV variant requires explicit typed RVV "
      "extension-family body");
}

llvm::Error validateRVVSelectedBodyVectorTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  if (!value)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be present before route construction");
  auto vectorType = llvm::dyn_cast<tcrv::rvv::VectorType>(value.getType());
  if (!vectorType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.vector value before route construction");

  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be an integer type");
  if (integerElementType.getWidth() != config.sew)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element width " + llvm::Twine(integerElementType.getWidth()) +
        " to match selected config SEW " + llvm::Twine(config.sew));
  if (vectorType.getLmul() != config.lmul)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + vectorType.getLmul() +
        "' to match selected config LMUL '" + config.lmul + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyMaskTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  auto maskType = llvm::dyn_cast<tcrv::rvv::MaskType>(value.getType());
  if (!maskType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.mask value before route construction");

  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(maskType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be an integer type");
  if (integerElementType.getWidth() != config.sew)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element width " + llvm::Twine(integerElementType.getWidth()) +
        " to match selected config SEW " + llvm::Twine(config.sew));
  if (maskType.getLmul() != config.lmul)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + maskType.getLmul() +
        "' to match selected config LMUL '" + config.lmul + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyIndexVectorTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  auto vectorType =
      llvm::dyn_cast<tcrv::rvv::IndexVectorType>(value.getType());
  if (!vectorType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.index_vector value before route "
        "construction");

  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be an integer type");
  if (integerElementType.getWidth() != 32)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " index element width 32 for the bounded indexed gather route");
  if (vectorType.getLmul() != config.lmul ||
      vectorType.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + vectorType.getLmul() +
        "' to match selected config LMUL m1 for indexed gather");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyTypedConfigFacts(
    const RVVSelectedBodyRouteSlice &slice,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI32ToI64 ||
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI16ToI32) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    if (slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI16ToI32) {
      sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
      sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    } else {
      sourceConfig.sew = tcrv::rvv::getRVVFirstSliceSEWBits();
      sourceConfig.lmul = tcrv::rvv::getRVVLMULM1();
    }
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "conversion source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "conversion result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "conversion stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningMAccAdd) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "widening macc lhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "widening macc rhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue, "widening macc accumulator vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "widening macc result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "widening macc stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::WideningDotReduceAdd) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "widening dot-reduction lhs source vector",
            sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "widening dot-reduction rhs source vector",
            sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "widening dot-reduction result vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "widening dot-reduction stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "strided-input dot-reduction lhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "strided-input dot-reduction rhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "strided-input dot-reduction result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "strided-input dot-reduction stored vector",
            config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "computed-mask dot-reduction compare lhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "computed-mask dot-reduction compare rhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask,
            "computed-mask dot-reduction predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotLHSValue,
            "computed-mask dot-reduction lhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotRHSValue,
            "computed-mask dot-reduction rhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "computed-mask dot-reduction result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "computed-mask dot-reduction stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          ComputedMaskStridedInputWideningDotReduceAdd) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "computed-mask strided-input dot-reduction compare lhs vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "computed-mask strided-input dot-reduction compare rhs vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask,
            "computed-mask strided-input dot-reduction predicate mask",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotLHSValue,
            "computed-mask strided-input dot-reduction lhs source vector",
            sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotRHSValue,
            "computed-mask strided-input dot-reduction rhs source vector",
            sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "computed-mask strided-input dot-reduction result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "computed-mask strided-input dot-reduction stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::StridedLoadUnitStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "strided source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "unit-stride stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::UnitLoadStridedStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "unit-load source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "strided-store stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
    if (llvm::Error error = validateRVVSelectedBodyIndexVectorTypeAgainstConfig(
            slice.indexValue, "indexed gather index vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "indexed gather loaded data vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "unit-stride stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::IndexedScatterUnitLoad) {
    if (llvm::Error error = validateRVVSelectedBodyIndexVectorTypeAgainstConfig(
            slice.indexValue, "indexed scatter index vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "indexed scatter source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "indexed-store stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::MaskedUnitLoadStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "masked memory source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue, "masked memory old destination vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.maskValue, "masked memory predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "masked movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "masked unit-stride stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedUnitStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "masked store payload source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.maskValue, "masked store predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "masked store payload vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "computed-mask compare lhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "computed-mask compare rhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.sourceValue, "computed-mask active source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue,
            "computed-mask old destination vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask, "computed-mask predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "computed-mask movement result vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "computed-mask unit-stride stored vector",
            config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskSelect) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "computed-mask select compare lhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "computed-mask select compare rhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.trueValue, "computed-mask select true-value vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.falseValue, "computed-mask select false-value vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask, "computed-mask select predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "computed-mask selected result vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "computed-mask select stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskStridedStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "computed-mask strided-store compare lhs vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "computed-mask strided-store compare rhs vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.sourceValue,
            "computed-mask strided-store active source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue,
            "computed-mask strided-store old destination vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask, "computed-mask strided-store predicate mask",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "computed-mask strided-store movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "computed-mask strided-store stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field0LoadedValue, "segment2 field0 loaded vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field1LoadedValue, "segment2 field1 loaded vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field0Value, "segment2 field0 movement result", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field1Value, "segment2 field1 movement result", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field0Value, "segment2 field0 source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field1Value, "segment2 field1 source vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::StandaloneReduceAdd) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "standalone reduction input vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "standalone reduction result vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "standalone reduction stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.lhsValue, "lhs vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.rhsValue, "rhs vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.arithmeticResult, "compute result vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.storeValue, "stored vector", config))
    return error;
  if (slice.compareOp)
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask, "compare mask", config))
      return error;
  if (slice.maskedBinaryOp)
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.maskedPassthrough, "masked passthrough vector", config))
      return error;
  if (slice.maccOp)
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue, "multiply-accumulate accumulator vector",
            config))
      return error;
  return llvm::Error::success();
}

std::string formatRuntimeABIExpectedRoles(
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles) {
  std::string expected;
  llvm::raw_string_ostream stream(expected);
  llvm::interleave(
      expectedRoles,
      [&](support::RuntimeABIParameterRole role) {
        stream << "'" << support::stringifyRuntimeABIParameterRole(role) << "'";
      },
      [&] { stream << " or "; });
  stream.flush();
  return expected;
}

llvm::Expected<support::RuntimeABIParameter>
getRuntimeABIParameterBindingFromValue(
    mlir::Value value, llvm::StringRef context,
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must be defined by explicit tcrv_rvv.runtime_abi_value before "
        "RVV EmitC route construction");
  if (llvm::Error error =
          verifyRVVRuntimeABIValueRoleOpInterface(binding.getOperation()))
    return std::move(error);

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI role '" +
        binding.getRole() + "'");
  if (!llvm::is_contained(expectedRoles, *role)) {
    std::string expectedRolesText =
        formatRuntimeABIExpectedRoles(expectedRoles);
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " must bind runtime ABI role " +
        expectedRolesText + "; got '" + binding.getRole() + "'");
  }

  std::optional<support::RuntimeABIParameterOwnership> ownership =
      support::symbolizeRuntimeABIParameterOwnership(binding.getOwnership());
  if (!ownership)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI ownership '" +
        binding.getOwnership() + "'");

  return support::RuntimeABIParameter(binding.getCName(), binding.getCType(),
                                      *role, *ownership);
}

llvm::Error
assignRVVGenericLoadBinding(RVVSelectedBodyRouteSlice &slice,
                            tcrv::rvv::LoadOp load,
                            const support::RuntimeABIParameter &parameter) {
  if (parameter.role == support::RuntimeABIParameterRole::LHSInputBuffer) {
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique lhs-input-buffer load");
    slice.lhsGenericLoad = load;
    slice.lhsLoadOperation = load.getOperation();
    slice.lhsBuffer = load.getBuffer();
    slice.lhsValue = load.getLoaded();
    slice.lhsABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::RHSInputBuffer) {
    if (slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique rhs-input-buffer load");
    slice.rhsGenericLoad = load;
    slice.rhsLoadOperation = load.getOperation();
    slice.rhsBuffer = load.getBuffer();
    slice.rhsValue = load.getLoaded();
    slice.rhsABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::TrueValueInputBuffer) {
    if (slice.trueValueLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV computed-mask select route requires a unique "
          "true-value-input-buffer load");
    slice.trueValueGenericLoad = load;
    slice.trueValueLoadOperation = load.getOperation();
    slice.trueValueBuffer = load.getBuffer();
    slice.trueValue = load.getLoaded();
    slice.trueValueABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::FalseValueInputBuffer) {
    if (slice.falseValueLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV computed-mask select route requires a unique "
          "false-value-input-buffer load");
    slice.falseValueGenericLoad = load;
    slice.falseValueLoadOperation = load.getOperation();
    slice.falseValueBuffer = load.getBuffer();
    slice.falseValue = load.getLoaded();
    slice.falseValueABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::DotLHSInputBuffer) {
    if (slice.dotLHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV computed-mask dot-reduction route requires a unique "
          "dot-lhs-input-buffer load");
    slice.dotLHSGenericLoad = load;
    slice.dotLHSLoadOperation = load.getOperation();
    slice.dotLHSBuffer = load.getBuffer();
    slice.dotLHSValue = load.getLoaded();
    slice.dotLHSABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::DotRHSInputBuffer) {
    if (slice.dotRHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV computed-mask dot-reduction route requires a unique "
          "dot-rhs-input-buffer load");
    slice.dotRHSGenericLoad = load;
    slice.dotRHSLoadOperation = load.getOperation();
    slice.dotRHSBuffer = load.getBuffer();
    slice.dotRHSValue = load.getLoaded();
    slice.dotRHSABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::SourceInputBuffer) {
    if (slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique source-input-buffer "
          "load");
    slice.sourceGenericLoad = load;
    slice.sourceLoadOperation = load.getOperation();
    slice.sourceBuffer = load.getBuffer();
    slice.sourceValue = load.getLoaded();
    slice.sourceABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::OutputBuffer) {
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique output-buffer "
          "accumulator load");
    slice.accumulatorLoadOperation = load.getOperation();
    slice.accumulatorBuffer = load.getBuffer();
    slice.accumulatorValue = load.getLoaded();
    slice.accumulatorABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::AccumulatorInputBuffer) {
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique "
          "accumulator-input-buffer load");
    slice.accumulatorLoadOperation = load.getOperation();
    slice.accumulatorBuffer = load.getBuffer();
    slice.accumulatorValue = load.getLoaded();
    slice.accumulatorABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::SegmentField0InputBuffer) {
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV segment2 interleave route requires a unique field0 "
          "source load");
    slice.lhsGenericLoad = load;
    slice.lhsLoadOperation = load.getOperation();
    slice.lhsBuffer = load.getBuffer();
    slice.lhsValue = load.getLoaded();
    slice.field0Value = load.getLoaded();
    slice.lhsABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::SegmentField1InputBuffer) {
    if (slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV segment2 interleave route requires a unique field1 "
          "source load");
    slice.rhsGenericLoad = load;
    slice.rhsLoadOperation = load.getOperation();
    slice.rhsBuffer = load.getBuffer();
    slice.rhsValue = load.getLoaded();
    slice.field1Value = load.getLoaded();
    slice.rhsABI = parameter;
    return llvm::Error::success();
  }

  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV generic load runtime ABI role '") +
      support::stringifyRuntimeABIParameterRole(parameter.role) +
      "' for bounded EmitC route");
}

llvm::Error assignRVVGenericBroadcastBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::BroadcastLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::RHSInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV generic broadcast runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) +
        "' for bounded EmitC route");
  if (slice.rhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires a unique RHS source load or "
        "broadcast");
  slice.rhsBroadcastLoad = load;
  slice.rhsLoadOperation = load.getOperation();
  slice.rhsBuffer = load.getBuffer();
  slice.rhsValue = load.getBroadcast();
  slice.rhsABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::RHSBroadcastLoad;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericScalarSplatBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::SplatOp splat,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::RHSScalarValue)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV generic scalar splat runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) +
        "' for bounded EmitC route");
  if (slice.rhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires a unique RHS source load, "
        "broadcast, or scalar splat");
  slice.rhsScalarSplat = splat;
  slice.rhsLoadOperation = splat.getOperation();
  slice.rhsBuffer = splat.getScalar();
  slice.rhsValue = splat.getBroadcast();
  slice.rhsABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::RHSScalarBroadcast;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericStridedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::StridedLoadOp load,
    const support::RuntimeABIParameter &bufferParameter,
    const support::RuntimeABIParameter &strideParameter) {
  if (bufferParameter.role ==
      support::RuntimeABIParameterRole::SourceInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::SourceByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV source byte-strided load requires "
          "source-byte-stride runtime ABI value");
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided-load to unit-stride-store route "
          "requires a unique source strided load");
    slice.lhsStridedLoad = load;
    slice.lhsLoadOperation = load.getOperation();
    slice.lhsBuffer = load.getBuffer();
    slice.lhsStride = load.getStride();
    slice.lhsValue = load.getLoaded();
    slice.lhsABI = bufferParameter;
    slice.lhsStrideABI = strideParameter;
    return llvm::Error::success();
  }

  if (bufferParameter.role == support::RuntimeABIParameterRole::LHSInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::LHSInputStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided lhs load requires "
          "lhs-input-stride runtime ABI value");
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided route requires a unique lhs load");
    slice.lhsStridedLoad = load;
    slice.lhsLoadOperation = load.getOperation();
    slice.lhsBuffer = load.getBuffer();
    slice.lhsStride = load.getStride();
    slice.lhsValue = load.getLoaded();
    slice.lhsABI = bufferParameter;
    slice.lhsStrideABI = strideParameter;
    return llvm::Error::success();
  }

  if (bufferParameter.role == support::RuntimeABIParameterRole::RHSInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::RHSInputStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided rhs load requires "
          "rhs-input-stride runtime ABI value");
    if (slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided route requires a unique rhs load");
    slice.rhsStridedLoad = load;
    slice.rhsLoadOperation = load.getOperation();
    slice.rhsBuffer = load.getBuffer();
    slice.rhsStride = load.getStride();
    slice.rhsValue = load.getLoaded();
    slice.rhsABI = bufferParameter;
    slice.rhsStrideABI = strideParameter;
    slice.memoryForm = RVVSelectedBodyMemoryForm::StridedLoadStore;
    return llvm::Error::success();
  }

  if (bufferParameter.role ==
      support::RuntimeABIParameterRole::DotLHSInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::LHSInputStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided dot lhs load requires "
          "lhs-input-stride runtime ABI value");
    if (slice.dotLHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided dot route requires a "
          "unique dot lhs load");
    slice.lhsStridedLoad = load;
    slice.dotLHSLoadOperation = load.getOperation();
    slice.dotLHSBuffer = load.getBuffer();
    slice.lhsStride = load.getStride();
    slice.dotLHSValue = load.getLoaded();
    slice.dotLHSABI = bufferParameter;
    slice.lhsStrideABI = strideParameter;
    return llvm::Error::success();
  }

  if (bufferParameter.role ==
      support::RuntimeABIParameterRole::DotRHSInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::RHSInputStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided dot rhs load requires "
          "rhs-input-stride runtime ABI value");
    if (slice.dotRHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided dot route requires a "
          "unique dot rhs load");
    slice.rhsStridedLoad = load;
    slice.dotRHSLoadOperation = load.getOperation();
    slice.dotRHSBuffer = load.getBuffer();
    slice.rhsStride = load.getStride();
    slice.dotRHSValue = load.getLoaded();
    slice.dotRHSABI = bufferParameter;
    slice.rhsStrideABI = strideParameter;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskStridedInputWideningDotReduce;
    return llvm::Error::success();
  }

  if (bufferParameter.role == support::RuntimeABIParameterRole::OutputBuffer) {
    if (strideParameter.role !=
            support::RuntimeABIParameterRole::OutputStride &&
        strideParameter.role !=
            support::RuntimeABIParameterRole::DestinationByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided old-destination load requires "
          "output-stride or destination-byte-stride runtime ABI value");
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided route requires a unique "
          "old-destination strided load");
    slice.lhsStridedLoad = load;
    slice.accumulatorLoadOperation = load.getOperation();
    slice.accumulatorBuffer = load.getBuffer();
    slice.accumulatorValue = load.getLoaded();
    slice.outBuffer = load.getBuffer();
    slice.outStride = load.getStride();
    slice.accumulatorABI = bufferParameter;
    slice.outABI = bufferParameter;
    slice.outStrideABI = strideParameter;
    return llvm::Error::success();
  }

  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV strided load buffer runtime ABI role '") +
      support::stringifyRuntimeABIParameterRole(bufferParameter.role) + "'");
}

llvm::Error assignRVVGenericStridedStoreBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::StridedStoreOp store,
    const support::RuntimeABIParameter &bufferParameter,
    const support::RuntimeABIParameter &strideParameter,
    bool requiresDestinationByteStride) {
  if (bufferParameter.role != support::RuntimeABIParameterRole::OutputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV strided store buffer runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(bufferParameter.role) + "'");
  const support::RuntimeABIParameterRole expectedStrideRole =
      requiresDestinationByteStride
          ? support::RuntimeABIParameterRole::DestinationByteStride
          : support::RuntimeABIParameterRole::OutputStride;
  if (strideParameter.role != expectedStrideRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded generic RVV strided store requires ") +
        support::stringifyRuntimeABIParameterRole(expectedStrideRole) +
        " runtime ABI value");
  slice.stridedStore = store;
  slice.storeOperation = store.getOperation();
  slice.outBuffer = store.getBuffer();
  slice.storeValue = store.getValue();
  slice.outStride = store.getStride();
  slice.outABI = bufferParameter;
  slice.outStrideABI = strideParameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::StridedLoadStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericIndexLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::IndexInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV index load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.indexLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires a unique "
        "tcrv_rvv.index_load op");
  if (static_cast<std::int64_t>(load.getIndexEew()) != 32)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route supports only index_eew 32");

  slice.indexLoad = load;
  slice.indexLoadOperation = load.getOperation();
  slice.indexBuffer = load.getIndex();
  slice.indexValue = load.getLoaded();
  slice.indexABI = parameter;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericIndexedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexedLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::LHSInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV indexed data load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.lhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires a unique data "
        "source tcrv_rvv.indexed_load op");
  if (static_cast<std::int64_t>(load.getIndexEew()) != 32)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route supports only index_eew 32");
  if (load.getOffsetUnit() != kRVVIndexedGatherOffsetUnit)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route supports only offset_unit "
        "'element'");

  slice.indexedLoad = load;
  slice.indexedLoadOperation = load.getOperation();
  slice.lhsLoadOperation = load.getOperation();
  slice.lhsBuffer = load.getData();
  slice.indexedDataBuffer = load.getData();
  slice.lhsValue = load.getLoaded();
  slice.lhsABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericIndexedStoreBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexedStoreOp store,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::OutputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV indexed data store runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.indexedStoreOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route requires a unique "
        "tcrv_rvv.indexed_store op");
  if (static_cast<std::int64_t>(store.getIndexEew()) != 32)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route supports only index_eew "
        "32");
  if (store.getOffsetUnit() != kRVVIndexedGatherOffsetUnit)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route supports only offset_unit "
        "'element'");

  slice.indexedStore = store;
  slice.indexedStoreOperation = store.getOperation();
  slice.storeOperation = store.getOperation();
  slice.outBuffer = store.getDestination();
  slice.indexedDestinationBuffer = store.getDestination();
  slice.storeValue = store.getValue();
  slice.outABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericMaskLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::MaskInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV mask load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.maskLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires a unique "
        "tcrv_rvv.mask_load op");
  if (load.getMaskRole() != kRVVMaskRole)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires mask_role "
        "'predicate-mask-input-buffer'");
  if (load.getMaskMemoryForm() != kRVVMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires mask_memory_form "
        "'unit-stride-mask-load'");

  slice.maskLoad = load;
  slice.maskLoadOperation = load.getOperation();
  slice.maskBuffer = load.getMask();
  slice.maskValue = load.getLoaded();
  slice.maskABI = parameter;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericSegment2LoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::Segment2LoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::LHSInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV segment2 load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.segment2LoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires a unique "
        "tcrv_rvv.segment2_load op");
  if (static_cast<std::int64_t>(load.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires "
        "segment_count 2");
  if (load.getSourceMemoryForm() != kRVVSegment2SourceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires "
        "source_memory_form 'segment2-interleaved-unit-stride-load'");
  if (load.getField0Role() != kRVVSegment2Field0Role ||
      load.getField1Role() != kRVVSegment2Field1Role)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires field0_role "
        "'segment-field0-output-buffer' and field1_role "
        "'segment-field1-output-buffer'");
  if (load.getField0Role() == load.getField1Role())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires distinct "
        "field roles");

  slice.segment2Load = load;
  slice.segment2LoadOperation = load.getOperation();
  slice.lhsLoadOperation = load.getOperation();
  slice.lhsBuffer = load.getSource();
  slice.lhsABI = parameter;
  slice.field0LoadedValue = load.getField0();
  slice.field1LoadedValue = load.getField1();
  slice.memoryForm = RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericSegment2StoreBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::Segment2StoreOp store,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role !=
      support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV segment2 store runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.segment2StoreOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires a unique "
        "tcrv_rvv.segment2_store op");
  if (static_cast<std::int64_t>(store.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires "
        "segment_count 2");
  if (store.getDestinationMemoryForm() !=
      kRVVSegment2InterleavedDestinationMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires "
        "destination_memory_form "
        "'segment2-interleaved-unit-stride-store'");
  if (store.getField0Role() != kRVVSegment2Field0InputRole ||
      store.getField1Role() != kRVVSegment2Field1InputRole)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires field0_role "
        "'segment-field0-input-buffer' and field1_role "
        "'segment-field1-input-buffer'");
  if (store.getField0Role() == store.getField1Role())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires distinct "
        "field roles");

  slice.segment2Store = store;
  slice.segment2StoreOperation = store.getOperation();
  slice.storeOperation = store.getOperation();
  slice.outBuffer = store.getDestination();
  slice.outABI = parameter;
  slice.field0Value = store.getField0();
  slice.field1Value = store.getField1();
  slice.storeValue = store.getField0();
  slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyRuntimeABIParameters(
    RVVSelectedBodyRouteSlice &slice,
    const RVVSelectedBodyConstructionRoute &constructionRoute,
    const RVVSelectedBodyConfigProfile &configProfile,
    const support::RuntimeABIParameter &runtimeElementCountABI,
    const support::RuntimeABIParameter &outABI) {
  slice.runtimeElementCountABI = runtimeElementCountABI;
  slice.outABI = outABI;
  const bool isWideningMAcc =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::WideningDotReduceAdd;
  const bool isStridedInputWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd;
  const bool isComputedMaskWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
  const bool isComputedMaskStridedInputWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          ComputedMaskStridedInputWideningDotReduceAdd;
  const bool isStandaloneReduction =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::StandaloneReduceAdd;

  llvm::SmallVector<support::RuntimeABIParameter, 9> ordered;
  ordered.push_back(slice.lhsABI);
  if (isComputedMaskWideningDotReduce) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.dotLHSABI);
    ordered.push_back(slice.dotRHSABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit computed-mask widening "
                "dot-product reduction runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isComputedMaskStridedInputWideningDotReduce) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.dotLHSABI);
    ordered.push_back(slice.dotRHSABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    ordered.push_back(slice.lhsStrideABI);
    ordered.push_back(slice.rhsStrideABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit computed-mask "
                "strided-input widening dot-product reduction runtime ABI "
                "values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::Segment2LoadUnitStore) {
    ordered.push_back(slice.field0ABI);
    ordered.push_back(slice.field1ABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit segment2 runtime ABI "
                "values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadSegment2Store) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit segment2 interleave "
                "runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::IndexedLoadUnitStore ||
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadIndexedStore)
    ordered.push_back(slice.indexABI);
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitLoadStore ||
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore)
    ordered.push_back(slice.maskABI);
  if (slice.memoryForm != RVVSelectedBodyMemoryForm::UnitStrideConversion &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::StridedLoadUnitStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::UnitLoadStridedStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::IndexedLoadUnitStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::UnitLoadIndexedStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::MaskedUnitLoadStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::MaskedUnitStore &&
      slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction)
    ordered.push_back(slice.rhsABI);
  if (isWideningMAcc || isWideningDotReduce ||
      isStridedInputWideningDotReduce || isStandaloneReduction)
    ordered.push_back(slice.accumulatorABI);
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect) {
    ordered.push_back(slice.trueValueABI);
    ordered.push_back(slice.falseValueABI);
  }
  if (slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore ||
      slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore)
    ordered.push_back(slice.sourceABI);
  ordered.push_back(slice.outABI);
  ordered.push_back(slice.runtimeElementCountABI);
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore) {
    ordered.push_back(slice.lhsStrideABI);
    ordered.push_back(slice.rhsStrideABI);
    ordered.push_back(slice.outStrideABI);
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce) {
    ordered.push_back(slice.lhsStrideABI);
    ordered.push_back(slice.rhsStrideABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit strided-input widening "
                "dot-product reduction runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedLoadUnitStore) {
    ordered.push_back(slice.lhsStrideABI);
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitLoadStridedStore) {
    ordered.push_back(slice.outStrideABI);
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore) {
    ordered.push_back(slice.outStrideABI);
  } else if (slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::RHSScalarBroadcast &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::UnitStrideConversion &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::StridedLoadUnitStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::UnitLoadStridedStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::IndexedLoadUnitStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::UnitLoadIndexedStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore &&
             slice.memoryForm != RVVSelectedBodyMemoryForm::MaskedUnitLoadStore &&
             slice.memoryForm != RVVSelectedBodyMemoryForm::MaskedUnitStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction) {
    if (isWideningMAcc || isWideningDotReduce) {
      if (llvm::Error error =
              tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                  ordered,
                  isWideningDotReduce
                      ? "selected RVV EmitC route explicit widening "
                        "dot-product reduction runtime ABI values"
                      : "selected RVV EmitC route explicit widening macc "
                        "runtime ABI values"))
        return makeRVVEmitCRouteProviderError(
            llvm::toString(std::move(error)));
      return llvm::Error::success();
    }
    support::FiniteBinaryRuntimeABIContract contract(
        support::FiniteBinaryRuntimeABIContractSpec{
            constructionRoute.runtimeABIContractName,
            configProfile.constInputPointerCType,
            configProfile.outputPointerCType});
    llvm::Expected<support::FiniteBinaryCallableRuntimeABIParameterBindings>
        bindings = support::bindFiniteBinaryCallableRuntimeABIParametersByRole(
            ordered, "RVV selected-body explicit runtime ABI values",
            contract);
    if (!bindings)
      return bindings.takeError();
  }

  if (isStandaloneReduction) {
    if (llvm::Error error = tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
            ordered,
            "selected RVV EmitC route explicit standalone reduction runtime "
            "ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }

  if (llvm::Error error = tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
          ordered, "selected RVV EmitC route explicit runtime ABI values"))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));

  return llvm::Error::success();
}

void addRouteOperandBinding(
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

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVRouteOperandBindingPlan(const RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyRouteSlice &slice = analysis.slice;
  RVVRouteOperandBindingPlan plan;
  llvm::StringRef expectedRuntimeABIOrder;
  llvm::StringRef context;

  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::Add ||
      slice.arithmeticKind == RVVSelectedBodyOperationKind::Sub ||
      slice.arithmeticKind == RVVSelectedBodyOperationKind::Mul) {
    if (slice.arithmeticKind == RVVSelectedBodyOperationKind::Add) {
      plan.planID = kRVVAddOperandBindingPlanID.str();
      context = "add route";
    } else if (slice.arithmeticKind == RVVSelectedBodyOperationKind::Sub) {
      plan.planID = kRVVSubOperandBindingPlanID.str();
      context = "sub route";
    } else {
      plan.planID = kRVVMulOperandBindingPlanID.str();
      context = "mul route";
    }
    expectedRuntimeABIOrder = kRVVGenericBinaryRuntimeABIOrder;
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "load-base", "binary-lhs-call"});
    addRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "load-base", "binary-rhs-call"});
    addRouteOperandBinding(plan, "out", slice.outABI,
                           {"abi", "store-base", "header"});
    addRouteOperandBinding(plan, "n", slice.runtimeElementCountABI,
                           {"abi", "setvl-avl", "loop-control", "header"});
  } else if (slice.arithmeticKind == RVVSelectedBodyOperationKind::CmpSelect) {
    plan.planID = kRVVCmpSelectOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVGenericBinaryRuntimeABIOrder;
    context = "cmp_select route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "load-base", "compare-lhs-call", "select-true-call"});
    addRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "load-base", "compare-rhs-call", "select-false-call"});
    addRouteOperandBinding(plan, "out", slice.outABI,
                           {"abi", "store-base", "header"});
    addRouteOperandBinding(plan, "n", slice.runtimeElementCountABI,
                           {"abi", "setvl-avl", "loop-control", "header"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::ComputedMaskSelect) {
    plan.planID = kRVVComputedMaskSelectOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVComputedMaskSelectRuntimeABIOrder;
    context = "computed_mask_select route";
    addRouteOperandBinding(
        plan, "cmp_lhs", slice.lhsABI,
        {"abi", "cmp-lhs-load", "compare-lhs-call"});
    addRouteOperandBinding(
        plan, "cmp_rhs", slice.rhsABI,
        {"abi", "cmp-rhs-load", "compare-rhs-call"});
    addRouteOperandBinding(
        plan, "true_value", slice.trueValueABI,
        {"abi", "true-load", "select-true-call"});
    addRouteOperandBinding(
        plan, "false_value", slice.falseValueABI,
        {"abi", "false-load", "select-false-call"});
    addRouteOperandBinding(plan, "out", slice.outABI,
                           {"abi", "store-base", "header"});
    addRouteOperandBinding(plan, "n", slice.runtimeElementCountABI,
                           {"abi", "setvl-avl", "loop-control", "header"});
  } else if (slice.arithmeticKind == RVVSelectedBodyOperationKind::ReduceAdd) {
    plan.planID = kRVVReduceAddOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVGenericBinaryRuntimeABIOrder;
    context = "reduce_add route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "load-base", "reduction-input-call"});
    addRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "load-base", "reduction-accumulator-call"});
    addRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "store-base", "reduction-result-store", "header"});
    addRouteOperandBinding(plan, "n", slice.runtimeElementCountABI,
                           {"abi", "setvl-avl", "loop-control", "header"});
  } else if (slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedAdd) {
    plan.planID = kRVVMaskedAddOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVGenericBinaryRuntimeABIOrder;
    context = "masked_add route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "load-base", "compare-lhs-call", "masked-add-lhs-call",
         "masked-merge-passthrough-call"});
    addRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "load-base", "compare-rhs-call", "masked-add-rhs-call"});
    addRouteOperandBinding(plan, "out", slice.outABI,
                           {"abi", "store-base", "header"});
    addRouteOperandBinding(plan, "n", slice.runtimeElementCountABI,
                           {"abi", "setvl-avl", "loop-control", "header"});
  } else if (slice.arithmeticKind == RVVSelectedBodyOperationKind::StridedAdd) {
    plan.planID = kRVVStridedAddOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVStridedRuntimeABIOrder;
    context = "strided_add route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "lhs-load-base", "binary-lhs-call"});
    addRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "rhs-load-base", "binary-rhs-call"});
    addRouteOperandBinding(plan, "out", slice.outABI,
                           {"abi", "store-base", "header"});
    addRouteOperandBinding(plan, "n", slice.runtimeElementCountABI,
                           {"abi", "setvl-avl", "loop-control", "header"});
    addRouteOperandBinding(
        plan, "lhs_stride", slice.lhsStrideABI,
        {"abi", "lhs-load-stride", "lhs-byte-addr", "header"});
    addRouteOperandBinding(
        plan, "rhs_stride", slice.rhsStrideABI,
        {"abi", "rhs-load-stride", "rhs-byte-addr", "header"});
    addRouteOperandBinding(
        plan, "out_stride", slice.outStrideABI,
        {"abi", "store-stride", "out-byte-addr", "header"});
  } else if (slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd) {
    plan.planID = kRVVMAccOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVMAccRuntimeABIOrder;
    context = "macc_add route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base", "macc-lhs-call"});
    addRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"runtime-abi-mirror", "materialized-load-base", "macc-rhs-call"});
    addRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"runtime-abi-mirror", "materialized-accumulator-load-base",
         "macc-accumulator-call"});
    addRouteOperandBinding(
        plan, "out", slice.outABI,
        {"runtime-abi-mirror", "materialized-store-base",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedLoadUnitStore) {
    plan.planID = kRVVStridedLoadUnitStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVStridedLoadUnitStoreRuntimeABIOrder;
    context = "strided_load_unit_store route";
    addRouteOperandBinding(
        plan, "src", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-strided-load-base",
         "move-source"});
    addRouteOperandBinding(
        plan, "out", slice.outABI,
        {"runtime-abi-mirror", "materialized-store-base",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "stride_bytes", slice.lhsStrideABI,
        {"runtime-abi-mirror", "materialized-strided-load-stride",
         "materialized-byte-address", "header-mirror"});
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitLoadStridedStore) {
    plan.planID = kRVVUnitLoadStridedStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVUnitLoadStridedStoreRuntimeABIOrder;
    context = "unit_load_strided_store route";
    addRouteOperandBinding(
        plan, "src", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base", "move-source"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"runtime-abi-mirror", "materialized-strided-store-base",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "dst_stride_bytes", slice.outStrideABI,
        {"runtime-abi-mirror", "materialized-strided-store-stride",
         "materialized-byte-address", "header-mirror"});
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::RHSScalarBroadcast) {
    plan.planID = kRVVScalarBroadcastOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVScalarBroadcastRuntimeABIOrder;
    context = "scalar_broadcast_add route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base",
         "scalar-broadcast-lhs-call"});
    addRouteOperandBinding(
        plan, "rhs_scalar", slice.rhsABI,
        {"runtime-abi-mirror", "scalar-broadcast-rhs-call"});
    addRouteOperandBinding(
        plan, "out", slice.outABI,
        {"runtime-abi-mirror", "materialized-store-base",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction) {
    plan.planID = kRVVStandaloneReductionOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVStandaloneReductionRuntimeABIOrder;
    context = "standalone_reduce_add route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base",
         "standalone-reduction-input-call"});
    addRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"runtime-abi-mirror", "standalone-initial-accumulator-call"});
    addRouteOperandBinding(
        plan, "out", slice.outABI,
        {"runtime-abi-mirror", "standalone-accumulator-state-load",
         "materialized-store-base", "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
  } else if (slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore) {
    plan.planID = kRVVMaskedUnitStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVMaskedStoreRuntimeABIOrder;
    context = "masked_unit_store route";
    addRouteOperandBinding(
        plan, "src", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base",
         "masked-store-source-call"});
    addRouteOperandBinding(
        plan, "mask", slice.maskABI,
        {"runtime-abi-mirror", "materialized-mask-load-base",
         "masked-store-mask-call"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"runtime-abi-mirror", "materialized-masked-store-base",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::ComputedMaskStridedStore) {
    plan.planID = kRVVComputedMaskStridedStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVComputedMaskStridedStoreRuntimeABIOrder;
    context = "computed_masked_strided_store route";
    addRouteOperandBinding(
        plan, "cmp_lhs", slice.lhsABI,
        {"abi-mirror", "cmp-lhs-load"});
    addRouteOperandBinding(
        plan, "cmp_rhs", slice.rhsABI,
        {"abi-mirror", "cmp-rhs-load"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"abi-mirror", "src-load", "active-source"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"abi-mirror", "old-dst-load", "strided-store", "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi-mirror", "setvl-avl", "loop-control", "header-mirror"});
    addRouteOperandBinding(
        plan, "dst_stride_bytes", slice.outStrideABI,
        {"abi-mirror", "old-dst-stride", "store-stride", "byte-addr",
         "header-mirror"});
  }

  if (plan.planID.empty())
    return plan;

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

llvm::Error recordRVVSelectedBodyOperation(RVVSelectedBodyRouteSlice &slice,
                                           mlir::Operation *op,
                                           RVVSelectedBodyOperationKind kind,
                                           mlir::Value lhs, mlir::Value rhs,
                                           mlir::Value result) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one supported "
        "generic tcrv_rvv.binary op");
  slice.arithmeticOp = op;
  slice.arithmeticKind = kind;
  slice.arithmeticLhs = lhs;
  slice.arithmeticRhs = rhs;
  slice.arithmeticResult = result;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyCompare(RVVSelectedBodyRouteSlice &slice,
                                         tcrv::rvv::CompareOp compare) {
  if (slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one generic "
        "tcrv_rvv.compare op for mask-producing routes");
  if (compare.getKind() != "eq" && compare.getKind() != "slt")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.compare kind '") +
        compare.getKind() + "' for bounded RVV EmitC route");
  slice.compareOp = compare;
  slice.compareLhs = compare.getLhs();
  slice.compareRhs = compare.getRhs();
  slice.compareMask = compare.getMask();
  return llvm::Error::success();
}

llvm::Error
recordRVVSelectedBodyMaskedBinary(RVVSelectedBodyRouteSlice &slice,
                                  tcrv::rvv::MaskedBinaryOp maskedBinary) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (maskedBinary.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_binary kind '") +
        maskedBinary.getKind() + "' for bounded RVV masked route");
  slice.maskedBinaryOp = maskedBinary;
  slice.arithmeticOp = maskedBinary.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::MaskedAdd;
  slice.compareMask = maskedBinary.getMask();
  slice.maskedPassthrough = maskedBinary.getPassthrough();
  slice.arithmeticLhs = maskedBinary.getLhs();
  slice.arithmeticRhs = maskedBinary.getRhs();
  slice.arithmeticResult = maskedBinary.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodySelect(RVVSelectedBodyRouteSlice &slice,
                                        tcrv::rvv::SelectOp select) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  slice.selectOp = select;
  slice.arithmeticOp = select.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::CmpSelect;
  slice.arithmeticLhs = select.getTrueValue();
  slice.arithmeticRhs = select.getFalseValue();
  slice.arithmeticResult = select.getSelected();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyReduction(RVVSelectedBodyRouteSlice &slice,
                                           tcrv::rvv::ReduceOp reduce) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (reduce.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.reduce kind '") +
        reduce.getKind() + "' for bounded RVV reduction route");
  std::optional<llvm::StringRef> accumulatorLayout =
      reduce.getAccumulatorLayout();
  if (!accumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV reduction route requires tcrv_rvv.reduce to carry "
        "accumulator_layout 'rhs-vector-seed-lane0-per-vl-chunk'");
  if (*accumulatorLayout != kRVVReductionAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.reduce accumulator_layout '") +
        *accumulatorLayout + "' for bounded RVV reduction route");
  std::optional<llvm::StringRef> resultLayout = reduce.getResultLayout();
  if (!resultLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV reduction route requires tcrv_rvv.reduce to carry "
        "result_layout 'store-reduction-lane0-to-output-chunk-base'");
  if (*resultLayout != kRVVReductionResultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.reduce result_layout '") +
        *resultLayout + "' for bounded RVV reduction route");
  slice.reduceOp = reduce;
  slice.arithmeticOp = reduce.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::ReduceAdd;
  slice.arithmeticLhs = reduce.getInput();
  slice.arithmeticRhs = reduce.getAccumulator();
  slice.arithmeticResult = reduce.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyStandaloneReduction(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::StandaloneReduceOp standaloneReduce) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (standaloneReduce.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.standalone_reduce kind '") +
        standaloneReduce.getKind() +
        "' for bounded RVV standalone reduction route");
  if (standaloneReduce.getAccumulatorLayout() !=
      kRVVStandaloneReductionAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.standalone_reduce "
                    "accumulator_layout '") +
        standaloneReduce.getAccumulatorLayout() +
        "' for bounded RVV standalone reduction route");
  if (standaloneReduce.getResultLayout() !=
      kRVVStandaloneReductionResultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.standalone_reduce "
                    "result_layout '") +
        standaloneReduce.getResultLayout() +
        "' for bounded RVV standalone reduction route");
  slice.standaloneReduceOp = standaloneReduce;
  slice.arithmeticOp = standaloneReduce.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::StandaloneReduceAdd;
  slice.arithmeticLhs = standaloneReduce.getInput();
  slice.arithmeticAccumulator = standaloneReduce.getAccumulatorSeed();
  slice.accumulatorBuffer = standaloneReduce.getAccumulatorSeed();
  slice.arithmeticResult = standaloneReduce.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMAcc(RVVSelectedBodyRouteSlice &slice,
                                      tcrv::rvv::MAccOp macc) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (macc.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.macc kind '") +
        macc.getKind() + "' for bounded RVV multiply-accumulate route");
  std::optional<llvm::StringRef> accumulatorLayout =
      macc.getAccumulatorLayout();
  if (!accumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV multiply-accumulate route requires tcrv_rvv.macc to "
        "carry accumulator_layout 'separate-i32-vector-accumulator-input'");
  if (*accumulatorLayout != kRVVMAccAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(
            "unsupported generic tcrv_rvv.macc accumulator_layout '") +
        *accumulatorLayout + "' for bounded RVV multiply-accumulate route");
  std::optional<llvm::StringRef> resultLayout = macc.getResultLayout();
  if (!resultLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV multiply-accumulate route requires tcrv_rvv.macc to "
        "carry result_layout "
        "'store-multiply-accumulate-result-to-output-buffer'");
  if (*resultLayout != kRVVMAccResultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.macc result_layout '") +
        *resultLayout + "' for bounded RVV multiply-accumulate route");
  slice.maccOp = macc;
  slice.arithmeticOp = macc.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::MAccAdd;
  slice.arithmeticLhs = macc.getLhs();
  slice.arithmeticRhs = macc.getRhs();
  slice.arithmeticAccumulator = macc.getAccumulator();
  slice.arithmeticResult = macc.getResult();
  return llvm::Error::success();
}

llvm::Error
recordRVVSelectedBodyWideningMAcc(RVVSelectedBodyRouteSlice &slice,
                                  tcrv::rvv::WideningMAccOp macc) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (macc.getKind() != "signed_widening_macc_add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_macc kind '") +
        macc.getKind() +
        "' for bounded RVV widening multiply-accumulate route");
  if (macc.getAccumulatorLayout() != kRVVWideningMAccAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_macc "
                    "accumulator_layout '") +
        macc.getAccumulatorLayout() +
        "' for bounded RVV widening multiply-accumulate route");
  if (macc.getResultLayout() != kRVVWideningMAccResultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_macc "
                    "result_layout '") +
        macc.getResultLayout() +
        "' for bounded RVV widening multiply-accumulate route");
  if (macc.getMaccRelation() != kRVVWideningMAccRelation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_macc "
                    "macc_relation '") +
        macc.getMaccRelation() +
        "' for bounded RVV widening multiply-accumulate route");
  slice.wideningMAccOp = macc;
  slice.arithmeticOp = macc.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::WideningMAccAdd;
  slice.arithmeticLhs = macc.getLhs();
  slice.arithmeticRhs = macc.getRhs();
  slice.arithmeticAccumulator = macc.getAccumulator();
  slice.arithmeticResult = macc.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyWideningDotReduce(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::WideningDotReduceOp dotReduce) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (dotReduce.getKind() != "signed_widening_dot_reduce_add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(
            "unsupported generic tcrv_rvv.widening_dot_reduce kind '") +
        dotReduce.getKind() +
        "' for bounded RVV widening dot-product reduction route");
  if (dotReduce.getAccumulatorLayout() !=
      kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_dot_reduce "
                    "accumulator_layout '") +
        dotReduce.getAccumulatorLayout() +
        "' for bounded RVV widening dot-product reduction route");
  if (dotReduce.getResultLayout() != kRVVWideningDotProductResultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_dot_reduce "
                    "result_layout '") +
        dotReduce.getResultLayout() +
        "' for bounded RVV widening dot-product reduction route");
  if (dotReduce.getDotProductRelation() !=
      kRVVWideningDotProductRelation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_dot_reduce "
                    "dot_product_relation '") +
        dotReduce.getDotProductRelation() +
        "' for bounded RVV widening dot-product reduction route");
  slice.wideningDotReduceOp = dotReduce;
  slice.arithmeticOp = dotReduce.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::WideningDotReduceAdd;
  slice.arithmeticLhs = dotReduce.getLhs();
  slice.arithmeticRhs = dotReduce.getRhs();
  slice.arithmeticAccumulator = dotReduce.getAccumulatorSeed();
  slice.accumulatorBuffer = dotReduce.getAccumulatorSeed();
  slice.arithmeticResult = dotReduce.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedWideningDotReduce(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::MaskedWideningDotReduceOp dotReduce) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (dotReduce.getKind() != "signed_masked_widening_dot_reduce_add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(
            "unsupported generic tcrv_rvv.masked_widening_dot_reduce kind '") +
        dotReduce.getKind() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getMaskRole() != kRVVMaskedPredicateMaskRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce mask_role '") +
        dotReduce.getMaskRole() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getMaskSource() != kRVVMaskedCompareMaskSource)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce mask_source '") +
        dotReduce.getMaskSource() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getMaskMemoryForm() != kRVVComputedMaskMemoryMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce mask_memory_form '") +
        dotReduce.getMaskMemoryForm() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getAccumulatorLayout() !=
      kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce "
                    "accumulator_layout '") +
        dotReduce.getAccumulatorLayout() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getResultLayout() != kRVVWideningDotProductResultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce result_layout '") +
        dotReduce.getResultLayout() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getDotProductRelation() !=
      kRVVWideningDotProductRelation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce "
                    "dot_product_relation '") +
        dotReduce.getDotProductRelation() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  slice.maskedWideningDotReduceOp = dotReduce;
  slice.arithmeticOp = dotReduce.getOperation();
  slice.arithmeticKind =
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
  slice.compareMask = dotReduce.getMask();
  slice.maskValue = dotReduce.getMask();
  slice.arithmeticLhs = dotReduce.getLhs();
  slice.arithmeticRhs = dotReduce.getRhs();
  slice.arithmeticAccumulator = dotReduce.getAccumulatorSeed();
  slice.accumulatorBuffer = dotReduce.getAccumulatorSeed();
  slice.arithmeticResult = dotReduce.getResult();
  slice.memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideWideningDotReduce;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyWideningConvert(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::WideningConvertOp conversion) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (conversion.getKind() != "widen_i32_to_i64" &&
      conversion.getKind() != "sign_extend_widen_vf2")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_convert kind '") +
        conversion.getKind() + "' for bounded RVV widening conversion route");
  slice.wideningConvertOp = conversion;
  slice.arithmeticOp = conversion.getOperation();
  slice.arithmeticKind =
      conversion.getKind() == "sign_extend_widen_vf2"
          ? RVVSelectedBodyOperationKind::WidenI16ToI32
          : RVVSelectedBodyOperationKind::WidenI32ToI64;
  slice.conversionSource = conversion.getSource();
  slice.arithmeticLhs = conversion.getSource();
  slice.arithmeticResult = conversion.getResult();
  slice.memoryForm = RVVSelectedBodyMemoryForm::UnitStrideConversion;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMove(RVVSelectedBodyRouteSlice &slice,
                                      tcrv::rvv::MoveOp move) {
  if (auto segment2Load =
          move.getSource().getDefiningOp<tcrv::rvv::Segment2LoadOp>()) {
    if (move.getKind() != "copy")
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("unsupported generic tcrv_rvv.move kind '") +
          move.getKind() +
          "' for bounded RVV segment2 deinterleave route");
    if (move.getSource() == segment2Load.getField0()) {
      if (slice.field0MoveOperation)
        return makeRVVEmitCRouteProviderError(
            "bounded RVV segment2 deinterleave route requires a unique "
            "field0 tcrv_rvv.move");
      slice.field0MoveOp = move;
      slice.field0MoveOperation = move.getOperation();
      slice.field0LoadedValue = move.getSource();
      slice.field0Value = move.getResult();
    } else if (move.getSource() == segment2Load.getField1()) {
      if (slice.field1MoveOperation)
        return makeRVVEmitCRouteProviderError(
            "bounded RVV segment2 deinterleave route requires a unique "
            "field1 tcrv_rvv.move");
      slice.field1MoveOp = move;
      slice.field1MoveOperation = move.getOperation();
      slice.field1LoadedValue = move.getSource();
      slice.field1Value = move.getResult();
    } else {
      return makeRVVEmitCRouteProviderError(
          "bounded RVV segment2 deinterleave route requires tcrv_rvv.move to "
          "consume field0 or field1 from tcrv_rvv.segment2_load");
    }
    if (!slice.arithmeticOp)
      slice.arithmeticOp = move.getOperation();
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore;
    slice.arithmeticLhs = slice.field0LoadedValue;
    slice.arithmeticResult = slice.field0Value;
    return llvm::Error::success();
  }

  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement op");
  if (move.getKind() != "copy")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.move kind '") +
        move.getKind() + "' for bounded RVV strided memory movement route");
  slice.moveOp = move;
  slice.arithmeticOp = move.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  slice.arithmeticLhs = move.getSource();
  slice.arithmeticResult = move.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedMove(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedMoveOp move) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement op");
  if (move.getKind() != "active-source-preserve-old-destination")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_move kind '") +
        move.getKind() + "' for bounded RVV masked memory route");
  slice.maskedMoveOp = move;
  slice.arithmeticOp = move.getOperation();
  slice.arithmeticKind = move.getMask().getDefiningOp<tcrv::rvv::CompareOp>()
                             ? RVVSelectedBodyOperationKind::
                                   ComputedMaskUnitLoadStore
                             : RVVSelectedBodyOperationKind::
                                   MaskedUnitLoadStore;
  slice.maskValue = move.getMask();
  slice.maskedActiveValue = move.getActiveValue();
  slice.maskedInactivePassthrough = move.getInactivePassthrough();
  slice.arithmeticLhs = move.getActiveValue();
  slice.arithmeticRhs = move.getInactivePassthrough();
  slice.arithmeticResult = move.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedStore(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedStoreOp store) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement/store op");
  if (store.getMemoryForm() != kRVVMaskedStoreDestinationMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_store memory_form '") +
        store.getMemoryForm() + "' for bounded RVV masked store route");
  if (store.getInactiveLanePolicy() !=
      "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_store "
                    "inactive_lane_policy '") +
        store.getInactiveLanePolicy() +
        "' for bounded RVV masked store route");
  slice.maskedStore = store;
  slice.maskedStoreOperation = store.getOperation();
  slice.storeOperation = store.getOperation();
  slice.arithmeticOp = store.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::MaskedUnitStore;
  slice.maskValue = store.getMask();
  slice.maskedActiveValue = store.getValue();
  slice.arithmeticLhs = store.getValue();
  slice.arithmeticResult = store.getValue();
  slice.outBuffer = store.getBuffer();
  slice.storeValue = store.getValue();
  slice.memoryForm = RVVSelectedBodyMemoryForm::MaskedUnitStore;
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyOperationKind>
parseRVVSelectedBodyBinaryKind(llvm::StringRef kind) {
  if (kind == "add")
    return RVVSelectedBodyOperationKind::Add;
  if (kind == "sub")
    return RVVSelectedBodyOperationKind::Sub;
  if (kind == "mul")
    return RVVSelectedBodyOperationKind::Mul;
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported generic tcrv_rvv.binary kind '") + kind +
      "' for bounded RVV EmitC route");
}

llvm::Expected<RVVSelectedBodyRouteSlice>
collectRVVSelectedBodyRouteSlice(tcrv::exec::VariantOp variant) {
  llvm::SmallVector<tcrv::rvv::SetVLOp, 2> setvls;
  llvm::SmallVector<tcrv::rvv::WithVLOp, 2> withVLs;
  unsigned rvvOpCount = 0;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    ++rvvOpCount;
    if (auto setvl = llvm::dyn_cast<tcrv::rvv::SetVLOp>(op))
      setvls.push_back(setvl);
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      withVLs.push_back(withVL);
  });

  if (setvls.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.setvl op");
  if (withVLs.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.with_vl op");

  RVVSelectedBodyRouteSlice slice;
  slice.setvl = setvls.front();
  slice.withVL = withVLs.front();

  tcrv::rvv::RVVConfigContractDiagnostic configDiagnostic =
      tcrv::rvv::validateRVVSelectedBodyConfigVLStructure(slice.setvl,
                                                          slice.withVL);
  if (!configDiagnostic.ok)
    return makeRVVEmitCRouteProviderError(configDiagnostic.message);

  llvm::Expected<support::RuntimeABIParameter> runtimeElementCountABI =
      getRuntimeABIParameterBindingFromValue(
          slice.setvl.getAvl(), "tcrv_rvv.setvl AVL operand",
          {support::RuntimeABIParameterRole::RuntimeElementCount});
  if (!runtimeElementCountABI)
    return runtimeElementCountABI.takeError();

  llvm::SmallVector<tcrv::rvv::LoadOp, 2> genericLoads;
  llvm::SmallVector<tcrv::rvv::StridedLoadOp, 2> genericStridedLoads;
  llvm::SmallVector<tcrv::rvv::IndexLoadOp, 1> genericIndexLoads;
  llvm::SmallVector<tcrv::rvv::IndexedLoadOp, 1> genericIndexedLoads;
  llvm::SmallVector<tcrv::rvv::IndexedStoreOp, 1> genericIndexedStores;
  llvm::SmallVector<tcrv::rvv::MaskLoadOp, 1> genericMaskLoads;
  llvm::SmallVector<tcrv::rvv::Segment2LoadOp, 1> genericSegment2Loads;
  llvm::SmallVector<tcrv::rvv::Segment2StoreOp, 1> genericSegment2Stores;
  llvm::SmallVector<tcrv::rvv::BroadcastLoadOp, 1> genericBroadcastLoads;
  llvm::SmallVector<tcrv::rvv::SplatOp, 1> genericScalarSplats;
  llvm::SmallVector<tcrv::rvv::StoreOp, 2> genericStores;
  unsigned storeCount = 0;
  unsigned stridedStoreCount = 0;
  for (mlir::Operation &op : slice.withVL.getBody().front()) {
    if (auto load = llvm::dyn_cast<tcrv::rvv::LoadOp>(op)) {
      genericLoads.push_back(load);
      continue;
    }
    if (auto stridedLoad = llvm::dyn_cast<tcrv::rvv::StridedLoadOp>(op)) {
      genericStridedLoads.push_back(stridedLoad);
      continue;
    }
    if (auto indexLoad = llvm::dyn_cast<tcrv::rvv::IndexLoadOp>(op)) {
      genericIndexLoads.push_back(indexLoad);
      continue;
    }
    if (auto indexedLoad = llvm::dyn_cast<tcrv::rvv::IndexedLoadOp>(op)) {
      genericIndexedLoads.push_back(indexedLoad);
      continue;
    }
    if (auto indexedStore = llvm::dyn_cast<tcrv::rvv::IndexedStoreOp>(op)) {
      genericIndexedStores.push_back(indexedStore);
      continue;
    }
    if (auto maskLoad = llvm::dyn_cast<tcrv::rvv::MaskLoadOp>(op)) {
      genericMaskLoads.push_back(maskLoad);
      continue;
    }
    if (auto segment2Load = llvm::dyn_cast<tcrv::rvv::Segment2LoadOp>(op)) {
      genericSegment2Loads.push_back(segment2Load);
      continue;
    }
    if (auto segment2Store = llvm::dyn_cast<tcrv::rvv::Segment2StoreOp>(op)) {
      genericSegment2Stores.push_back(segment2Store);
      continue;
    }
    if (auto broadcast = llvm::dyn_cast<tcrv::rvv::BroadcastLoadOp>(op)) {
      genericBroadcastLoads.push_back(broadcast);
      continue;
    }
    if (auto splat = llvm::dyn_cast<tcrv::rvv::SplatOp>(op)) {
      genericScalarSplats.push_back(splat);
      continue;
    }
    if (auto binary = llvm::dyn_cast<tcrv::rvv::BinaryOp>(op)) {
      llvm::Expected<RVVSelectedBodyOperationKind> kind =
          parseRVVSelectedBodyBinaryKind(binary.getKind());
      if (!kind)
        return kind.takeError();
      if (llvm::Error error = recordRVVSelectedBodyOperation(
              slice, binary.getOperation(), *kind, binary.getLhs(),
              binary.getRhs(), binary.getResult()))
        return std::move(error);
      continue;
    }
    if (auto compare = llvm::dyn_cast<tcrv::rvv::CompareOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyCompare(slice, compare))
        return std::move(error);
      continue;
    }
    if (auto maskedBinary = llvm::dyn_cast<tcrv::rvv::MaskedBinaryOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedBinary(slice, maskedBinary))
        return std::move(error);
      continue;
    }
    if (auto select = llvm::dyn_cast<tcrv::rvv::SelectOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodySelect(slice, select))
        return std::move(error);
      continue;
    }
    if (auto reduce = llvm::dyn_cast<tcrv::rvv::ReduceOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyReduction(slice, reduce))
        return std::move(error);
      continue;
    }
    if (auto standaloneReduce =
            llvm::dyn_cast<tcrv::rvv::StandaloneReduceOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyStandaloneReduction(slice,
                                                       standaloneReduce))
        return std::move(error);
      continue;
    }
    if (auto macc = llvm::dyn_cast<tcrv::rvv::MAccOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyMAcc(slice, macc))
        return std::move(error);
      continue;
    }
    if (auto wideningMAcc =
            llvm::dyn_cast<tcrv::rvv::WideningMAccOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyWideningMAcc(slice, wideningMAcc))
        return std::move(error);
      continue;
    }
    if (auto dotReduce =
            llvm::dyn_cast<tcrv::rvv::WideningDotReduceOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyWideningDotReduce(slice, dotReduce))
        return std::move(error);
      continue;
    }
    if (auto maskedDotReduce =
            llvm::dyn_cast<tcrv::rvv::MaskedWideningDotReduceOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyMaskedWideningDotReduce(
              slice, maskedDotReduce))
        return std::move(error);
      continue;
    }
    if (auto conversion = llvm::dyn_cast<tcrv::rvv::WideningConvertOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyWideningConvert(slice, conversion))
        return std::move(error);
      continue;
    }
    if (auto move = llvm::dyn_cast<tcrv::rvv::MoveOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyMove(slice, move))
        return std::move(error);
      continue;
    }
    if (auto maskedMove = llvm::dyn_cast<tcrv::rvv::MaskedMoveOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedMove(slice, maskedMove))
        return std::move(error);
      continue;
    }
    if (auto maskedStore = llvm::dyn_cast<tcrv::rvv::MaskedStoreOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedStore(slice, maskedStore))
        return std::move(error);
      continue;
    }
    if (auto store = llvm::dyn_cast<tcrv::rvv::StoreOp>(op)) {
      slice.genericStore = store;
      genericStores.push_back(store);
      ++storeCount;
      continue;
    }
    if (auto stridedStore = llvm::dyn_cast<tcrv::rvv::StridedStoreOp>(op)) {
      slice.stridedStore = stridedStore;
      ++stridedStoreCount;
      continue;
    }
    if (op.getName().getStringRef().starts_with("tcrv_rvv.i32_"))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("legacy selected-body op '") +
          op.getName().getStringRef() +
          "' is fail-closed during RVV Stage1; Stage2 routes must use generic "
          "tcrv_rvv.load, tcrv_rvv.broadcast_load, "
          "tcrv_rvv.splat, tcrv_rvv.strided_load, tcrv_rvv.binary, "
          "tcrv_rvv.index_load, tcrv_rvv.indexed_load, tcrv_rvv.segment2_load, "
          "tcrv_rvv.segment2_store, "
          "tcrv_rvv.indexed_store, tcrv_rvv.mask_load, tcrv_rvv.compare, "
          "tcrv_rvv.masked_binary, tcrv_rvv.select, tcrv_rvv.reduce, "
          "tcrv_rvv.standalone_reduce, "
          "tcrv_rvv.macc, tcrv_rvv.widening_convert, tcrv_rvv.move, "
          "tcrv_rvv.widening_dot_reduce, "
          "tcrv_rvv.masked_widening_dot_reduce, "
          "tcrv_rvv.masked_move, tcrv_rvv.masked_store, tcrv_rvv.store, "
          "and tcrv_rvv.strided_store body structure");
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded RVV EmitC route does not support op '") +
        op.getName().getStringRef() +
        "' inside tcrv_rvv.with_vl; expected generic load, broadcast_load, "
        "splat, strided_load, index_load, indexed_load, indexed_store, "
        "mask_load, segment2_load, segment2_store, binary, compare, "
        "masked_binary, select, reduce, standalone_reduce, macc, "
        "widening_convert, widening_dot_reduce, "
        "masked_widening_dot_reduce, move, masked_move, store, and "
        "strided_store only");
  }

  const bool hasIndexedMemory =
      !genericIndexLoads.empty() || !genericIndexedLoads.empty() ||
      !genericIndexedStores.empty();
  const bool hasMaskedMemory =
      !genericMaskLoads.empty() || static_cast<bool>(slice.maskedMoveOp) ||
      static_cast<bool>(slice.maskedStore);
  const bool hasSegmentedMemory = !genericSegment2Loads.empty() ||
                                  !genericSegment2Stores.empty() ||
                                  static_cast<bool>(slice.field0MoveOp) ||
                                  static_cast<bool>(slice.field1MoveOp);
  if (hasIndexedMemory) {
    if (!slice.moveOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed memory route requires exactly one "
          "tcrv_rvv.move {kind = \"copy\"} movement op");
    slice.arithmeticKind =
        genericIndexedStores.empty()
            ? RVVSelectedBodyOperationKind::IndexedGatherUnitStore
            : RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
  }
  if (!genericSegment2Stores.empty()) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
    slice.arithmeticOp = genericSegment2Stores.front().getOperation();
  }
  const bool hasStridedMemory =
      !genericStridedLoads.empty() || static_cast<bool>(slice.stridedStore);
  if (hasStridedMemory && genericStridedLoads.empty() &&
      stridedStoreCount == 1 && slice.moveOp)
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::UnitLoadStridedStore;
  if (hasStridedMemory && genericStridedLoads.size() == 1 &&
      stridedStoreCount == 1 && slice.maskedMoveOp && slice.compareOp &&
      genericLoads.size() == 3)
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
  if (hasStridedMemory && genericStridedLoads.size() == 2 &&
      stridedStoreCount == 0 && storeCount == 1 &&
      slice.wideningDotReduceOp && genericLoads.empty()) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce;
  }
  if (hasStridedMemory && genericStridedLoads.size() == 2 &&
      stridedStoreCount == 0 && storeCount == 1 &&
      slice.maskedWideningDotReduceOp && slice.compareOp &&
      genericLoads.size() == 2) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::
            ComputedMaskStridedInputWideningDotReduceAdd;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::
            ComputedMaskStridedInputWideningDotReduce;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      slice.selectOp && slice.compareOp && genericLoads.size() == 4 &&
      storeCount == 1) {
    slice.arithmeticKind = RVVSelectedBodyOperationKind::ComputedMaskSelect;
    slice.memoryForm = RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect;
  }

  const bool isCompareSelect =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::CmpSelect;
  const bool isComputedMaskSelect =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::ComputedMaskSelect;
  const bool isReduction =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::ReduceAdd;
  const bool isStandaloneReduction =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::StandaloneReduceAdd;
  const bool isMaskedAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedAdd;
  const bool isMAccAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd;
  const bool isWideningMAccAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isWideningDotReduceAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningDotReduceAdd;
  const bool isStridedInputWideningDotReduceAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd;
  const bool isComputedMaskWideningDotReduceAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
  const bool isComputedMaskStridedInputWideningDotReduceAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::
              ComputedMaskStridedInputWideningDotReduceAdd;
  const bool isStridedLoadUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  const bool isUnitLoadStridedStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::UnitLoadStridedStore;
  const bool isIndexedGatherUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::IndexedGatherUnitStore;
  const bool isIndexedScatterUnitLoad =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
  const bool isMaskedUnitLoadStore =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  const bool isMaskedUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedUnitStore;
  const bool isComputedMaskUnitLoadStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore;
  const bool isComputedMaskStridedStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
  const bool isSegment2DeinterleaveUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore;
  const bool isSegment2InterleaveUnitLoad =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
  const bool isWideningConversion =
      slice.arithmeticOp &&
      (slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI32ToI64 ||
       slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI16ToI32);
  const bool hasScalarBroadcast = !genericScalarSplats.empty();
  if (hasIndexedMemory && isIndexedGatherUnitStore &&
      (!genericStridedLoads.empty() || stridedStoreCount != 0 ||
       !genericLoads.empty() || !genericBroadcastLoads.empty() ||
       hasScalarBroadcast))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route cannot mix indexed memory "
        "ops with unit-stride input loads, broadcast, scalar splat, strided "
        "loads, or strided stores");
  if (hasIndexedMemory && isIndexedScatterUnitLoad &&
      (!genericStridedLoads.empty() || stridedStoreCount != 0 ||
       !genericBroadcastLoads.empty() || hasScalarBroadcast || storeCount != 0))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route cannot mix indexed store "
        "memory ops with broadcast, scalar splat, unit-stride stores, "
        "strided loads, or strided stores");
  if (genericScalarSplats.size() > 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires at most one "
        "tcrv_rvv.splat op");
  if (genericBroadcastLoads.size() > 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires at most one "
        "tcrv_rvv.broadcast_load op");
  const bool hasRHSBroadcastLike =
      !genericBroadcastLoads.empty() || hasScalarBroadcast;
  if (!genericBroadcastLoads.empty() && hasScalarBroadcast)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route cannot mix RHS buffer broadcast and "
        "RHS scalar splat in one slice");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isUnitLoadStridedStore &&
      !isComputedMaskStridedStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      (!genericBroadcastLoads.empty() || hasScalarBroadcast ||
       !genericLoads.empty() || slice.genericStore))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route cannot mix strided memory ops with "
        "unit-stride load/store, broadcast, or scalar-splat memory forms");
  if (hasStridedMemory && isUnitLoadStridedStore &&
      (!genericBroadcastLoads.empty() || hasScalarBroadcast ||
       !genericStridedLoads.empty() || slice.genericStore))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV unit-load to strided-store route cannot mix "
        "unit-stride source memory with strided source loads, unit-stride "
        "stores, broadcast, or scalar-splat memory forms");
  if (hasStridedMemory && isStridedLoadUnitStore &&
      (!genericBroadcastLoads.empty() || hasScalarBroadcast ||
       !genericLoads.empty()))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route cannot "
        "mix strided source memory with unit-stride input loads, broadcast, or "
        "scalar-splat memory forms");
  if (hasStridedMemory &&
      (isMAccAdd || isWideningMAccAdd || isWideningDotReduceAdd ||
       isComputedMaskWideningDotReduceAdd))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route supports only add in this slice, "
        "not multiply-accumulate or dot-product reduction");
  if (hasStridedMemory && isWideningConversion)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion route requires unit-stride "
        "source load and destination store");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isUnitLoadStridedStore &&
      !isComputedMaskStridedStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      (!slice.arithmeticOp ||
                           slice.arithmeticKind !=
                               RVVSelectedBodyOperationKind::Add))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route requires exactly one "
        "tcrv_rvv.binary {kind = \"add\"} compute op");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isUnitLoadStridedStore &&
      !isComputedMaskStridedStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      genericStridedLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route requires exactly two "
        "tcrv_rvv.strided_load ops for lhs and rhs");
  if (hasStridedMemory && isUnitLoadStridedStore &&
      genericStridedLoads.size() != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV unit-load to strided-store route must use "
        "unit-stride tcrv_rvv.load source, not tcrv_rvv.strided_load");
  if (hasStridedMemory && isUnitLoadStridedStore && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV unit-load to strided-store route requires "
        "exactly one unit-stride tcrv_rvv.load source op");
  if (hasStridedMemory && isStridedLoadUnitStore &&
      genericStridedLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route requires "
        "exactly one tcrv_rvv.strided_load source op");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isUnitLoadStridedStore && !isComputedMaskStridedStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      stridedStoreCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route requires exactly one "
        "tcrv_rvv.strided_store op");
  if (hasStridedMemory && isUnitLoadStridedStore && stridedStoreCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV unit-load to strided-store route requires "
        "exactly one tcrv_rvv.strided_store op");
  if (hasStridedMemory && isStridedLoadUnitStore && stridedStoreCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route must use "
        "unit-stride tcrv_rvv.store, not tcrv_rvv.strided_store");
  if (hasStridedMemory && isStridedLoadUnitStore && storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route requires "
        "exactly one unit-stride tcrv_rvv.store op");
  if ((isStridedInputWideningDotReduceAdd ||
       isComputedMaskStridedInputWideningDotReduceAdd) &&
      stridedStoreCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-input widening dot-product reduction "
        "route must use unit-stride scalar tcrv_rvv.store, not "
        "tcrv_rvv.strided_store");
  if ((isStridedInputWideningDotReduceAdd ||
       isComputedMaskStridedInputWideningDotReduceAdd) &&
      storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-input widening dot-product reduction "
        "route requires exactly one unit-stride scalar tcrv_rvv.store op");
  if (hasStridedMemory && isUnitLoadStridedStore && storeCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV unit-load to strided-store route must use "
        "tcrv_rvv.strided_store, not unit-stride tcrv_rvv.store");
  if (hasIndexedMemory && genericIndexLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed memory route requires exactly one "
        "tcrv_rvv.index_load op");
  if (isIndexedGatherUnitStore && genericIndexedLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires exactly one "
        "tcrv_rvv.indexed_load op");
  if (isIndexedGatherUnitStore && genericIndexedStores.size() != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route must use unit-stride "
        "tcrv_rvv.store, not tcrv_rvv.indexed_store");
  if (isIndexedGatherUnitStore && storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires exactly one "
        "unit-stride tcrv_rvv.store op");
  if (isIndexedScatterUnitLoad && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route requires exactly one "
        "unit-stride tcrv_rvv.load source op");
  if (isIndexedScatterUnitLoad && genericIndexedLoads.size() != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route must use unit-stride "
        "tcrv_rvv.load source, not tcrv_rvv.indexed_load");
  if (isIndexedScatterUnitLoad && genericIndexedStores.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route requires exactly one "
        "tcrv_rvv.indexed_store op");
  if (hasIndexedMemory &&
      (isMAccAdd || isWideningMAccAdd || isWideningConversion ||
       isWideningDotReduceAdd || isComputedMaskWideningDotReduceAdd ||
       isCompareSelect || isComputedMaskSelect || isMaskedAdd || isReduction ||
       isStandaloneReduction))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed memory route supports only "
        "index_load/indexed_load_or_unit_load/move/store_or_indexed_store "
        "memory movement in this slice");
  if (hasMaskedMemory &&
      ((!isMaskedUnitLoadStore && !isMaskedUnitStore &&
        !isComputedMaskUnitLoadStore &&
        !isComputedMaskStridedStore) ||
       hasIndexedMemory || (hasStridedMemory && !isComputedMaskStridedStore) ||
       !genericBroadcastLoads.empty() ||
       hasScalarBroadcast || hasSegmentedMemory || slice.maskedBinaryOp ||
       slice.selectOp ||
       slice.reduceOp || slice.standaloneReduceOp || slice.maccOp ||
       slice.wideningMAccOp ||
       slice.wideningDotReduceOp ||
       isWideningConversion))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route supports only runtime-mask "
        "unit-stride load/mask_load/old-destination-load/masked_move/store, "
        "runtime-mask unit-stride load/mask_load/masked_store, "
        "or computed-mask compare/source/old-destination/masked_move/store "
        "memory movement in this slice");
  if (hasSegmentedMemory && isSegment2DeinterleaveUnitStore &&
      (hasIndexedMemory ||
       hasStridedMemory || hasMaskedMemory || !genericLoads.empty() ||
       !genericBroadcastLoads.empty() || hasScalarBroadcast ||
       !genericSegment2Stores.empty() ||
       stridedStoreCount != 0 || genericIndexedStores.size() != 0 ||
       slice.maskedBinaryOp || slice.selectOp || slice.reduceOp ||
       slice.standaloneReduceOp || slice.maccOp || slice.wideningMAccOp ||
       slice.wideningDotReduceOp ||
       slice.maskedWideningDotReduceOp || slice.compareOp ||
       isWideningConversion))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route supports only "
        "segment2_load/field0 move/field1 move/field0 store/field1 store "
        "memory movement in this slice");
  if (hasSegmentedMemory && isSegment2InterleaveUnitLoad &&
      (hasIndexedMemory || hasStridedMemory || hasMaskedMemory ||
       !genericSegment2Loads.empty() || !genericBroadcastLoads.empty() ||
       hasScalarBroadcast || stridedStoreCount != 0 ||
       genericIndexedStores.size() != 0 || storeCount != 0 ||
       slice.maskedBinaryOp || slice.selectOp || slice.reduceOp ||
       slice.standaloneReduceOp || slice.maccOp || slice.wideningMAccOp ||
       slice.wideningDotReduceOp ||
       slice.maskedWideningDotReduceOp || slice.compareOp ||
       isWideningConversion))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route supports only "
        "field0 load/field1 load/segment2_store memory movement in this "
        "slice");
  if (hasSegmentedMemory && !isSegment2DeinterleaveUnitStore &&
      !isSegment2InterleaveUnitLoad)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 route requires either deinterleave "
        "segment2_load or interleave segment2_store typed body structure");
  if (isSegment2DeinterleaveUnitStore && genericSegment2Loads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires exactly "
        "one tcrv_rvv.segment2_load op");
  if (isSegment2DeinterleaveUnitStore &&
      (!slice.field0MoveOp || !slice.field1MoveOp))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires exactly one "
        "field0 move and one field1 move from tcrv_rvv.segment2_load");
  if (isSegment2DeinterleaveUnitStore && storeCount != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires exactly two "
        "unit-stride tcrv_rvv.store ops for field0 and field1");
  if (isSegment2InterleaveUnitLoad && genericSegment2Stores.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires exactly one "
        "tcrv_rvv.segment2_store op");
  if (isSegment2InterleaveUnitLoad && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires exactly two "
        "unit-stride tcrv_rvv.load ops for field0 and field1");
  if (isMaskedUnitLoadStore && slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime-mask memory route must not contain "
        "tcrv_rvv.compare; computed masks require "
        "computed_masked_unit_load_store");
  if (isMaskedUnitStore && slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime-mask store route must not contain "
        "tcrv_rvv.compare; this slice requires explicit mask_load authority");
  if (isComputedMaskUnitLoadStore && !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask memory route requires one "
        "tcrv_rvv.compare producer before tcrv_rvv.masked_move");
  if (isComputedMaskStridedStore && !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route requires one "
        "tcrv_rvv.compare producer before tcrv_rvv.masked_move");
  if ((isMaskedUnitLoadStore || isMaskedUnitStore) &&
      genericMaskLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires exactly one "
        "tcrv_rvv.mask_load op");
  if (isComputedMaskUnitLoadStore && !genericMaskLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask memory route must not consume "
        "tcrv_rvv.mask_load; the mask must be produced by tcrv_rvv.compare");
  if (isComputedMaskStridedStore && !genericMaskLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route must not "
        "consume tcrv_rvv.mask_load; the mask must be produced by "
        "tcrv_rvv.compare");
  if (isMaskedUnitLoadStore && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires exactly two "
        "tcrv_rvv.load ops for source and old destination");
  if (isMaskedUnitStore && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked store route requires exactly one "
        "tcrv_rvv.load op for the payload source");
  if (isComputedMaskUnitLoadStore && genericLoads.size() != 4)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask memory route requires exactly four "
        "tcrv_rvv.load ops for compare lhs, compare rhs, active source, and "
        "old destination");
  if (isComputedMaskStridedStore && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route requires "
        "exactly three unit-stride tcrv_rvv.load ops for compare lhs, "
        "compare rhs, and active source");
  if (isComputedMaskStridedStore && genericStridedLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route requires "
        "exactly one tcrv_rvv.strided_load old-destination op");
  if (isMaskedUnitStore && storeCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked store route must use "
        "tcrv_rvv.masked_store, not unit-stride tcrv_rvv.store");
  if (isMaskedUnitStore && !slice.maskedStoreOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked store route requires exactly one "
        "tcrv_rvv.masked_store op");
  if ((isMaskedUnitLoadStore || isComputedMaskUnitLoadStore) &&
      storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires exactly one "
        "unit-stride tcrv_rvv.store op");
  if (isComputedMaskStridedStore && storeCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route must use "
        "tcrv_rvv.strided_store, not unit-stride tcrv_rvv.store");
  if (isComputedMaskStridedStore && stridedStoreCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route requires "
        "exactly one tcrv_rvv.strided_store op");
  if (isMAccAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV multiply-accumulate route requires explicit "
        "vector lhs, rhs, and accumulator loads; broadcast/splat macc is not "
        "in this bounded slice");
  if (isWideningMAccAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening multiply-accumulate route requires "
        "explicit vector lhs, rhs, and accumulator loads; broadcast/splat "
        "macc is not in this bounded slice");
  if (isWideningDotReduceAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening dot-product reduction route requires "
        "explicit lhs/rhs unit-stride vector loads and an accumulator seed "
        "runtime ABI boundary; broadcast/splat dot-reduction is not in this "
        "bounded slice");
  if (isStridedInputWideningDotReduceAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-input widening dot-product reduction "
        "route requires explicit lhs/rhs strided vector loads and an "
        "accumulator seed runtime ABI boundary; broadcast/splat "
        "dot-reduction is not in this bounded slice");
  if (isComputedMaskWideningDotReduceAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask widening dot-product reduction "
        "route requires explicit compare lhs/rhs and dot lhs/rhs unit-stride "
        "loads; broadcast/splat dot-reduction is not in this bounded slice");
  if (isComputedMaskStridedInputWideningDotReduceAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-input widening "
        "dot-product reduction route requires explicit compare lhs/rhs "
        "unit-stride loads and dot lhs/rhs strided loads; broadcast/splat "
        "dot-reduction is not in this bounded slice");
  if (isWideningConversion && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion route does not consume an "
        "RHS broadcast or scalar splat");
  if (isMAccAdd && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV multiply-accumulate route requires exactly three "
        "tcrv_rvv.load ops for lhs, rhs, and accumulator-input-buffer");
  if (isWideningMAccAdd && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening multiply-accumulate route requires "
        "exactly three tcrv_rvv.load ops for lhs, rhs, and "
        "accumulator-input-buffer");
  if (isWideningDotReduceAdd && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening dot-product reduction route requires "
        "exactly two tcrv_rvv.load ops for lhs and rhs; the accumulator seed "
        "is a scalar runtime ABI boundary");
  if (isStandaloneReduction && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV standalone reduction route requires exactly one "
        "tcrv_rvv.load op for lhs; the accumulator seed is a scalar runtime "
        "ABI boundary");
  if (isStridedInputWideningDotReduceAdd && genericLoads.size() != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-input widening dot-product reduction "
        "route must use tcrv_rvv.strided_load for lhs/rhs sources, not "
        "unit-stride tcrv_rvv.load");
  if (isStridedInputWideningDotReduceAdd && genericStridedLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-input widening dot-product reduction "
        "route requires exactly two tcrv_rvv.strided_load ops for lhs and rhs; "
        "the accumulator seed is a scalar runtime ABI boundary");
  if (isComputedMaskWideningDotReduceAdd && genericLoads.size() != 4)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask widening dot-product reduction "
        "route requires exactly four tcrv_rvv.load ops for compare lhs, "
        "compare rhs, dot lhs, and dot rhs; the accumulator seed is a scalar "
        "runtime ABI boundary");
  if (isComputedMaskStridedInputWideningDotReduceAdd &&
      genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-input widening "
        "dot-product reduction route requires exactly two unit-stride "
        "tcrv_rvv.load ops for compare lhs and compare rhs");
  if (isComputedMaskStridedInputWideningDotReduceAdd &&
      genericStridedLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-input widening "
        "dot-product reduction route requires exactly two tcrv_rvv.strided_load "
        "ops for dot lhs and dot rhs; the accumulator seed is a scalar "
        "runtime ABI boundary");
  if (isComputedMaskSelect && genericLoads.size() != 4)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask select route requires exactly four "
        "tcrv_rvv.load ops for compare lhs, compare rhs, true value, and "
        "false value");
  if (hasScalarBroadcast && (!slice.arithmeticOp ||
                             slice.arithmeticKind !=
                                 RVVSelectedBodyOperationKind::Add))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV scalar-broadcast route currently requires "
        "exactly one tcrv_rvv.binary {kind = \"add\"} compute op");
  if (isWideningConversion && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion route requires exactly one "
        "tcrv_rvv.load source op");
  if (isWideningConversion &&
      (slice.compareOp || slice.maskedBinaryOp || slice.selectOp ||
       slice.reduceOp || slice.maccOp || slice.wideningMAccOp ||
       slice.wideningDotReduceOp))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion route cannot mix "
        "compare/select/masked/reduce/macc compute ops");
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      !isMAccAdd &&
      !isWideningMAccAdd &&
      !isWideningDotReduceAdd &&
      !isStandaloneReduction &&
      !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isComputedMaskSelect &&
      !isMaskedUnitLoadStore &&
      !isMaskedUnitStore &&
      !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore &&
      !hasRHSBroadcastLike &&
      !isWideningConversion &&
      genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV vector-load route requires exactly two "
        "tcrv_rvv.load ops");
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      !isMAccAdd &&
      !isWideningMAccAdd &&
      !isWideningDotReduceAdd &&
      !isStandaloneReduction &&
      !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isComputedMaskSelect &&
      !isMaskedUnitLoadStore &&
      !isMaskedUnitStore &&
      !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore &&
      !genericBroadcastLoads.empty() && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV broadcast route requires exactly one "
        "tcrv_rvv.load op and one tcrv_rvv.broadcast_load op");
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      !isMAccAdd &&
      !isWideningMAccAdd &&
      !isWideningDotReduceAdd &&
      !isStandaloneReduction &&
      !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isComputedMaskSelect &&
      !isMaskedUnitLoadStore &&
      !isMaskedUnitStore &&
      !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore &&
      hasScalarBroadcast && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV scalar-broadcast route requires exactly one "
        "tcrv_rvv.load op and one tcrv_rvv.splat op");
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      !isMaskedUnitStore && !slice.genericStore)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one "
        "tcrv_rvv.store op");
  if (!slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one supported "
        "tcrv_rvv.binary, tcrv_rvv.select, tcrv_rvv.reduce, tcrv_rvv.macc, "
        "tcrv_rvv.standalone_reduce, tcrv_rvv.widening_macc, "
        "tcrv_rvv.widening_dot_reduce, "
        "tcrv_rvv.masked_widening_dot_reduce, tcrv_rvv.widening_convert, "
        "tcrv_rvv.move, or "
        "tcrv_rvv.masked_move, or tcrv_rvv.masked_store op");
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      !isMaskedUnitStore && storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one tcrv_rvv.store "
        "op");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd && storeCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route must use tcrv_rvv.strided_store "
        "instead of tcrv_rvv.store");
  if ((isCompareSelect || isComputedMaskSelect || isMaskedAdd ||
       isComputedMaskUnitLoadStore ||
       isComputedMaskStridedStore || isComputedMaskWideningDotReduceAdd ||
       isComputedMaskStridedInputWideningDotReduceAdd) &&
      !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV mask-consuming route requires one "
        "tcrv_rvv.compare op before the mask-consuming compute op");
  if (!isCompareSelect && !isComputedMaskSelect && !isMaskedAdd &&
      !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore && !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV non-mask route does not support a standalone "
        "tcrv_rvv.compare op");
  if ((isCompareSelect || isComputedMaskSelect) && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV compare/select route requires an explicit RHS "
        "vector load; broadcast/splat compare/select is not in this bounded "
        "slice");
  if (isMaskedAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked add route requires an explicit RHS vector "
        "load; broadcast/splat masked add is not in this bounded slice");
  if (isReduction && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV reduction route requires explicit vector input "
        "and accumulator loads; broadcast/splat reduction is not in this "
        "bounded slice");
  if (isStandaloneReduction && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV standalone reduction route requires explicit "
        "input load and accumulator-input-buffer scalar seed boundary; "
        "broadcast/splat reduction is not in this bounded slice");
  const unsigned expectedRVVOps =
      isWideningConversion
          ? 8
      : isStandaloneReduction
          ? 9
      : isSegment2DeinterleaveUnitStore
          ? 11
      : isSegment2InterleaveUnitLoad
          ? 9
      : isWideningDotReduceAdd
          ? 11
      : isStridedInputWideningDotReduceAdd
          ? 13
      : isComputedMaskWideningDotReduceAdd
          ? 16
      : isComputedMaskStridedInputWideningDotReduceAdd
          ? 18
      : isMAccAdd
          ? 12
      : isWideningMAccAdd
          ? 12
      : isComputedMaskUnitLoadStore
          ? 14
      : isComputedMaskStridedStore
          ? 15
      : isComputedMaskSelect
          ? 15
      : isMaskedUnitStore
          ? 9
      : isMaskedUnitLoadStore
          ? 11
          : (isIndexedGatherUnitStore || isIndexedScatterUnitLoad
                 ? 10
                 : (isStridedLoadUnitStore || isUnitLoadStridedStore
                        ? 9
          : (hasStridedMemory
                 ? 13
                 : ((isCompareSelect || isMaskedAdd) ? 11 : 10))));
  if (rvvOpCount != expectedRVVOps)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route supports only runtime_abi_value/"
        "runtime_abi_value/runtime_abi_value plus optional runtime_abi_value "
        "and optional strided runtime_abi_value/runtime_abi_value/"
        "runtime_abi_value, "
        "setvl/with_vl, and generic load/broadcast_load/splat/strided_load/"
        "index_load/indexed_load/mask_load/segment2_load/segment2_store/"
        "binary/compare/select/masked_binary/reduce/macc/"
        "standalone_reduce/widening_dot_reduce/widening_convert/move/"
        "masked_move/masked_store/store/strided_store body structure");

  for (tcrv::rvv::LoadOp load : genericLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getBuffer(), "tcrv_rvv.load buffer operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer,
             support::RuntimeABIParameterRole::RHSInputBuffer,
             support::RuntimeABIParameterRole::SourceInputBuffer,
             support::RuntimeABIParameterRole::TrueValueInputBuffer,
             support::RuntimeABIParameterRole::FalseValueInputBuffer,
             support::RuntimeABIParameterRole::OutputBuffer,
             support::RuntimeABIParameterRole::AccumulatorInputBuffer,
             support::RuntimeABIParameterRole::DotLHSInputBuffer,
             support::RuntimeABIParameterRole::DotRHSInputBuffer,
             support::RuntimeABIParameterRole::SegmentField0InputBuffer,
             support::RuntimeABIParameterRole::SegmentField1InputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error = assignRVVGenericLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::BroadcastLoadOp broadcast : genericBroadcastLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            broadcast.getBuffer(), "tcrv_rvv.broadcast_load buffer operand",
            {support::RuntimeABIParameterRole::RHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericBroadcastBinding(slice, broadcast, *parameter))
      return error;
  }
  for (tcrv::rvv::SplatOp splat : genericScalarSplats) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            splat.getScalar(), "tcrv_rvv.splat scalar operand",
            {support::RuntimeABIParameterRole::RHSScalarValue});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericScalarSplatBinding(slice, splat, *parameter))
      return error;
  }
  for (tcrv::rvv::StridedLoadOp load : genericStridedLoads) {
    llvm::Expected<support::RuntimeABIParameter> bufferParameter =
        getRuntimeABIParameterBindingFromValue(
            load.getBuffer(), "tcrv_rvv.strided_load buffer operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer,
             support::RuntimeABIParameterRole::RHSInputBuffer,
             support::RuntimeABIParameterRole::DotLHSInputBuffer,
             support::RuntimeABIParameterRole::DotRHSInputBuffer,
             support::RuntimeABIParameterRole::SourceInputBuffer,
             support::RuntimeABIParameterRole::OutputBuffer});
    if (!bufferParameter)
      return bufferParameter.takeError();
    llvm::Expected<support::RuntimeABIParameter> strideParameter =
        getRuntimeABIParameterBindingFromValue(
            load.getStride(), "tcrv_rvv.strided_load stride operand",
            {support::RuntimeABIParameterRole::LHSInputStride,
             support::RuntimeABIParameterRole::RHSInputStride,
             support::RuntimeABIParameterRole::SourceByteStride,
             support::RuntimeABIParameterRole::DestinationByteStride,
             support::RuntimeABIParameterRole::OutputStride});
    if (!strideParameter)
      return strideParameter.takeError();
    if (llvm::Error error = assignRVVGenericStridedLoadBinding(
            slice, load, *bufferParameter, *strideParameter))
      return error;
  }
  for (tcrv::rvv::IndexLoadOp load : genericIndexLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getIndex(), "tcrv_rvv.index_load index operand",
            {support::RuntimeABIParameterRole::IndexInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericIndexLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::IndexedLoadOp load : genericIndexedLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getData(), "tcrv_rvv.indexed_load data operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericIndexedLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::IndexedStoreOp store : genericIndexedStores) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            store.getDestination(), "tcrv_rvv.indexed_store destination operand",
            {support::RuntimeABIParameterRole::OutputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericIndexedStoreBinding(slice, store, *parameter))
      return error;
  }
  for (tcrv::rvv::MaskLoadOp load : genericMaskLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getMask(), "tcrv_rvv.mask_load mask operand",
            {support::RuntimeABIParameterRole::MaskInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericMaskLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::Segment2LoadOp load : genericSegment2Loads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getSource(), "tcrv_rvv.segment2_load source operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericSegment2LoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::Segment2StoreOp store : genericSegment2Stores) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            store.getDestination(),
            "tcrv_rvv.segment2_store destination operand",
            {support::RuntimeABIParameterRole::
                 SegmentInterleavedOutputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericSegment2StoreBinding(slice, store, *parameter))
      return error;
  }
  if (isIndexedGatherUnitStore) {
    if (slice.indexedLoad.getIndices() != slice.indexValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route requires "
          "tcrv_rvv.indexed_load to consume the index vector produced by "
          "tcrv_rvv.index_load");
    if (slice.indexLoad.getVl() != slice.setvl.getVl() ||
        slice.indexedLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route requires index_load and "
          "indexed_load to consume the selected !tcrv_rvv.vl token");
  }
  if (isIndexedScatterUnitLoad) {
    if (slice.indexedStore.getIndices() != slice.indexValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route requires "
          "tcrv_rvv.indexed_store to consume the index vector produced by "
          "tcrv_rvv.index_load");
    if (slice.indexLoad.getVl() != slice.setvl.getVl() ||
        slice.indexedStore.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route requires index_load and "
          "indexed_store to consume the selected !tcrv_rvv.vl token");
  }
  if (isMaskedUnitLoadStore) {
    if (slice.maskedMoveOp.getMask() != slice.maskValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires "
          "tcrv_rvv.masked_move to consume the mask produced by "
          "tcrv_rvv.mask_load");
    if (slice.maskLoad.getVl() != slice.setvl.getVl() ||
        slice.maskedMoveOp.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires mask_load and "
          "masked_move to consume the selected !tcrv_rvv.vl token");
  }
  if (isMaskedUnitStore) {
    if (slice.maskedStore.getMask() != slice.maskValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route requires "
          "tcrv_rvv.masked_store to consume the mask produced by "
          "tcrv_rvv.mask_load");
    if (slice.maskLoad.getVl() != slice.setvl.getVl() ||
        slice.maskedStore.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route requires mask_load and "
          "masked_store to consume the selected !tcrv_rvv.vl token");
  }
  if (isComputedMaskUnitLoadStore) {
    if (slice.maskedMoveOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires "
          "tcrv_rvv.masked_move to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.maskedMoveOp.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires compare "
          "and masked_move to consume the selected !tcrv_rvv.vl token");
  }
  if (isComputedMaskStridedStore) {
    if (slice.maskedMoveOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires "
          "tcrv_rvv.masked_move to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.maskedMoveOp.getVl() != slice.setvl.getVl() ||
        slice.lhsStridedLoad.getVl() != slice.setvl.getVl() ||
        slice.stridedStore.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires "
          "compare, masked_move, and strided_store to consume the selected "
          "!tcrv_rvv.vl token");
  }
  if (isSegment2DeinterleaveUnitStore) {
    if (slice.segment2Load.getVl() != slice.setvl.getVl() ||
        slice.field0MoveOp.getVl() != slice.setvl.getVl() ||
        slice.field1MoveOp.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 deinterleave route requires "
          "segment2_load and both move ops to consume the selected "
          "!tcrv_rvv.vl token");
    if (slice.field0MoveOp.getSource() != slice.segment2Load.getField0() ||
        slice.field1MoveOp.getSource() != slice.segment2Load.getField1())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 deinterleave route requires field0 "
          "and field1 moves to consume matching segment2_load field results");
  }
  if (isSegment2InterleaveUnitLoad) {
    if (slice.lhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.rhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.segment2Store.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 interleave route requires field0 "
          "load, field1 load, and segment2_store to consume the selected "
          "!tcrv_rvv.vl token");
    if (slice.segment2Store.getField0() != slice.field0Value ||
        slice.segment2Store.getField1() != slice.field1Value)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 interleave route requires "
          "segment2_store to consume matching field0 and field1 load results");
  }

  if (!slice.lhsLoadOperation ||
      (!isWideningConversion && !isStridedLoadUnitStore &&
       !isUnitLoadStridedStore &&
       !isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
       !isMaskedUnitLoadStore &&
       !isMaskedUnitStore &&
       !isComputedMaskUnitLoadStore &&
       !isComputedMaskStridedStore &&
       !isSegment2DeinterleaveUnitStore &&
       !isSegment2InterleaveUnitLoad &&
       !isStandaloneReduction &&
       !slice.rhsLoadOperation))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires lhs-input-buffer and "
        "rhs-input-buffer or rhs-scalar-value generic load, broadcast, or "
        "scalar-splat dataflow");
  support::RuntimeABIParameter resolvedOutABI;
  if (isSegment2DeinterleaveUnitStore) {
    for (tcrv::rvv::StoreOp store : genericStores) {
      llvm::Expected<support::RuntimeABIParameter> parameter =
          getRuntimeABIParameterBindingFromValue(
              store.getBuffer(), "tcrv_rvv.segment2 store buffer operand",
              {support::RuntimeABIParameterRole::SegmentField0OutputBuffer,
               support::RuntimeABIParameterRole::SegmentField1OutputBuffer});
      if (!parameter)
        return parameter.takeError();
      if (parameter->role ==
          support::RuntimeABIParameterRole::SegmentField0OutputBuffer) {
        if (slice.field0StoreOperation)
          return makeRVVEmitCRouteProviderError(
              "bounded generic RVV segment2 deinterleave route requires a "
              "unique field0 tcrv_rvv.store");
        slice.field0Store = store;
        slice.field0StoreOperation = store.getOperation();
        slice.field0Buffer = store.getBuffer();
        slice.field0ABI = *parameter;
      } else {
        if (slice.field1StoreOperation)
          return makeRVVEmitCRouteProviderError(
              "bounded generic RVV segment2 deinterleave route requires a "
              "unique field1 tcrv_rvv.store");
        slice.field1Store = store;
        slice.field1StoreOperation = store.getOperation();
        slice.field1Buffer = store.getBuffer();
        slice.field1ABI = *parameter;
      }
    }
    if (!slice.field0StoreOperation || !slice.field1StoreOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 deinterleave route requires field0 "
          "and field1 destination stores with explicit segment field ABI "
          "roles");
    resolvedOutABI = slice.field0ABI;
  } else if (isSegment2InterleaveUnitLoad) {
    if (!slice.segment2StoreOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 interleave route requires "
          "interleaved destination segment2_store with explicit ABI role");
    resolvedOutABI = slice.outABI;
  } else {
    llvm::Expected<support::RuntimeABIParameter> outABI =
        hasStridedMemory && !isStridedLoadUnitStore &&
                !isStridedInputWideningDotReduceAdd &&
                !isComputedMaskStridedInputWideningDotReduceAdd
            ? getRuntimeABIParameterBindingFromValue(
                  slice.stridedStore.getBuffer(),
                  "tcrv_rvv.strided_store buffer operand",
                  {support::RuntimeABIParameterRole::OutputBuffer})
        : isIndexedScatterUnitLoad
            ? getRuntimeABIParameterBindingFromValue(
                  slice.indexedStore.getDestination(),
                  "tcrv_rvv.indexed_store destination operand",
                  {support::RuntimeABIParameterRole::OutputBuffer})
        : isMaskedUnitStore
            ? getRuntimeABIParameterBindingFromValue(
                  slice.maskedStore.getBuffer(),
                  "tcrv_rvv.masked_store destination operand",
                  {support::RuntimeABIParameterRole::OutputBuffer})
            : getRuntimeABIParameterBindingFromValue(
                  slice.genericStore.getBuffer(),
                  "tcrv_rvv.store buffer operand",
                  {support::RuntimeABIParameterRole::OutputBuffer});
    if (!outABI)
      return outABI.takeError();
    resolvedOutABI = *outABI;
  }
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd) {
    llvm::Expected<support::RuntimeABIParameter> outStrideABI =
        getRuntimeABIParameterBindingFromValue(
            slice.stridedStore.getStride(),
            "tcrv_rvv.strided_store stride operand",
            {support::RuntimeABIParameterRole::OutputStride,
             support::RuntimeABIParameterRole::DestinationByteStride});
    if (!outStrideABI)
      return outStrideABI.takeError();
    if (llvm::Error error = assignRVVGenericStridedStoreBinding(
            slice, slice.stridedStore, resolvedOutABI, *outStrideABI,
            isUnitLoadStridedStore || isComputedMaskStridedStore))
      return error;
    if (isUnitLoadStridedStore)
      slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadStridedStore;
    else if (isComputedMaskStridedStore)
      slice.memoryForm =
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
    else
      slice.arithmeticKind = RVVSelectedBodyOperationKind::StridedAdd;
  } else if (isStridedLoadUnitStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  } else if (isStridedInputWideningDotReduceAdd) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce;
  } else if (isComputedMaskStridedInputWideningDotReduceAdd) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::
            ComputedMaskStridedInputWideningDotReduce;
  } else if (isIndexedGatherUnitStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  } else if (isIndexedScatterUnitLoad) {
    slice.storeOperation = slice.indexedStore.getOperation();
    slice.outBuffer = slice.indexedStore.getDestination();
    slice.storeValue = slice.indexedStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  } else if (isMaskedUnitLoadStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  } else if (isMaskedUnitStore) {
    slice.storeOperation = slice.maskedStore.getOperation();
    slice.outBuffer = slice.maskedStore.getBuffer();
    slice.storeValue = slice.maskedStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::MaskedUnitStore;
  } else if (isComputedMaskUnitLoadStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore;
  } else if (isSegment2DeinterleaveUnitStore) {
    slice.storeOperation = slice.field0StoreOperation;
    slice.outBuffer = slice.field0Buffer;
    slice.storeValue = slice.field0Store.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  } else if (isSegment2InterleaveUnitLoad) {
    slice.storeOperation = slice.segment2StoreOperation;
    slice.outBuffer = slice.segment2Store.getDestination();
    slice.storeValue = slice.segment2Store.getField0();
    slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  } else if (isStandaloneReduction) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction;
  } else {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    if (hasScalarBroadcast)
      slice.arithmeticKind =
          RVVSelectedBodyOperationKind::ScalarBroadcastAdd;
  }
  llvm::StringRef operationMnemonic =
      stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind);
  slice.runtimeElementCountABI = *runtimeElementCountABI;
  slice.outABI = resolvedOutABI;
  if (isWideningDotReduceAdd || isStridedInputWideningDotReduceAdd) {
    llvm::Expected<support::RuntimeABIParameter> accumulatorABI =
        getRuntimeABIParameterBindingFromValue(
            slice.wideningDotReduceOp.getAccumulatorSeed(),
            "tcrv_rvv.widening_dot_reduce accumulator_seed operand",
            {support::RuntimeABIParameterRole::AccumulatorInputBuffer});
    if (!accumulatorABI)
      return accumulatorABI.takeError();
    slice.accumulatorABI = *accumulatorABI;
    slice.accumulatorBuffer =
        slice.wideningDotReduceOp.getAccumulatorSeed();
  }
  if (isComputedMaskWideningDotReduceAdd ||
      isComputedMaskStridedInputWideningDotReduceAdd) {
    llvm::Expected<support::RuntimeABIParameter> accumulatorABI =
        getRuntimeABIParameterBindingFromValue(
            slice.maskedWideningDotReduceOp.getAccumulatorSeed(),
            "tcrv_rvv.masked_widening_dot_reduce accumulator_seed operand",
            {support::RuntimeABIParameterRole::AccumulatorInputBuffer});
    if (!accumulatorABI)
      return accumulatorABI.takeError();
    slice.accumulatorABI = *accumulatorABI;
    slice.accumulatorBuffer =
        slice.maskedWideningDotReduceOp.getAccumulatorSeed();
  }
  if (isStandaloneReduction) {
    llvm::Expected<support::RuntimeABIParameter> accumulatorABI =
        getRuntimeABIParameterBindingFromValue(
            slice.standaloneReduceOp.getAccumulatorSeed(),
            "tcrv_rvv.standalone_reduce accumulator_seed operand",
            {support::RuntimeABIParameterRole::AccumulatorInputBuffer});
    if (!accumulatorABI)
      return accumulatorABI.takeError();
    slice.accumulatorABI = *accumulatorABI;
    slice.accumulatorBuffer =
        slice.standaloneReduceOp.getAccumulatorSeed();
  }
  if (isComputedMaskSelect) {
    if (!slice.trueValueLoadOperation || !slice.falseValueLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route requires "
          "true-value-input-buffer and false-value-input-buffer loads");
    if (slice.compareOp.getKind() != "slt")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route currently supports "
          "only tcrv_rvv.compare {kind = \"slt\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route requires "
          "tcrv_rvv.compare to consume compare lhs/rhs generic load results");
    if (slice.selectOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route requires "
          "tcrv_rvv.select to consume the mask produced by tcrv_rvv.compare");
    if (slice.arithmeticLhs != slice.trueValue ||
        slice.arithmeticRhs != slice.falseValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route requires "
          "tcrv_rvv.select to consume true-value load as true value and "
          "false-value load as false value");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.selectOp.getVl() != slice.setvl.getVl() ||
        slice.trueValueGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.falseValueGenericLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route requires compare, "
          "select, and true/false loads to consume the selected "
          "!tcrv_rvv.vl token");
    if (slice.sourceLoadOperation || slice.accumulatorLoadOperation ||
        slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route does not support "
          "source, accumulator, or runtime mask_load inputs");
  } else if (isCompareSelect) {
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.compare "
          "to consume lhs/rhs generic load results");
    if (slice.selectOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.select "
          "to consume the mask produced by tcrv_rvv.compare");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.select "
          "to consume lhs as true value and rhs as false value");
  } else if (isMaskedAdd) {
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires tcrv_rvv.compare to "
          "consume lhs/rhs generic load results");
    if (slice.maskedBinaryOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires "
          "tcrv_rvv.masked_binary to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.maskedPassthrough != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires "
          "tcrv_rvv.masked_binary passthrough to consume the lhs load result");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires "
          "tcrv_rvv.masked_binary to consume lhs/rhs generic load results");
  } else if (isMAccAdd) {
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV multiply-accumulate route requires one "
          "accumulator-input-buffer tcrv_rvv.load");
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV multiply-accumulate route requires the "
          "accumulator load to bind accumulator-input-buffer, not output "
          "buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV multiply-accumulate route requires separate "
          "accumulator input and output destination ABI values");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue ||
        slice.arithmeticAccumulator != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
        "bounded generic RVV multiply-accumulate route requires "
        "tcrv_rvv.macc to consume lhs/rhs generic load results and the "
        "accumulator-input-buffer load result");
  } else if (isWideningMAccAdd) {
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening multiply-accumulate route requires "
          "one accumulator-input-buffer tcrv_rvv.load");
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening multiply-accumulate route requires "
          "the accumulator load to bind accumulator-input-buffer, not output "
          "buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening multiply-accumulate route requires "
          "separate accumulator input and output destination ABI values");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue ||
        slice.arithmeticAccumulator != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening multiply-accumulate route requires "
          "tcrv_rvv.widening_macc to consume lhs/rhs source loads and the "
          "accumulator-input-buffer load result");
  } else if (isWideningDotReduceAdd || isStridedInputWideningDotReduceAdd) {
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening dot-product reduction route requires "
          "the accumulator seed to bind accumulator-input-buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening dot-product reduction route requires "
          "separate accumulator seed input and output destination ABI values");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue ||
        slice.arithmeticAccumulator != slice.accumulatorBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening dot-product reduction route requires "
          "tcrv_rvv.widening_dot_reduce to consume lhs/rhs source loads and "
          "the accumulator-input-buffer scalar seed boundary");
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening dot-product reduction route does not "
          "load the accumulator seed as a vector; it must remain a scalar "
          "runtime ABI boundary");
    if (isStridedInputWideningDotReduceAdd &&
        (slice.lhsStridedLoad.getVl() != slice.setvl.getVl() ||
         slice.rhsStridedLoad.getVl() != slice.setvl.getVl()))
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided-input widening dot-product reduction "
          "route requires both strided source loads to consume the selected "
          "!tcrv_rvv.vl token");
  } else if (isStandaloneReduction) {
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV standalone reduction route requires the "
          "accumulator seed to bind accumulator-input-buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV standalone reduction route requires separate "
          "accumulator seed input and scalar output destination ABI values");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticAccumulator != slice.accumulatorBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV standalone reduction route requires "
          "tcrv_rvv.standalone_reduce to consume the input vector load and "
          "the accumulator-input-buffer scalar seed boundary");
    if (slice.accumulatorLoadOperation || slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV standalone reduction route does not load RHS "
          "or accumulator seed as vectors; the seed must remain a scalar "
          "runtime ABI boundary");
    if (slice.standaloneReduceOp.getVl() != slice.setvl.getVl() ||
        slice.lhsGenericLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV standalone reduction route requires input load "
          "and standalone_reduce to consume the selected !tcrv_rvv.vl token");
  } else if (isComputedMaskWideningDotReduceAdd ||
             isComputedMaskStridedInputWideningDotReduceAdd) {
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires the accumulator seed to bind "
          "accumulator-input-buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires separate accumulator seed input and output "
          "destination ABI values");
    if (!slice.compareOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires one tcrv_rvv.compare producer");
    if (!slice.dotLHSLoadOperation || !slice.dotRHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires dot-lhs-input-buffer and dot-rhs-input-buffer "
          "loads");
    if (slice.dotLHSABI.role !=
            support::RuntimeABIParameterRole::DotLHSInputBuffer ||
        slice.dotRHSABI.role !=
            support::RuntimeABIParameterRole::DotRHSInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires dot lhs/rhs loads to bind dedicated dot input ABI "
          "roles");
    if (slice.compareOp.getKind() != "slt")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route currently supports only tcrv_rvv.compare {kind = \"slt\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires tcrv_rvv.compare to consume compare lhs/rhs "
          "generic load results");
    if (slice.maskedWideningDotReduceOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires tcrv_rvv.masked_widening_dot_reduce to consume the "
          "mask produced by tcrv_rvv.compare");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.maskedWideningDotReduceOp.getVl() != slice.setvl.getVl() ||
        slice.lhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.rhsGenericLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires compare, masked dot-reduction, and compare loads "
          "to consume the selected !tcrv_rvv.vl token");
    if (isComputedMaskWideningDotReduceAdd &&
        (slice.dotLHSGenericLoad.getVl() != slice.setvl.getVl() ||
         slice.dotRHSGenericLoad.getVl() != slice.setvl.getVl()))
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires dot lhs/rhs unit-stride loads to consume the "
          "selected !tcrv_rvv.vl token");
    if (isComputedMaskStridedInputWideningDotReduceAdd &&
        (slice.lhsStridedLoad.getVl() != slice.setvl.getVl() ||
         slice.rhsStridedLoad.getVl() != slice.setvl.getVl()))
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-input widening "
          "dot-product reduction route requires dot lhs/rhs strided loads to "
          "consume the selected !tcrv_rvv.vl token");
    if (slice.arithmeticLhs != slice.dotLHSValue ||
        slice.arithmeticRhs != slice.dotRHSValue ||
        slice.arithmeticAccumulator != slice.accumulatorBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires tcrv_rvv.masked_widening_dot_reduce to consume "
          "dot lhs/rhs source loads and the accumulator-input-buffer scalar "
          "seed boundary");
    if (slice.accumulatorLoadOperation || slice.maskLoadOperation ||
        slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route does not support vector accumulator loads, runtime mask_load "
          "input, or source-input-buffer loads");
  } else if (isWideningConversion) {
    if (slice.conversionSource != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening conversion route requires "
          "tcrv_rvv.widening_convert to consume the lhs source load result");
    if (slice.accumulatorLoadOperation || slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening conversion route does not support RHS "
          "or accumulator loads");
  } else if (isStridedLoadUnitStore) {
    if (slice.arithmeticLhs != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided-load to unit-stride-store route "
          "requires tcrv_rvv.move to consume the strided source load result");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided-load to unit-stride-store route does "
          "not support RHS or accumulator loads");
  } else if (isUnitLoadStridedStore) {
    if (slice.arithmeticLhs != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV unit-load to strided-store route requires "
          "tcrv_rvv.move to consume the unit-stride source load result");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV unit-load to strided-store route does not "
          "support RHS or accumulator loads");
  } else if (isIndexedGatherUnitStore) {
    if (slice.arithmeticLhs != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route requires tcrv_rvv.move "
          "to consume the indexed data load result");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route does not support RHS or "
          "accumulator loads");
  } else if (isIndexedScatterUnitLoad) {
    if (slice.arithmeticLhs != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route requires tcrv_rvv.move "
          "to consume the unit-stride source load result");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route does not support RHS or "
          "accumulator loads");
  } else if (isMaskedUnitLoadStore) {
    if (!slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires one "
          "tcrv_rvv.mask_load");
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires an "
          "output-buffer tcrv_rvv.load for old-destination preservation");
    if (slice.accumulatorBuffer != slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires the "
          "old-destination load to consume the same output buffer as "
          "tcrv_rvv.store");
    if (slice.maskedActiveValue != slice.lhsValue ||
        slice.maskedInactivePassthrough != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires "
          "tcrv_rvv.masked_move to consume source load as active value and "
          "old-destination load as inactive passthrough");
    if (slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route does not support an RHS "
          "data load");
  } else if (isMaskedUnitStore) {
    if (!slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route requires one "
          "tcrv_rvv.mask_load");
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route must not load the old "
          "destination; false lanes are preserved by masked_store");
    if (slice.maskedActiveValue != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route requires "
          "tcrv_rvv.masked_store to consume the payload source load value");
    if (slice.rhsLoadOperation || slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route does not support RHS or "
          "extra source data loads");
  } else if (isComputedMaskUnitLoadStore) {
    if (!slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires one "
          "source-input-buffer tcrv_rvv.load for active source values");
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires an "
          "output-buffer tcrv_rvv.load for old-destination preservation");
    if (slice.accumulatorBuffer != slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires the "
          "old-destination load to consume the same output buffer as "
          "tcrv_rvv.store");
    if (slice.compareOp.getKind() != "slt")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route currently supports "
          "only tcrv_rvv.compare {kind = \"slt\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires "
          "tcrv_rvv.compare to consume compare lhs/rhs generic load results");
    if (slice.maskedActiveValue != slice.sourceValue ||
        slice.maskedInactivePassthrough != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires "
          "tcrv_rvv.masked_move to consume source-input load as active value "
          "and old-destination load as inactive passthrough");
    if (slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route does not support "
          "runtime mask_load input");
  } else if (isComputedMaskStridedStore) {
    if (!slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires one "
          "source-input-buffer tcrv_rvv.load for active source values");
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires an "
          "output-buffer tcrv_rvv.strided_load for old-destination "
          "preservation");
    if (slice.accumulatorBuffer != slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires the "
          "old-destination strided load to consume the same output buffer as "
          "tcrv_rvv.strided_store");
    if (slice.outStride != slice.stridedStore.getStride())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires the "
          "old-destination strided load and strided_store to consume the same "
          "destination stride SSA value");
    if (slice.compareOp.getKind() != "slt")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route currently "
          "supports only tcrv_rvv.compare {kind = \"slt\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires "
          "tcrv_rvv.compare to consume compare lhs/rhs generic load results");
    if (slice.maskedActiveValue != slice.sourceValue ||
        slice.maskedInactivePassthrough != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires "
          "tcrv_rvv.masked_move to consume source-input load as active value "
          "and old strided destination load as inactive passthrough");
    if (slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route does not "
          "support runtime mask_load input");
  } else if (isSegment2DeinterleaveUnitStore) {
    if (slice.field0Store.getValue() != slice.field0Value ||
        slice.field1Store.getValue() != slice.field1Value)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 deinterleave route requires field0 "
          "and field1 stores to consume their matching field move results");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation ||
        slice.sourceLoadOperation || slice.indexLoadOperation ||
        slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 deinterleave route does not support "
          "RHS, source, index, mask, or accumulator loads");
  } else if (isSegment2InterleaveUnitLoad) {
    if (slice.segment2Store.getField0() != slice.field0Value ||
        slice.segment2Store.getField1() != slice.field1Value)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 interleave route requires "
          "segment2_store to consume its matching field source load results");
    if (slice.accumulatorLoadOperation || slice.sourceLoadOperation ||
        slice.indexLoadOperation || slice.maskLoadOperation ||
        slice.segment2LoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 interleave route does not support "
          "source, index, mask, accumulator, or segment2_load inputs");
  } else if (slice.arithmeticLhs != slice.lhsValue ||
             slice.arithmeticRhs != slice.rhsValue) {
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV non-multiply-accumulate route does not support "
          "an output-buffer accumulator load");
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded generic RVV EmitC route requires selected-body ") +
        operationMnemonic +
        " to consume lhs/rhs generic load or broadcast results");
  }
  if (!isSegment2DeinterleaveUnitStore && !isSegment2InterleaveUnitLoad &&
      slice.storeValue != slice.arithmeticResult)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires selected-body store to "
        "consume the selected compute result");

  return slice;
}

struct RVVOrderedRoleOperations {
  llvm::SmallVector<mlir::Operation *, 12> operations;
  llvm::SmallVector<unsigned, 12> constructionOrders;
};

unsigned getRVVCanonicalRoleOrder(RVVSelectedBodyRouteSlice &slice,
                                  mlir::Operation *op) {
  auto getRuntimeABI =
      [](mlir::Value value) -> tcrv::rvv::RuntimeABIValueOp {
    if (!value)
      return nullptr;
    return value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  };
  auto lhsABI =
      getRuntimeABI(slice.lhsBuffer);
  auto rhsABI =
      getRuntimeABI(slice.rhsBuffer);
  auto trueValueABI =
      getRuntimeABI(slice.trueValueBuffer);
  auto falseValueABI =
      getRuntimeABI(slice.falseValueBuffer);
  auto sourceABI =
      getRuntimeABI(slice.sourceBuffer);
  auto dotLHSABI =
      getRuntimeABI(slice.dotLHSBuffer);
  auto dotRHSABI =
      getRuntimeABI(slice.dotRHSBuffer);
  auto indexABI =
      getRuntimeABI(slice.indexBuffer);
  auto maskABI =
      getRuntimeABI(slice.maskBuffer);
  auto field0ABI =
      getRuntimeABI(slice.field0Buffer);
  auto field1ABI =
      getRuntimeABI(slice.field1Buffer);
  auto accumulatorABI =
      getRuntimeABI(slice.accumulatorBuffer);
  auto outABI =
      getRuntimeABI(slice.outBuffer);
  auto nABI =
      getRuntimeABI(slice.setvl ? slice.setvl.getAvl() : mlir::Value());
  auto lhsStrideABI =
      slice.lhsStride
          ? slice.lhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto rhsStrideABI =
      slice.rhsStride
          ? slice.rhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto outStrideABI =
      slice.outStride
          ? slice.outStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  const bool isStrided =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore;
  const bool isStridedLoadUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  const bool isUnitLoadStridedStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadStridedStore;
  const bool isIndexedGatherUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  const bool isIndexedScatterUnitLoad =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  const bool isMaskedUnitLoadStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  const bool isMaskedUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore;
  const bool isComputedMaskUnitLoadStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore;
  const bool isComputedMaskSelect =
      slice.memoryForm == RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect;
  const bool isComputedMaskStridedStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
  const bool isSegment2DeinterleaveUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  const bool isSegment2InterleaveUnitLoad =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  const bool isConversion =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitStrideConversion;
  const bool isMAcc =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd;
  const bool isWideningMAcc =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::WideningDotReduceAdd;
  const bool isStridedInputWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd;
  const bool isComputedMaskWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
  const bool isComputedMaskStridedInputWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          ComputedMaskStridedInputWideningDotReduceAdd;
  const bool isStandaloneReduction =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::StandaloneReduceAdd;
  if (lhsABI && op == lhsABI.getOperation())
    return 0;
  if (isComputedMaskSelect) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (trueValueABI && op == trueValueABI.getOperation())
      return 2;
    if (falseValueABI && op == falseValueABI.getOperation())
      return 3;
    if (outABI && op == outABI.getOperation())
      return 4;
    if (nABI && op == nABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.trueValueLoadOperation)
      return 10;
    if (op == slice.falseValueLoadOperation)
      return 11;
    if (op == slice.compareOp.getOperation())
      return 12;
    if (op == slice.arithmeticOp)
      return 13;
    if (op == slice.storeOperation)
      return 14;
    return 15;
  }
  if (isComputedMaskStridedInputWideningDotReduce) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (dotLHSABI && op == dotLHSABI.getOperation())
      return 2;
    if (dotRHSABI && op == dotRHSABI.getOperation())
      return 3;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 4;
    if (outABI && op == outABI.getOperation())
      return 5;
    if (nABI && op == nABI.getOperation())
      return 6;
    if (lhsStrideABI && op == lhsStrideABI.getOperation())
      return 7;
    if (rhsStrideABI && op == rhsStrideABI.getOperation())
      return 8;
    if (op == slice.setvl.getOperation())
      return 9;
    if (op == slice.withVL.getOperation())
      return 10;
    if (op == slice.lhsLoadOperation)
      return 11;
    if (op == slice.rhsLoadOperation)
      return 12;
    if (op == slice.dotLHSLoadOperation)
      return 13;
    if (op == slice.dotRHSLoadOperation)
      return 14;
    if (op == slice.compareOp.getOperation())
      return 15;
    if (op == slice.arithmeticOp)
      return 16;
    if (op == slice.storeOperation)
      return 17;
    return 18;
  }
  if (isComputedMaskWideningDotReduce) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (dotLHSABI && op == dotLHSABI.getOperation())
      return 2;
    if (dotRHSABI && op == dotRHSABI.getOperation())
      return 3;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 4;
    if (outABI && op == outABI.getOperation())
      return 5;
    if (nABI && op == nABI.getOperation())
      return 6;
    if (op == slice.setvl.getOperation())
      return 7;
    if (op == slice.withVL.getOperation())
      return 8;
    if (op == slice.lhsLoadOperation)
      return 9;
    if (op == slice.rhsLoadOperation)
      return 10;
    if (op == slice.dotLHSLoadOperation)
      return 11;
    if (op == slice.dotRHSLoadOperation)
      return 12;
    if (op == slice.compareOp.getOperation())
      return 13;
    if (op == slice.arithmeticOp)
      return 14;
    if (op == slice.storeOperation)
      return 15;
    return 16;
  }
  if (isMAcc || isWideningMAcc) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (op == slice.setvl.getOperation())
      return 5;
    if (op == slice.withVL.getOperation())
      return 6;
    if (op == slice.lhsLoadOperation)
      return 7;
    if (op == slice.rhsLoadOperation)
      return 8;
    if (op == slice.accumulatorLoadOperation)
      return 9;
    if (op == slice.arithmeticOp)
      return 10;
    if (op == slice.storeOperation)
      return 11;
    return 12;
  }
  if (isWideningDotReduce) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (op == slice.setvl.getOperation())
      return 5;
    if (op == slice.withVL.getOperation())
      return 6;
    if (op == slice.lhsLoadOperation)
      return 7;
    if (op == slice.rhsLoadOperation)
      return 8;
    if (op == slice.arithmeticOp)
      return 9;
    if (op == slice.storeOperation)
      return 10;
    return 11;
  }
  if (isStridedInputWideningDotReduce) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (lhsStrideABI && op == lhsStrideABI.getOperation())
      return 5;
    if (rhsStrideABI && op == rhsStrideABI.getOperation())
      return 6;
    if (op == slice.setvl.getOperation())
      return 7;
    if (op == slice.withVL.getOperation())
      return 8;
    if (op == slice.lhsLoadOperation)
      return 9;
    if (op == slice.rhsLoadOperation)
      return 10;
    if (op == slice.arithmeticOp)
      return 11;
    if (op == slice.storeOperation)
      return 12;
    return 13;
  }
  if (isStandaloneReduction) {
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.arithmeticOp)
      return 7;
    if (op == slice.storeOperation)
      return 8;
    return 9;
  }
  if (isSegment2InterleaveUnitLoad) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.rhsLoadOperation)
      return 7;
    if (op == slice.segment2StoreOperation)
      return 8;
    return 9;
  }
  if (isSegment2DeinterleaveUnitStore) {
    if (field0ABI && op == field0ABI.getOperation())
      return 1;
    if (field1ABI && op == field1ABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.segment2LoadOperation)
      return 6;
    if (op == slice.field0MoveOperation)
      return 7;
    if (op == slice.field1MoveOperation)
      return 8;
    if (op == slice.field0StoreOperation)
      return 9;
    if (op == slice.field1StoreOperation)
      return 10;
    return 11;
  }
  if (isComputedMaskUnitLoadStore) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (op == slice.setvl.getOperation())
      return 5;
    if (op == slice.withVL.getOperation())
      return 6;
    if (op == slice.lhsLoadOperation)
      return 7;
    if (op == slice.rhsLoadOperation)
      return 8;
    if (op == slice.sourceLoadOperation)
      return 9;
    if (op == slice.accumulatorLoadOperation)
      return 10;
    if (op == slice.compareOp.getOperation())
      return 11;
    if (op == slice.arithmeticOp)
      return 12;
    if (op == slice.storeOperation)
      return 13;
    return 14;
  }
  if (isComputedMaskStridedStore) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (outStrideABI && op == outStrideABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.sourceLoadOperation)
      return 10;
    if (op == slice.accumulatorLoadOperation)
      return 11;
    if (op == slice.compareOp.getOperation())
      return 12;
    if (op == slice.arithmeticOp)
      return 13;
    if (op == slice.storeOperation)
      return 14;
    return 15;
  }
  if (isMaskedUnitLoadStore) {
    if (maskABI && op == maskABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.maskLoadOperation)
      return 7;
    if (op == slice.accumulatorLoadOperation)
      return 8;
    if (op == slice.arithmeticOp)
      return 9;
    if (op == slice.storeOperation)
      return 10;
    return 11;
  }
  if (isMaskedUnitStore) {
    if (maskABI && op == maskABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.maskLoadOperation)
      return 7;
    if (op == slice.storeOperation)
      return 8;
    return 9;
  }
  if (isIndexedGatherUnitStore) {
    if (indexABI && op == indexABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.indexLoadOperation)
      return 6;
    if (op == slice.indexedLoadOperation)
      return 7;
    if (op == slice.arithmeticOp)
      return 8;
    if (op == slice.storeOperation)
      return 9;
    return 10;
  }
  if (isIndexedScatterUnitLoad) {
    if (indexABI && op == indexABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.indexLoadOperation)
      return 7;
    if (op == slice.arithmeticOp)
      return 8;
    if (op == slice.indexedStoreOperation)
      return 9;
    return 10;
  }
  if (isStridedLoadUnitStore) {
    if (outABI && op == outABI.getOperation())
      return 1;
    if (nABI && op == nABI.getOperation())
      return 2;
    if (lhsStrideABI && op == lhsStrideABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.arithmeticOp)
      return 7;
    if (op == slice.storeOperation)
      return 8;
    return 9;
  }
  if (isUnitLoadStridedStore) {
    if (outABI && op == outABI.getOperation())
      return 1;
    if (nABI && op == nABI.getOperation())
      return 2;
    if (outStrideABI && op == outStrideABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.arithmeticOp)
      return 7;
    if (op == slice.storeOperation)
      return 8;
    return 9;
  }
  if (isConversion) {
    if (outABI && op == outABI.getOperation())
      return 1;
    if (nABI && op == nABI.getOperation())
      return 2;
    if (op == slice.setvl.getOperation())
      return 3;
    if (op == slice.withVL.getOperation())
      return 4;
    if (op == slice.lhsLoadOperation)
      return 5;
    if (op == slice.arithmeticOp)
      return 6;
    if (op == slice.storeOperation)
      return 7;
    return 8;
  }
  if (rhsABI && op == rhsABI.getOperation())
    return 1;
  if (outABI && op == outABI.getOperation())
    return 2;
  if (nABI && op == nABI.getOperation())
    return 3;
  if (isStrided) {
    if (lhsStrideABI && op == lhsStrideABI.getOperation())
      return 4;
    if (rhsStrideABI && op == rhsStrideABI.getOperation())
      return 5;
    if (outStrideABI && op == outStrideABI.getOperation())
      return 6;
  }
  if (op == slice.setvl.getOperation())
    return isStrided ? 7 : 4;
  if (op == slice.withVL.getOperation())
    return isStrided ? 8 : 5;
  if (op == slice.lhsLoadOperation)
    return isStrided ? 9 : 6;
  if (op == slice.rhsLoadOperation)
    return isStrided ? 10 : 7;
  if (isStrided && op == slice.arithmeticOp)
    return 11;
  if (isStrided && op == slice.storeOperation)
    return 12;
  if (slice.accumulatorLoadOperation && op == slice.accumulatorLoadOperation)
    return 8;
  if (slice.compareOp && op == slice.compareOp.getOperation())
    return 8;
  if (op == slice.arithmeticOp)
    return (slice.compareOp || slice.accumulatorLoadOperation) ? 9 : 8;
  if (op == slice.storeOperation)
    return (slice.compareOp || slice.accumulatorLoadOperation) ? 10 : 9;
  return (slice.compareOp || slice.accumulatorLoadOperation) ? 11 : 10;
}

RVVOrderedRoleOperations
collectRVVRoleOperationsInBodyOrder(tcrv::exec::VariantOp variant,
                                    RVVSelectedBodyRouteSlice &slice) {
  RVVOrderedRoleOperations ordered;
  if (!variant || variant.getBody().empty())
    return ordered;

  for (mlir::Operation &op : variant.getBody().front()) {
    if (op.getName().getDialectNamespace() != "tcrv_rvv")
      continue;
    ordered.operations.push_back(&op);
    ordered.constructionOrders.push_back(getRVVCanonicalRoleOrder(slice, &op));
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      for (mlir::Operation &nested : withVL.getBody().front())
        if (nested.getName().getDialectNamespace() == "tcrv_rvv") {
          ordered.operations.push_back(&nested);
          ordered.constructionOrders.push_back(
              getRVVCanonicalRoleOrder(slice, &nested));
        }
  }
  return ordered;
}

llvm::Error verifySelectedRVVRoleSequence(
    RVVSelectedBodyRouteSlice &slice,
    const VariantEmitCLowerableRequest &request,
    const RVVSelectedBodyConstructionRoute &constructionRoute) {
  auto getRuntimeABI =
      [](mlir::Value value) -> tcrv::rvv::RuntimeABIValueOp {
    if (!value)
      return nullptr;
    return value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  };
  auto lhsABI =
      getRuntimeABI(slice.lhsBuffer);
  auto rhsABI =
      getRuntimeABI(slice.rhsBuffer);
  auto trueValueABI =
      getRuntimeABI(slice.trueValueBuffer);
  auto falseValueABI =
      getRuntimeABI(slice.falseValueBuffer);
  auto sourceABI =
      getRuntimeABI(slice.sourceBuffer);
  auto dotLHSABI =
      getRuntimeABI(slice.dotLHSBuffer);
  auto dotRHSABI =
      getRuntimeABI(slice.dotRHSBuffer);
  auto indexABI =
      getRuntimeABI(slice.indexBuffer);
  auto maskABI =
      getRuntimeABI(slice.maskBuffer);
  auto field0ABI =
      getRuntimeABI(slice.field0Buffer);
  auto field1ABI =
      getRuntimeABI(slice.field1Buffer);
  auto accumulatorABI =
      getRuntimeABI(slice.accumulatorBuffer);
  auto outABI =
      getRuntimeABI(slice.outBuffer);
  auto nABI =
      getRuntimeABI(slice.setvl ? slice.setvl.getAvl() : mlir::Value());
  auto lhsStrideABI =
      slice.lhsStride
          ? slice.lhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto rhsStrideABI =
      slice.rhsStride
          ? slice.rhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto outStrideABI =
      slice.outStride
          ? slice.outStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  const bool isStrided =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore;
  const bool isStridedLoadUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  const bool isUnitLoadStridedStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadStridedStore;
  const bool isIndexedGatherUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  const bool isIndexedScatterUnitLoad =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  const bool isMaskedUnitLoadStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  const bool isMaskedUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore;
  const bool isComputedMaskUnitLoadStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore;
  const bool isComputedMaskSelect =
      slice.memoryForm == RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect;
  const bool isComputedMaskStridedStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
  const bool isSegment2DeinterleaveUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  const bool isSegment2InterleaveUnitLoad =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  const bool isConversion =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitStrideConversion;
  const bool isMAcc =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd;
  const bool isWideningMAcc =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::WideningDotReduceAdd;
  const bool isStridedInputWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd;
  const bool isComputedMaskWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
  const bool isComputedMaskStridedInputWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          ComputedMaskStridedInputWideningDotReduceAdd;
  const bool isStandaloneReduction =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::StandaloneReduceAdd;
  if (!lhsABI ||
      (!isConversion && !isStridedLoadUnitStore &&
       !isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
       !isMaskedUnitLoadStore && !isMaskedUnitStore &&
       !isComputedMaskUnitLoadStore &&
       !isComputedMaskSelect &&
       !isComputedMaskStridedStore &&
       !isSegment2DeinterleaveUnitStore && !isSegment2InterleaveUnitLoad &&
       !isUnitLoadStridedStore &&
       !isStandaloneReduction &&
       !rhsABI) ||
      (!isSegment2DeinterleaveUnitStore && !outABI) ||
      !nABI || (isStrided && (!lhsStrideABI || !rhsStrideABI || !outStrideABI)) ||
      (isStridedLoadUnitStore && !lhsStrideABI) ||
      (isUnitLoadStridedStore && !outStrideABI) ||
      ((isIndexedGatherUnitStore || isIndexedScatterUnitLoad) && !indexABI) ||
      ((isMaskedUnitLoadStore || isMaskedUnitStore) && !maskABI) ||
      (isComputedMaskUnitLoadStore && (!rhsABI || !sourceABI)) ||
      (isComputedMaskSelect &&
       (!rhsABI || !trueValueABI || !falseValueABI)) ||
      (isComputedMaskStridedStore &&
       (!rhsABI || !sourceABI || !outStrideABI)) ||
      (isMAcc && (!rhsABI || !accumulatorABI)) ||
      (isWideningMAcc && (!rhsABI || !accumulatorABI)) ||
      (isWideningDotReduce && (!rhsABI || !accumulatorABI)) ||
      (isStandaloneReduction && !accumulatorABI) ||
      (isStridedInputWideningDotReduce &&
       (!rhsABI || !accumulatorABI || !lhsStrideABI || !rhsStrideABI)) ||
      (isComputedMaskWideningDotReduce &&
       (!rhsABI || !dotLHSABI || !dotRHSABI || !accumulatorABI)) ||
      (isComputedMaskStridedInputWideningDotReduce &&
       (!rhsABI || !dotLHSABI || !dotRHSABI || !accumulatorABI ||
        !lhsStrideABI || !rhsStrideABI)) ||
      (isSegment2DeinterleaveUnitStore && (!field0ABI || !field1ABI)) ||
      (isSegment2InterleaveUnitLoad && (!rhsABI || !outABI)))
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires runtime ABI values "
        "to be explicit tcrv_rvv.runtime_abi_value ops");

  if (isStridedLoadUnitStore) {
    if (slice.lhsABI.role !=
        support::RuntimeABIParameterRole::SourceInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV byte-strided-load to unit-stride-store route "
          "requires source-input-buffer runtime ABI value");
    if (slice.lhsStrideABI.role !=
        support::RuntimeABIParameterRole::SourceByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV byte-strided-load to unit-stride-store route "
          "requires source-byte-stride runtime ABI value");
  }
  if (isUnitLoadStridedStore) {
    if (slice.outStrideABI.role !=
        support::RuntimeABIParameterRole::DestinationByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV unit-load to byte-strided-store route "
          "requires destination-byte-stride runtime ABI value");
  }
  if (isComputedMaskStridedStore) {
    if (slice.outStrideABI.role !=
        support::RuntimeABIParameterRole::DestinationByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask byte-strided-store route "
          "requires destination-byte-stride runtime ABI value");
  }

  mlir::ArrayAttr requires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requires || requires.empty())
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires non-empty selected "
        "variant requires metadata");

  RVVOrderedRoleOperations ordered =
      collectRVVRoleOperationsInBodyOrder(request.getVariant(), slice);
  return verifyRVVSelectedBodySelectedRoleSequence(
      ordered.operations, ordered.constructionOrders,
      request.getVariant().getSymName(),
      stringifyVariantEmissionRole(request.getRole()),
      constructionRoute.operationMnemonic,
      slice.arithmeticOp->getName().getStringRef(),
      isIndexedGatherUnitStore
          ? slice.indexedLoadOperation->getName().getStringRef()
      : isIndexedScatterUnitLoad
          ? slice.indexedStoreOperation->getName().getStringRef()
      : isMaskedUnitLoadStore
          ? slice.maskLoadOperation->getName().getStringRef()
      : isMaskedUnitStore
          ? slice.maskLoadOperation->getName().getStringRef()
      : isComputedMaskUnitLoadStore
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskSelect
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskStridedStore
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskWideningDotReduce
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskStridedInputWideningDotReduce
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isStridedInputWideningDotReduce
          ? slice.rhsLoadOperation->getName().getStringRef()
      : isSegment2DeinterleaveUnitStore
          ? slice.segment2LoadOperation->getName().getStringRef()
      : isSegment2InterleaveUnitLoad
          ? slice.segment2StoreOperation->getName().getStringRef()
      : isStandaloneReduction
          ? slice.lhsLoadOperation->getName().getStringRef()
      : isStridedLoadUnitStore
          ? slice.lhsLoadOperation->getName().getStringRef()
      : isUnitLoadStridedStore
          ? slice.storeOperation->getName().getStringRef()
          : (slice.rhsLoadOperation
                 ? slice.rhsLoadOperation->getName().getStringRef()
                 : llvm::StringRef()),
      "selected RVV EmitC route");
}

} // namespace

llvm::Expected<RVVSelectedBodyRouteAnalysis>
analyzeRVVSelectedBodyRoute(const VariantEmitCLowerableRequest &request) {
  if (!request.getVariant())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");

  if (llvm::Error error = requireRVVVariantLegality(request.getVariant()))
    return std::move(error);
  if (llvm::Error error = verifyRVVConstructionProtocolReady())
    return std::move(error);

  llvm::Expected<RVVSelectedBodyRouteSlice> slice =
      collectRVVSelectedBodyRouteSlice(request.getVariant());
  if (!slice)
    return slice.takeError();

  tcrv::rvv::RVVCompileTimeConfig config =
      tcrv::rvv::getRVVSetVLCompileTimeConfig(slice->setvl);
  if (llvm::Error error =
          validateRVVSelectedBodyTypedConfigFacts(*slice, config))
    return std::move(error);
  const auto &configContract =
      tcrv::rvv::getRVVSelectedBodyConfigVLContract(config.sew, config.lmul,
                                                    config.policy);

  RVVSelectedBodyRouteAnalysis analysis;
  analysis.slice = std::move(*slice);
  analysis.description.operation = analysis.slice.arithmeticKind;
  analysis.description.memoryForm = analysis.slice.memoryForm;
  analysis.description.sew = config.sew;
  analysis.description.lmul = config.lmul;
  analysis.description.tailPolicy =
      stringifyRVVTailPolicy(config.policy.getTail());
  analysis.description.maskPolicy =
      stringifyRVVMaskPolicy(config.policy.getMask());
  analysis.description.configContractID = configContract.configContractID;
  analysis.description.runtimeVLContractID = configContract.runtimeVLContractID;
  analysis.description.runtimeAVLASource = configContract.runtimeAVLASource;
  switch (analysis.slice.memoryForm) {
  case RVVSelectedBodyMemoryForm::VectorRHSLoad:
  case RVVSelectedBodyMemoryForm::RHSBroadcastLoad:
    analysis.description.runtimeABIOrder = configContract.runtimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::RHSScalarBroadcast:
    analysis.description.runtimeABIOrder = kRVVScalarBroadcastRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::StridedLoadStore:
    analysis.description.runtimeABIOrder = kRVVStridedRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::StridedLoadUnitStore:
    analysis.description.runtimeABIOrder =
        kRVVStridedLoadUnitStoreRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::UnitLoadStridedStore:
    analysis.description.runtimeABIOrder =
        kRVVUnitLoadStridedStoreRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::IndexedLoadUnitStore:
    analysis.description.runtimeABIOrder = kRVVIndexedGatherRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::UnitLoadIndexedStore:
    analysis.description.runtimeABIOrder = kRVVIndexedScatterRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::MaskedUnitLoadStore:
    analysis.description.runtimeABIOrder = kRVVMaskedMemoryRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::MaskedUnitStore:
    analysis.description.runtimeABIOrder = kRVVMaskedStoreRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore:
    analysis.description.runtimeABIOrder =
        kRVVComputedMaskMemoryRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect:
    analysis.description.runtimeABIOrder =
        kRVVComputedMaskSelectRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore:
    analysis.description.runtimeABIOrder =
        kRVVComputedMaskStridedStoreRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::Segment2LoadUnitStore:
    analysis.description.runtimeABIOrder = kRVVSegment2RuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::UnitLoadSegment2Store:
    analysis.description.runtimeABIOrder =
        kRVVSegment2InterleaveRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction:
    analysis.description.runtimeABIOrder = kRVVStandaloneReductionRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::UnitStrideConversion:
    analysis.description.runtimeABIOrder =
        kRVVWideningConversionRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideWideningDotReduce:
    analysis.description.runtimeABIOrder =
        kRVVComputedMaskWideningDotProductRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce:
    analysis.description.runtimeABIOrder =
        kRVVStridedInputWideningDotProductRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskStridedInputWideningDotReduce:
    analysis.description.runtimeABIOrder =
        kRVVComputedMaskStridedInputWideningDotProductRuntimeABIOrder;
    break;
  }
  if (isRVVSelectedBodyContractionRouteOperation(
          analysis.slice.arithmeticKind))
    analysis.description.runtimeABIOrder =
        getRVVSelectedBodyContractionRuntimeABIOrder(
            analysis.slice.arithmeticKind);
  if (analysis.slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd)
    analysis.description.runtimeABIOrder = kRVVMAccRuntimeABIOrder;
  analysis.description.vlDefOpName = configContract.vlDefOpName;
  analysis.description.vlScopeOpName = configContract.vlScopeOpName;
  analysis.description.vlUses = configContract.vlUses;
  analysis.description.emitCLoopKind = configContract.emitCLoopKind;
  analysis.description.emitCLoopInductionName =
      configContract.emitCLoopInductionName;
  analysis.description.emitCFullChunkVLName =
      configContract.emitCFullChunkVLName;
  analysis.description.emitCLoopVLName =
      tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName();
  analysis.description.remainingAVLMetadata =
      configContract.remainingAVLMetadata;
  analysis.description.pointerAdvanceMetadata =
      configContract.pointerAdvanceMetadata;
  analysis.description.boundedSlice = configContract.boundedSlice;
  analysis.description.multiVL = configContract.multiVL;
  if (analysis.slice.memoryForm ==
      RVVSelectedBodyMemoryForm::UnitStrideConversion) {
    if (analysis.slice.arithmeticKind ==
        RVVSelectedBodyOperationKind::WidenI16ToI32) {
      analysis.description.sourceSEW = tcrv::rvv::getRVVSEW16Bits();
      analysis.description.sourceLMUL = tcrv::rvv::getRVVLMULMF2();
      analysis.description.sourceVectorTypeName =
          "!tcrv_rvv.vector<i16, \"mf2\">";
      analysis.description.sourceVectorCType = "vint16mf2_t";
      analysis.description.sourceVectorLoadIntrinsic =
          "__riscv_vle16_v_i16mf2";
      analysis.description.conversionRelation =
          kRVVWidenI16ToI32ConversionRelation;
    } else {
      analysis.description.sourceSEW = tcrv::rvv::getRVVFirstSliceSEWBits();
      analysis.description.sourceLMUL = tcrv::rvv::getRVVLMULM1();
      analysis.description.sourceVectorTypeName =
          "!tcrv_rvv.vector<i32, \"m1\">";
      analysis.description.sourceVectorCType = "vint32m1_t";
      analysis.description.sourceVectorLoadIntrinsic =
          "__riscv_vle32_v_i32m1";
      analysis.description.conversionRelation = kRVVWideningConversionRelation;
    }
  } else if (isRVVSelectedBodyContractionRouteOperation(
                 analysis.slice.arithmeticKind)) {
    analysis.description.sourceSEW = tcrv::rvv::getRVVSEW16Bits();
    analysis.description.sourceLMUL = tcrv::rvv::getRVVLMULMF2();
    analysis.description.sourceVectorTypeName =
        "!tcrv_rvv.vector<i16, \"mf2\">";
    analysis.description.sourceVectorCType = "vint16mf2_t";
    analysis.description.sourceVectorLoadIntrinsic =
        "__riscv_vle16_v_i16mf2";
    if (analysis.slice.arithmeticKind ==
        RVVSelectedBodyOperationKind::WideningMAccAdd)
      analysis.description.wideningMAccRelation = kRVVWideningMAccRelation;
    else
      analysis.description.wideningDotProductRelation =
          kRVVWideningDotProductRelation;
  }
  analysis.description.boundaryOpName = kRVVSelectedBodyLoweringBoundaryOpName;
  analysis.description.targetArtifactRouteID =
      getRVVSelectedBodyTargetArtifactRouteID();
  analysis.description.targetArtifactKind =
      getRVVSelectedBodyTargetArtifactKind();
  analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  if (analysis.slice.memoryForm ==
      RVVSelectedBodyMemoryForm::Segment2LoadUnitStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.field0ABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.field1ABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  }
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::IndexedLoadUnitStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::UnitLoadIndexedStore)
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.indexABI);
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::MaskedUnitLoadStore ||
      analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore)
    analysis.description.runtimeABIParameters.push_back(analysis.slice.maskABI);
  if (analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitStrideConversion &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::StridedLoadUnitStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitLoadStridedStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::IndexedLoadUnitStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitLoadIndexedStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::MaskedUnitLoadStore &&
      analysis.slice.memoryForm != RVVSelectedBodyMemoryForm::MaskedUnitStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::Segment2LoadUnitStore)
    analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction ||
      analysis.slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd)
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.accumulatorABI);
  if (analysis.slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.trueValueABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.falseValueABI);
  }
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore)
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.sourceABI);
  if (analysis.slice.memoryForm !=
      RVVSelectedBodyMemoryForm::Segment2LoadUnitStore) {
    analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  }
  if (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.lhsStrideABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.rhsStrideABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.outStrideABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedLoadUnitStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.lhsStrideABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitLoadStridedStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.outStrideABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.outStrideABI);
  }
  llvm::Expected<RVVSelectedBodyRouteProfile> routeProfile =
      deriveRVVSelectedBodyRouteProfile(analysis.description);
  if (!routeProfile)
    return routeProfile.takeError();
  llvm::Expected<const RVVSelectedBodyConstructionRoute *> constructionRoute =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          routeProfile->operation.operationMnemonic);
  if (!constructionRoute)
    return constructionRoute.takeError();
  analysis.constructionRoute = *constructionRoute;

  analysis.description.typedComputeOpName =
      analysis.slice.arithmeticOp->getName().getStringRef();
  analysis.description.emitCRouteID = analysis.constructionRoute->emitCRouteID;
  analysis.description.runtimeABIName =
      analysis.constructionRoute->runtimeABIName;
  analysis.description.runtimeABIContractName =
      analysis.constructionRoute->runtimeABIContractName;
  analysis.description.vlCType = routeProfile->config.vlCType;
  analysis.description.vectorTypeName =
      routeProfile->config.vectorTypeName;
  analysis.description.indexVectorTypeName =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexVectorTypeName
          : "";
  analysis.description.maskTypeName =
      (routeProfile->operation.isCompareSelect ||
       routeProfile->operation.isMaskedArithmetic ||
       routeProfile->operation.isMaskedMemoryMovement ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::
               ComputedMaskStridedInputWideningDotReduceAdd)
          ? routeProfile->config.maskTypeName
          : "";
  analysis.description.vectorCType = routeProfile->config.vectorCType;
  analysis.description.indexVectorCType =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexVectorCType
          : "";
  analysis.description.maskCType =
      (routeProfile->operation.isCompareSelect ||
       routeProfile->operation.isMaskedArithmetic ||
       routeProfile->operation.isMaskedMemoryMovement ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::
               ComputedMaskStridedInputWideningDotReduceAdd)
          ? routeProfile->config.maskCType
          : "";
  analysis.description.setVLIntrinsic =
      routeProfile->config.setVLIntrinsic;
  analysis.description.vectorLoadIntrinsic =
      routeProfile->config.vectorLoadIntrinsic;
  analysis.description.indexLoadIntrinsic =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexLoadIntrinsic
          : "";
  analysis.description.indexScaleIntrinsic =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexScaleIntrinsic
          : "";
  analysis.description.indexedLoadIntrinsic =
      routeProfile->operation.operation ==
              RVVSelectedBodyOperationKind::IndexedGatherUnitStore
          ? routeProfile->config.indexedLoadIntrinsic
          : "";
  analysis.description.indexedStoreIntrinsic =
      routeProfile->operation.operation ==
              RVVSelectedBodyOperationKind::IndexedScatterUnitLoad
          ? routeProfile->config.indexedStoreIntrinsic
          : "";
  analysis.description.stridedLoadIntrinsic =
      (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::StridedLoadUnitStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore)
          ? routeProfile->config.stridedLoadIntrinsic
          : "";
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::
              ComputedMaskStridedInputWideningDotReduce)
    analysis.description.stridedLoadIntrinsic = "__riscv_vlse16_v_i16mf2";
  analysis.description.rhsBroadcastIntrinsic =
      routeProfile->targetLeaves.rhsBroadcastIntrinsic;
  analysis.description.storeIntrinsic =
      analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore
          ? llvm::StringRef(kRVVMaskedStoreIntrinsic)
          : routeProfile->config.storeIntrinsic;
  analysis.description.stridedStoreIntrinsic =
      (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::UnitLoadStridedStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore)
          ? routeProfile->config.stridedStoreIntrinsic
          : "";
  analysis.description.intrinsic = routeProfile->targetLeaves.intrinsic;
  analysis.description.compareIntrinsic =
      routeProfile->targetLeaves.compareIntrinsic;
  analysis.description.maskedMergeIntrinsic =
      routeProfile->operation.operation ==
              RVVSelectedBodyOperationKind::MaskedUnitStore
          ? llvm::StringRef()
          : routeProfile->targetLeaves.maskedMergeIntrinsic;
  analysis.description.resultName = routeProfile->operation.resultName;
  analysis.description.maskName = routeProfile->operation.maskName;
  if (isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan>
        scalarBroadcastPlan =
            deriveRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
                analysis, routeProfile->config, routeProfile->targetLeaves);
    if (!scalarBroadcastPlan)
      return scalarBroadcastPlan.takeError();
    analysis.scalarBroadcastElementwiseRouteFamilyPlan =
        std::move(*scalarBroadcastPlan);
    applyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
        *analysis.scalarBroadcastElementwiseRouteFamilyPlan,
        analysis.description);
  }
  if (isRVVSelectedBodyStandaloneReductionRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodyStandaloneReductionRouteFamilyPlan>
        standaloneReductionPlan =
            deriveRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
                analysis, routeProfile->config, routeProfile->targetLeaves);
    if (!standaloneReductionPlan)
      return standaloneReductionPlan.takeError();
    analysis.standaloneReductionRouteFamilyPlan =
        std::move(*standaloneReductionPlan);
    applyRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
        *analysis.standaloneReductionRouteFamilyPlan, analysis.description);
  }
  if (routeProfile->operation.isMaskedArithmetic) {
    analysis.description.maskRole = kRVVMaskedPredicateMaskRole;
    analysis.description.maskSource = kRVVMaskedCompareMaskSource;
    analysis.description.inactiveLaneContract =
        kRVVMaskedInactiveLaneContract;
    analysis.description.maskedPassthroughLayout =
        kRVVMaskedPassthroughLayout;
  }
  if (routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSelect) {
    analysis.description.maskRole = kRVVMaskedPredicateMaskRole;
    analysis.description.maskSource = kRVVMaskedCompareMaskSource;
    analysis.description.maskMemoryForm = kRVVComputedMaskMemoryMaskMemoryForm;
    analysis.description.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
    analysis.description.indexedMemoryLayout = kRVVComputedMaskSelectMemoryLayout;
  }
  if (routeProfile->operation.isMaskedMemoryMovement) {
    const bool isComputedMask =
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
    const bool isMaskedStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::MaskedUnitStore;
    analysis.description.maskRole =
        isComputedMask ? kRVVMaskedPredicateMaskRole : kRVVMaskRole;
    analysis.description.maskSource =
        isComputedMask ? kRVVMaskedCompareMaskSource : kRVVMaskSource;
    analysis.description.maskMemoryForm =
        isComputedMask ? kRVVComputedMaskMemoryMaskMemoryForm
                       : kRVVMaskMemoryForm;
    analysis.description.inactiveLaneContract =
        isMaskedStore ? kRVVMaskedStoreInactiveLaneContract
                      : kRVVMaskedMemoryInactiveLaneContract;
    analysis.description.maskedPassthroughLayout =
        isMaskedStore ? kRVVMaskedStorePassthroughLayout
                      : kRVVMaskedMemoryPassthroughLayout;
  }
  if (routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::ReduceAdd) {
    analysis.description.reductionAccumulatorLayout =
        *analysis.slice.reduceOp.getAccumulatorLayout();
    analysis.description.reductionResultLayout =
        *analysis.slice.reduceOp.getResultLayout();
    analysis.description.reductionStoreVL = kRVVReductionStoreVL;
  }
  if (routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::MAccAdd) {
    analysis.description.maccAccumulatorLayout =
        *analysis.slice.maccOp.getAccumulatorLayout();
    analysis.description.maccResultLayout =
        *analysis.slice.maccOp.getResultLayout();
  }
  if (isRVVSelectedBodyContractionRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodyContractionRouteFamilyPlan>
        contractionPlan = deriveRVVSelectedBodyContractionRouteFamilyPlan(
            analysis, routeProfile->config);
    if (!contractionPlan)
      return contractionPlan.takeError();
    analysis.contractionRouteFamilyPlan = std::move(*contractionPlan);
    applyRVVSelectedBodyContractionRouteFamilyPlan(
        *analysis.contractionRouteFamilyPlan, analysis.description);
  }
  if (routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::StridedAdd) {
    analysis.description.stridedMemoryLayout = kRVVStridedMemoryLayout;
    analysis.description.lhsStrideSource = kRVVLHSStrideSource;
    analysis.description.rhsStrideSource = kRVVRHSStrideSource;
    analysis.description.outStrideSource = kRVVOutStrideSource;
  }
  if (routeProfile->operation.isMemoryMovement) {
    const bool isUnitLoadStridedStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::UnitLoadStridedStore;
    analysis.description.stridedMemoryLayout =
        isUnitLoadStridedStore ? kRVVUnitLoadStridedStoreMemoryLayout
                               : kRVVStridedLoadUnitStoreMemoryLayout;
    if (isUnitLoadStridedStore) {
      analysis.description.outStrideSource = kRVVDestinationByteStrideSource;
      analysis.description.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
      analysis.description.destinationMemoryForm = "strided-store";
    } else {
      analysis.description.sourceStrideSource = kRVVSourceStrideSource;
      analysis.description.sourceMemoryForm = kRVVSourceMemoryForm;
      analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
    }
  }
  if (routeProfile->operation.isIndexedMemoryMovement) {
    const bool isScatter =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
    analysis.description.indexedMemoryLayout =
        isScatter ? kRVVIndexedScatterMemoryLayout
                  : kRVVIndexedGatherMemoryLayout;
    analysis.description.indexEEW =
        static_cast<std::int64_t>(analysis.slice.indexLoad.getIndexEew());
    analysis.description.offsetUnit =
        isScatter ? analysis.slice.indexedStore.getOffsetUnit()
                  : analysis.slice.indexedLoad.getOffsetUnit();
    analysis.description.indexSource = kRVVIndexSource;
    if (isScatter) {
      analysis.description.indexUniqueness =
          analysis.slice.indexedStore.getIndexUniqueness();
      analysis.description.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
      analysis.description.indexedDestinationMemoryForm =
          kRVVIndexedDestinationMemoryForm;
      analysis.description.destinationMemoryForm =
          kRVVIndexedDestinationMemoryForm;
    } else {
      analysis.description.indexedDataMemoryForm = kRVVIndexedDataMemoryForm;
      analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
    }
  }
  if (routeProfile->operation.isMaskedMemoryMovement) {
    const bool isMaskedStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::MaskedUnitStore;
    const bool isComputedMaskStridedStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
    if (isMaskedStore)
      analysis.description.indexedMemoryLayout = kRVVMaskedStoreMemoryLayout;
    else if (isComputedMaskStridedStore)
      analysis.description.indexedMemoryLayout =
          kRVVComputedMaskStridedStoreMemoryLayout;
    else if (routeProfile->operation.operation ==
             RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore)
      analysis.description.indexedMemoryLayout = kRVVComputedMaskMemoryLayout;
    else
      analysis.description.indexedMemoryLayout = kRVVMaskedMemoryLayout;
    if (isComputedMaskStridedStore) {
      analysis.description.stridedMemoryLayout =
          kRVVComputedMaskStridedStoreMemoryLayout;
      analysis.description.outStrideSource = kRVVDestinationByteStrideSource;
    }
    analysis.description.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    if (isMaskedStore)
      analysis.description.destinationMemoryForm =
          kRVVMaskedStoreDestinationMemoryForm;
    else if (isComputedMaskStridedStore)
      analysis.description.destinationMemoryForm = "strided-store";
    else
      analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
  }
  if (routeProfile->operation.isSegmentedMemoryMovement) {
    const bool isInterleave =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
    analysis.description.segmentMemoryLayout =
        isInterleave ? kRVVSegment2InterleaveMemoryLayout
                     : kRVVSegment2MemoryLayout;
    analysis.description.segmentCount =
        isInterleave
            ? static_cast<std::int64_t>(
                  analysis.slice.segment2Store.getSegmentCount())
            : static_cast<std::int64_t>(
                  analysis.slice.segment2Load.getSegmentCount());
    analysis.description.segmentTupleCType = kRVVSegment2TupleCType;
    analysis.description.segmentLoadIntrinsic =
        isInterleave ? llvm::StringRef()
                     : llvm::StringRef(kRVVSegment2LoadIntrinsic);
    analysis.description.segmentStoreIntrinsic =
        isInterleave ? llvm::StringRef(kRVVSegment2StoreIntrinsic)
                     : llvm::StringRef();
    analysis.description.segmentFieldExtractIntrinsic =
        isInterleave ? kRVVSegment2TupleCreateIntrinsic
                     : kRVVSegment2FieldExtractIntrinsic;
    if (isInterleave) {
      analysis.description.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
      analysis.description.destinationMemoryForm =
          analysis.slice.segment2Store.getDestinationMemoryForm();
      analysis.description.field0Role =
          analysis.slice.segment2Store.getField0Role();
      analysis.description.field1Role =
          analysis.slice.segment2Store.getField1Role();
      analysis.description.field0Name = "field0_vec";
      analysis.description.field1Name = "field1_vec";
      analysis.description.field0SourceMemoryForm =
          kRVVUnitStrideSourceMemoryForm;
      analysis.description.field1SourceMemoryForm =
          kRVVUnitStrideSourceMemoryForm;
    } else {
      analysis.description.sourceMemoryForm =
          analysis.slice.segment2Load.getSourceMemoryForm();
      analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
      analysis.description.field0Role =
          analysis.slice.segment2Load.getField0Role();
      analysis.description.field1Role =
          analysis.slice.segment2Load.getField1Role();
      analysis.description.field0Name = "field0_vec";
      analysis.description.field1Name = "field1_vec";
      analysis.description.field0DestinationMemoryForm =
          kRVVDestinationMemoryForm;
      analysis.description.field1DestinationMemoryForm =
          kRVVDestinationMemoryForm;
    }
  }

  if (llvm::Error error = validateRVVSelectedBodyRuntimeABIParameters(
          analysis.slice, *analysis.constructionRoute,
          routeProfile->config, analysis.slice.runtimeElementCountABI,
          analysis.slice.outABI))
    return error;
  llvm::Expected<RVVRouteOperandBindingPlan> operandBindingPlan =
      deriveRVVRouteOperandBindingPlan(analysis);
  if (!operandBindingPlan)
    return operandBindingPlan.takeError();
  analysis.routeOperandBindingPlan = std::move(*operandBindingPlan);
  if (!analysis.routeOperandBindingPlan.planID.empty()) {
    analysis.description.routeOperandBindingPlanID =
        analysis.routeOperandBindingPlan.planID;
    analysis.description.routeOperandBindingSummary =
        stringifyRVVRouteOperandBindingPlan(
            analysis.routeOperandBindingPlan);
  }
  if (llvm::Error error = verifyRVVSelectedBodyConstructionRouteMapping(
          routeProfile->operation.operationMnemonic,
          analysis.description.typedComputeOpName,
          analysis.constructionRoute->emitCRouteID,
          analysis.constructionRoute->runtimeABIName))
    return std::move(error);
  if (llvm::Error error = verifySelectedRVVRoleSequence(
          analysis.slice, request, *analysis.constructionRoute))
    return std::move(error);
  if (llvm::Error error = verifyRVVSelectedBodyEmitCRouteDescription(
          analysis.description, "selected RVV EmitC route description"))
    return std::move(error);
  return analysis;
}

llvm::ArrayRef<RVVSelectedBodyOperationKind>
getRVVSelectedBodyOperationKinds() {
  return kRVVSelectedBodyOperationKinds;
}

llvm::StringRef
stringifyRVVSelectedBodyOperationKind(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyOperationProfile(op).operationMnemonic;
}

llvm::StringRef
stringifyRVVSelectedBodyMemoryForm(RVVSelectedBodyMemoryForm form) {
  switch (form) {
  case RVVSelectedBodyMemoryForm::VectorRHSLoad:
    return "vector-rhs-load";
  case RVVSelectedBodyMemoryForm::RHSBroadcastLoad:
    return "rhs-broadcast-load";
  case RVVSelectedBodyMemoryForm::RHSScalarBroadcast:
    return "rhs-scalar-broadcast";
  case RVVSelectedBodyMemoryForm::StridedLoadStore:
    return "strided-load-store";
  case RVVSelectedBodyMemoryForm::StridedLoadUnitStore:
    return "strided-load-unit-store";
  case RVVSelectedBodyMemoryForm::UnitLoadStridedStore:
    return "unit-load-strided-store";
  case RVVSelectedBodyMemoryForm::IndexedLoadUnitStore:
    return "indexed-load-unit-store";
  case RVVSelectedBodyMemoryForm::UnitLoadIndexedStore:
    return "unit-load-indexed-store";
  case RVVSelectedBodyMemoryForm::MaskedUnitLoadStore:
    return "masked-unit-load-store";
  case RVVSelectedBodyMemoryForm::MaskedUnitStore:
    return "masked-unit-store";
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore:
    return "computed-mask-unit-load-store";
  case RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect:
    return "computed-mask-vector-select";
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore:
    return "computed-mask-unit-load-strided-store";
  case RVVSelectedBodyMemoryForm::Segment2LoadUnitStore:
    return "segment2-load-unit-store";
  case RVVSelectedBodyMemoryForm::UnitLoadSegment2Store:
    return "unit-load-segment2-store";
  case RVVSelectedBodyMemoryForm::UnitStrideConversion:
    return "unit-stride-conversion";
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideWideningDotReduce:
    return "computed-mask-unit-stride-widening-dot-reduce";
  case RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce:
    return "strided-input-widening-dot-reduce";
  case RVVSelectedBodyMemoryForm::ComputedMaskStridedInputWideningDotReduce:
    return "computed-mask-strided-input-widening-dot-reduce";
  case RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction:
    return "unit-stride-standalone-reduction";
  }
  llvm_unreachable("unknown RVV selected-body memory form");
}

llvm::StringRef
getRVVSelectedBodyEmitCRouteID(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyConstructionRouteOrDie(op).emitCRouteID;
}

llvm::StringRef getRVVSelectedBodyEmissionKind() {
  return kRVVSelectedBodyEmissionKind;
}

llvm::StringRef getRVVSelectedBodyLoweringBoundaryOpName() {
  return kRVVSelectedBodyLoweringBoundaryOpName;
}

llvm::StringRef getRVVSelectedBodyRuntimeABIKind() {
  return kRVVSelectedBodyRuntimeABIKind;
}

llvm::StringRef
getRVVSelectedBodyRuntimeABIName(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyConstructionRouteOrDie(op).runtimeABIName;
}

llvm::StringRef getRVVSelectedBodyRuntimeGlueRole() {
  return kRVVSelectedBodyRuntimeGlueRole;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters() {
  return tcrv::rvv::getRVVSelectedBodyRuntimeABIParameters();
}

RVVSelectedBodyConstructionMetadataFacts
getRVVSelectedBodyConstructionMetadataFacts(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  RVVSelectedBodyConstructionMetadataFacts facts;
  facts.operationMnemonic =
      stringifyRVVSelectedBodyOperationKind(description.operation);
  facts.typedComputeOpName = description.typedComputeOpName;
  facts.emitCRouteID = description.emitCRouteID;
  facts.targetArtifactRouteID = description.targetArtifactRouteID;
  facts.targetArtifactKind = description.targetArtifactKind;
  facts.runtimeABIName = description.runtimeABIName;
  facts.runtimeABIContractName = description.runtimeABIContractName;
  facts.runtimeABIParameters = description.runtimeABIParameters;
  return facts;
}

llvm::Error verifyRVVSelectedBodyEmitCRouteDescription(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (context.trim().empty())
    return makeRVVEmitCRouteProviderError(
        "selected-body route description verification requires a non-empty "
        "context");

  llvm::Expected<RVVSelectedBodyRouteProfile> profile =
      deriveRVVSelectedBodyRouteProfile(description);
  if (!profile)
    return profile.takeError();
  const RVVSelectedBodyOperationProfile &operationProfile =
      profile->operation;
  const RVVSelectedBodyConfigProfile &configProfile = profile->config;
  const RVVSelectedBodyTargetLeafProfile &targetLeaves =
      profile->targetLeaves;
  const tcrv::rvv::RVVSelectedBodyConfigVLContract &configContract =
      *configProfile.configContract;
  const bool isComputedMaskWideningDotReduce =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
  const bool isComputedMaskStridedInputWideningDotReduce =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::
          ComputedMaskStridedInputWideningDotReduceAdd;
  const bool isComputedMaskSelect =
      operationProfile.operation == RVVSelectedBodyOperationKind::ComputedMaskSelect;
  const bool isMaskedUnitStore =
      operationProfile.operation == RVVSelectedBodyOperationKind::MaskedUnitStore;
  const bool isStridedInputWideningDotReduce =
      operationProfile.operation ==
      RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd;
  const bool isContractionRoute =
      isRVVSelectedBodyContractionRouteOperation(operationProfile.operation);
  const bool isScalarBroadcastElementwiseRoute =
      isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
          operationProfile.operation);
  const bool isStandaloneReductionRoute =
      isRVVSelectedBodyStandaloneReductionRouteOperation(
          operationProfile.operation);

  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          operationProfile.operationMnemonic);
  if (!route)
    return route.takeError();
  const RVVSelectedBodyConstructionRoute &constructionRoute = **route;

  const bool usesGenericBinary =
      description.typedComputeOpName == "tcrv_rvv.binary";
  if (usesGenericBinary && operationProfile.isCompareSelect)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " compare/select cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && operationProfile.isReduction)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && operationProfile.isMaskedArithmetic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " masked arithmetic cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && operationProfile.isMultiplyAccumulate)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " multiply-accumulate cannot use generic tcrv_rvv.binary");
  if (!usesGenericBinary)
    if (llvm::Error error = requireRouteDescriptionField(
            context, "typed compute op", description.typedComputeOpName,
            constructionRoute.typedComputeOpName))
      return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC route id", description.emitCRouteID,
          constructionRoute.emitCRouteID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "target artifact route id",
          description.targetArtifactRouteID,
          getRVVSelectedBodyTargetArtifactRouteID()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "target artifact kind", description.targetArtifactKind,
          getRVVSelectedBodyTargetArtifactKind()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI name", description.runtimeABIName,
          constructionRoute.runtimeABIName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI contract", description.runtimeABIContractName,
          constructionRoute.runtimeABIContractName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "lowering boundary op", description.boundaryOpName,
          kRVVSelectedBodyLoweringBoundaryOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "config contract", description.configContractID,
          configContract.configContractID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime VL contract", description.runtimeVLContractID,
          configContract.runtimeVLContractID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime AVL source", description.runtimeAVLASource,
          configContract.runtimeAVLASource))
    return error;
  if (isContractionRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            kRVVContractionTargetLeafProfile))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            kRVVContractionProviderSupportedMirror))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            kRVVContractionRequiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            kRVVContractionCTypeMappingSummary))
      return error;
  } else if (isScalarBroadcastElementwiseRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            kRVVScalarBroadcastElementwiseTargetLeafProfile))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            kRVVScalarBroadcastElementwiseProviderSupportedMirror))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            kRVVScalarBroadcastElementwiseRequiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            kRVVScalarBroadcastElementwiseCTypeMappingSummary))
      return error;
  } else if (isStandaloneReductionRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            kRVVStandaloneReductionTargetLeafProfile))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            kRVVStandaloneReductionProviderSupportedMirror))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            kRVVStandaloneReductionRequiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            kRVVStandaloneReductionCTypeMappingSummary))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            ""))
      return error;
  }
  llvm::StringRef expectedRuntimeABIOrder = configContract.runtimeABIOrder;
  if (isComputedMaskStridedInputWideningDotReduce) {
    expectedRuntimeABIOrder =
        kRVVComputedMaskStridedInputWideningDotProductRuntimeABIOrder;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout,
            kRVVComputedMaskStridedInputWideningDotMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVStridedInputDotSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
  } else if (isStridedInputWideningDotReduce) {
    expectedRuntimeABIOrder = kRVVStridedInputWideningDotProductRuntimeABIOrder;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout,
            kRVVStridedInputWideningDotMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVStridedInputDotSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for strided-input dot-reduction routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.operation ==
             RVVSelectedBodyOperationKind::StridedAdd) {
    expectedRuntimeABIOrder = kRVVStridedRuntimeABIOrder;
  } else if (operationProfile.isSegmentedMemoryMovement) {
    expectedRuntimeABIOrder =
        operationProfile.operation ==
                RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad
            ? kRVVSegment2InterleaveRuntimeABIOrder
            : kRVVSegment2RuntimeABIOrder;
  } else if (operationProfile.isMemoryMovement) {
    expectedRuntimeABIOrder =
        operationProfile.operation ==
                RVVSelectedBodyOperationKind::UnitLoadStridedStore
            ? kRVVUnitLoadStridedStoreRuntimeABIOrder
            : kRVVStridedLoadUnitStoreRuntimeABIOrder;
  } else if (isComputedMaskSelect) {
    expectedRuntimeABIOrder = kRVVComputedMaskSelectRuntimeABIOrder;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            kRVVComputedMaskSelectMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for computed-mask select routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isMaskedMemoryMovement) {
    expectedRuntimeABIOrder =
        description.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore
            ? kRVVMaskedStoreRuntimeABIOrder
        : description.memoryForm ==
                RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore
            ? kRVVComputedMaskStridedStoreRuntimeABIOrder
            : (description.memoryForm ==
                       RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore
                   ? kRVVComputedMaskMemoryRuntimeABIOrder
                   : kRVVMaskedMemoryRuntimeABIOrder);
  } else if (operationProfile.isIndexedMemoryMovement) {
    expectedRuntimeABIOrder =
        operationProfile.operation ==
                RVVSelectedBodyOperationKind::IndexedScatterUnitLoad
            ? kRVVIndexedScatterRuntimeABIOrder
            : kRVVIndexedGatherRuntimeABIOrder;
  } else if (operationProfile.isWideningConversion) {
    expectedRuntimeABIOrder = kRVVWideningConversionRuntimeABIOrder;
  } else if (operationProfile.operation == RVVSelectedBodyOperationKind::MAccAdd) {
    expectedRuntimeABIOrder = kRVVMAccRuntimeABIOrder;
  } else if (operationProfile.operation ==
             RVVSelectedBodyOperationKind::WideningMAccAdd) {
    expectedRuntimeABIOrder = kRVVWideningMAccRuntimeABIOrder;
  } else if (operationProfile.operation ==
             RVVSelectedBodyOperationKind::WideningDotReduceAdd) {
    expectedRuntimeABIOrder = kRVVWideningDotProductRuntimeABIOrder;
  } else if (isStridedInputWideningDotReduce) {
    expectedRuntimeABIOrder = kRVVStridedInputWideningDotProductRuntimeABIOrder;
  } else if (isComputedMaskWideningDotReduce) {
    expectedRuntimeABIOrder = kRVVComputedMaskWideningDotProductRuntimeABIOrder;
  } else if (isComputedMaskStridedInputWideningDotReduce) {
    expectedRuntimeABIOrder =
        kRVVComputedMaskStridedInputWideningDotProductRuntimeABIOrder;
  } else if (description.memoryForm ==
             RVVSelectedBodyMemoryForm::RHSScalarBroadcast) {
    expectedRuntimeABIOrder = kRVVScalarBroadcastRuntimeABIOrder;
  } else if (isStandaloneReductionRoute) {
    expectedRuntimeABIOrder = kRVVStandaloneReductionRuntimeABIOrder;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI order", description.runtimeABIOrder,
          expectedRuntimeABIOrder))
    return error;
  llvm::StringRef expectedOperandBindingPlanID;
  if (operationProfile.operation == RVVSelectedBodyOperationKind::Add)
    expectedOperandBindingPlanID = kRVVAddOperandBindingPlanID;
  else if (operationProfile.operation == RVVSelectedBodyOperationKind::Sub)
    expectedOperandBindingPlanID = kRVVSubOperandBindingPlanID;
  else if (operationProfile.operation == RVVSelectedBodyOperationKind::Mul)
    expectedOperandBindingPlanID = kRVVMulOperandBindingPlanID;
  else if (operationProfile.operation ==
           RVVSelectedBodyOperationKind::CmpSelect)
    expectedOperandBindingPlanID = kRVVCmpSelectOperandBindingPlanID;
  else if (operationProfile.operation ==
           RVVSelectedBodyOperationKind::ComputedMaskSelect)
    expectedOperandBindingPlanID =
        kRVVComputedMaskSelectOperandBindingPlanID;
  else if (operationProfile.operation ==
           RVVSelectedBodyOperationKind::ReduceAdd)
    expectedOperandBindingPlanID = kRVVReduceAddOperandBindingPlanID;
  else if (operationProfile.operation ==
           RVVSelectedBodyOperationKind::MaskedAdd)
    expectedOperandBindingPlanID = kRVVMaskedAddOperandBindingPlanID;
  else if (operationProfile.operation ==
           RVVSelectedBodyOperationKind::StridedAdd)
    expectedOperandBindingPlanID = kRVVStridedAddOperandBindingPlanID;
  else if (operationProfile.operation == RVVSelectedBodyOperationKind::MAccAdd)
    expectedOperandBindingPlanID = kRVVMAccOperandBindingPlanID;
  else if (operationProfile.operation ==
           RVVSelectedBodyOperationKind::StridedLoadUnitStore)
    expectedOperandBindingPlanID =
        kRVVStridedLoadUnitStoreOperandBindingPlanID;
  else if (operationProfile.operation ==
           RVVSelectedBodyOperationKind::UnitLoadStridedStore)
    expectedOperandBindingPlanID =
        kRVVUnitLoadStridedStoreOperandBindingPlanID;
  else if (operationProfile.operation ==
           RVVSelectedBodyOperationKind::ScalarBroadcastAdd)
    expectedOperandBindingPlanID = kRVVScalarBroadcastOperandBindingPlanID;
  else if (operationProfile.operation ==
           RVVSelectedBodyOperationKind::StandaloneReduceAdd)
    expectedOperandBindingPlanID = kRVVStandaloneReductionOperandBindingPlanID;
  else if (operationProfile.operation ==
           RVVSelectedBodyOperationKind::MaskedUnitStore)
    expectedOperandBindingPlanID = kRVVMaskedUnitStoreOperandBindingPlanID;
  else if (operationProfile.operation ==
           RVVSelectedBodyOperationKind::ComputedMaskStridedStore)
    expectedOperandBindingPlanID =
        kRVVComputedMaskStridedStoreOperandBindingPlanID;
  if (!expectedOperandBindingPlanID.empty()) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "route operand ABI binding plan",
            description.routeOperandBindingPlanID,
            expectedOperandBindingPlanID))
      return error;
    if (description.routeOperandBindingSummary.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " requires a non-empty route operand ABI binding mirror summary");
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "route operand ABI binding plan",
            description.routeOperandBindingPlanID, ""))
      return error;
    if (!description.routeOperandBindingSummary.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " must not carry a route operand ABI binding summary for this "
          "unconverted route");
  }
  if (isScalarBroadcastElementwiseRoute || isStandaloneReductionRoute)
    if (llvm::Error error = requireRouteDescriptionField(
            context, "runtime control plan", description.runtimeControlPlanID,
            getRVVRuntimeAVLVLControlPlanID()))
      return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL def op", description.vlDefOpName,
          configContract.vlDefOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL scope op", description.vlScopeOpName,
          configContract.vlScopeOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL uses", description.vlUses, configContract.vlUses))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop kind", description.emitCLoopKind,
          configContract.emitCLoopKind))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop induction", description.emitCLoopInductionName,
          configContract.emitCLoopInductionName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC full-chunk VL", description.emitCFullChunkVLName,
          configContract.emitCFullChunkVLName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop VL", description.emitCLoopVLName,
          tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "remaining AVL metadata",
          description.remainingAVLMetadata, configContract.remainingAVLMetadata))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "pointer advance metadata",
          description.pointerAdvanceMetadata,
          configContract.pointerAdvanceMetadata))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "bounded slice", description.boundedSlice,
          configContract.boundedSlice))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "multi-VL support", description.multiVL,
          configContract.multiVL))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL C type", description.vlCType, configProfile.vlCType))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector type", description.vectorTypeName,
          configProfile.vectorTypeName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector C type", description.vectorCType,
          configProfile.vectorCType))
    return error;
  if (operationProfile.isIndexedMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector type", description.indexVectorTypeName,
            configProfile.indexVectorTypeName))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector C type", description.indexVectorCType,
            configProfile.indexVectorCType))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector type", description.indexVectorTypeName, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index vector C type", description.indexVectorCType, ""))
      return error;
  }
  if (operationProfile.operation ==
          RVVSelectedBodyOperationKind::WideningMAccAdd ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
      isStridedInputWideningDotReduce ||
      isComputedMaskWideningDotReduce ||
      isComputedMaskStridedInputWideningDotReduce) {
    if (description.sourceSEW != tcrv::rvv::getRVVSEW16Bits())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " source SEW must be provider-derived from typed source vectors "
          "for widening mixed-width RVV routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source LMUL", description.sourceLMUL,
            tcrv::rvv::getRVVLMULMF2()))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector type", description.sourceVectorTypeName,
            "!tcrv_rvv.vector<i16, \"mf2\">"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector C type", description.sourceVectorCType,
            "vint16mf2_t"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector-load intrinsic",
            description.sourceVectorLoadIntrinsic,
            "__riscv_vle16_v_i16mf2"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "conversion relation", description.conversionRelation,
            ""))
      return error;
  } else if (operationProfile.isWideningConversion) {
    const bool isI16ToI32 =
        operationProfile.operation == RVVSelectedBodyOperationKind::WidenI16ToI32;
    std::int64_t expectedSourceSEW =
        isI16ToI32 ? tcrv::rvv::getRVVSEW16Bits()
                   : tcrv::rvv::getRVVFirstSliceSEWBits();
    llvm::StringRef expectedSourceLMUL =
        isI16ToI32 ? tcrv::rvv::getRVVLMULMF2()
                   : tcrv::rvv::getRVVLMULM1();
    llvm::StringRef expectedSourceVectorType =
        isI16ToI32 ? "!tcrv_rvv.vector<i16, \"mf2\">"
                   : "!tcrv_rvv.vector<i32, \"m1\">";
    llvm::StringRef expectedSourceVectorCType =
        isI16ToI32 ? "vint16mf2_t" : "vint32m1_t";
    llvm::StringRef expectedSourceLoadIntrinsic =
        isI16ToI32 ? "__riscv_vle16_v_i16mf2"
                   : "__riscv_vle32_v_i32m1";
    llvm::StringRef expectedConversionRelation =
        isI16ToI32 ? kRVVWidenI16ToI32ConversionRelation
                   : kRVVWideningConversionRelation;
    if (description.sourceSEW != expectedSourceSEW)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " source SEW must be provider-derived from typed source vector for "
          "widening conversion");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source LMUL", description.sourceLMUL,
            expectedSourceLMUL))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector type", description.sourceVectorTypeName,
            expectedSourceVectorType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector C type", description.sourceVectorCType,
            expectedSourceVectorCType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector-load intrinsic",
            description.sourceVectorLoadIntrinsic,
            expectedSourceLoadIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "conversion relation", description.conversionRelation,
            expectedConversionRelation))
      return error;
  } else {
    if (description.sourceSEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " source SEW must be empty for non-conversion routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source LMUL", description.sourceLMUL, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector type", description.sourceVectorTypeName,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector C type", description.sourceVectorCType,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source vector-load intrinsic",
            description.sourceVectorLoadIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "conversion relation", description.conversionRelation,
            ""))
      return error;
  }
  if (operationProfile.isCompareSelect || operationProfile.isMaskedArithmetic ||
      operationProfile.isMaskedMemoryMovement ||
      isComputedMaskWideningDotReduce ||
      isComputedMaskStridedInputWideningDotReduce) {
    if (llvm::Error error =
            requireRouteDescriptionField(context, "mask type",
                                         description.maskTypeName,
                                         configProfile.maskTypeName))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "mask C type",
                                         description.maskCType,
                                         configProfile.maskCType))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask type", description.maskTypeName, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask C type", description.maskCType, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "setvl intrinsic", description.setVLIntrinsic,
          configProfile.setVLIntrinsic))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector-load intrinsic", description.vectorLoadIntrinsic,
          configProfile.vectorLoadIntrinsic))
    return error;
  if (operationProfile.isIndexedMemoryMovement) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-load intrinsic", description.indexLoadIntrinsic,
            configProfile.indexLoadIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-scale intrinsic", description.indexScaleIntrinsic,
            configProfile.indexScaleIntrinsic))
      return error;
    if (operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-load intrinsic",
              description.indexedLoadIntrinsic,
              configProfile.indexedLoadIntrinsic))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-store intrinsic",
              description.indexedStoreIntrinsic, ""))
        return error;
    } else {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-load intrinsic",
              description.indexedLoadIntrinsic, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "indexed-store intrinsic",
              description.indexedStoreIntrinsic,
              configProfile.indexedStoreIntrinsic))
        return error;
    }
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-load intrinsic", description.indexLoadIntrinsic,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index-scale intrinsic", description.indexScaleIntrinsic,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed-load intrinsic",
            description.indexedLoadIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed-store intrinsic",
            description.indexedStoreIntrinsic, ""))
      return error;
  }
  if (operationProfile.operation == RVVSelectedBodyOperationKind::StridedAdd ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::StridedLoadUnitStore ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedStore) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided-load intrinsic",
            description.stridedLoadIntrinsic,
            configProfile.stridedLoadIntrinsic))
      return error;
  } else if (isStridedInputWideningDotReduce ||
             isComputedMaskStridedInputWideningDotReduce) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided-load intrinsic",
            description.stridedLoadIntrinsic, "__riscv_vlse16_v_i16mf2"))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided-load intrinsic",
            description.stridedLoadIntrinsic, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "store intrinsic", description.storeIntrinsic,
          isMaskedUnitStore ? llvm::StringRef(kRVVMaskedStoreIntrinsic)
                            : configProfile.storeIntrinsic))
    return error;
  if (operationProfile.operation == RVVSelectedBodyOperationKind::StridedAdd ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::UnitLoadStridedStore ||
      operationProfile.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedStore) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided-store intrinsic",
            description.stridedStoreIntrinsic,
            configProfile.stridedStoreIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided-store intrinsic",
            description.stridedStoreIntrinsic, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "compute intrinsic", description.intrinsic,
          targetLeaves.intrinsic))
    return error;
  if (operationProfile.isCompareSelect) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic,
            targetLeaves.compareIntrinsic))
      return error;
  } else if (operationProfile.isMaskedArithmetic ||
             operationProfile.isMaskedMemoryMovement ||
             isComputedMaskWideningDotReduce ||
             isComputedMaskStridedInputWideningDotReduce) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic,
            targetLeaves.compareIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic, ""))
      return error;
  }
  if (operationProfile.isMaskedArithmetic ||
      operationProfile.isMaskedMemoryMovement ||
      isComputedMaskWideningDotReduce ||
      isComputedMaskStridedInputWideningDotReduce) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked merge intrinsic",
            description.maskedMergeIntrinsic,
            isMaskedUnitStore ? llvm::StringRef()
                              : targetLeaves.maskedMergeIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked merge intrinsic",
            description.maskedMergeIntrinsic, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "result value name", description.resultName,
          operationProfile.resultName))
    return error;
  if (operationProfile.isCompareSelect || operationProfile.isMaskedArithmetic ||
      operationProfile.isMaskedMemoryMovement ||
      isComputedMaskWideningDotReduce ||
      isComputedMaskStridedInputWideningDotReduce) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask value name", description.maskName,
            operationProfile.maskName))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask value name", description.maskName, ""))
      return error;
  }
  if (operationProfile.isMaskedArithmetic || isComputedMaskSelect) {
    llvm::StringRef expectedMaskMemoryForm =
        isComputedMaskSelect
            ? llvm::StringRef(kRVVComputedMaskMemoryMaskMemoryForm)
            : llvm::StringRef();
    llvm::StringRef expectedInactiveLaneContract =
        isComputedMaskSelect
            ? llvm::StringRef()
            : llvm::StringRef(kRVVMaskedInactiveLaneContract);
    llvm::StringRef expectedMaskedPassthroughLayout =
        isComputedMaskSelect ? llvm::StringRef()
                             : llvm::StringRef(kRVVMaskedPassthroughLayout);
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask role", description.maskRole,
            kRVVMaskedPredicateMaskRole))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource,
            kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask memory form", description.maskMemoryForm,
            expectedMaskMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive lane contract",
            description.inactiveLaneContract, expectedInactiveLaneContract))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout,
            expectedMaskedPassthroughLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive-lane zeroing requirement",
            description.inactiveLaneZeroingRequirement, ""))
      return error;
  } else if (isComputedMaskWideningDotReduce ||
             isComputedMaskStridedInputWideningDotReduce) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask role", description.maskRole,
            kRVVMaskedPredicateMaskRole))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource,
            kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask memory form", description.maskMemoryForm,
            kRVVComputedMaskMemoryMaskMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive lane contract",
            description.inactiveLaneContract, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive-lane zeroing requirement",
            description.inactiveLaneZeroingRequirement,
            kRVVContractionMaskedInactiveLaneZeroingRequirement))
      return error;
  } else if (operationProfile.isMaskedMemoryMovement) {
    const bool isComputedMask =
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
    const bool isMaskedStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::MaskedUnitStore;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask role", description.maskRole,
            isComputedMask ? kRVVMaskedPredicateMaskRole : kRVVMaskRole))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource,
            isComputedMask ? kRVVMaskedCompareMaskSource : kRVVMaskSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask memory form", description.maskMemoryForm,
            isComputedMask ? kRVVComputedMaskMemoryMaskMemoryForm
                           : kRVVMaskMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive lane contract",
            description.inactiveLaneContract,
            isMaskedStore ? kRVVMaskedStoreInactiveLaneContract
                          : kRVVMaskedMemoryInactiveLaneContract))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout,
            isMaskedStore ? kRVVMaskedStorePassthroughLayout
                          : kRVVMaskedMemoryPassthroughLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive-lane zeroing requirement",
            description.inactiveLaneZeroingRequirement, ""))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask role", description.maskRole, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask memory form", description.maskMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive lane contract",
            description.inactiveLaneContract, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "inactive-lane zeroing requirement",
            description.inactiveLaneZeroingRequirement, ""))
      return error;
  }
  if (operationProfile.operation == RVVSelectedBodyOperationKind::ReduceAdd) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction accumulator layout",
            description.reductionAccumulatorLayout,
            kRVVReductionAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction result layout",
            description.reductionResultLayout, kRVVReductionResultLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction store VL", description.reductionStoreVL,
            kRVVReductionStoreVL))
      return error;
  } else if (isStandaloneReductionRoute) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction accumulator layout",
            description.reductionAccumulatorLayout,
            kRVVStandaloneReductionAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction result layout",
            description.reductionResultLayout,
            kRVVStandaloneReductionResultLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction store VL", description.reductionStoreVL,
            kRVVStandaloneReductionStoreVL))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction accumulator layout",
            description.reductionAccumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction result layout",
            description.reductionResultLayout, ""))
      return error;
    if (operationProfile.operation !=
            RVVSelectedBodyOperationKind::WideningDotReduceAdd &&
        !isStridedInputWideningDotReduce &&
        !isComputedMaskWideningDotReduce &&
        !isComputedMaskStridedInputWideningDotReduce)
      if (llvm::Error error = requireRouteDescriptionField(
              context, "reduction store VL", description.reductionStoreVL, ""))
        return error;
  }
  if (operationProfile.operation == RVVSelectedBodyOperationKind::MAccAdd) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "multiply-accumulate accumulator layout",
            description.maccAccumulatorLayout, kRVVMAccAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "multiply-accumulate result layout",
            description.maccResultLayout, kRVVMAccResultLayout))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "multiply-accumulate accumulator layout",
            description.maccAccumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "multiply-accumulate result layout",
            description.maccResultLayout, ""))
      return error;
  }
  if (operationProfile.operation == RVVSelectedBodyOperationKind::WideningMAccAdd) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening multiply-accumulate accumulator layout",
            description.wideningMAccAccumulatorLayout,
            kRVVWideningMAccAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening multiply-accumulate result layout",
            description.wideningMAccResultLayout,
            kRVVWideningMAccResultLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening multiply-accumulate relation",
            description.wideningMAccRelation, kRVVWideningMAccRelation))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening multiply-accumulate accumulator layout",
            description.wideningMAccAccumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening multiply-accumulate result layout",
            description.wideningMAccResultLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening multiply-accumulate relation",
            description.wideningMAccRelation, ""))
      return error;
  }
  if (operationProfile.operation ==
          RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
      isStridedInputWideningDotReduce ||
      isComputedMaskWideningDotReduce ||
      isComputedMaskStridedInputWideningDotReduce) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening dot-product accumulator layout",
            description.wideningDotProductAccumulatorLayout,
            kRVVWideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening dot-product result layout",
            description.wideningDotProductResultLayout,
            kRVVWideningDotProductResultLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening dot-product relation",
            description.wideningDotProductRelation,
            kRVVWideningDotProductRelation))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening product intrinsic",
            description.wideningProductIntrinsic,
            "__riscv_vwmul_vv_i32m1"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked widening product intrinsic",
            description.maskedWideningProductIntrinsic,
            (isComputedMaskWideningDotReduce ||
             isComputedMaskStridedInputWideningDotReduce)
                ? "__riscv_vwmul_vv_i32m1_m"
                : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "scalar seed splat intrinsic",
            description.scalarSeedSplatIntrinsic,
            configProfile.rhsBroadcastIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening dot-product reduction store VL",
            description.reductionStoreVL, kRVVWideningDotProductStoreVL))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening dot-product accumulator layout",
            description.wideningDotProductAccumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening dot-product result layout",
            description.wideningDotProductResultLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening dot-product relation",
            description.wideningDotProductRelation, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "widening product intrinsic",
            description.wideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked widening product intrinsic",
            description.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "scalar seed splat intrinsic",
            description.scalarSeedSplatIntrinsic,
            isStandaloneReductionRoute ? configProfile.rhsBroadcastIntrinsic
                                       : ""))
      return error;
  }
  if (isComputedMaskStridedInputWideningDotReduce) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            kRVVComputedMaskStridedInputWideningDotMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVStridedInputDotSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for computed-mask strided-input "
          "dot-reduction routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isStridedInputWideningDotReduce) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            kRVVStridedInputWideningDotMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVStridedInputDotSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for strided-input dot-reduction routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.operation ==
             RVVSelectedBodyOperationKind::StridedAdd) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout, kRVVStridedMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource,
            kRVVOutStrideSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm, ""))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm, ""))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for non-indexed routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (isComputedMaskSelect) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            kRVVComputedMaskSelectMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for computed-mask select routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isMemoryMovement) {
    const bool isUnitLoadStridedStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::UnitLoadStridedStore;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout,
            isUnitLoadStridedStore ? kRVVUnitLoadStridedStoreMemoryLayout
                                   : kRVVStridedLoadUnitStoreMemoryLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (isUnitLoadStridedStore) {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "out stride source", description.outStrideSource,
              kRVVDestinationByteStrideSource))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "source stride source", description.sourceStrideSource,
              ""))
        return error;
    } else {
      if (llvm::Error error = requireRouteDescriptionField(
              context, "out stride source", description.outStrideSource, ""))
        return error;
      if (llvm::Error error = requireRouteDescriptionField(
              context, "source stride source", description.sourceStrideSource,
              kRVVSourceStrideSource))
        return error;
    }
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            isUnitLoadStridedStore ? kRVVUnitStrideSourceMemoryForm
                                   : kRVVSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm,
            isUnitLoadStridedStore ? "strided-store"
                                   : kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for non-indexed routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isSegmentedMemoryMovement) {
    const bool isSegmentInterleave =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            isSegmentInterleave ? kRVVUnitStrideSourceMemoryForm
                                : kRVVSegment2SourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         isSegmentInterleave
                                             ? kRVVSegment2InterleavedDestinationMemoryForm
                                             : kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for segment2 memory routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isMaskedMemoryMovement) {
    const bool isComputedMask =
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
        operationProfile.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
    const bool isMaskedStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::MaskedUnitStore;
    const bool isComputedMaskStridedStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout", description.stridedMemoryLayout,
            isComputedMaskStridedStore
                ? kRVVComputedMaskStridedStoreMemoryLayout
                : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            isMaskedStore
                ? kRVVMaskedStoreMemoryLayout
            : isComputedMaskStridedStore
                ? kRVVComputedMaskStridedStoreMemoryLayout
                : (isComputedMask ? kRVVComputedMaskMemoryLayout
                                  : kRVVMaskedMemoryLayout)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource,
            isComputedMaskStridedStore ? kRVVDestinationByteStrideSource
                                       : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm,
                                         isMaskedStore
                                             ? kRVVMaskedStoreDestinationMemoryForm
                                         : isComputedMaskStridedStore
                                             ? "strided-store"
                                             : kRVVDestinationMemoryForm))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for masked memory routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  } else if (operationProfile.isIndexedMemoryMovement) {
    const bool isScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            isScatter ? kRVVIndexedScatterMemoryLayout
                      : kRVVIndexedGatherMemoryLayout))
      return error;
    if (description.indexEEW != 32)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be provider-derived as 32 for indexed memory");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit,
            kRVVIndexedGatherOffsetUnit))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource,
            kRVVIndexSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index uniqueness", description.indexUniqueness,
            isScatter ? kRVVIndexedScatterIndexUniqueness : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm,
            isScatter ? "" : kRVVIndexedDataMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm,
            isScatter ? kRVVIndexedDestinationMemoryForm : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            isScatter ? kRVVUnitStrideSourceMemoryForm : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm,
            isScatter ? kRVVIndexedDestinationMemoryForm
                      : kRVVDestinationMemoryForm))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed memory layout", description.indexedMemoryLayout,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "lhs stride source", description.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "rhs stride source", description.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "out stride source", description.outStrideSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source stride source", description.sourceStrideSource,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "source memory form", description.sourceMemoryForm, ""))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "destination memory form",
                                         description.destinationMemoryForm, ""))
      return error;
    if (description.indexEEW != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " index EEW must be empty for non-indexed routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "offset unit", description.offsetUnit, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "index source", description.indexSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed data memory form",
            description.indexedDataMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "indexed destination memory form",
            description.indexedDestinationMemoryForm, ""))
      return error;
  }
  if (operationProfile.isSegmentedMemoryMovement) {
    const bool isSegmentInterleave =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment memory layout",
            description.segmentMemoryLayout,
            isSegmentInterleave ? kRVVSegment2InterleaveMemoryLayout
                                : kRVVSegment2MemoryLayout))
      return error;
    if (description.segmentCount != 2)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment count must be provider-derived as 2 for segment2 "
          "memory");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment tuple C type", description.segmentTupleCType,
            kRVVSegment2TupleCType))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment-load intrinsic",
            description.segmentLoadIntrinsic,
            isSegmentInterleave ? "" : kRVVSegment2LoadIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment-store intrinsic",
            description.segmentStoreIntrinsic,
            isSegmentInterleave ? kRVVSegment2StoreIntrinsic : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment field extract intrinsic",
            description.segmentFieldExtractIntrinsic,
            isSegmentInterleave ? kRVVSegment2TupleCreateIntrinsic
                                : kRVVSegment2FieldExtractIntrinsic))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 role", description.field0Role,
            isSegmentInterleave ? kRVVSegment2Field0InputRole
                                : kRVVSegment2Field0Role))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 role", description.field1Role,
            isSegmentInterleave ? kRVVSegment2Field1InputRole
                                : kRVVSegment2Field1Role))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 result name", description.field0Name,
            "field0_vec"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 result name", description.field1Name,
            "field1_vec"))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 source memory form",
            description.field0SourceMemoryForm,
            isSegmentInterleave ? kRVVUnitStrideSourceMemoryForm : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 source memory form",
            description.field1SourceMemoryForm,
            isSegmentInterleave ? kRVVUnitStrideSourceMemoryForm : ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 destination memory form",
            description.field0DestinationMemoryForm,
            isSegmentInterleave ? "" : kRVVDestinationMemoryForm))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 destination memory form",
            description.field1DestinationMemoryForm,
            isSegmentInterleave ? "" : kRVVDestinationMemoryForm))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment memory layout",
            description.segmentMemoryLayout, ""))
      return error;
    if (description.segmentCount != 0)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment count must be empty for non-segmented routes");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment tuple C type", description.segmentTupleCType,
            ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment-load intrinsic",
            description.segmentLoadIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment-store intrinsic",
            description.segmentStoreIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "segment field extract intrinsic",
            description.segmentFieldExtractIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 role", description.field0Role, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 role", description.field1Role, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 result name", description.field0Name, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 result name", description.field1Name, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 source memory form",
            description.field0SourceMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 source memory form",
            description.field1SourceMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field0 destination memory form",
            description.field0DestinationMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "field1 destination memory form",
            description.field1DestinationMemoryForm, ""))
      return error;
  }
  if (description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad ||
      description.memoryForm == RVVSelectedBodyMemoryForm::RHSScalarBroadcast)
    if (llvm::Error error = requireRouteDescriptionText(
            context, "RHS broadcast intrinsic",
            description.rhsBroadcastIntrinsic))
      return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "RHS broadcast intrinsic",
          description.rhsBroadcastIntrinsic,
          targetLeaves.rhsBroadcastIntrinsic))
    return error;

  if (llvm::Error error = verifyRVVSelectedBodyConstructionRuntimeABIParameters(
          description.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));

  return llvm::Error::success();
}

llvm::SmallVector<support::ArtifactMetadataEntry, 16>
getRVVSelectedBodyConfigArtifactMetadata(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> metadata;
  metadata.push_back(
      {"tcrv_rvv.config_contract", description.configContractID});
  metadata.push_back({"tcrv_rvv.sew", llvm::Twine(description.sew).str()});
  metadata.push_back({"tcrv_rvv.lmul", description.lmul});
  metadata.push_back({"tcrv_rvv.tail_policy", description.tailPolicy});
  metadata.push_back({"tcrv_rvv.mask_policy", description.maskPolicy});
  if (!description.runtimeControlPlanID.empty())
    metadata.push_back(
        {"tcrv_rvv.runtime_control_plan", description.runtimeControlPlanID});
  metadata.push_back({"tcrv_rvv.memory_form",
                      stringifyRVVSelectedBodyMemoryForm(
                          description.memoryForm)});
  metadata.push_back(
      {"tcrv_rvv.runtime_vl_contract", description.runtimeVLContractID});
  metadata.push_back(
      {"tcrv_rvv.runtime_avl_source", description.runtimeAVLASource});
  metadata.push_back({"tcrv_rvv.vl_def", description.vlDefOpName});
  metadata.push_back({"tcrv_rvv.vl_scope", description.vlScopeOpName});
  metadata.push_back({"tcrv_rvv.vl_uses", description.vlUses});
  metadata.push_back(
      {"tcrv_rvv.runtime_abi_order", description.runtimeABIOrder});
  metadata.push_back({"tcrv_rvv.runtime_avl_abi_parameter",
                      tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName()});
  if (!description.routeOperandBindingPlanID.empty()) {
    metadata.push_back({"tcrv_rvv.route_operand_binding_plan",
                        description.routeOperandBindingPlanID});
    metadata.push_back({"tcrv_rvv.route_operand_binding_operands",
                        description.routeOperandBindingSummary});
  }
  metadata.push_back({"tcrv_rvv.emitc_loop", description.emitCLoopKind});
  metadata.push_back(
      {"tcrv_rvv.loop_induction", description.emitCLoopInductionName});
  metadata.push_back({"tcrv_rvv.loop_step", description.emitCFullChunkVLName});
  metadata.push_back(
      {"tcrv_rvv.remaining_avl", description.remainingAVLMetadata});
  metadata.push_back(
      {"tcrv_rvv.pointer_advance", description.pointerAdvanceMetadata});
  metadata.push_back({"tcrv_rvv.bounded_slice", description.boundedSlice});
  metadata.push_back({"tcrv_rvv.multi_vl", description.multiVL});
  if (isRVVSelectedBodyContractionRouteOperation(description.operation) ||
      isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
          description.operation) ||
      isRVVSelectedBodyStandaloneReductionRouteOperation(
          description.operation)) {
    metadata.push_back(
        {"tcrv_rvv.target_leaf_profile", description.targetLeafProfile});
    metadata.push_back({"tcrv_rvv.provider_supported_mirror",
                        description.providerSupportedMirror});
    metadata.push_back({"tcrv_rvv.required_header_declarations",
                        description.requiredHeaderDeclarations});
    metadata.push_back(
        {"tcrv_rvv.c_type_mapping", description.cTypeMappingSummary});
    if (!description.inactiveLaneZeroingRequirement.empty())
      metadata.push_back({"tcrv_rvv.inactive_lane_zeroing_requirement",
                          description.inactiveLaneZeroingRequirement});
  }
  if (description.operation == RVVSelectedBodyOperationKind::MaskedAdd) {
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back({"tcrv_rvv.inactive_lane_contract",
                        description.inactiveLaneContract});
    metadata.push_back({"tcrv_rvv.masked_passthrough_layout",
                        description.maskedPassthroughLayout});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSelect) {
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"tcrv_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back(
        {"tcrv_rvv.select_layout",
         "select-true-value-when-mask-else-false-value"});
  }
  if (description.operation == RVVSelectedBodyOperationKind::MaskedUnitLoadStore ||
      description.operation == RVVSelectedBodyOperationKind::MaskedUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedStore) {
    metadata.push_back(
        {"tcrv_rvv.masked_memory_layout", description.indexedMemoryLayout});
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"tcrv_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"tcrv_rvv.inactive_lane_contract",
                        description.inactiveLaneContract});
    metadata.push_back({"tcrv_rvv.masked_passthrough_layout",
                        description.maskedPassthroughLayout});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    if (description.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskStridedStore) {
      metadata.push_back({"tcrv_rvv.strided_memory_layout",
                          description.stridedMemoryLayout});
      metadata.push_back({"tcrv_rvv.destination_stride_source",
                          description.outStrideSource});
    }
  }
  if (description.operation == RVVSelectedBodyOperationKind::ReduceAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::StandaloneReduceAdd) {
    metadata.push_back({"tcrv_rvv.reduction_accumulator_layout",
                        description.reductionAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.reduction_result_layout",
                        description.reductionResultLayout});
    metadata.push_back(
        {"tcrv_rvv.reduction_store_vl", description.reductionStoreVL});
  }
  if (description.operation == RVVSelectedBodyOperationKind::MAccAdd) {
    metadata.push_back({"tcrv_rvv.macc_accumulator_layout",
                        description.maccAccumulatorLayout});
    metadata.push_back(
        {"tcrv_rvv.macc_result_layout", description.maccResultLayout});
  }
  if (description.operation == RVVSelectedBodyOperationKind::WideningMAccAdd) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.result_lmul", description.lmul});
    metadata.push_back({"tcrv_rvv.widening_macc_accumulator_layout",
                        description.wideningMAccAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.widening_macc_result_layout",
                        description.wideningMAccResultLayout});
    metadata.push_back({"tcrv_rvv.widening_macc_relation",
                        description.wideningMAccRelation});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::WideningDotReduceAdd) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.result_lmul", description.lmul});
    metadata.push_back({"tcrv_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back({"tcrv_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back({"tcrv_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.result_lmul", description.lmul});
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.lhs_stride_source", description.lhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.rhs_stride_source", description.rhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back({"tcrv_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.strided_load_intrinsic", description.stridedLoadIntrinsic});
    metadata.push_back({"tcrv_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.result_lmul", description.lmul});
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"tcrv_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"tcrv_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back({"tcrv_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back({"tcrv_rvv.masked_widening_product_intrinsic",
                        description.maskedWideningProductIntrinsic});
    metadata.push_back({"tcrv_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::
          ComputedMaskStridedInputWideningDotReduceAdd) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.accumulator_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.accumulator_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.result_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.result_lmul", description.lmul});
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.lhs_stride_source", description.lhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.rhs_stride_source", description.rhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.mask_role", description.maskRole});
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back(
        {"tcrv_rvv.mask_memory_form", description.maskMemoryForm});
    metadata.push_back({"tcrv_rvv.widening_dot_accumulator_layout",
                        description.wideningDotProductAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_result_layout",
                        description.wideningDotProductResultLayout});
    metadata.push_back({"tcrv_rvv.widening_dot_relation",
                        description.wideningDotProductRelation});
    metadata.push_back({"tcrv_rvv.widening_product_intrinsic",
                        description.wideningProductIntrinsic});
    metadata.push_back({"tcrv_rvv.masked_widening_product_intrinsic",
                        description.maskedWideningProductIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.strided_load_intrinsic", description.stridedLoadIntrinsic});
    metadata.push_back({"tcrv_rvv.widening_dot_reduction_store_vl",
                        description.reductionStoreVL});
  }
  if (description.operation == RVVSelectedBodyOperationKind::StridedAdd) {
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.lhs_stride_source", description.lhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.rhs_stride_source", description.rhsStrideSource});
    metadata.push_back(
        {"tcrv_rvv.out_stride_source", description.outStrideSource});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::StridedLoadUnitStore) {
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.source_stride_source", description.sourceStrideSource});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::UnitLoadStridedStore) {
    metadata.push_back({"tcrv_rvv.strided_memory_layout",
                        description.stridedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.destination_stride_source", description.outStrideSource});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
    metadata.push_back({"tcrv_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.index_source", description.indexSource});
    metadata.push_back({"tcrv_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"tcrv_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"tcrv_rvv.indexed_data_memory_form",
                        description.indexedDataMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::IndexedScatterUnitLoad) {
    metadata.push_back({"tcrv_rvv.indexed_memory_layout",
                        description.indexedMemoryLayout});
    metadata.push_back(
        {"tcrv_rvv.index_source", description.indexSource});
    metadata.push_back({"tcrv_rvv.index_eew",
                        llvm::Twine(description.indexEEW).str()});
    metadata.push_back({"tcrv_rvv.offset_unit", description.offsetUnit});
    metadata.push_back({"tcrv_rvv.index_uniqueness",
                        description.indexUniqueness});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.indexed_destination_memory_form",
                        description.indexedDestinationMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore) {
    metadata.push_back({"tcrv_rvv.segment_memory_layout",
                        description.segmentMemoryLayout});
    metadata.push_back({"tcrv_rvv.segment_count",
                        llvm::Twine(description.segmentCount).str()});
    metadata.push_back({"tcrv_rvv.segment_tuple_c_type",
                        description.segmentTupleCType});
    metadata.push_back({"tcrv_rvv.segment_load_intrinsic",
                        description.segmentLoadIntrinsic});
    metadata.push_back({"tcrv_rvv.segment_field_extract_intrinsic",
                        description.segmentFieldExtractIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.field0_role", description.field0Role});
    metadata.push_back({"tcrv_rvv.field1_role", description.field1Role});
    metadata.push_back({"tcrv_rvv.field0_name", description.field0Name});
    metadata.push_back({"tcrv_rvv.field1_name", description.field1Name});
    metadata.push_back({"tcrv_rvv.field0_destination_memory_form",
                        description.field0DestinationMemoryForm});
    metadata.push_back({"tcrv_rvv.field1_destination_memory_form",
                        description.field1DestinationMemoryForm});
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad) {
    metadata.push_back({"tcrv_rvv.segment_memory_layout",
                        description.segmentMemoryLayout});
    metadata.push_back({"tcrv_rvv.segment_count",
                        llvm::Twine(description.segmentCount).str()});
    metadata.push_back({"tcrv_rvv.segment_tuple_c_type",
                        description.segmentTupleCType});
    metadata.push_back({"tcrv_rvv.segment_store_intrinsic",
                        description.segmentStoreIntrinsic});
    metadata.push_back({"tcrv_rvv.segment_tuple_create_intrinsic",
                        description.segmentFieldExtractIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.field0_role", description.field0Role});
    metadata.push_back({"tcrv_rvv.field1_role", description.field1Role});
    metadata.push_back({"tcrv_rvv.field0_name", description.field0Name});
    metadata.push_back({"tcrv_rvv.field1_name", description.field1Name});
    metadata.push_back({"tcrv_rvv.field0_source_memory_form",
                        description.field0SourceMemoryForm});
    metadata.push_back({"tcrv_rvv.field1_source_memory_form",
                        description.field1SourceMemoryForm});
  }
  if (description.operation == RVVSelectedBodyOperationKind::WidenI32ToI64 ||
      description.operation == RVVSelectedBodyOperationKind::WidenI16ToI32) {
    metadata.push_back(
        {"tcrv_rvv.source_sew", llvm::Twine(description.sourceSEW).str()});
    metadata.push_back({"tcrv_rvv.source_lmul", description.sourceLMUL});
    metadata.push_back(
        {"tcrv_rvv.dest_sew", llvm::Twine(description.sew).str()});
    metadata.push_back({"tcrv_rvv.dest_lmul", description.lmul});
    metadata.push_back(
        {"tcrv_rvv.conversion_relation", description.conversionRelation});
  }
  return metadata;
}


} // namespace tianchenrv::plugin::rvv
