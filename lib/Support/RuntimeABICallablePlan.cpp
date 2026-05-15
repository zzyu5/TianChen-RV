#include "TianChenRV/Support/RuntimeABICallablePlan.h"

#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <optional>

namespace tianchenrv::support {
namespace {

using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::MemWindowOp;
using tianchenrv::tcrv::exec::RuntimeParamOp;

llvm::Error makeCallablePlanError(
    KernelOp kernel, llvm::StringRef familyID, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "finite binary runtime ABI callable plan validation failed";
  if (!familyID.empty())
    stream << " for family '" << familyID << "'";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::StringRef getCallableBufferCName(
    llvm::ArrayRef<RuntimeABIParameter> callableParameters,
    RuntimeABIParameterRole role) {
  for (const RuntimeABIParameter &parameter : callableParameters)
    if (parameter.role == role)
      return parameter.cName;
  return {};
}

llvm::StringRef getStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op ? op->getAttrOfType<mlir::StringAttr>(attrName)
                 : mlir::StringAttr();
  if (!attr)
    return {};
  return attr.getValue();
}

llvm::Error requireBoundedInvocationContractText(
    KernelOp kernel, llvm::StringRef familyID, llvm::StringRef fieldName,
    llvm::StringRef value) {
  llvm::StringRef trimmed = value.trim();
  if (trimmed.empty())
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine("runtime ABI invocation contract field '") + fieldName +
            "' requires non-empty bounded text");
  if (trimmed.size() > 160)
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine("runtime ABI invocation contract field '") + fieldName +
            "' is too long");
  if (trimmed.contains('\n') || trimmed.contains('\r'))
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine("runtime ABI invocation contract field '") + fieldName +
            "' must be single-line text");
  return llvm::Error::success();
}

llvm::Expected<RuntimeABIParameter>
makeParameterFromMemWindow(KernelOp kernel, MemWindowOp window,
                           RuntimeABIParameterRole expectedRole,
                           llvm::ArrayRef<RuntimeABIParameter> callableParams,
                           llvm::StringRef familyID) {
  llvm::StringRef role =
      getStringAttr(window.getOperation(), kMemWindowABIRoleAttrName);
  std::optional<RuntimeABIParameterRole> parsedRole =
      symbolizeRuntimeABIParameterRole(role);
  if (!parsedRole || *parsedRole != expectedRole)
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine("tcrv.exec.mem_window @") + window.getSymName() +
            " must carry ABI role '" +
            stringifyRuntimeABIParameterRole(expectedRole) + "'");

  llvm::StringRef cType =
      getStringAttr(window.getOperation(), kMemWindowCTypeAttrName);
  llvm::StringRef ownership =
      getStringAttr(window.getOperation(), kMemWindowOwnershipAttrName);
  std::optional<RuntimeABIParameterOwnership> parsedOwnership =
      symbolizeRuntimeABIParameterOwnership(ownership);
  if (!parsedOwnership)
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine("tcrv.exec.mem_window @") + window.getSymName() +
            " has unsupported ownership '" + ownership + "'");

  llvm::StringRef cName =
      getCallableBufferCName(callableParams, expectedRole);
  if (cName.empty())
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine("tcrv.exec.mem_window @") + window.getSymName() +
            " has no callable ABI C parameter name for role '" +
            stringifyRuntimeABIParameterRole(expectedRole) + "'");

  return RuntimeABIParameter(cName, cType, expectedRole, *parsedOwnership);
}

llvm::Expected<RuntimeABIParameter>
makeParameterFromRuntimeParam(KernelOp kernel, RuntimeParamOp param,
                              RuntimeABIParameterRole expectedRole,
                              llvm::StringRef familyID) {
  llvm::StringRef role =
      getStringAttr(param.getOperation(), kRuntimeParamABIRoleAttrName);
  std::optional<RuntimeABIParameterRole> parsedRole =
      symbolizeRuntimeABIParameterRole(role);
  if (!parsedRole || *parsedRole != expectedRole)
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine("tcrv.exec.runtime_param @") + param.getSymName() +
            " must carry ABI role '" +
            stringifyRuntimeABIParameterRole(expectedRole) + "'");

  llvm::StringRef cName =
      getStringAttr(param.getOperation(), kRuntimeParamCNameAttrName);
  if (cName.trim().empty())
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine("tcrv.exec.runtime_param @") + param.getSymName() +
            " requires non-empty C parameter name for ABI role '" +
            stringifyRuntimeABIParameterRole(expectedRole) + "'");

  llvm::StringRef cType =
      getStringAttr(param.getOperation(), kRuntimeParamCTypeAttrName);
  llvm::StringRef ownership =
      getStringAttr(param.getOperation(), kRuntimeParamOwnershipAttrName);
  std::optional<RuntimeABIParameterOwnership> parsedOwnership =
      symbolizeRuntimeABIParameterOwnership(ownership);
  if (!parsedOwnership)
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine("tcrv.exec.runtime_param @") + param.getSymName() +
            " has unsupported ownership '" + ownership + "'");

  return RuntimeABIParameter(cName, cType, expectedRole, *parsedOwnership);
}

llvm::Error requireSameParameter(KernelOp kernel,
                                 const RuntimeABIParameter &metadata,
                                 const RuntimeABIParameter &irBacked,
                                 llvm::StringRef metadataSource,
                                 llvm::StringRef familyID) {
  if (metadata.cName == irBacked.cName && metadata.cType == irBacked.cType &&
      metadata.role == irBacked.role && metadata.ownership == irBacked.ownership)
    return llvm::Error::success();

  return makeCallablePlanError(
      kernel, familyID,
      llvm::Twine(metadataSource) + " runtime ABI parameter role '" +
          stringifyRuntimeABIParameterRole(irBacked.role) +
          "' must mirror IR-backed callable ABI parameter c_name='" +
          irBacked.cName + "', c_type='" + irBacked.cType + "', ownership='" +
          stringifyRuntimeABIParameterOwnership(irBacked.ownership) + "'");
}

template <typename ContractT>
llvm::Expected<FiniteBinaryCallableABIPlan>
buildFiniteBinaryCallableABIPlanFromContract(KernelOp kernel,
                                             const ContractT &contract) {
  llvm::StringRef familyID = contract.getFamilyID();
  if (!kernel || kernel.getBody().empty())
    return makeCallablePlanError(
        kernel, familyID, "requires a materialized tcrv.exec.kernel body");

  FiniteBinaryCallableABIPlan plan;
  llvm::ArrayRef<RuntimeABIMemWindowSpec> windowSpecs =
      contract.getBufferMemWindowSpecs();
  if (llvm::Error error = collectRuntimeABIBufferMemWindows(
          kernel, windowSpecs, plan.bufferWindows))
    return std::move(error);

  llvm::ArrayRef<RuntimeABIParameter> callableParams =
      contract.getCallableParameters();
  for (MemWindowOp window : plan.bufferWindows) {
    llvm::StringRef role =
        getStringAttr(window.getOperation(), kMemWindowABIRoleAttrName);
    std::optional<RuntimeABIParameterRole> parsedRole =
        symbolizeRuntimeABIParameterRole(role);
    if (!parsedRole)
      return makeCallablePlanError(
          kernel, familyID,
          llvm::Twine("unsupported tcrv.exec.mem_window ABI role '") + role +
              "'");

    llvm::Expected<RuntimeABIParameter> parameter =
        makeParameterFromMemWindow(kernel, window, *parsedRole, callableParams,
                                   familyID);
    if (!parameter)
      return parameter.takeError();
    plan.parameters.push_back(std::move(*parameter));
  }

  RuntimeABIParamSpec countSpec =
      contract.getRuntimeElementCountParamSpec(/*cName=*/"");
  llvm::SmallVector<RuntimeABIParamSpec, 1> countSpecs;
  countSpecs.push_back(countSpec);
  llvm::SmallVector<RuntimeParamOp, 1> runtimeParams;
  if (llvm::Error error =
          collectRuntimeABIParams(kernel, countSpecs, runtimeParams))
    return std::move(error);
  plan.runtimeElementCountParam = runtimeParams.front();

  llvm::Expected<RuntimeABIParameter> runtimeCount =
      makeParameterFromRuntimeParam(
          kernel, plan.runtimeElementCountParam,
          RuntimeABIParameterRole::RuntimeElementCount, familyID);
  if (!runtimeCount)
    return runtimeCount.takeError();
  plan.parameters.push_back(std::move(*runtimeCount));

  return plan;
}

template <typename ContractT>
llvm::Error validateFiniteBinaryCallableABIParameterMirrorFromContract(
    KernelOp kernel, llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource, const ContractT &contract) {
  llvm::StringRef familyID = contract.getFamilyID();
  if (metadataParameters.empty())
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine(metadataSource) +
            " requires runtime_abi_parameters metadata mirroring the "
            "IR-backed callable ABI plan");

  std::size_t expectedParameterCount =
      contract.getCallableParameters().size();
  if (irBackedParameters.size() != expectedParameterCount)
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine("IR-backed finite binary callable ABI plan must contain "
                    "exactly ") +
            llvm::Twine(expectedParameterCount) + " parameters for family '" +
            familyID + "'");

  for (const RuntimeABIParameter &expected : irBackedParameters) {
    const RuntimeABIParameter *actual = nullptr;
    unsigned count = 0;
    for (const RuntimeABIParameter &candidate : metadataParameters) {
      if (candidate.role != expected.role)
        continue;
      actual = &candidate;
      ++count;
    }

    if (count == 0)
      return makeCallablePlanError(
          kernel, familyID,
          llvm::Twine(metadataSource) +
              " requires runtime ABI parameter role '" +
              stringifyRuntimeABIParameterRole(expected.role) +
              "' to mirror the IR-backed callable ABI plan");
    if (count > 1)
      return makeCallablePlanError(
          kernel, familyID,
          llvm::Twine(metadataSource) +
              " contains duplicate runtime ABI parameter role '" +
              stringifyRuntimeABIParameterRole(expected.role) + "'");

    if (llvm::Error error =
            requireSameParameter(kernel, *actual, expected, metadataSource,
                                 familyID))
      return error;
  }

  for (const RuntimeABIParameter &actual : metadataParameters) {
    bool expectedRole =
        llvm::any_of(irBackedParameters, [&](const RuntimeABIParameter &param) {
          return param.role == actual.role;
        });
    if (!expectedRole)
      return makeCallablePlanError(
          kernel, familyID,
          llvm::Twine(metadataSource) +
              " contains unsupported runtime ABI parameter role '" +
              stringifyRuntimeABIParameterRole(actual.role) + "'");
  }

  if (metadataParameters.size() != irBackedParameters.size())
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine(metadataSource) +
            " must contain exactly the ordered IR-backed callable ABI "
            "parameters");

  for (std::size_t index = 0; index < irBackedParameters.size(); ++index) {
    const RuntimeABIParameter &actual = metadataParameters[index];
    const RuntimeABIParameter &expected = irBackedParameters[index];
    if (actual.role == expected.role)
      continue;
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine(metadataSource) +
            " must preserve IR-backed callable ABI parameter order at index " +
            llvm::Twine(index) + ": expected role '" +
            stringifyRuntimeABIParameterRole(expected.role) +
            "' but found role '" +
            stringifyRuntimeABIParameterRole(actual.role) + "'");
  }

  return llvm::Error::success();
}

} // namespace

std::string
formatRuntimeABIOrderedRoles(llvm::ArrayRef<RuntimeABIParameter> parameters) {
  std::string roles;
  llvm::raw_string_ostream stream(roles);
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    if (index != 0)
      stream << "->";
    stream << stringifyRuntimeABIParameterRole(parameter.role);
  }
  stream.flush();
  return roles;
}

std::string formatRuntimeABIInvocationContractCommentBody(
    llvm::StringRef label, const RuntimeABIInvocationContract &contract) {
  std::string body;
  llvm::raw_string_ostream stream(body);
  stream << label << ": source=" << contract.sourceOwner
         << ", callable_symbol=" << contract.callableSymbol
         << ", family=" << contract.familyID
         << ", runtime_abi_kind=" << contract.runtimeABIKind
         << ", runtime_abi_name=" << contract.runtimeABIName;
  if (!contract.runtimeGlueRole.empty())
    stream << ", runtime_glue_role=" << contract.runtimeGlueRole;
  stream << ", parameter_count=" << contract.parameters.size()
         << ", ordered_roles="
         << formatRuntimeABIOrderedRoles(contract.parameters)
         << ", runtime_element_count_c_name="
         << contract.runtimeElementCountCName;
  if (!contract.dispatchGuardCName.empty())
    stream << ", dispatch_guard_c_name=" << contract.dispatchGuardCName;
  stream << ", production_owner=" << contract.productionOwner;
  stream.flush();
  return body;
}

llvm::Expected<RuntimeABIInvocationContract>
buildRuntimeABIInvocationContract(
    KernelOp kernel, llvm::StringRef familyID,
    llvm::ArrayRef<RuntimeABIParameter> parameters,
    llvm::StringRef sourceOwner, llvm::StringRef callableSymbol,
    llvm::StringRef runtimeABIKind, llvm::StringRef runtimeABIName,
    llvm::StringRef runtimeGlueRole, llvm::StringRef runtimeElementCountCName,
    llvm::StringRef productionOwner, llvm::StringRef dispatchGuardCName) {
  if (parameters.empty())
    return makeCallablePlanError(
        kernel, familyID,
        "runtime ABI invocation contract requires ordered ABI parameters");

  const llvm::StringRef requiredFields[] = {
      sourceOwner,     callableSymbol,           familyID,
      runtimeABIKind,  runtimeABIName,           runtimeElementCountCName,
      productionOwner};
  const llvm::StringLiteral fieldNames[] = {
      "source",
      "callable_symbol",
      "family",
      "runtime_abi_kind",
      "runtime_abi_name",
      "runtime_element_count_c_name",
      "production_owner"};
  for (auto [field, fieldName] : llvm::zip(requiredFields, fieldNames))
    if (llvm::Error error = requireBoundedInvocationContractText(
            kernel, familyID, fieldName, field))
      return std::move(error);
  if (!runtimeGlueRole.empty())
    if (llvm::Error error = requireBoundedInvocationContractText(
            kernel, familyID, "runtime_glue_role", runtimeGlueRole))
      return std::move(error);
  if (!dispatchGuardCName.empty())
    if (llvm::Error error = requireBoundedInvocationContractText(
            kernel, familyID, "dispatch_guard_c_name", dispatchGuardCName))
      return std::move(error);

  llvm::Expected<const RuntimeABIParameter *> runtimeCount =
      findUniqueRuntimeABIParameterByRole(
          parameters, RuntimeABIParameterRole::RuntimeElementCount,
          "runtime ABI invocation contract");
  if (!runtimeCount) {
    std::string message = llvm::toString(runtimeCount.takeError());
    return makeCallablePlanError(kernel, familyID, message);
  }
  if ((*runtimeCount)->cName != runtimeElementCountCName)
    return makeCallablePlanError(
        kernel, familyID,
        llvm::Twine("runtime ABI invocation contract "
                    "runtime_element_count_c_name '") +
            runtimeElementCountCName +
            "' must match ordered ABI parameter role 'runtime-element-count' "
            "C name '" +
            (*runtimeCount)->cName + "'");

  const RuntimeABIParameter *dispatchGuard = nullptr;
  unsigned dispatchGuardCount = 0;
  for (const RuntimeABIParameter &parameter : parameters) {
    if (parameter.role != RuntimeABIParameterRole::DispatchAvailabilityGuard)
      continue;
    dispatchGuard = &parameter;
    ++dispatchGuardCount;
  }
  if (dispatchGuardCName.empty()) {
    if (dispatchGuardCount != 0)
      return makeCallablePlanError(
          kernel, familyID,
          "runtime ABI invocation contract for a direct callable must not "
          "carry a dispatch-availability-guard ABI parameter");
  } else {
    if (dispatchGuardCount != 1)
      return makeCallablePlanError(
          kernel, familyID,
          "runtime ABI invocation contract requires exactly one "
          "dispatch-availability-guard ABI parameter when "
          "dispatch_guard_c_name is present");
    if (dispatchGuard->cName != dispatchGuardCName)
      return makeCallablePlanError(
          kernel, familyID,
          llvm::Twine("runtime ABI invocation contract dispatch_guard_c_name "
                      "'") +
              dispatchGuardCName +
              "' must match ordered ABI parameter role "
              "'dispatch-availability-guard' C name '" +
              dispatchGuard->cName + "'");
  }

  RuntimeABIInvocationContract contract;
  contract.sourceOwner = sourceOwner.str();
  contract.callableSymbol = callableSymbol.str();
  contract.familyID = familyID.str();
  contract.runtimeABIKind = runtimeABIKind.str();
  contract.runtimeABIName = runtimeABIName.str();
  contract.runtimeGlueRole = runtimeGlueRole.str();
  contract.parameters.append(parameters.begin(), parameters.end());
  contract.runtimeElementCountCName = runtimeElementCountCName.str();
  contract.dispatchGuardCName = dispatchGuardCName.str();
  contract.productionOwner = productionOwner.str();
  return contract;
}

llvm::Expected<FiniteBinaryCallableABIPlan>
buildFiniteBinaryCallableABIPlan(
    KernelOp kernel, const FiniteBinaryRuntimeABIContract &contract) {
  return buildFiniteBinaryCallableABIPlanFromContract(kernel, contract);
}

llvm::Error validateFiniteBinaryCallableABIParameterMirror(
    KernelOp kernel, llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource,
    const FiniteBinaryRuntimeABIContract &contract) {
  return validateFiniteBinaryCallableABIParameterMirrorFromContract(
      kernel, metadataParameters, irBackedParameters, metadataSource,
      contract);
}

} // namespace tianchenrv::support
