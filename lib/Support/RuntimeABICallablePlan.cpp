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
    KernelOp kernel, const I32BinaryRuntimeABIContract *contract,
    llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "i32 binary runtime ABI callable plan validation failed";
  if (contract)
    stream << " for family '" << contract->getFamilyID() << "'";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::StringRef getStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op ? op->getAttrOfType<mlir::StringAttr>(attrName)
                 : mlir::StringAttr();
  if (!attr)
    return {};
  return attr.getValue();
}

llvm::Expected<RuntimeABIParameter>
makeParameterFromMemWindow(KernelOp kernel, MemWindowOp window,
                           RuntimeABIParameterRole expectedRole,
                           const I32BinaryRuntimeABIContract &contract) {
  llvm::StringRef role =
      getStringAttr(window.getOperation(), kMemWindowABIRoleAttrName);
  std::optional<RuntimeABIParameterRole> parsedRole =
      symbolizeRuntimeABIParameterRole(role);
  if (!parsedRole || *parsedRole != expectedRole)
    return makeCallablePlanError(
        kernel, &contract,
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
        kernel, &contract,
        llvm::Twine("tcrv.exec.mem_window @") + window.getSymName() +
            " has unsupported ownership '" + ownership + "'");

  llvm::StringRef cName =
      contract.getCallableBufferCName(expectedRole);
  if (cName.empty())
    return makeCallablePlanError(
        kernel, &contract,
        llvm::Twine("tcrv.exec.mem_window @") + window.getSymName() +
            " has no callable ABI C parameter name for role '" +
            stringifyRuntimeABIParameterRole(expectedRole) + "'");

  return RuntimeABIParameter(cName, cType, expectedRole, *parsedOwnership);
}

llvm::Expected<RuntimeABIParameter>
makeParameterFromRuntimeParam(KernelOp kernel, RuntimeParamOp param,
                              RuntimeABIParameterRole expectedRole,
                              const I32BinaryRuntimeABIContract &contract) {
  llvm::StringRef role =
      getStringAttr(param.getOperation(), kRuntimeParamABIRoleAttrName);
  std::optional<RuntimeABIParameterRole> parsedRole =
      symbolizeRuntimeABIParameterRole(role);
  if (!parsedRole || *parsedRole != expectedRole)
    return makeCallablePlanError(
        kernel, &contract,
        llvm::Twine("tcrv.exec.runtime_param @") + param.getSymName() +
            " must carry ABI role '" +
            stringifyRuntimeABIParameterRole(expectedRole) + "'");

  llvm::StringRef cName =
      getStringAttr(param.getOperation(), kRuntimeParamCNameAttrName);
  llvm::StringRef cType =
      getStringAttr(param.getOperation(), kRuntimeParamCTypeAttrName);
  llvm::StringRef ownership =
      getStringAttr(param.getOperation(), kRuntimeParamOwnershipAttrName);
  std::optional<RuntimeABIParameterOwnership> parsedOwnership =
      symbolizeRuntimeABIParameterOwnership(ownership);
  if (!parsedOwnership)
    return makeCallablePlanError(
        kernel, &contract,
        llvm::Twine("tcrv.exec.runtime_param @") + param.getSymName() +
            " has unsupported ownership '" + ownership + "'");

  return RuntimeABIParameter(cName, cType, expectedRole, *parsedOwnership);
}

llvm::Error requireSameParameter(KernelOp kernel,
                                 const RuntimeABIParameter &metadata,
                                 const RuntimeABIParameter &irBacked,
                                 llvm::StringRef metadataSource,
                                 const I32BinaryRuntimeABIContract &contract) {
  if (metadata.cName == irBacked.cName && metadata.cType == irBacked.cType &&
      metadata.role == irBacked.role && metadata.ownership == irBacked.ownership)
    return llvm::Error::success();

  return makeCallablePlanError(
      kernel, &contract,
      llvm::Twine(metadataSource) + " runtime ABI parameter role '" +
          stringifyRuntimeABIParameterRole(irBacked.role) +
          "' must mirror IR-backed callable ABI parameter c_name='" +
          irBacked.cName + "', c_type='" + irBacked.cType + "', ownership='" +
          stringifyRuntimeABIParameterOwnership(irBacked.ownership) + "'");
}

} // namespace

llvm::Expected<I32BinaryCallableABIPlan>
buildI32BinaryCallableABIPlan(KernelOp kernel,
                              const I32BinaryRuntimeABIContract &contract) {
  if (!kernel || kernel.getBody().empty())
    return makeCallablePlanError(
        kernel, &contract,
        "requires a materialized tcrv.exec.kernel body");

  I32BinaryCallableABIPlan plan;
  llvm::ArrayRef<RuntimeABIMemWindowSpec> windowSpecs =
      contract.getBufferMemWindowSpecs();
  if (llvm::Error error = collectRuntimeABIBufferMemWindows(
          kernel, windowSpecs, plan.bufferWindows))
    return std::move(error);

  for (MemWindowOp window : plan.bufferWindows) {
    llvm::StringRef role =
        getStringAttr(window.getOperation(), kMemWindowABIRoleAttrName);
    std::optional<RuntimeABIParameterRole> parsedRole =
        symbolizeRuntimeABIParameterRole(role);
    if (!parsedRole)
      return makeCallablePlanError(
          kernel, &contract,
          llvm::Twine("unsupported tcrv.exec.mem_window ABI role '") + role +
              "'");

    llvm::Expected<RuntimeABIParameter> parameter =
        makeParameterFromMemWindow(kernel, window, *parsedRole, contract);
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
          RuntimeABIParameterRole::RuntimeElementCount, contract);
  if (!runtimeCount)
    return runtimeCount.takeError();
  plan.parameters.push_back(std::move(*runtimeCount));

  return plan;
}

llvm::Expected<I32BinaryCallableABIPlan>
buildI32BinaryCallableABIPlan(
    KernelOp kernel,
    const target::i32_binary::I32BinaryFamilyDescriptor &family) {
  return buildI32BinaryCallableABIPlan(
      kernel, getI32BinaryRuntimeABIContract(family));
}

llvm::Error validateI32BinaryCallableABIParameterMirror(
    KernelOp kernel, llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource,
    const I32BinaryRuntimeABIContract &contract) {
  if (metadataParameters.empty())
    return makeCallablePlanError(
        kernel, &contract,
        llvm::Twine(metadataSource) +
            " requires runtime_abi_parameters metadata mirroring the "
            "IR-backed callable ABI plan");

  std::size_t expectedParameterCount =
      contract.getCallableParameters().size();
  if (irBackedParameters.size() != expectedParameterCount)
    return makeCallablePlanError(
        kernel, &contract,
        llvm::Twine("IR-backed i32 binary callable ABI plan must contain "
                    "exactly ") +
            llvm::Twine(expectedParameterCount) + " parameters for family '" +
            contract.getFamilyID() + "'");

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
          kernel, &contract,
          llvm::Twine(metadataSource) +
              " requires runtime ABI parameter role '" +
              stringifyRuntimeABIParameterRole(expected.role) +
              "' to mirror the IR-backed callable ABI plan");
    if (count > 1)
      return makeCallablePlanError(
          kernel, &contract,
          llvm::Twine(metadataSource) +
              " contains duplicate runtime ABI parameter role '" +
              stringifyRuntimeABIParameterRole(expected.role) + "'");

    if (llvm::Error error =
            requireSameParameter(kernel, *actual, expected, metadataSource,
                                 contract))
      return error;
  }

  for (const RuntimeABIParameter &actual : metadataParameters) {
    bool expectedRole =
        llvm::any_of(irBackedParameters, [&](const RuntimeABIParameter &param) {
          return param.role == actual.role;
        });
    if (!expectedRole)
      return makeCallablePlanError(
          kernel, &contract,
          llvm::Twine(metadataSource) +
              " contains unsupported runtime ABI parameter role '" +
              stringifyRuntimeABIParameterRole(actual.role) + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateI32BinaryCallableABIParameterMirror(
    KernelOp kernel, llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource,
    const target::i32_binary::I32BinaryFamilyDescriptor &family) {
  return validateI32BinaryCallableABIParameterMirror(
      kernel, metadataParameters, irBackedParameters, metadataSource,
      getI32BinaryRuntimeABIContract(family));
}

llvm::Expected<I32VAddCallableABIPlan>
buildI32VAddCallableABIPlan(KernelOp kernel) {
  return buildI32BinaryCallableABIPlan(
      kernel, getI32BinaryRuntimeABIContract(
                  target::i32_binary::I32BinaryFamilyKind::Add));
}

llvm::Error validateI32VAddCallableABIParameterMirror(
    KernelOp kernel, llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource) {
  return validateI32BinaryCallableABIParameterMirror(
      kernel, metadataParameters, irBackedParameters, metadataSource,
      getI32BinaryRuntimeABIContract(
          target::i32_binary::I32BinaryFamilyKind::Add));
}

} // namespace tianchenrv::support
