#include "TianChenRV/Support/RuntimeABICallablePlan.h"

#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace tianchenrv::support {
namespace {

using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::MemWindowOp;
using tianchenrv::tcrv::exec::RuntimeParamOp;

llvm::Error makeCallablePlanError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "runtime ABI callable plan validation failed";
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

llvm::StringRef getDefaultCNameForBufferRole(RuntimeABIParameterRole role) {
  switch (role) {
  case RuntimeABIParameterRole::LHSInputBuffer:
    return "lhs";
  case RuntimeABIParameterRole::RHSInputBuffer:
    return "rhs";
  case RuntimeABIParameterRole::OutputBuffer:
    return "out";
  case RuntimeABIParameterRole::RuntimeElementCount:
  case RuntimeABIParameterRole::DispatchAvailabilityGuard:
    break;
  }
  return {};
}

llvm::Expected<RuntimeABIParameter>
makeParameterFromMemWindow(KernelOp kernel, MemWindowOp window,
                           RuntimeABIParameterRole expectedRole) {
  llvm::StringRef role =
      getStringAttr(window.getOperation(), kMemWindowABIRoleAttrName);
  std::optional<RuntimeABIParameterRole> parsedRole =
      symbolizeRuntimeABIParameterRole(role);
  if (!parsedRole || *parsedRole != expectedRole)
    return makeCallablePlanError(
        kernel, llvm::Twine("tcrv.exec.mem_window @") + window.getSymName() +
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
        kernel, llvm::Twine("tcrv.exec.mem_window @") + window.getSymName() +
                    " has unsupported ownership '" + ownership + "'");

  return RuntimeABIParameter(getDefaultCNameForBufferRole(expectedRole), cType,
                             expectedRole, *parsedOwnership);
}

llvm::Expected<RuntimeABIParameter>
makeParameterFromRuntimeParam(KernelOp kernel, RuntimeParamOp param,
                              RuntimeABIParameterRole expectedRole) {
  llvm::StringRef role =
      getStringAttr(param.getOperation(), kRuntimeParamABIRoleAttrName);
  std::optional<RuntimeABIParameterRole> parsedRole =
      symbolizeRuntimeABIParameterRole(role);
  if (!parsedRole || *parsedRole != expectedRole)
    return makeCallablePlanError(
        kernel, llvm::Twine("tcrv.exec.runtime_param @") + param.getSymName() +
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
        kernel, llvm::Twine("tcrv.exec.runtime_param @") + param.getSymName() +
                    " has unsupported ownership '" + ownership + "'");

  return RuntimeABIParameter(cName, cType, expectedRole, *parsedOwnership);
}

llvm::Error requireSameParameter(KernelOp kernel,
                                 const RuntimeABIParameter &metadata,
                                 const RuntimeABIParameter &irBacked,
                                 llvm::StringRef metadataSource) {
  if (metadata.cName == irBacked.cName && metadata.cType == irBacked.cType &&
      metadata.role == irBacked.role && metadata.ownership == irBacked.ownership)
    return llvm::Error::success();

  return makeCallablePlanError(
      kernel, llvm::Twine(metadataSource) +
                  " runtime ABI parameter role '" +
                  stringifyRuntimeABIParameterRole(irBacked.role) +
                  "' must mirror IR-backed callable ABI parameter "
                  "c_name='" +
                  irBacked.cName + "', c_type='" + irBacked.cType +
                  "', ownership='" +
                  stringifyRuntimeABIParameterOwnership(irBacked.ownership) +
                  "'");
}

} // namespace

llvm::Expected<I32VAddCallableABIPlan>
buildI32VAddCallableABIPlan(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return makeCallablePlanError(kernel,
                                 "requires a materialized tcrv.exec.kernel "
                                 "body");

  I32VAddCallableABIPlan plan;
  llvm::SmallVector<RuntimeABIMemWindowSpec, 3> windowSpecs =
      getI32VAddBufferMemWindowSpecs();
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
          kernel, llvm::Twine("unsupported tcrv.exec.mem_window ABI role '") +
                      role + "'");

    llvm::Expected<RuntimeABIParameter> parameter =
        makeParameterFromMemWindow(kernel, window, *parsedRole);
    if (!parameter)
      return parameter.takeError();
    plan.parameters.push_back(std::move(*parameter));
  }

  RuntimeABIParamSpec countSpec =
      getI32VAddRuntimeElementCountParamSpec(/*cName=*/"");
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
          RuntimeABIParameterRole::RuntimeElementCount);
  if (!runtimeCount)
    return runtimeCount.takeError();
  plan.parameters.push_back(std::move(*runtimeCount));

  return plan;
}

llvm::Error validateI32VAddCallableABIParameterMirror(
    KernelOp kernel, llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource) {
  if (metadataParameters.empty())
    return makeCallablePlanError(
        kernel, llvm::Twine(metadataSource) +
                    " requires runtime_abi_parameters metadata mirroring the "
                    "IR-backed callable ABI plan");

  if (irBackedParameters.size() != 4)
    return makeCallablePlanError(
        kernel,
        "IR-backed i32-vadd callable ABI plan must contain exactly four "
        "parameters");

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
          kernel, llvm::Twine(metadataSource) +
                      " requires runtime ABI parameter role '" +
                      stringifyRuntimeABIParameterRole(expected.role) +
                      "' to mirror the IR-backed callable ABI plan");
    if (count > 1)
      return makeCallablePlanError(
          kernel, llvm::Twine(metadataSource) +
                      " contains duplicate runtime ABI parameter role '" +
                      stringifyRuntimeABIParameterRole(expected.role) + "'");

    if (llvm::Error error =
            requireSameParameter(kernel, *actual, expected, metadataSource))
      return error;
  }

  for (const RuntimeABIParameter &actual : metadataParameters) {
    bool expectedRole =
        llvm::any_of(irBackedParameters, [&](const RuntimeABIParameter &param) {
          return param.role == actual.role;
        });
    if (!expectedRole)
      return makeCallablePlanError(
          kernel, llvm::Twine(metadataSource) +
                      " contains unsupported runtime ABI parameter role '" +
                      stringifyRuntimeABIParameterRole(actual.role) + "'");
  }

  return llvm::Error::success();
}

} // namespace tianchenrv::support
