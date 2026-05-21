#include "TianChenRV/Support/RuntimeABIContract.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::support {
namespace {

llvm::Error makeCallableBindingError(llvm::StringRef context,
                                     llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "finite binary runtime ABI callable parameter role binding failed";
  if (!context.empty())
    stream << " for " << context;
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

bool isFiniteBinaryCallableRole(RuntimeABIParameterRole role) {
  switch (role) {
  case RuntimeABIParameterRole::LHSInputBuffer:
  case RuntimeABIParameterRole::RHSInputBuffer:
  case RuntimeABIParameterRole::AccumulatorInputBuffer:
  case RuntimeABIParameterRole::OutputBuffer:
  case RuntimeABIParameterRole::RuntimeElementCount:
    return true;
  case RuntimeABIParameterRole::SourceInputBuffer:
  case RuntimeABIParameterRole::TrueValueInputBuffer:
  case RuntimeABIParameterRole::FalseValueInputBuffer:
  case RuntimeABIParameterRole::DotLHSInputBuffer:
  case RuntimeABIParameterRole::DotRHSInputBuffer:
  case RuntimeABIParameterRole::IndexInputBuffer:
  case RuntimeABIParameterRole::MaskInputBuffer:
  case RuntimeABIParameterRole::SegmentField0InputBuffer:
  case RuntimeABIParameterRole::SegmentField1InputBuffer:
  case RuntimeABIParameterRole::RHSScalarValue:
  case RuntimeABIParameterRole::SegmentField0OutputBuffer:
  case RuntimeABIParameterRole::SegmentField1OutputBuffer:
  case RuntimeABIParameterRole::SegmentInterleavedOutputBuffer:
  case RuntimeABIParameterRole::LHSInputStride:
  case RuntimeABIParameterRole::RHSInputStride:
  case RuntimeABIParameterRole::SourceByteStride:
  case RuntimeABIParameterRole::OutputStride:
  case RuntimeABIParameterRole::DispatchAvailabilityGuard:
    return false;
  }
  return false;
}

const RuntimeABIParameter *
getFiniteBinaryCallableRoleRequirement(
    const FiniteBinaryRuntimeABIContract &contract,
    RuntimeABIParameterRole role) {
  for (const RuntimeABIParameter &requirement :
       contract.getCallableRoleRequirements())
    if (requirement.role == role)
      return &requirement;
  return nullptr;
}

llvm::Expected<const RuntimeABIParameter *>
bindOneFiniteBinaryCallableParameterByRole(
    llvm::ArrayRef<RuntimeABIParameter> parameters,
    RuntimeABIParameterRole role, llvm::StringRef context,
    const FiniteBinaryRuntimeABIContract &contract) {
  llvm::Expected<const RuntimeABIParameter *> parameter =
      findUniqueRuntimeABIParameterByRole(parameters, role, context);
  if (!parameter) {
    std::string message = llvm::toString(parameter.takeError());
    return makeCallableBindingError(context, message);
  }

  const RuntimeABIParameter *bound = *parameter;
  if (llvm::StringRef(bound->cName).trim().empty())
    return makeCallableBindingError(
        context, llvm::Twine("runtime ABI parameter role '") +
                     stringifyRuntimeABIParameterRole(role) +
                     "' requires non-empty C name");

  const RuntimeABIParameter *requirement =
      getFiniteBinaryCallableRoleRequirement(contract, role);
  if (!requirement)
    return makeCallableBindingError(
        context,
        llvm::Twine("missing finite binary callable role requirement for "
                    "family '") +
            contract.getFamilyID() + "' and role '" +
            stringifyRuntimeABIParameterRole(role) + "'");

  if (bound->cType != requirement->cType)
    return makeCallableBindingError(
        context, llvm::Twine("runtime ABI parameter role '") +
                     stringifyRuntimeABIParameterRole(role) +
                     "' must use C type '" + requirement->cType + "'");

  if (bound->ownership != requirement->ownership)
    return makeCallableBindingError(
        context, llvm::Twine("runtime ABI parameter role '") +
                     stringifyRuntimeABIParameterRole(role) +
                     "' must use ownership '" +
                     stringifyRuntimeABIParameterOwnership(
                         requirement->ownership) +
                     "'");

  return bound;
}

} // namespace

FiniteBinaryRuntimeABIContract::FiniteBinaryRuntimeABIContract(
    const FiniteBinaryRuntimeABIContractSpec &spec)
    : familyID(spec.familyID) {
  callableParameters.push_back(makeTargetExportABIParameter(
      "lhs", spec.constInputPointerCType,
      RuntimeABIParameterRole::LHSInputBuffer));
  callableParameters.push_back(makeTargetExportABIParameter(
      "rhs", spec.constInputPointerCType,
      RuntimeABIParameterRole::RHSInputBuffer));
  callableParameters.push_back(makeTargetExportABIParameter(
      "out", spec.outputPointerCType, RuntimeABIParameterRole::OutputBuffer));
  callableParameters.push_back(makeTargetExportABIParameter(
      "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount));

  for (const RuntimeABIParameter &parameter : callableParameters)
    callableRoleRequirements.push_back(makeTargetExportABIRoleRequirement(
        parameter.cType, parameter.role));

  bufferMemWindowSpecs.push_back(RuntimeABIMemWindowSpec(
      "abi_lhs_input_buffer", RuntimeABIParameterRole::LHSInputBuffer,
      kRuntimeABIReadAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      spec.constInputPointerCType));
  bufferMemWindowSpecs.push_back(RuntimeABIMemWindowSpec(
      "abi_rhs_input_buffer", RuntimeABIParameterRole::RHSInputBuffer,
      kRuntimeABIReadAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      spec.constInputPointerCType));
  bufferMemWindowSpecs.push_back(RuntimeABIMemWindowSpec(
      "abi_output_buffer", RuntimeABIParameterRole::OutputBuffer,
      kRuntimeABIWriteAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      spec.outputPointerCType));
}

llvm::SmallVector<RuntimeABIParameter, 4>
FiniteBinaryRuntimeABIContract::getCallableParameters(
    llvm::StringRef runtimeCountCName) const {
  llvm::SmallVector<RuntimeABIParameter, 4> parameters(callableParameters.begin(),
                                                       callableParameters.end());
  for (RuntimeABIParameter &parameter : parameters) {
    if (parameter.role == RuntimeABIParameterRole::RuntimeElementCount) {
      parameter.cName = runtimeCountCName.str();
      break;
    }
  }
  return parameters;
}

RuntimeABIParamSpec
FiniteBinaryRuntimeABIContract::getRuntimeElementCountParamSpec(
    llvm::StringRef cName) const {
  return RuntimeABIParamSpec(
      "abi_runtime_element_count",
      RuntimeABIParameterRole::RuntimeElementCount, cName, "size_t",
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned));
}

RuntimeABIParamSpec
FiniteBinaryRuntimeABIContract::getDispatchAvailabilityGuardParamSpec(
    llvm::StringRef cName) const {
  return ::tianchenrv::support::getDispatchAvailabilityGuardParamSpec(cName);
}

RuntimeABIParameter
FiniteBinaryRuntimeABIContract::getDispatchAvailabilityGuardParameter(
    llvm::StringRef cName) const {
  return makeTargetExportABIParameter(
      cName, "int", RuntimeABIParameterRole::DispatchAvailabilityGuard);
}

llvm::SmallVector<RuntimeABIParamSpec, 1>
FiniteBinaryRuntimeABIContract::getRuntimeElementCountParamSpecs(
    llvm::StringRef cName) const {
  llvm::SmallVector<RuntimeABIParamSpec, 1> specs;
  specs.push_back(getRuntimeElementCountParamSpec(cName));
  return specs;
}

llvm::SmallVector<RuntimeABIParamSpec, 2>
FiniteBinaryRuntimeABIContract::getDispatchRuntimeParamSpecs(
    llvm::StringRef runtimeCountCName, llvm::StringRef guardCName) const {
  llvm::SmallVector<RuntimeABIParamSpec, 2> specs;
  specs.push_back(getRuntimeElementCountParamSpec(runtimeCountCName));
  specs.push_back(getDispatchAvailabilityGuardParamSpec(guardCName));
  return specs;
}

llvm::SmallVector<RuntimeABIParameter, 5>
FiniteBinaryRuntimeABIContract::getDispatchRuntimeABIParameters(
    llvm::StringRef runtimeCountCName, llvm::StringRef guardCName) const {
  llvm::SmallVector<RuntimeABIParameter, 5> parameters;
  llvm::SmallVector<RuntimeABIParameter, 4> callable =
      getCallableParameters(runtimeCountCName);
  parameters.append(callable.begin(), callable.end());
  parameters.push_back(getDispatchAvailabilityGuardParameter(guardCName));
  return parameters;
}

llvm::StringRef FiniteBinaryRuntimeABIContract::getCallableBufferCName(
    RuntimeABIParameterRole role) const {
  for (const RuntimeABIParameter &parameter : callableParameters) {
    if (parameter.role == role)
      return parameter.cName;
  }
  return {};
}

llvm::Expected<const RuntimeABIParameter *> findUniqueRuntimeABIParameterByRole(
    llvm::ArrayRef<RuntimeABIParameter> parameters,
    RuntimeABIParameterRole role, llvm::StringRef context) {
  const RuntimeABIParameter *matched = nullptr;
  unsigned count = 0;
  for (const RuntimeABIParameter &parameter : parameters) {
    if (parameter.role != role)
      continue;
    matched = &parameter;
    ++count;
  }

  if (count == 1)
    return matched;

  std::string message;
  llvm::raw_string_ostream stream(message);
  stream << "runtime ABI parameter role lookup failed";
  if (!context.empty())
    stream << " for " << context;
  stream << ": requires exactly one runtime ABI parameter with role '"
         << stringifyRuntimeABIParameterRole(role) << "'";
  if (count == 0)
    stream << "; found none";
  else
    stream << "; found duplicate parameters";
  stream.flush();
  return llvm::make_error<llvm::StringError>(message,
                                             llvm::errc::invalid_argument);
}

llvm::Expected<FiniteBinaryCallableRuntimeABIParameterBindings>
bindFiniteBinaryCallableRuntimeABIParametersByRole(
    llvm::ArrayRef<RuntimeABIParameter> parameters, llvm::StringRef context,
    const FiniteBinaryRuntimeABIContract &contract) {
  using Role = RuntimeABIParameterRole;
  FiniteBinaryCallableRuntimeABIParameterBindings bindings;

  llvm::Expected<const RuntimeABIParameter *> lhs =
      bindOneFiniteBinaryCallableParameterByRole(
          parameters, Role::LHSInputBuffer, context, contract);
  if (!lhs)
    return lhs.takeError();
  bindings.lhs = *lhs;

  llvm::Expected<const RuntimeABIParameter *> rhs =
      bindOneFiniteBinaryCallableParameterByRole(
          parameters, Role::RHSInputBuffer, context, contract);
  if (!rhs)
    return rhs.takeError();
  bindings.rhs = *rhs;

  llvm::Expected<const RuntimeABIParameter *> out =
      bindOneFiniteBinaryCallableParameterByRole(
          parameters, Role::OutputBuffer, context, contract);
  if (!out)
    return out.takeError();
  bindings.out = *out;

  llvm::Expected<const RuntimeABIParameter *> runtimeElementCount =
      bindOneFiniteBinaryCallableParameterByRole(
          parameters, Role::RuntimeElementCount, context, contract);
  if (!runtimeElementCount)
    return runtimeElementCount.takeError();
  bindings.runtimeElementCount = *runtimeElementCount;

  for (const RuntimeABIParameter &parameter : parameters) {
    if (isFiniteBinaryCallableRole(parameter.role))
      continue;
    return makeCallableBindingError(
        context, llvm::Twine("contains unsupported direct callable runtime ABI "
                             "parameter role '") +
                     stringifyRuntimeABIParameterRole(parameter.role) + "'");
  }

  return bindings;
}

} // namespace tianchenrv::support
