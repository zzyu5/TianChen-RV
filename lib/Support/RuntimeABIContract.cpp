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
  stream << "i32 binary runtime ABI callable parameter role binding failed";
  if (!context.empty())
    stream << " for " << context;
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

bool isI32BinaryCallableRole(RuntimeABIParameterRole role) {
  switch (role) {
  case RuntimeABIParameterRole::LHSInputBuffer:
  case RuntimeABIParameterRole::RHSInputBuffer:
  case RuntimeABIParameterRole::OutputBuffer:
  case RuntimeABIParameterRole::RuntimeElementCount:
    return true;
  case RuntimeABIParameterRole::DispatchAvailabilityGuard:
    return false;
  }
  return false;
}

const RuntimeABIParameter *
getI32BinaryCallableRoleRequirement(
    const I32BinaryRuntimeABIContract &contract, RuntimeABIParameterRole role) {
  for (const RuntimeABIParameter &requirement :
       contract.getCallableRoleRequirements())
    if (requirement.role == role)
      return &requirement;
  return nullptr;
}

llvm::Expected<const RuntimeABIParameter *>
bindOneI32BinaryCallableParameterByRole(
    llvm::ArrayRef<RuntimeABIParameter> parameters,
    RuntimeABIParameterRole role, llvm::StringRef context,
    const I32BinaryRuntimeABIContract &contract) {
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
      getI32BinaryCallableRoleRequirement(contract, role);
  if (!requirement)
    return makeCallableBindingError(
        context,
        llvm::Twine("missing i32 binary callable role requirement for family "
                    "'") +
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
    : familyID(spec.familyID), callableIdentity(spec.callableIdentity),
      rvvCallableIdentity(spec.rvvCallableIdentity),
      scalarCallableIdentity(spec.scalarCallableIdentity),
      dispatchIdentity(spec.dispatchIdentity),
      externalABIComponentGroup(spec.externalABIComponentGroup) {
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
    llvm::StringRef guardCName) const {
  llvm::SmallVector<RuntimeABIParameter, 5> parameters;
  parameters.append(callableParameters.begin(), callableParameters.end());
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

const I32BinaryRuntimeABIContract &getI32BinaryRuntimeABIContract(
    llvm::StringRef familyID) {
  static const I32BinaryRuntimeABIContract addContract(
      FiniteBinaryRuntimeABIContractSpec{
          "i32-vadd",
          "const int32_t *",
          "int32_t *",
          {"rvv-i32-vadd-runtime-callable-c-abi.v1",
           "rvv-runtime-callable-c-abi",
           "rvv-i32-vadd-runtime-callable-c-function.v1",
           "runtime-callable-i32-vadd-function"},
          {"rvv-i32-vadd-runtime-callable-c-abi.v1",
           "rvv-runtime-callable-c-abi",
           "rvv-i32-vadd-runtime-callable-c-function.v1",
           "runtime-callable-i32-vadd-function"},
          {"scalar-i32-vadd-runtime-callable-c-abi.v1",
           "scalar-runtime-callable-c-abi",
           "scalar-i32-vadd-runtime-callable-c-function.v1",
           "runtime-callable-i32-vadd-fallback-function"},
          {"rvv-scalar-dispatch-runtime-callable-c-abi",
           "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"},
          "rvv-i32-vadd-microkernel-external-abi.v1"});
  static const I32BinaryRuntimeABIContract subContract(
      FiniteBinaryRuntimeABIContractSpec{
          "i32-vsub",
          "const int32_t *",
          "int32_t *",
          {"rvv-i32-vsub-runtime-callable-c-abi.v1",
           "rvv-runtime-callable-c-abi",
           "rvv-i32-vsub-runtime-callable-c-function.v1",
           "runtime-callable-i32-vsub-function"},
          {"rvv-i32-vsub-runtime-callable-c-abi.v1",
           "rvv-runtime-callable-c-abi",
           "rvv-i32-vsub-runtime-callable-c-function.v1",
           "runtime-callable-i32-vsub-function"},
          {"scalar-i32-vsub-runtime-callable-c-abi.v1",
           "scalar-runtime-callable-c-abi",
           "scalar-i32-vsub-runtime-callable-c-function.v1",
           "runtime-callable-i32-vsub-fallback-function"},
          {"rvv-scalar-dispatch-runtime-callable-c-abi",
           "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"},
          "rvv-i32-vsub-microkernel-external-abi.v1"});
  static const I32BinaryRuntimeABIContract mulContract(
      FiniteBinaryRuntimeABIContractSpec{
          "i32-vmul",
          "const int32_t *",
          "int32_t *",
          {"rvv-i32-vmul-runtime-callable-c-abi.v1",
           "rvv-runtime-callable-c-abi",
           "rvv-i32-vmul-runtime-callable-c-function.v1",
           "runtime-callable-i32-vmul-function"},
          {"rvv-i32-vmul-runtime-callable-c-abi.v1",
           "rvv-runtime-callable-c-abi",
           "rvv-i32-vmul-runtime-callable-c-function.v1",
           "runtime-callable-i32-vmul-function"},
          {"scalar-i32-vmul-runtime-callable-c-abi.v1",
           "scalar-runtime-callable-c-abi",
           "scalar-i32-vmul-runtime-callable-c-function.v1",
           "runtime-callable-i32-vmul-fallback-function"},
          {"rvv-scalar-dispatch-runtime-callable-c-abi",
           "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1"},
          "rvv-i32-vmul-microkernel-external-abi.v1"});

  familyID = familyID.trim();
  if (familyID == "i32-vadd")
    return addContract;
  if (familyID == "i32-vsub")
    return subContract;
  if (familyID == "i32-vmul")
    return mulContract;
  llvm_unreachable("unknown i32 binary runtime ABI family id");
}

void appendI32BinaryRuntimeABIParameters(
    llvm::SmallVectorImpl<RuntimeABIParameter> &out) {
  llvm::ArrayRef<RuntimeABIParameter> parameters =
      getI32BinaryRuntimeABIContract("i32-vadd")
          .getCallableParameters();
  out.append(parameters.begin(), parameters.end());
}

llvm::SmallVector<RuntimeABIParameter, 4> getI32BinaryRuntimeABIParameters() {
  llvm::SmallVector<RuntimeABIParameter, 4> parameters;
  appendI32BinaryRuntimeABIParameters(parameters);
  return parameters;
}

void appendI32BinaryRuntimeABIRoleRequirements(
    llvm::SmallVectorImpl<RuntimeABIParameter> &out) {
  llvm::ArrayRef<RuntimeABIParameter> parameters =
      getI32BinaryRuntimeABIContract("i32-vadd")
          .getCallableRoleRequirements();
  out.append(parameters.begin(), parameters.end());
}

llvm::SmallVector<RuntimeABIParameter, 4>
getI32BinaryRuntimeABIRoleRequirements() {
  llvm::SmallVector<RuntimeABIParameter, 4> parameters;
  appendI32BinaryRuntimeABIRoleRequirements(parameters);
  return parameters;
}

RuntimeABIParameter
makeI32BinaryDispatchAvailabilityGuard(llvm::StringRef cName) {
  return getI32BinaryRuntimeABIContract(
             "i32-vadd")
      .getDispatchAvailabilityGuardParameter(cName);
}

llvm::SmallVector<RuntimeABIParameter, 5>
getI32BinaryDispatchRuntimeABIParameters() {
  return getI32BinaryRuntimeABIContract(
             "i32-vadd")
      .getDispatchRuntimeABIParameters();
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

llvm::Expected<I32BinaryCallableRuntimeABIParameterBindings>
bindI32BinaryCallableRuntimeABIParametersByRole(
    llvm::ArrayRef<RuntimeABIParameter> parameters, llvm::StringRef context) {
  using Role = RuntimeABIParameterRole;
  const I32BinaryRuntimeABIContract &contract = getI32BinaryRuntimeABIContract(
      "i32-vadd");
  I32BinaryCallableRuntimeABIParameterBindings bindings;

  llvm::Expected<const RuntimeABIParameter *> lhs =
      bindOneI32BinaryCallableParameterByRole(
          parameters, Role::LHSInputBuffer, context, contract);
  if (!lhs)
    return lhs.takeError();
  bindings.lhs = *lhs;

  llvm::Expected<const RuntimeABIParameter *> rhs =
      bindOneI32BinaryCallableParameterByRole(
          parameters, Role::RHSInputBuffer, context, contract);
  if (!rhs)
    return rhs.takeError();
  bindings.rhs = *rhs;

  llvm::Expected<const RuntimeABIParameter *> out =
      bindOneI32BinaryCallableParameterByRole(
          parameters, Role::OutputBuffer, context, contract);
  if (!out)
    return out.takeError();
  bindings.out = *out;

  llvm::Expected<const RuntimeABIParameter *> runtimeElementCount =
      bindOneI32BinaryCallableParameterByRole(
          parameters, Role::RuntimeElementCount, context, contract);
  if (!runtimeElementCount)
    return runtimeElementCount.takeError();
  bindings.runtimeElementCount = *runtimeElementCount;

  for (const RuntimeABIParameter &parameter : parameters) {
    if (isI32BinaryCallableRole(parameter.role))
      continue;
    return makeCallableBindingError(
        context, llvm::Twine("contains unsupported direct callable runtime ABI "
                             "parameter role '") +
                     stringifyRuntimeABIParameterRole(parameter.role) + "'");
  }

  return bindings;
}

llvm::SmallVector<RuntimeABIMemWindowSpec, 3>
getI32BinaryBufferMemWindowSpecs() {
  llvm::ArrayRef<RuntimeABIMemWindowSpec> specs =
      getI32BinaryRuntimeABIContract("i32-vadd")
          .getBufferMemWindowSpecs();
  return llvm::SmallVector<RuntimeABIMemWindowSpec, 3>(specs.begin(),
                                                       specs.end());
}

RuntimeABIParamSpec
getI32BinaryRuntimeElementCountParamSpec(llvm::StringRef cName) {
  return getI32BinaryRuntimeABIContract(
             "i32-vadd")
      .getRuntimeElementCountParamSpec(cName);
}

RuntimeABIParamSpec
getI32BinaryDispatchAvailabilityGuardParamSpec(llvm::StringRef cName) {
  return getI32BinaryRuntimeABIContract(
             "i32-vadd")
      .getDispatchAvailabilityGuardParamSpec(cName);
}

llvm::SmallVector<RuntimeABIParamSpec, 1>
getI32BinaryRuntimeElementCountParamSpecs(llvm::StringRef cName) {
  return getI32BinaryRuntimeABIContract(
             "i32-vadd")
      .getRuntimeElementCountParamSpecs(cName);
}

llvm::SmallVector<RuntimeABIParamSpec, 2>
getI32BinaryDispatchRuntimeParamSpecs(llvm::StringRef runtimeCountCName,
                                      llvm::StringRef guardCName) {
  return getI32BinaryRuntimeABIContract(
             "i32-vadd")
      .getDispatchRuntimeParamSpecs(runtimeCountCName, guardCName);
}

} // namespace tianchenrv::support
