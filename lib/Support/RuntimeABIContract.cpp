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

I32BinaryRuntimeABIContract::I32BinaryRuntimeABIContract(
    const target::i32_binary::I32BinaryFamilyDescriptor &family)
    : family(&family),
      rvvCallableIdentity{family.rvv.runtimeABI, family.rvv.runtimeABIKind,
                          family.rvv.runtimeABIName,
                          family.rvv.runtimeGlueRole},
      scalarCallableIdentity{family.scalar.runtimeABI,
                             family.scalar.runtimeABIKind,
                             family.scalar.runtimeABIName,
                             family.scalar.runtimeGlueRole},
      dispatchIdentity{family.dispatch.dispatchRuntimeABIKind,
                       family.dispatch.dispatchRuntimeABIName} {
  callableParameters.push_back(makeTargetExportABIParameter(
      "lhs", "const int32_t *", RuntimeABIParameterRole::LHSInputBuffer));
  callableParameters.push_back(makeTargetExportABIParameter(
      "rhs", "const int32_t *", RuntimeABIParameterRole::RHSInputBuffer));
  callableParameters.push_back(makeTargetExportABIParameter(
      "out", "int32_t *", RuntimeABIParameterRole::OutputBuffer));
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
      "const int32_t *"));
  bufferMemWindowSpecs.push_back(RuntimeABIMemWindowSpec(
      "abi_rhs_input_buffer", RuntimeABIParameterRole::RHSInputBuffer,
      kRuntimeABIReadAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "const int32_t *"));
  bufferMemWindowSpecs.push_back(RuntimeABIMemWindowSpec(
      "abi_output_buffer", RuntimeABIParameterRole::OutputBuffer,
      kRuntimeABIWriteAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "int32_t *"));
}

RuntimeABIParamSpec
I32BinaryRuntimeABIContract::getRuntimeElementCountParamSpec(
    llvm::StringRef cName) const {
  return RuntimeABIParamSpec(
      "abi_runtime_element_count",
      RuntimeABIParameterRole::RuntimeElementCount, cName, "size_t",
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned));
}

RuntimeABIParamSpec
I32BinaryRuntimeABIContract::getDispatchAvailabilityGuardParamSpec(
    llvm::StringRef cName) const {
  return ::tianchenrv::support::getDispatchAvailabilityGuardParamSpec(cName);
}

RuntimeABIParameter
I32BinaryRuntimeABIContract::getDispatchAvailabilityGuardParameter(
    llvm::StringRef cName) const {
  return makeTargetExportABIParameter(
      cName, "int", RuntimeABIParameterRole::DispatchAvailabilityGuard);
}

llvm::SmallVector<RuntimeABIParamSpec, 1>
I32BinaryRuntimeABIContract::getRuntimeElementCountParamSpecs(
    llvm::StringRef cName) const {
  llvm::SmallVector<RuntimeABIParamSpec, 1> specs;
  specs.push_back(getRuntimeElementCountParamSpec(cName));
  return specs;
}

llvm::SmallVector<RuntimeABIParamSpec, 2>
I32BinaryRuntimeABIContract::getDispatchRuntimeParamSpecs(
    llvm::StringRef runtimeCountCName, llvm::StringRef guardCName) const {
  llvm::SmallVector<RuntimeABIParamSpec, 2> specs;
  specs.push_back(getRuntimeElementCountParamSpec(runtimeCountCName));
  specs.push_back(getDispatchAvailabilityGuardParamSpec(guardCName));
  return specs;
}

llvm::SmallVector<RuntimeABIParameter, 5>
I32BinaryRuntimeABIContract::getDispatchRuntimeABIParameters(
    llvm::StringRef guardCName) const {
  llvm::SmallVector<RuntimeABIParameter, 5> parameters;
  parameters.append(callableParameters.begin(), callableParameters.end());
  parameters.push_back(getDispatchAvailabilityGuardParameter(guardCName));
  return parameters;
}

llvm::StringRef I32BinaryRuntimeABIContract::getCallableBufferCName(
    RuntimeABIParameterRole role) const {
  for (const RuntimeABIParameter &parameter : callableParameters) {
    if (parameter.role == role)
      return parameter.cName;
  }
  return {};
}

const I32BinaryRuntimeABIContract &getI32BinaryRuntimeABIContract(
    const target::i32_binary::I32BinaryFamilyDescriptor &family) {
  using Kind = target::i32_binary::I32BinaryFamilyKind;
  switch (family.kind) {
  case Kind::Add: {
    static const I32BinaryRuntimeABIContract contract(
        target::i32_binary::getI32VAddFamilyRegistrationRecord());
    return contract;
  }
  case Kind::Sub: {
    static const I32BinaryRuntimeABIContract contract(
        target::i32_binary::getI32VSubFamilyRegistrationRecord());
    return contract;
  }
  case Kind::Mul: {
    static const I32BinaryRuntimeABIContract contract(
        target::i32_binary::getI32VMulFamilyRegistrationRecord());
    return contract;
  }
  }
  llvm_unreachable("unknown i32 binary family kind");
}

const I32BinaryRuntimeABIContract &getI32BinaryRuntimeABIContract(
    target::i32_binary::I32BinaryFamilyKind kind) {
  using Kind = target::i32_binary::I32BinaryFamilyKind;
  switch (kind) {
  case Kind::Add:
    return getI32BinaryRuntimeABIContract(
        target::i32_binary::getI32VAddFamilyRegistrationRecord());
  case Kind::Sub:
    return getI32BinaryRuntimeABIContract(
        target::i32_binary::getI32VSubFamilyRegistrationRecord());
  case Kind::Mul:
    return getI32BinaryRuntimeABIContract(
        target::i32_binary::getI32VMulFamilyRegistrationRecord());
  }
  llvm_unreachable("unknown i32 binary family kind");
}

void appendI32BinaryRuntimeABIParameters(
    llvm::SmallVectorImpl<RuntimeABIParameter> &out) {
  llvm::ArrayRef<RuntimeABIParameter> parameters =
      getI32BinaryRuntimeABIContract(
          target::i32_binary::I32BinaryFamilyKind::Add)
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
      getI32BinaryRuntimeABIContract(
          target::i32_binary::I32BinaryFamilyKind::Add)
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
             target::i32_binary::I32BinaryFamilyKind::Add)
      .getDispatchAvailabilityGuardParameter(cName);
}

llvm::SmallVector<RuntimeABIParameter, 5>
getI32BinaryDispatchRuntimeABIParameters() {
  return getI32BinaryRuntimeABIContract(
             target::i32_binary::I32BinaryFamilyKind::Add)
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
      target::i32_binary::I32BinaryFamilyKind::Add);
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
      getI32BinaryRuntimeABIContract(
          target::i32_binary::I32BinaryFamilyKind::Add)
          .getBufferMemWindowSpecs();
  return llvm::SmallVector<RuntimeABIMemWindowSpec, 3>(specs.begin(),
                                                       specs.end());
}

RuntimeABIParamSpec
getI32BinaryRuntimeElementCountParamSpec(llvm::StringRef cName) {
  return getI32BinaryRuntimeABIContract(
             target::i32_binary::I32BinaryFamilyKind::Add)
      .getRuntimeElementCountParamSpec(cName);
}

RuntimeABIParamSpec
getI32BinaryDispatchAvailabilityGuardParamSpec(llvm::StringRef cName) {
  return getI32BinaryRuntimeABIContract(
             target::i32_binary::I32BinaryFamilyKind::Add)
      .getDispatchAvailabilityGuardParamSpec(cName);
}

llvm::SmallVector<RuntimeABIParamSpec, 1>
getI32BinaryRuntimeElementCountParamSpecs(llvm::StringRef cName) {
  return getI32BinaryRuntimeABIContract(
             target::i32_binary::I32BinaryFamilyKind::Add)
      .getRuntimeElementCountParamSpecs(cName);
}

llvm::SmallVector<RuntimeABIParamSpec, 2>
getI32BinaryDispatchRuntimeParamSpecs(llvm::StringRef runtimeCountCName,
                                      llvm::StringRef guardCName) {
  return getI32BinaryRuntimeABIContract(
             target::i32_binary::I32BinaryFamilyKind::Add)
      .getDispatchRuntimeParamSpecs(runtimeCountCName, guardCName);
}

} // namespace tianchenrv::support
