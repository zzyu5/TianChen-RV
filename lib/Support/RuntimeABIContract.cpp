#include "TianChenRV/Support/RuntimeABIContract.h"

namespace tianchenrv::support {

I32VAddRuntimeABIContract::I32VAddRuntimeABIContract()
    : rvvCallableIdentity{
          "rvv-i32-vadd-runtime-callable-c-abi.v1",
          "rvv-runtime-callable-c-abi",
          "rvv-i32-vadd-runtime-callable-c-function.v1",
          "runtime-callable-i32-vadd-function"},
      scalarCallableIdentity{
          "scalar-i32-vadd-runtime-callable-c-abi.v1",
          "scalar-runtime-callable-c-abi",
          "scalar-i32-vadd-runtime-callable-c-function.v1",
          "runtime-callable-i32-vadd-fallback-function"},
      dispatchIdentity{
          "rvv-scalar-dispatch-runtime-callable-c-abi",
          "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"} {
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

RuntimeABIParamSpec I32VAddRuntimeABIContract::getRuntimeElementCountParamSpec(
    llvm::StringRef cName) const {
  return RuntimeABIParamSpec(
      "abi_runtime_element_count",
      RuntimeABIParameterRole::RuntimeElementCount, cName, "size_t",
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned));
}

RuntimeABIParamSpec
I32VAddRuntimeABIContract::getDispatchAvailabilityGuardParamSpec(
    llvm::StringRef cName) const {
  return ::tianchenrv::support::getDispatchAvailabilityGuardParamSpec(cName);
}

RuntimeABIParameter
I32VAddRuntimeABIContract::getDispatchAvailabilityGuardParameter(
    llvm::StringRef cName) const {
  return makeTargetExportABIParameter(
      cName, "int", RuntimeABIParameterRole::DispatchAvailabilityGuard);
}

llvm::SmallVector<RuntimeABIParamSpec, 1>
I32VAddRuntimeABIContract::getRuntimeElementCountParamSpecs(
    llvm::StringRef cName) const {
  llvm::SmallVector<RuntimeABIParamSpec, 1> specs;
  specs.push_back(getRuntimeElementCountParamSpec(cName));
  return specs;
}

llvm::SmallVector<RuntimeABIParamSpec, 2>
I32VAddRuntimeABIContract::getDispatchRuntimeParamSpecs(
    llvm::StringRef runtimeCountCName, llvm::StringRef guardCName) const {
  llvm::SmallVector<RuntimeABIParamSpec, 2> specs;
  specs.push_back(getRuntimeElementCountParamSpec(runtimeCountCName));
  specs.push_back(getDispatchAvailabilityGuardParamSpec(guardCName));
  return specs;
}

llvm::SmallVector<RuntimeABIParameter, 5>
I32VAddRuntimeABIContract::getDispatchRuntimeABIParameters(
    llvm::StringRef guardCName) const {
  llvm::SmallVector<RuntimeABIParameter, 5> parameters;
  parameters.append(callableParameters.begin(), callableParameters.end());
  parameters.push_back(getDispatchAvailabilityGuardParameter(guardCName));
  return parameters;
}

llvm::StringRef I32VAddRuntimeABIContract::getCallableBufferCName(
    RuntimeABIParameterRole role) const {
  for (const RuntimeABIParameter &parameter : callableParameters) {
    if (parameter.role == role)
      return parameter.cName;
  }
  return {};
}

const I32VAddRuntimeABIContract &getI32VAddRuntimeABIContract() {
  static const I32VAddRuntimeABIContract contract;
  return contract;
}

void appendI32VAddRuntimeABIParameters(
    llvm::SmallVectorImpl<RuntimeABIParameter> &out) {
  llvm::ArrayRef<RuntimeABIParameter> parameters =
      getI32VAddRuntimeABIContract().getCallableParameters();
  out.append(parameters.begin(), parameters.end());
}

llvm::SmallVector<RuntimeABIParameter, 4> getI32VAddRuntimeABIParameters() {
  llvm::SmallVector<RuntimeABIParameter, 4> parameters;
  appendI32VAddRuntimeABIParameters(parameters);
  return parameters;
}

void appendI32VAddRuntimeABIRoleRequirements(
    llvm::SmallVectorImpl<RuntimeABIParameter> &out) {
  llvm::ArrayRef<RuntimeABIParameter> parameters =
      getI32VAddRuntimeABIContract().getCallableRoleRequirements();
  out.append(parameters.begin(), parameters.end());
}

llvm::SmallVector<RuntimeABIParameter, 4>
getI32VAddRuntimeABIRoleRequirements() {
  llvm::SmallVector<RuntimeABIParameter, 4> parameters;
  appendI32VAddRuntimeABIRoleRequirements(parameters);
  return parameters;
}

RuntimeABIParameter
makeI32VAddDispatchAvailabilityGuard(llvm::StringRef cName) {
  return getI32VAddRuntimeABIContract().getDispatchAvailabilityGuardParameter(
      cName);
}

llvm::SmallVector<RuntimeABIParameter, 5>
getI32VAddDispatchRuntimeABIParameters() {
  return getI32VAddRuntimeABIContract().getDispatchRuntimeABIParameters();
}

llvm::SmallVector<RuntimeABIMemWindowSpec, 3>
getI32VAddBufferMemWindowSpecs() {
  llvm::ArrayRef<RuntimeABIMemWindowSpec> specs =
      getI32VAddRuntimeABIContract().getBufferMemWindowSpecs();
  return llvm::SmallVector<RuntimeABIMemWindowSpec, 3>(specs.begin(),
                                                       specs.end());
}

RuntimeABIParamSpec
getI32VAddRuntimeElementCountParamSpec(llvm::StringRef cName) {
  return getI32VAddRuntimeABIContract().getRuntimeElementCountParamSpec(cName);
}

RuntimeABIParamSpec
getI32VAddDispatchAvailabilityGuardParamSpec(llvm::StringRef cName) {
  return getI32VAddRuntimeABIContract().getDispatchAvailabilityGuardParamSpec(
      cName);
}

llvm::SmallVector<RuntimeABIParamSpec, 1>
getI32VAddRuntimeElementCountParamSpecs(llvm::StringRef cName) {
  return getI32VAddRuntimeABIContract().getRuntimeElementCountParamSpecs(cName);
}

llvm::SmallVector<RuntimeABIParamSpec, 2>
getI32VAddDispatchRuntimeParamSpecs(llvm::StringRef runtimeCountCName,
                                    llvm::StringRef guardCName) {
  return getI32VAddRuntimeABIContract().getDispatchRuntimeParamSpecs(
      runtimeCountCName, guardCName);
}

} // namespace tianchenrv::support
