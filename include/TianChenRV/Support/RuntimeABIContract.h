#ifndef TIANCHENRV_SUPPORT_RUNTIMEABICONTRACT_H
#define TIANCHENRV_SUPPORT_RUNTIMEABICONTRACT_H

#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace tianchenrv::support {

struct FiniteBinaryRuntimeABIContractSpec {
  llvm::StringRef familyID;
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
};

// Extension-agnostic runtime ABI contract for a bounded finite binary callable
// shape. It carries parameter/mem-window/runtime-param shape only. ABI route
// identity, artifact identity, hardware facts, descriptor metadata, bundle
// records, and evidence data stay with their owning plugin/target layers.
class FiniteBinaryRuntimeABIContract {
public:
  explicit FiniteBinaryRuntimeABIContract(
      const FiniteBinaryRuntimeABIContractSpec &spec);

  llvm::StringRef getFamilyID() const { return familyID; }

  llvm::ArrayRef<RuntimeABIParameter> getCallableParameters() const {
    return callableParameters;
  }

  llvm::SmallVector<RuntimeABIParameter, 4>
  getCallableParameters(llvm::StringRef runtimeCountCName) const;

  llvm::ArrayRef<RuntimeABIParameter> getCallableRoleRequirements() const {
    return callableRoleRequirements;
  }

  llvm::ArrayRef<RuntimeABIMemWindowSpec> getBufferMemWindowSpecs() const {
    return bufferMemWindowSpecs;
  }

  RuntimeABIParamSpec
  getRuntimeElementCountParamSpec(llvm::StringRef cName = "n") const;

  RuntimeABIParamSpec getDispatchAvailabilityGuardParamSpec(
      llvm::StringRef cName) const;

  RuntimeABIParameter
  getDispatchAvailabilityGuardParameter(llvm::StringRef cName) const;

  llvm::SmallVector<RuntimeABIParamSpec, 1>
  getRuntimeElementCountParamSpecs(llvm::StringRef cName = "n") const;

  llvm::SmallVector<RuntimeABIParamSpec, 2> getDispatchRuntimeParamSpecs(
      llvm::StringRef runtimeCountCName, llvm::StringRef guardCName) const;

  llvm::SmallVector<RuntimeABIParameter, 5> getDispatchRuntimeABIParameters(
      llvm::StringRef runtimeCountCName, llvm::StringRef guardCName) const;

  llvm::StringRef
  getCallableBufferCName(RuntimeABIParameterRole role) const;

protected:
  FiniteBinaryRuntimeABIContract() = default;

private:
  llvm::StringRef familyID;
  llvm::SmallVector<RuntimeABIParameter, 4> callableParameters;
  llvm::SmallVector<RuntimeABIParameter, 4> callableRoleRequirements;
  llvm::SmallVector<RuntimeABIMemWindowSpec, 3> bufferMemWindowSpecs;
};

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_RUNTIMEABICONTRACT_H
