#ifndef TIANCHENRV_SUPPORT_RUNTIMEABICONTRACT_H
#define TIANCHENRV_SUPPORT_RUNTIMEABICONTRACT_H

#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace tianchenrv::support {

struct RuntimeABICallableIdentity {
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
};

struct RuntimeABIDispatchIdentity {
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
};

// Single support-layer owner for the current bounded i32-vadd callable ABI.
// It centralizes ABI metadata only; route ids, artifact kinds, hardware facts,
// descriptor metadata, bundle records, and evidence data stay with their owners.
class I32VAddRuntimeABIContract {
public:
  llvm::ArrayRef<RuntimeABIParameter> getCallableParameters() const {
    return callableParameters;
  }

  llvm::ArrayRef<RuntimeABIParameter> getCallableRoleRequirements() const {
    return callableRoleRequirements;
  }

  llvm::ArrayRef<RuntimeABIMemWindowSpec> getBufferMemWindowSpecs() const {
    return bufferMemWindowSpecs;
  }

  RuntimeABIParamSpec
  getRuntimeElementCountParamSpec(llvm::StringRef cName = "n") const;

  RuntimeABIParamSpec getDispatchAvailabilityGuardParamSpec(
      llvm::StringRef cName = "rvv_available") const;

  RuntimeABIParameter
  getDispatchAvailabilityGuardParameter(llvm::StringRef cName =
                                            "rvv_available") const;

  llvm::SmallVector<RuntimeABIParamSpec, 1>
  getRuntimeElementCountParamSpecs(llvm::StringRef cName = "n") const;

  llvm::SmallVector<RuntimeABIParamSpec, 2> getDispatchRuntimeParamSpecs(
      llvm::StringRef runtimeCountCName = "n",
      llvm::StringRef guardCName = "rvv_available") const;

  llvm::SmallVector<RuntimeABIParameter, 5>
  getDispatchRuntimeABIParameters(llvm::StringRef guardCName =
                                      "rvv_available") const;

  llvm::StringRef
  getCallableBufferCName(RuntimeABIParameterRole role) const;

  const RuntimeABICallableIdentity &getRVVCallableIdentity() const {
    return rvvCallableIdentity;
  }

  const RuntimeABICallableIdentity &getScalarCallableIdentity() const {
    return scalarCallableIdentity;
  }

  const RuntimeABIDispatchIdentity &getDispatchIdentity() const {
    return dispatchIdentity;
  }

private:
  friend const I32VAddRuntimeABIContract &getI32VAddRuntimeABIContract();

  I32VAddRuntimeABIContract();

  llvm::SmallVector<RuntimeABIParameter, 4> callableParameters;
  llvm::SmallVector<RuntimeABIParameter, 4> callableRoleRequirements;
  llvm::SmallVector<RuntimeABIMemWindowSpec, 3> bufferMemWindowSpecs;
  RuntimeABICallableIdentity rvvCallableIdentity;
  RuntimeABICallableIdentity scalarCallableIdentity;
  RuntimeABIDispatchIdentity dispatchIdentity;
};

const I32VAddRuntimeABIContract &getI32VAddRuntimeABIContract();

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_RUNTIMEABICONTRACT_H
