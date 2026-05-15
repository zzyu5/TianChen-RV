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

struct FiniteBinaryRuntimeABIContractSpec {
  llvm::StringRef familyID;
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
  RuntimeABICallableIdentity callableIdentity;
  RuntimeABIDispatchIdentity dispatchIdentity;
  llvm::StringRef externalABIComponentGroup;
};

// Support-layer runtime ABI contract for the current bounded finite binary
// callable shape. It carries ABI shape and selected family identity only; route
// ids, artifact kinds, hardware facts, descriptor metadata, bundle records, and
// evidence data stay with their owning target/export layers.
class FiniteBinaryRuntimeABIContract {
public:
  explicit FiniteBinaryRuntimeABIContract(
      const FiniteBinaryRuntimeABIContractSpec &spec);

  llvm::StringRef getFamilyID() const { return familyID; }
  llvm::StringRef getRuntimeABI() const { return callableIdentity.runtimeABI; }
  llvm::StringRef getRuntimeABIKind() const {
    return callableIdentity.runtimeABIKind;
  }
  llvm::StringRef getRuntimeABIName() const {
    return callableIdentity.runtimeABIName;
  }
  llvm::StringRef getRuntimeGlueRole() const {
    return callableIdentity.runtimeGlueRole;
  }
  llvm::StringRef getExternalABIComponentGroup() const {
    return externalABIComponentGroup;
  }

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

  const RuntimeABIDispatchIdentity &getDispatchIdentity() const {
    return dispatchIdentity;
  }

protected:
  FiniteBinaryRuntimeABIContract() = default;

private:
  llvm::StringRef familyID;
  llvm::SmallVector<RuntimeABIParameter, 4> callableParameters;
  llvm::SmallVector<RuntimeABIParameter, 4> callableRoleRequirements;
  llvm::SmallVector<RuntimeABIMemWindowSpec, 3> bufferMemWindowSpecs;
  RuntimeABICallableIdentity callableIdentity;
  RuntimeABIDispatchIdentity dispatchIdentity;
  llvm::StringRef externalABIComponentGroup;
};

using I32BinaryRuntimeABIContract = FiniteBinaryRuntimeABIContract;

const I32BinaryRuntimeABIContract &getI32BinaryRuntimeABIContract(
    llvm::StringRef familyID);

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_RUNTIMEABICONTRACT_H
