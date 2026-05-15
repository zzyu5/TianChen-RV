#ifndef TIANCHENRV_SUPPORT_RUNTIMEABICALLABLEPLAN_H
#define TIANCHENRV_SUPPORT_RUNTIMEABICALLABLEPLAN_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace tianchenrv::support {

struct FiniteBinaryCallableABIPlan {
  llvm::SmallVector<RuntimeABIParameter, 4> parameters;
  llvm::SmallVector<tcrv::exec::MemWindowOp, 3> bufferWindows;
  tcrv::exec::RuntimeParamOp runtimeElementCountParam;
};

struct RuntimeABIInvocationContract {
  std::string sourceOwner;
  std::string callableSymbol;
  std::string familyID;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  std::string runtimeGlueRole;
  llvm::SmallVector<RuntimeABIParameter, 5> parameters;
  std::string runtimeElementCountCName;
  std::string dispatchGuardCName;
  std::string productionOwner;
};

std::string
formatRuntimeABIOrderedRoles(llvm::ArrayRef<RuntimeABIParameter> parameters);

std::string formatRuntimeABIInvocationContractCommentBody(
    llvm::StringRef label, const RuntimeABIInvocationContract &contract);

llvm::Expected<RuntimeABIInvocationContract>
buildRuntimeABIInvocationContract(
    tcrv::exec::KernelOp kernel, llvm::StringRef familyID,
    llvm::ArrayRef<RuntimeABIParameter> parameters,
    llvm::StringRef sourceOwner, llvm::StringRef callableSymbol,
    llvm::StringRef runtimeABIKind, llvm::StringRef runtimeABIName,
    llvm::StringRef runtimeGlueRole, llvm::StringRef runtimeElementCountCName,
    llvm::StringRef productionOwner,
    llvm::StringRef dispatchGuardCName = llvm::StringRef());

// Bounded bridge for the current finite binary callable ABI. It is not a
// general ABI subsystem.
llvm::Expected<FiniteBinaryCallableABIPlan>
buildFiniteBinaryCallableABIPlan(
    tcrv::exec::KernelOp kernel,
    const FiniteBinaryRuntimeABIContract &contract);

llvm::Error validateFiniteBinaryCallableABIParameterMirror(
    tcrv::exec::KernelOp kernel,
    llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource,
    const FiniteBinaryRuntimeABIContract &contract);

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_RUNTIMEABICALLABLEPLAN_H
