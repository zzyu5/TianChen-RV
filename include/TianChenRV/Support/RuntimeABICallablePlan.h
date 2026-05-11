#ifndef TIANCHENRV_SUPPORT_RUNTIMEABICALLABLEPLAN_H
#define TIANCHENRV_SUPPORT_RUNTIMEABICALLABLEPLAN_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Target/I32BinaryFamilyRegistry.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::support {

struct FiniteBinaryCallableABIPlan {
  llvm::SmallVector<RuntimeABIParameter, 4> parameters;
  llvm::SmallVector<tcrv::exec::MemWindowOp, 3> bufferWindows;
  tcrv::exec::RuntimeParamOp runtimeElementCountParam;
};

using I32BinaryCallableABIPlan = FiniteBinaryCallableABIPlan;

// Bounded bridge for the current finite binary callable ABI. It is not a
// general ABI subsystem.
llvm::Expected<FiniteBinaryCallableABIPlan>
buildFiniteBinaryCallableABIPlan(
    tcrv::exec::KernelOp kernel,
    const target::rvv::RVVBinaryRuntimeABIContract &contract);

llvm::Expected<FiniteBinaryCallableABIPlan>
buildFiniteBinaryCallableABIPlan(
    tcrv::exec::KernelOp kernel,
    const target::rvv::RVVBinaryFamilyDescriptor &family);

llvm::Error validateFiniteBinaryCallableABIParameterMirror(
    tcrv::exec::KernelOp kernel,
    llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource,
    const target::rvv::RVVBinaryRuntimeABIContract &contract);

llvm::Error validateFiniteBinaryCallableABIParameterMirror(
    tcrv::exec::KernelOp kernel,
    llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource,
    const target::rvv::RVVBinaryFamilyDescriptor &family);

// Compatibility wrappers for the current i32 binary callable ABI.
llvm::Expected<I32BinaryCallableABIPlan>
buildI32BinaryCallableABIPlan(tcrv::exec::KernelOp kernel,
                              const I32BinaryRuntimeABIContract &contract);

llvm::Expected<I32BinaryCallableABIPlan>
buildI32BinaryCallableABIPlan(
    tcrv::exec::KernelOp kernel,
    const target::i32_binary::I32BinaryFamilyDescriptor &family);

llvm::Error validateI32BinaryCallableABIParameterMirror(
    tcrv::exec::KernelOp kernel,
    llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource,
    const I32BinaryRuntimeABIContract &contract);

llvm::Error validateI32BinaryCallableABIParameterMirror(
    tcrv::exec::KernelOp kernel,
    llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource,
    const target::i32_binary::I32BinaryFamilyDescriptor &family);

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_RUNTIMEABICALLABLEPLAN_H
