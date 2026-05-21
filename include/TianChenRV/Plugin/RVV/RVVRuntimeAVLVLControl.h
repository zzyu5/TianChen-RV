#ifndef TIANCHENRV_PLUGIN_RVV_RVVRUNTIMEAVLVLCONTROL_H
#define TIANCHENRV_PLUGIN_RVV_RVVRUNTIMEAVLVLCONTROL_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace tianchenrv::plugin::rvv {

struct RVVRuntimeAVLVLControlPlan {
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  tcrv::rvv::PolicyAttr policy;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef controlPlanID;
  llvm::StringRef configContractID;
  llvm::StringRef runtimeVLContractID;
  llvm::StringRef runtimeAVLABIParameterName;
  llvm::StringRef runtimeAVLASource;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef vlDefOpName;
  llvm::StringRef vlScopeOpName;
  llvm::StringRef vlUses;
  llvm::StringRef emitCLoopKind;
  llvm::StringRef emitCLoopInductionName;
  llvm::StringRef emitCFullChunkVLName;
  llvm::StringRef emitCLoopVLName;
  llvm::StringRef remainingAVLMetadata;
  llvm::StringRef pointerAdvanceMetadata;
  llvm::StringRef boundedSlice;
  llvm::StringRef multiVL;
  mlir::Value runtimeAVLValue;
  tianchenrv::support::RuntimeABIParameter runtimeAVLParameter;
};

llvm::StringRef getRVVRuntimeAVLVLControlPlanID();

llvm::Expected<RVVRuntimeAVLVLControlPlan>
deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
    tcrv::exec::VariantOp variant, mlir::Value runtimeAVLValue,
    std::int64_t sew, llvm::StringRef lmul, tcrv::rvv::PolicyAttr policy,
    llvm::StringRef runtimeABIOrder, llvm::StringRef context);

llvm::Expected<RVVRuntimeAVLVLControlPlan>
deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
    tcrv::exec::VariantOp variant, tcrv::rvv::SetVLOp setvl,
    tcrv::rvv::WithVLOp withVL, llvm::StringRef runtimeABIOrder,
    llvm::StringRef context);

llvm::Error verifyRVVRuntimeAVLVLControlPlan(
    const RVVRuntimeAVLVLControlPlan &plan, llvm::StringRef context);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVRUNTIMEAVLVLCONTROL_H
