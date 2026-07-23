#ifndef WEFT_PLUGIN_RVV_RVVRUNTIMEAVLVLCONTROL_H
#define WEFT_PLUGIN_RVV_RVVRUNTIMEAVLVLCONTROL_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"
#include "Weft/Support/RuntimeABI.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace weft::plugin::rvv {

/// The construction-time RVV loop control that is still code-affecting after
/// formula evaluation.  It deliberately contains no EmitC spelling, route id,
/// metadata mirror, or artifact contract: body realization consumes these
/// four facts directly to build setvl/with_vl.
struct RVVBodyRuntimeControl {
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  weft::rvv::PolicyAttr policy;
  mlir::Value runtimeAVLValue;
};

llvm::Expected<RVVBodyRuntimeControl> deriveRVVBodyRuntimeControl(
    weft::exec::VariantOp variant, mlir::Value runtimeAVLValue,
    std::int64_t sew, llvm::StringRef lmul, weft::rvv::PolicyAttr policy,
    llvm::StringRef context);

struct RVVRuntimeAVLVLControlPlan {
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  weft::rvv::PolicyAttr policy;
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
  weft::support::RuntimeABIParameter runtimeAVLParameter;
};

struct RVVRuntimeAVLVLSelectedBoundaryContract {
  llvm::StringRef consumerLabel;
  std::int64_t sew = 0;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeVLContractID;
  std::string runtimeAVLABIParameterName;
  std::string runtimeAVLASource;
  std::string runtimeABIOrder;
  std::string selectedBoundaryOpName;
  std::string selectedBodyProvenance;
  std::string vlDefOpName;
  std::string vlScopeOpName;
  std::string vlUses;
  std::string setVLIntrinsic;
  std::string vlCType;
  std::string emitCLoopKind;
  std::string emitCLoopInductionName;
  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string remainingAVLMetadata;
  std::string pointerAdvanceMetadata;
  std::string boundedSlice;
  std::string multiVL;
  weft::support::RuntimeABIParameter runtimeAVLParameter;
};

llvm::StringRef getRVVRuntimeAVLVLControlPlanID();

std::optional<RVVRuntimeAVLVLSelectedBoundaryContract>
getRVVRuntimeAVLVLSelectedBoundaryContract(
    std::int64_t sew, llvm::StringRef lmul, llvm::StringRef tailPolicy,
    llvm::StringRef maskPolicy, llvm::StringRef configContractID,
    llvm::StringRef setVLIntrinsic, llvm::StringRef vlCType,
    llvm::StringRef runtimeABIOrder,
    llvm::ArrayRef<weft::support::RuntimeABIParameter>
        runtimeABIParameters,
    llvm::StringRef consumerLabel);

llvm::Expected<RVVRuntimeAVLVLControlPlan>
deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
    weft::exec::VariantOp variant, mlir::Value runtimeAVLValue,
    std::int64_t sew, llvm::StringRef lmul, weft::rvv::PolicyAttr policy,
    llvm::StringRef runtimeABIOrder, llvm::StringRef context);

llvm::Expected<RVVRuntimeAVLVLControlPlan>
deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
    weft::exec::VariantOp variant, weft::rvv::SetVLOp setvl,
    weft::rvv::WithVLOp withVL, llvm::StringRef runtimeABIOrder,
    llvm::StringRef context);

llvm::Error verifyRVVRuntimeAVLVLControlPlan(
    const RVVRuntimeAVLVLControlPlan &plan, llvm::StringRef context);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVRUNTIMEAVLVLCONTROL_H
