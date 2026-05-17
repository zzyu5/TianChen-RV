#ifndef TIANCHENRV_DIALECT_RVV_IR_RVVCONFIGCONTRACT_H
#define TIANCHENRV_DIALECT_RVV_IR_RVVCONFIGCONTRACT_H

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Support/ArtifactMetadata.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::tcrv::rvv {

struct RVVCompileTimeConfig {
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  PolicyAttr policy;
};

struct RVVI32M1ArithmeticConfigVLContract {
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  TailPolicy tailPolicy = TailPolicy::Agnostic;
  MaskPolicy maskPolicy = MaskPolicy::Agnostic;
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
  llvm::StringRef remainingAVLMetadata;
  llvm::StringRef pointerAdvanceMetadata;
  llvm::StringRef boundedSlice;
  llvm::StringRef multiVL;
};

struct RVVConfigContractDiagnostic {
  bool ok = true;
  std::string message;

  static RVVConfigContractDiagnostic success();
  static RVVConfigContractDiagnostic failure(llvm::StringRef message);
};

std::int64_t getRVVFirstSliceSEWBits();
llvm::StringRef getRVVI32M1LMUL();
llvm::StringRef getRVVI32M2LMUL();
const RVVI32M1ArithmeticConfigVLContract &
getRVVI32M1ArithmeticConfigVLContract();
PolicyAttr getRVVI32M1ArithmeticPolicy(mlir::MLIRContext *context);
void populateRVVI32M1ArithmeticConfigAttrs(mlir::Builder &builder,
                                           mlir::OperationState &state);

bool isRVVFirstSliceDataflowConfig(std::int64_t sew, llvm::StringRef lmul);
bool isRVVI32M1ArithmeticConfig(std::int64_t sew, llvm::StringRef lmul);
bool isRVVAgnosticPolicy(PolicyAttr policy);

RVVCompileTimeConfig getRVVSetVLCompileTimeConfig(SetVLOp setvl);
std::optional<RVVCompileTimeConfig>
getRVVWithVLCompileTimeConfig(WithVLOp withVL);

bool areRVVCompileTimeConfigsEqual(const RVVCompileTimeConfig &lhs,
                                   const RVVCompileTimeConfig &rhs);

RVVConfigContractDiagnostic
validateRVVI32M1ArithmeticConfigVLContract(SetVLOp setvl, WithVLOp withVL);

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVI32M1ArithmeticRuntimeABIParameters();
llvm::Error verifyRVVI32M1ArithmeticRuntimeABIParameters(
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameter> parameters,
    llvm::StringRef context);
llvm::StringRef getRVVI32M1ArithmeticRuntimeAVLParameterName();
llvm::StringRef getRVVI32M1ArithmeticEmitCLoopInductionName();
llvm::StringRef getRVVI32M1ArithmeticEmitCFullChunkVLName();
llvm::StringRef getRVVI32M1ArithmeticEmitCLoopVLName();
std::string getRVVI32M1ArithmeticEmitCRemainingAVLExpression(
    llvm::StringRef runtimeCountName, llvm::StringRef inductionName);

llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry>
getRVVI32M1ArithmeticArtifactMetadata();

llvm::Error verifyRVVI32M1ArithmeticArtifactMetadata(
    llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context);

} // namespace tianchenrv::tcrv::rvv

#endif // TIANCHENRV_DIALECT_RVV_IR_RVVCONFIGCONTRACT_H
