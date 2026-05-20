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

struct RVVSelectedBodyConfigVLContract {
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
std::int64_t getRVVSEW64Bits();
llvm::StringRef getRVVLMULM1();
llvm::StringRef getRVVLMULM2();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM1ConfigVLContract();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM2ConfigVLContract();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyI64M1ConfigVLContract();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyI64M2ConfigVLContract();
PolicyAttr getRVVSelectedBodyDefaultPolicy(mlir::MLIRContext *context);
void populateRVVSelectedBodyDefaultConfigAttrs(mlir::Builder &builder,
                                               mlir::OperationState &state);
void populateRVVSelectedBodyConfigAttrs(mlir::Builder &builder,
                                        mlir::OperationState &state,
                                        std::int64_t sew,
                                        llvm::StringRef lmul,
                                        PolicyAttr policy);

bool isRVVFirstSliceDataflowConfig(std::int64_t sew, llvm::StringRef lmul);
bool isRVVSelectedBodyM1Config(std::int64_t sew, llvm::StringRef lmul);
bool isRVVSelectedBodyI64M1Config(std::int64_t sew, llvm::StringRef lmul);
bool isRVVSelectedBodyI64M2Config(std::int64_t sew, llvm::StringRef lmul);
bool isRVVAgnosticPolicy(PolicyAttr policy);

RVVCompileTimeConfig getRVVSetVLCompileTimeConfig(SetVLOp setvl);
std::optional<RVVCompileTimeConfig>
getRVVWithVLCompileTimeConfig(WithVLOp withVL);

bool areRVVCompileTimeConfigsEqual(const RVVCompileTimeConfig &lhs,
                                   const RVVCompileTimeConfig &rhs);

RVVConfigContractDiagnostic
validateRVVSelectedBodyConfigVLStructure(SetVLOp setvl, WithVLOp withVL);
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyConfigVLContract();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyConfigVLContract(llvm::StringRef lmul);
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyConfigVLContract(std::int64_t sew, llvm::StringRef lmul);
RVVConfigContractDiagnostic
validateRVVSelectedBodyM1ConfigVLContract(SetVLOp setvl, WithVLOp withVL);

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyI64RuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyScalarBroadcastRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 3>
getRVVSelectedBodyWideningConversionRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 7>
getRVVSelectedBodyStridedRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyStridedLoadUnitStoreRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyUnitLoadStridedStoreRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyIndexedGatherRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyIndexedScatterRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyMaskedMemoryRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 5>
getRVVSelectedBodyComputedMaskMemoryRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodySegment2DeinterleaveRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodySegment2InterleaveRuntimeABIParameters();
llvm::Error verifyRVVSelectedBodyRuntimeABIParameters(
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameter> parameters,
    llvm::StringRef context);
llvm::StringRef getRVVSelectedBodyRuntimeAVLParameterName();
llvm::StringRef getRVVSelectedBodyEmitCLoopInductionName();
llvm::StringRef getRVVSelectedBodyEmitCFullChunkVLName();
llvm::StringRef getRVVSelectedBodyEmitCLoopVLName();
std::string
getRVVSelectedBodyEmitCRemainingAVLExpression(llvm::StringRef runtimeCountName,
                                              llvm::StringRef inductionName);

llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry>
getRVVSelectedBodyConfigArtifactMetadata();

llvm::Error verifyRVVSelectedBodyConfigArtifactMetadata(
    llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context);

} // namespace tianchenrv::tcrv::rvv

#endif // TIANCHENRV_DIALECT_RVV_IR_RVVCONFIGCONTRACT_H
