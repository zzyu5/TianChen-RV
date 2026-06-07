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
std::int64_t getRVVSEW8Bits();
std::int64_t getRVVSEW16Bits();
std::int64_t getRVVSEW64Bits();
llvm::StringRef getRVVLMULMF4();
llvm::StringRef getRVVLMULMF2();
llvm::StringRef getRVVLMULM1();
llvm::StringRef getRVVLMULM2();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyI16MF2ConfigVLContract();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM1ConfigVLContract();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM2ConfigVLContract();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyI64M1ConfigVLContract();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyI64M2ConfigVLContract();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM1UndisturbedConfigVLContract();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM2UndisturbedConfigVLContract();
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyI64M1UndisturbedConfigVLContract();
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
bool isRVVUndisturbedPolicy(PolicyAttr policy);

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
const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyConfigVLContract(std::int64_t sew, llvm::StringRef lmul,
                                   PolicyAttr policy);
RVVConfigContractDiagnostic
validateRVVSelectedBodyM1ConfigVLContract(SetVLOp setvl, WithVLOp withVL);

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyI64RuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyScalarBroadcastRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 5>
getRVVSelectedBodyScalarBroadcastMAccRuntimeABIParameters();

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeSplatStoreRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 3>
getRVVSelectedBodyWideningConversionRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 3>
getRVVSelectedBodyWidenI16ToI32RuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyDequantizationRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyDequantClampF32EpilogueRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyWideningProductRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 5>
getRVVSelectedBodyWideningProductReductionRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyWideningProductReductionDequantizationRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
getRVVSelectedBodyWideningProductReductionDequantClampF32RuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 5>
getRVVSelectedBodyMAccRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 7>
getRVVSelectedBodyComputedMaskMAccRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 7>
getRVVSelectedBodyRuntimeScalarComputedMaskMAccRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherMAccScatterRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 5>
getRVVSelectedBodyWideningMAccRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyStandaloneReductionRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyWideningStandaloneReductionRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskStandaloneReductionRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
buildRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters(
    llvm::StringRef elementCType);
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 7>
getRVVSelectedBodyStridedInputWideningDotReduceRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 7>
getRVVSelectedBodyComputedMaskWideningDotReduceRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 9>
getRVVSelectedBodyComputedMaskStridedInputWideningDotReduceRuntimeABIParameters();
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
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskSelectRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
buildRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters(
    llvm::StringRef elementCType);
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
buildRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters(
    llvm::StringRef elementCType);
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
getRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 5>
getRVVSelectedBodyRuntimeScalarF32ClampSelectRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 5>
buildRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters(
    llvm::StringRef elementCType);
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 5>
getRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskStridedStoreRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskStridedLoadRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskIndexedGatherRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskIndexedScatterRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyRuntimeScalarComputedMaskIndexedScatterRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskSegment2LoadRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyRuntimeScalarComputedMaskSegment2LoadRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskSegment2StoreRuntimeABIParameters();
llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 6>
getRVVSelectedBodyRuntimeScalarComputedMaskSegment2StoreRuntimeABIParameters();
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
