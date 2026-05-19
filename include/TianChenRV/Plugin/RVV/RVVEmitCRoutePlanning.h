#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPLANNING_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPLANNING_H

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Support/RuntimeABIContract.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::plugin::rvv {

struct RVVSelectedBodyRouteSlice {
  tcrv::rvv::SetVLOp setvl;
  tcrv::rvv::WithVLOp withVL;
  tcrv::rvv::LoadOp lhsGenericLoad;
  tcrv::rvv::LoadOp rhsGenericLoad;
  tcrv::rvv::StridedLoadOp lhsStridedLoad;
  tcrv::rvv::StridedLoadOp rhsStridedLoad;
  tcrv::rvv::BroadcastLoadOp rhsBroadcastLoad;
  tcrv::rvv::SplatOp rhsScalarSplat;
  tcrv::rvv::CompareOp compareOp;
  tcrv::rvv::SelectOp selectOp;
  tcrv::rvv::ReduceOp reduceOp;
  tcrv::rvv::MaskedBinaryOp maskedBinaryOp;
  tcrv::rvv::MAccOp maccOp;
  mlir::Operation *arithmeticOp = nullptr;
  mlir::Value arithmeticLhs;
  mlir::Value arithmeticRhs;
  mlir::Value arithmeticAccumulator;
  mlir::Value arithmeticResult;
  mlir::Value compareLhs;
  mlir::Value compareRhs;
  mlir::Value compareMask;
  mlir::Value maskedPassthrough;
  RVVSelectedBodyOperationKind arithmeticKind;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  tcrv::rvv::StoreOp genericStore;
  tcrv::rvv::StridedStoreOp stridedStore;
  mlir::Operation *lhsLoadOperation = nullptr;
  mlir::Operation *rhsLoadOperation = nullptr;
  mlir::Operation *accumulatorLoadOperation = nullptr;
  mlir::Operation *storeOperation = nullptr;
  mlir::Value lhsBuffer;
  mlir::Value rhsBuffer;
  mlir::Value accumulatorBuffer;
  mlir::Value outBuffer;
  mlir::Value lhsStride;
  mlir::Value rhsStride;
  mlir::Value outStride;
  mlir::Value lhsValue;
  mlir::Value rhsValue;
  mlir::Value accumulatorValue;
  mlir::Value storeValue;
  support::RuntimeABIParameter lhsABI;
  support::RuntimeABIParameter rhsABI;
  support::RuntimeABIParameter accumulatorABI;
  support::RuntimeABIParameter outABI;
  support::RuntimeABIParameter runtimeElementCountABI;
  support::RuntimeABIParameter lhsStrideABI;
  support::RuntimeABIParameter rhsStrideABI;
  support::RuntimeABIParameter outStrideABI;
};

struct RVVSelectedBodyRouteAnalysis {
  RVVSelectedBodyRouteSlice slice;
  const RVVSelectedBodyConstructionRoute *constructionRoute = nullptr;
  RVVSelectedBodyEmitCRouteDescription description;
};

llvm::Error makeRVVEmitCRouteProviderError(llvm::Twine message);

llvm::Expected<RVVSelectedBodyRouteAnalysis>
analyzeRVVSelectedBodyRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPLANNING_H
