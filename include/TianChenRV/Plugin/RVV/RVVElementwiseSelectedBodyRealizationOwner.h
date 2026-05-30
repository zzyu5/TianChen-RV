#ifndef TIANCHENRV_PLUGIN_RVV_RVVELEMENTWISESELECTEDBODYREALIZATIONOWNER_H
#define TIANCHENRV_PLUGIN_RVV_RVVELEMENTWISESELECTEDBODYREALIZATIONOWNER_H

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::plugin::rvv {

struct RVVElementwiseCompareSelectRealizationResult {
  tcrv::rvv::WithVLOp boundary;

  bool applies() const { return static_cast<bool>(boundary); }
};

bool isPreRealizedRVVElementwiseCompareSelectClusterOp(mlir::Operation *op);

bool variantContainsPreRealizedRVVElementwiseCompareSelectSelectedBody(
    tcrv::exec::VariantOp variant);

llvm::Expected<RVVElementwiseCompareSelectRealizationResult>
realizePreRealizedRVVElementwiseCompareSelectCluster(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVElementwiseCompareSelectOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVElementwiseCompareSelectSelectedBody(
    const VariantLoweringBoundaryRequest &request);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVELEMENTWISESELECTEDBODYREALIZATIONOWNER_H
