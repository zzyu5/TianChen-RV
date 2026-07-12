#ifndef WEFT_PLUGIN_RVV_RVVELEMENTWISESELECTEDBODYREALIZATIONOWNER_H
#define WEFT_PLUGIN_RVV_RVVELEMENTWISESELECTEDBODYREALIZATIONOWNER_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/Error.h"

namespace weft::plugin::rvv {

struct RVVElementwiseCompareSelectRealizationResult {
  weft::rvv::WithVLOp boundary;

  bool applies() const { return static_cast<bool>(boundary); }
};

bool isPreRealizedRVVElementwiseCompareSelectClusterOp(mlir::Operation *op);

bool variantContainsPreRealizedRVVElementwiseCompareSelectSelectedBody(
    weft::exec::VariantOp variant);

llvm::Expected<RVVElementwiseCompareSelectRealizationResult>
realizePreRealizedRVVElementwiseCompareSelectCluster(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVElementwiseCompareSelectOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVElementwiseCompareSelectSelectedBody(
    const VariantLoweringBoundaryRequest &request);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVELEMENTWISESELECTEDBODYREALIZATIONOWNER_H
