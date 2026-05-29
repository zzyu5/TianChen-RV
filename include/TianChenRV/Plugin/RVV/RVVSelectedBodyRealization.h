#ifndef TIANCHENRV_PLUGIN_RVV_RVVSELECTEDBODYREALIZATION_H
#define TIANCHENRV_PLUGIN_RVV_RVVSELECTEDBODYREALIZATION_H

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::plugin::rvv {

struct RVVSelectedBodyRealizationOwner {
  using ConsumerPredicate = bool (*)(mlir::Operation *);
  using RealizationHook = llvm::Expected<tcrv::rvv::WithVLOp> (*)(
      const VariantLoweringBoundaryRequest &, mlir::Operation *);

  llvm::StringLiteral familyName;
  ConsumerPredicate isConsumer = nullptr;
  RealizationHook realize = nullptr;
};

llvm::ArrayRef<RVVSelectedBodyRealizationOwner>
getRVVSelectedBodyRealizationOwners();

llvm::Expected<const RVVSelectedBodyRealizationOwner *>
getRVVSelectedBodyRealizationOwnerForBody(mlir::Operation *bodyOp,
                                          llvm::StringRef context);

bool variantContainsPreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant);

bool variantContainsPreRealizedRVVElementwiseCompareSelectSelectedBody(
    tcrv::exec::VariantOp variant);

struct RVVElementwiseCompareSelectRealizationResult {
  tcrv::rvv::WithVLOp boundary;

  bool applies() const { return static_cast<bool>(boundary); }
};

llvm::Expected<RVVElementwiseCompareSelectRealizationResult>
realizePreRealizedRVVElementwiseCompareSelectCluster(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVElementwiseCompareSelectSelectedBody(
    const VariantLoweringBoundaryRequest &request);

llvm::Error diagnoseRetiredPreRealizedRVVRouteEntrySelectedBody(
    const VariantLoweringBoundaryRequest &request);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVSELECTEDBODYREALIZATION_H
