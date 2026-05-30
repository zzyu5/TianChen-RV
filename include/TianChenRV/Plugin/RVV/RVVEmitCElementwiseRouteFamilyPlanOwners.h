#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCELEMENTWISEROUTEFAMILYPLANOWNERS_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCELEMENTWISEROUTEFAMILYPLANOWNERS_H

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace tianchenrv::plugin::rvv {

struct RVVElementwiseCompareSelectRealizationResult {
  tcrv::rvv::WithVLOp boundary;

  bool applies() const { return static_cast<bool>(boundary); }
};

bool isPreRealizedRVVElementwiseCompareSelectClusterOp(mlir::Operation *op);

llvm::Expected<RVVElementwiseCompareSelectRealizationResult>
realizePreRealizedRVVElementwiseCompareSelectCluster(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVElementwiseCompareSelectOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVElementwiseCompareSelectSelectedBody(
    const VariantLoweringBoundaryRequest &request);

bool isRVVSelectedBodyElementwiseArithmeticRouteOperation(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyPlainElementwiseArithmeticRouteOperation(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyMaskedElementwiseArithmeticRouteOperation(
    RVVSelectedBodyOperationKind operation);

bool isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation,
    RVVSelectedBodyMemoryForm memoryForm);

llvm::Expected<RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan>
deriveRVVSelectedBodyElementwiseArithmeticRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis);

void applyRVVSelectedBodyElementwiseArithmeticRouteFamilyPlan(
    const RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);

llvm::Error validateRVVSelectedBodyElementwiseArithmeticRouteFamilyPlan(
    const RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan &plan);

bool isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
    RVVSelectedBodyOperationKind operation);

llvm::Expected<RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan>
deriveRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis);

void applyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
    const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description);

llvm::Error validateRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
    const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan &plan);

llvm::Error verifyRVVSelectedBodyElementwiseRouteDescriptionMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyElementwiseRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation);

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodyElementwiseRouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand);

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyElementwiseRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCELEMENTWISEROUTEFAMILYPLANOWNERS_H
