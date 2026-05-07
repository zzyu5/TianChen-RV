#ifndef TIANCHENRV_TRANSFORMS_VARIANTSELECTION_H
#define TIANCHENRV_TRANSFORMS_VARIANTSELECTION_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>

namespace mlir {
class OpBuilder;
} // namespace mlir

namespace tianchenrv::transforms {

enum class VariantSelectionKind {
  StaticVariant,
  RuntimeDispatch,
  FallbackOnly,
  NoViableVariant,
};

struct VariantSelectionCase {
  tcrv::exec::VariantOp variant;
  plugin::VariantCostEstimate cost;
  std::size_t originalIndex = 0;
  bool genericallyAvailable = false;
  bool hasGenericDecisionMetadata = false;
};

struct VariantSelectionPlan {
  VariantSelectionKind kind = VariantSelectionKind::NoViableVariant;
  tcrv::exec::KernelOp kernel;
  tcrv::exec::VariantOp selectedVariant;
  tcrv::exec::VariantOp fallback;
  bool missingFallbackCoverage = false;
  llvm::SmallVector<VariantSelectionCase, 4> dispatchCases;
  llvm::SmallVector<VariantSelectionCase, 4> rankedVariants;
};

llvm::Expected<VariantSelectionPlan> planKernelVariantSelection(
    tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Expected<VariantSelectionPlan> planKernelVariantSelection(
    tcrv::exec::KernelOp kernel,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Error materializeRuntimeDispatchPlan(
    mlir::OpBuilder &builder, const VariantSelectionPlan &plan,
    tcrv::exec::DispatchOp *createdDispatch = nullptr);

llvm::Error materializeSelectedVariantMarker(
    mlir::OpBuilder &builder, const VariantSelectionPlan &plan,
    tcrv::exec::DiagnosticOp *createdMarker = nullptr);

} // namespace tianchenrv::transforms

#endif // TIANCHENRV_TRANSFORMS_VARIANTSELECTION_H
