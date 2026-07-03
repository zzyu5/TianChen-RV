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
  bool conflictFree = false;
  bool hasGenericDecisionMetadata = false;
  bool requiresRuntimeCapabilityGuard = false;
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

// Build ONE canonical-JSON compile-time selection attribution record (science
// target [D-4] (1)) for a planned kernel: sorted keys, no incidental whitespace,
// candidates in rank order. The returned string is a single JSONL line WITHOUT a
// trailing newline. This is a pure, side-effect-free derivation over the plan +
// the target capability fact set (used to re-derive keys_evaluated and to compute
// the declared-instance-hash), so it is directly unit-testable without running
// the pass or touching a file.
//
// reason is DERIVED (not the in-IR reason attr) and puts the distinction on the
// PRIMARY key: only_feasible = exactly one feasible candidate; static_order = >=2
// feasible chosen by today's capability-blind constant-score cold-start ordering.
// prior (capability-DERIVED prior) and measured (memoized-measurement winner) are
// valid enum values but are NEVER emitted at this selection stage today -- prior
// awaits [SEL-1]/G3, measured awaits [SEL-3]. Every feasible candidate always
// carries its constant ranking score so a static_order decision is fully
// reconstructible from the record.
//
// When noTimestamp is true the ts field is the fixed sentinel "0" for
// byte-deterministic lit output; otherwise it is a wall-clock ISO-8601 string.
std::string buildSelectionAttributionRecord(
    const VariantSelectionPlan &plan,
    const support::TargetCapabilitySet &capabilities, bool noTimestamp);

} // namespace tianchenrv::transforms

#endif // TIANCHENRV_TRANSFORMS_VARIANTSELECTION_H
