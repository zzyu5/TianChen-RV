//===- RVVProbedCapabilityAxesMaterialization.cpp -------------------------===//
//
// Materializes the RVV plugin-local capability authority's derived target-
// support axes (supported_sew / supported_lmul) onto the in-kernel
// tcrv.exec.capability / tcrv.exec.target provider ops that the EmitC legality
// gate already queries. This closes the LIVE probe->gate seam: a selected RVV
// -march (a profile selection) drives the in-IR capability-gate divergence
// automatically, with no hand-authored supported_sew / supported_lmul fixture
// attributes.
//
// The derived values are a TARGET-CAPABILITY support allow-list ("what element
// widths / LMUL groupings the configured target supports"), not a plugin-
// selected compile-time SEW/LMUL config: the typed body still owns its single
// chosen config, and the gate queries the typed capability object this pass
// writes. Per core-invariants:
//   * I1 -- capability stays a first-class queryable object; the pass writes the
//     facts onto the provider op the gate queries, it does not invent a route.
//   * I4 -- the materialized facts MIRROR the plugin-local C++ authority
//     (deriveSupported*AllowList); the authority is the source of truth, the IR
//     attribute is the mirror the gate reads.
//   * I5 -- the axes are derived from the validated ISA tier (the -march /
//     isa-vector-hints evidence), never inferred from ABI strings, family names,
//     route ids, or fabricated selected config; the pass derives NOTHING from
//     clang/cmake/compile-run toolchain facts and probes no hardware.
//   * I7 -- a constrained tier (zve32x) yields a narrower allow-list so an
//     unsupported (SEW=64) body is gated out fail-closed downstream.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZERVVPROBEDCAPABILITYAXES
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

// True iff `op` is an RVV-kind capability/target provider, i.e. one carrying the
// RVV capability id ("rvv") or the RVV capability kind ("isa-vector"). These are
// the providers whose support allow-lists the EmitC legality gate queries.
bool isRVVCapabilityProvider(mlir::Operation *op) {
  llvm::StringRef rvvID = plugin::rvv::getRVVCapabilityID();
  llvm::StringRef rvvKind = plugin::rvv::getRVVCapabilityKind();
  auto id = op->getAttrOfType<mlir::StringAttr>("id");
  if (id && id.getValue() == rvvID)
    return true;
  auto kind = op->getAttrOfType<mlir::StringAttr>("kind");
  if (kind && kind.getValue() == rvvKind)
    return true;
  return false;
}

// Stamps `axisName` = `derived` onto `op` unless the provider already carries
// that axis (so a hand-authored fixture attr is never clobbered) or the derived
// allow-list is empty (the evidence names no concrete RVV element-width tier ->
// no restriction, the historical silent-gate behaviour). Returns true when the
// pass wrote the attribute.
bool materializeAxis(mlir::Operation *op, llvm::StringRef axisName,
                     const std::string &derived) {
  if (derived.empty())
    return false;
  if (op->hasAttrOfType<mlir::StringAttr>(axisName))
    return false;
  op->setAttr(axisName,
              mlir::StringAttr::get(op->getContext(), derived));
  return true;
}

class MaterializeRVVProbedCapabilityAxesPass final
    : public impl::MaterializeRVVProbedCapabilityAxesBase<
          MaterializeRVVProbedCapabilityAxesPass> {
public:
  using impl::MaterializeRVVProbedCapabilityAxesBase<
      MaterializeRVVProbedCapabilityAxesPass>::
      MaterializeRVVProbedCapabilityAxesBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Derive the support allow-lists once from the selected -march (+ optional
    // probed isa/vector hints) through the SAME plugin-local authority the
    // probe->capability conversion uses. No toolchain-probe facts are consulted.
    std::string supportedSEW =
        plugin::rvv::deriveSupportedSEWAllowList(march, isaVectorHints);
    std::string supportedLMUL =
        plugin::rvv::deriveSupportedLMULAllowList(march, isaVectorHints);

    // A march that names no concrete RVV element-width tier derives no axes:
    // nothing to materialize, leave the IR (and the historically silent gate)
    // unchanged.
    if (supportedSEW.empty() && supportedLMUL.empty())
      return;

    module.walk([&](mlir::Operation *op) {
      if (!llvm::isa<tcrv::exec::CapabilityOp, tcrv::exec::TargetOp>(op))
        return;
      if (!isRVVCapabilityProvider(op))
        return;
      materializeAxis(op, "supported_sew", supportedSEW);
      materializeAxis(op, "supported_lmul", supportedLMUL);
    });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeRVVProbedCapabilityAxesPass() {
  return std::make_unique<MaterializeRVVProbedCapabilityAxesPass>();
}

} // namespace tianchenrv::transforms
