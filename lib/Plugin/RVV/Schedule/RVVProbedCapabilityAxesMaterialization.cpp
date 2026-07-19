//===- RVVProbedCapabilityAxesMaterialization.cpp -------------------------===//
//
// Materializes the RVV plugin-local capability authority's derived target-
// support axes (supported_sew / supported_lmul) onto the in-kernel
// weft.exec.capability / weft.exec.target provider ops that the EmitC legality
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

#include "Weft/Transforms/Passes.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
#include "Weft/Plugin/RVV/RVVExtensionPlugin.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <string>

namespace weft::transforms {

#define GEN_PASS_DEF_MATERIALIZERVVPROBEDCAPABILITYAXES
#include "Weft/Transforms/Passes.h.inc"

namespace {

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
    // The RVV ISA generation (the deepest N1 axis): RVV0.7 (xtheadvector / C920)
    // vs RVV1.0 (rv64gcv). The version stamp drives the legality gate's
    // ratified-policy (ta/ma) divergence -- an agnostic-policy body is RVV1.0-
    // only, so it is gated out on an rvv_version=0.7 provider. "" (Unknown) ->
    // no fact, mirroring the empty-allow-list silent-gate behaviour.
    std::string rvvVersion =
        plugin::rvv::stringifyRVVVersion(
            plugin::rvv::deriveRVVVersion(march, isaVectorHints))
            .str();

    // The guaranteed minimum VLEN (bits): the TYPED quantitative capability fact
    // the resource-aware consumers (repack strip width, block-dot schedule, the
    // front-door bridges) read back off this provider op INSTEAD of re-parsing
    // -march locally. This is the ONE producer of the minimum-VLEN fact; -march is
    // parsed here and the divergence flows through the typed provider attribute
    // (I1/I4). 0 (no concrete >= 128 floor, e.g. an embedded zve32x tier) -> no
    // fact, mirroring the empty-allow-list silent skip: the consumer then leaves
    // any hand-authored width intact.
    std::int64_t minimumVLEN =
        plugin::rvv::deriveMinimumVLEN(march, isaVectorHints);

    // A march that names no concrete RVV tier derives no axes AND no version AND
    // no VLEN floor: nothing to materialize, leave the IR (and the historically
    // silent gate) unchanged.
    if (supportedSEW.empty() && supportedLMUL.empty() && rvvVersion.empty() &&
        minimumVLEN <= 0)
      return;

    module.walk([&](mlir::Operation *op) {
      if (!plugin::rvv::isRVVCapabilityProvider(op))
        return;
      materializeAxis(op, "supported_sew", supportedSEW);
      materializeAxis(op, "supported_lmul", supportedLMUL);
      materializeAxis(op, "rvv_version", rvvVersion);
      // The minimum-VLEN fact is a TYPED i64 IntegerAttr (not a string mirror):
      // the resource-aware consumers reason over it numerically. No-clobber: a
      // hand-authored minimum_vlen (the decisive-experiment conflict fixture)
      // wins, so the provider fact -- not -march -- drives the divergence.
      llvm::StringRef vlenName =
          plugin::rvv::getRVVMinimumVLENProviderPropertyName();
      if (minimumVLEN > 0 && !op->hasAttrOfType<mlir::IntegerAttr>(vlenName))
        op->setAttr(vlenName,
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(op->getContext(), 64),
                        minimumVLEN));
    });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeRVVProbedCapabilityAxesPass() {
  return std::make_unique<MaterializeRVVProbedCapabilityAxesPass>();
}

} // namespace weft::transforms
