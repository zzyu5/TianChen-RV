//===- RVVProbedCapabilityAxesMaterialization.cpp -------------------------===//
//
// Materializes the RVV plugin-local capability authority's derived target-
// support axes (supported_sew / supported_lmul) onto the in-kernel
// weft.exec.capability / weft.exec.target provider ops that family-local
// construction legality queries. This closes the LIVE probe->gate seam: a selected RVV
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

class MaterializeRVVProbedCapabilityAxesPass final
    : public impl::MaterializeRVVProbedCapabilityAxesBase<
          MaterializeRVVProbedCapabilityAxesPass> {
public:
  using impl::MaterializeRVVProbedCapabilityAxesBase<
      MaterializeRVVProbedCapabilityAxesPass>::
      MaterializeRVVProbedCapabilityAxesBase;

  void runOnOperation() override {
    // The four support axes (supported_sew / supported_lmul / rvv_version +
    // typed minimum_vlen) are materialized through the ONE plugin-local producer
    // (materializeRVVProviderCapabilityAxes) so this probe pass and the RVV
    // source front doors share a single stamping home (byte-identical to the
    // prior in-pass logic: same derivations, same no-clobber, same empty-skip).
    (void)plugin::rvv::materializeRVVProviderCapabilityAxes(getOperation(), march,
                                                            isaVectorHints);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeRVVProbedCapabilityAxesPass() {
  return std::make_unique<MaterializeRVVProbedCapabilityAxesPass>();
}

} // namespace weft::transforms
