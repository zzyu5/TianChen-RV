//===- RVVDequantizeRowStreamFrontDoor.cpp ------------------------------===//
//
// The CERT-FD dequant首族 PRE-EMITC front door. See the header for the full WHY.
//
// In one line: it runs ONLY the CONSTRUCTION half of the streaming dequantize_row
// front door (the shared byte-exact tcrv::rvv::constructTypedDequantizeRowLoopBody)
// and STOPS at the realized typed region -- BEFORE --tcrv-rvv-lower-to-emitc -- so
// the certification walker can walk (and hence machine-certify) the constructed
// tcrv_rvv.typed_dequantize_row_loop_body region. NO emit here; the emit half
// (emitTypedDequantizeRowLoopBody) is byte-exact-unchanged and consumes the region
// when --tcrv-rvv-lower-to-emitc runs next. Numerical semantics: zero change.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVDequantizeRowStreamFrontDoor.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDequantizeRowConstruction.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>

namespace tianchenrv::plugin::rvv {
namespace {

namespace tcrvrvv = ::tianchenrv::tcrv::rvv;

class MaterializeRVVDequantizeRowStreamFrontDoorPass final
    : public mlir::PassWrapper<
          MaterializeRVVDequantizeRowStreamFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVDequantizeRowStreamFrontDoorPass() = default;

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-dequantize-row-stream-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Pre-emitc CONSTRUCT the typed tcrv_rvv.typed_dequantize_row_loop_body "
           "{ dequantize_row_decode_core; typed_dequantize_row_loop_yield } region "
           "in place of each abstract tcrv_rvv.dequantize_row (one of the 21 "
           "constructed streaming formats) and STOP before --tcrv-rvv-lower-to-emitc "
           "so the realized region is walkable (the shared byte-exact construction; "
           "the emit half is unchanged). tq1_0/tq2_0 stay dispatch-wired.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<tcrvrvv::TCRVRVVDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::IRRewriter rewriter(module.getContext());

    // Collect first, then rewrite: constructTypedDequantizeRowLoopBody erases each
    // abstract op, so mutating during the walk would be unsafe.
    llvm::SmallVector<tcrvrvv::GgmlDequantizeRowOp> deqOps;
    module.walk(
        [&](tcrvrvv::GgmlDequantizeRowOp op) { deqOps.push_back(op); });

    for (tcrvrvv::GgmlDequantizeRowOp deqOp : deqOps) {
      std::optional<tcrvrvv::DequantizeRowStreamFacts> facts =
          tcrvrvv::lookupDequantizeRowStreamFacts(deqOp.getFormat());
      if (!facts)
        continue; // tq1_0 / tq2_0: leave abstract (dispatch-wired monolith).
      if (mlir::failed(tcrvrvv::constructTypedDequantizeRowLoopBody(
              rewriter, deqOp, *facts))) {
        deqOp.emitError()
            << "dequantize_row-stream front door failed to construct the typed "
               "region for decode_model '"
            << deqOp.getFormat() << "'";
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
createMaterializeRVVDequantizeRowStreamFrontDoorPass() {
  return std::make_unique<MaterializeRVVDequantizeRowStreamFrontDoorPass>();
}

llvm::Error registerRVVDequantizeRowStreamFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry & /*registry*/,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin, "tcrv-rvv-materialize-dequantize-row-stream-front-door",
      "Pre-emitc construct the typed streaming dequantize_row loop-body region "
      "(tcrv_rvv.typed_dequantize_row_loop_body { dequantize_row_decode_core; "
      "yield }) in place of the abstract tcrv_rvv.dequantize_row so the realized "
      "region is walkable before --tcrv-rvv-lower-to-emitc (the shared byte-exact "
      "construction; tq1_0/tq2_0 stay dispatch-wired)",
      [] { return createMaterializeRVVDequantizeRowStreamFrontDoorPass(); },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
