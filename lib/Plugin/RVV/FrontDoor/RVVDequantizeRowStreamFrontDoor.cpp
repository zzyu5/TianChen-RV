//===- RVVDequantizeRowStreamFrontDoor.cpp ------------------------------===//
//
// The CERT-FD dequant首族 PRE-EMITC front door. See the header for the full WHY.
//
// In one line: it runs ONLY the CONSTRUCTION half of the streaming dequantize_row
// front door (the shared byte-exact weft::rvv::constructTypedDequantizeRowLoopBody)
// and STOPS at the realized typed region -- BEFORE --weft-rvv-lower-to-emitc -- so
// the certification walker can walk (and hence machine-certify) the constructed
// weft_rvv.typed_dequantize_row_loop_body region. NO emit here; the emit half
// (emitTypedDequantizeRowLoopBody) is byte-exact-unchanged and consumes the region
// when --weft-rvv-lower-to-emitc runs next. Numerical semantics: zero change.
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVDequantizeRowStreamFrontDoor.h"

#include "Weft/Dialect/RVV/IR/RVVDequantizeRowConstruction.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>

namespace weft::plugin::rvv {
namespace {

namespace weftrvv = ::weft::rvv;

class MaterializeRVVDequantizeRowStreamFrontDoorPass final
    : public mlir::PassWrapper<
          MaterializeRVVDequantizeRowStreamFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVDequantizeRowStreamFrontDoorPass() = default;

  llvm::StringRef getArgument() const final {
    return "weft-rvv-materialize-dequantize-row-stream-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Pre-emitc CONSTRUCT the typed weft_rvv.typed_dequantize_row_loop_body "
           "{ dequantize_row_decode_core; typed_dequantize_row_loop_yield } region "
           "in place of each abstract weft_rvv.dequantize_row (one of the 24 "
           "constructed streaming formats) and STOP before --weft-rvv-lower-to-emitc "
           "so the realized region is walkable (the shared byte-exact construction; "
           "the emit half is unchanged). The whole dequantize_row spectrum is now "
           "front-door CONSTRUCTED (q1_0 was the last flat leaf flipped).";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<weftrvv::WEFTRVVDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::IRRewriter rewriter(module.getContext());

    // Collect first, then rewrite: constructTypedDequantizeRowLoopBody erases each
    // abstract op, so mutating during the walk would be unsafe.
    llvm::SmallVector<weftrvv::GgmlDequantizeRowOp> deqOps;
    module.walk(
        [&](weftrvv::GgmlDequantizeRowOp op) { deqOps.push_back(op); });

    for (weftrvv::GgmlDequantizeRowOp deqOp : deqOps) {
      std::optional<weftrvv::DequantizeRowStreamFacts> facts =
          weftrvv::lookupDequantizeRowStreamFacts(deqOp.getFormat());
      if (!facts)
        continue; // unrecognized format: leave abstract (dispatch-wired monolith).
      if (mlir::failed(weftrvv::constructTypedDequantizeRowLoopBody(
              rewriter, deqOp, *facts))) {
        deqOp.emitError()
            << "dequantize_row-stream front door failed to construct the typed "
               "region for decode_model '"
            << deqOp.getFormat() << "'";
        signalPassFailure();
        return;
      }
    }

    // Deliberately stop after construction-owned typed g.  Capability selection
    // is the RVV backend preparation hook's responsibility, shared by the direct
    // wrapper and registry/artifact entry points.  Keeping it out of this plugin
    // front door avoids a Plugin -> Conversion dependency and leaves preselection
    // typed IR inspectable without pretending that an emission target was chosen.
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
      ownerPlugin, "weft-rvv-materialize-dequantize-row-stream-front-door",
      "Pre-emitc construct the typed streaming dequantize_row loop-body region "
      "(weft_rvv.typed_dequantize_row_loop_body { dequantize_row_decode_core; "
      "yield }) in place of the abstract weft_rvv.dequantize_row so the realized "
      "region is walkable before --weft-rvv-lower-to-emitc (the shared byte-exact "
      "construction; the whole dequantize_row spectrum is front-door constructed)",
      [] { return createMaterializeRVVDequantizeRowStreamFrontDoorPass(); },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
