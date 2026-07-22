//===- RVVElementwiseStreamFrontDoor.cpp --------------------------------===//
//
// The CERT-FD forward殿后族 PRE-EMITC front door. See the header for the full WHY.
//
// In one line: it runs ONLY the CONSTRUCTION half of the streaming
// forward-elementwise front door (the shared byte-exact
// weft::rvv::constructTypedElementwiseLoopBody) and STOPS at the realized typed
// region -- BEFORE --weft-rvv-lower-to-emitc -- so the certification walker can walk
// (and hence machine-certify) the constructed weft_rvv.typed_elementwise_loop_body
// region. NO emit here; the emit half (emitTypedElementwiseLoopBody) is byte-exact
// unchanged and consumes the region when --weft-rvv-lower-to-emitc runs next.
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVElementwiseStreamFrontDoor.h"
#include "Weft/Plugin/RVV/RVVFormulaCatalog.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Dialect/RVV/IR/RVVElementwiseStreamConstruction.h"
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

class MaterializeRVVElementwiseStreamFrontDoorPass final
    : public mlir::PassWrapper<
          MaterializeRVVElementwiseStreamFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVElementwiseStreamFrontDoorPass() = default;

  llvm::StringRef getArgument() const final {
    return "weft-rvv-materialize-forward-elementwise-stream-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Pre-emitc CONSTRUCT the typed weft_rvv.typed_elementwise_loop_body "
           "{ <map/reduce/rotate core brick>; typed_elementwise_loop_yield } region "
           "in place of each abstract weft_rvv.ggml_forward_elementwise (one of the "
           "5 constructed forward operators scale/silu/rms_norm/soft_max/rope) and "
           "STOP before --weft-rvv-lower-to-emitc so the realized region is walkable "
           "(the shared byte-exact construction; the emit half is unchanged).";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<weftrvv::WEFTRVVDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::IRRewriter rewriter(module.getContext());

    // Collect first, then rewrite: constructTypedElementwiseLoopBody erases each
    // abstract op, so mutating during the walk would be unsafe.
    llvm::SmallVector<weftrvv::GgmlForwardElementwiseOp> fwdOps;
    module.walk(
        [&](weftrvv::GgmlForwardElementwiseOp op) { fwdOps.push_back(op); });

    for (weftrvv::GgmlForwardElementwiseOp fwdOp : fwdOps) {
      std::optional<weftrvv::ForwardElementwiseFacts> facts =
          weftrvv::lookupForwardElementwiseFacts(fwdOp.getElementwiseModel());
      if (!facts)
        continue; // an unknown model: leave abstract.
      if (mlir::failed(weftrvv::constructTypedElementwiseLoopBody(rewriter, fwdOp,
                                                                 *facts))) {
        fwdOp.emitError()
            << "forward-elementwise-stream front door failed to construct the "
               "typed region for elementwise_model '"
            << fwdOp.getElementwiseModel() << "'";
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
createMaterializeRVVElementwiseStreamFrontDoorPass() {
  return std::make_unique<MaterializeRVVElementwiseStreamFrontDoorPass>();
}

llvm::Error registerRVVElementwiseStreamFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry & /*registry*/,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin, formula_catalog::kElementwiseSourceEntry,
      "Pre-emitc construct the typed streaming forward-elementwise loop-body "
      "region (weft_rvv.typed_elementwise_loop_body { <map/reduce/rotate core "
      "brick>; yield }) in place of the abstract weft_rvv.ggml_forward_elementwise "
      "so the realized region is walkable before --weft-rvv-lower-to-emitc (the "
      "shared byte-exact construction; scale/silu/rms_norm/soft_max/rope)",
      formula_catalog::kElementwiseConstruction,
      [] { return createMaterializeRVVElementwiseStreamFrontDoorPass(); },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
